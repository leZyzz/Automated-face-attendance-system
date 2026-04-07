import numpy as np
import cv2
from PySide6.QtCore import Signal,  Slot , QObject
from PySide6.QtGui import QImage
from ai_models.ArcFaceOnnx import ArcFaceEmbedderOnnx
from ai_models.FaceNetOnnx import FaceNetEmbedderOnnx
from ai_models.YoloOnnx import YoloOnnx

   
   


class AiWorker(QObject) :
    frame_ready = Signal(QImage)
    status_update = Signal(str)
    registration_status= Signal(str)
    
    def __init__(self,db) : 
        
        super().__init__()
        self.db = db
        self.active = True
        self.mode = 0 #attendance 
    def start_registration(self,name) : 

        self.mode = 1 #registration
        self.person_name = name 
        self.collected_embeddings = []
        self.registration_status.emit("please look straight at the camera")
    def check_quality(self,box,frame_shape) : 
        x1,y1,x2,y2 = box[0]
        h , w ,_=frame_shape

        box_width = x2-x1
        box_height=y2-y1

        # RULE 1 : must be big enough 
        if box_width<120 or box_height<120 : 
            return False,"move closer to the camera"
        #RULE 2 : Must be roughly centered 
        cx = x1 - (box_width/2)
        if cx < (w*0.2) or cx>(w*0.8) : 
             return False,"center your face"
        return True , "hold still...."
  

    def cos_sim(self,a,b):
            return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def recognize_face(self,emb, database, threshold=0.7):
        """
        Compares the query embedding (emb) against all stored embeddings 
        in the database using vectorized NumPy operations.
        """
        if emb is None: 
            return "unknown"
        
        # 1. Prepare the Query Vector (q)
        # Reshape the 1D query vector (e.g., shape (512,)) into a 
        # 2D row vector (shape (1, 512)) for matrix multiplication.
        emb_norm = np.linalg.norm(emb)
        
        # Initialize best score starting at the threshold to avoid unnecessary updates
        best_score, best_name = threshold, "unknown" 

        # We still loop over names, but the comparison for each name is vectorized
        for name, info in database.items():
            if not info['embeddings'] :
                continue
                
            # 2. Prepare the Database Matrix (D)
            # Stack all embeddings for the current person into a single matrix.
            # Shape becomes (N, 512), where N is the number of samples for that person.
            embeddings_matrix = np.array(info["embeddings"])
            
            # 3. Calculate Dot Products (Numerator: q * D^T)
            # np.dot(1x512, 512xN) -> result is 1xN array of dot products
            # Note: embeddings_matrix.T performs the transpose (D^T)
            dot_products = np.dot(emb, embeddings_matrix.T)
            
            # 4. Calculate Norms (Denominator: ||D||)
            # Calculate the L2-norm for every row (embedding) in the matrix.
            matrix_norms = np.linalg.norm(embeddings_matrix, axis=1)
            
            # 5. Calculate Cosine Similarities
            # The numerator (dot_products) is divided by the product of the two norms.
            # NumPy automatically handles broadcasting the division.
            # We flatten the 1xN result back to 1D.
            sim_scores = dot_products.flatten() / (emb_norm * matrix_norms)
            
            # 6. Find the best match for this person
            max_sim_score = np.max(sim_scores) if sim_scores.size > 0 else 0
            
            # 7. Update the overall best match
            if max_sim_score > best_score:
                best_score = max_sim_score
                best_name = name

        # Return the name only if the best score meets the threshold
        return best_name 
    
    def cvt2Qimage(self,img) : 
        rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h , w , ch = rgb_img.shape
        bytes_per_line = ch * w 
        Qformat_img = QImage(rgb_img.data,w,h,bytes_per_line,QImage.Format_RGB888)
        return Qformat_img.copy()
    @Slot()
    def start_camera_loop(self):
        yolo = YoloOnnx()
        embedder = FaceNetEmbedderOnnx()
        cap = cv2.VideoCapture(0)
    
        while self.active:
            success, frame = cap.read()

            if not success:
                self.status_update.emit("Failed to get frames")
                break

            results = yolo.detect(frame)
            if self.mode == 1 :
                try : 
                    if(len(results)==0) : 
                        self.registration_status.emit("No face detected ")
                    elif(len(results)>1) :
                        self.registration_status.emit("Multiple faces detected ")
                    else : 
                        is_good , message = self.check_quality(results,frame.shape)
                        self.registration_status.emit(message)
                        if is_good : 
                            x1, y1, x2, y2 = results[0]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            
                            # Get embedding and save it
                            face_crop = frame[max(0, y1-10):min(frame.shape[0], y2+10), 
                                            max(0, x1-10):min(frame.shape[1], x2+10)]
                            emb = embedder.get_embedding(face_crop)
                            self.collected_embeddings.append(emb)
                            
                            self.registration_status.emit(f"Captured {len(self.collected_embeddings)}/5")
                            
                            # Once we have 5, finish the registration!
                            if len(self.collected_embeddings) == 5:
                                # Calculate the mathematical average
                        
                                master_embedding = np.mean(self.collected_embeddings, axis=0)
                                
                                # Save to your DB dictionary
                                self.db[self.person_name] = master_embedding.tolist()
                                
                                self.registration_status.emit("Registration Complete!")
                                self.mode = 0
                                
                            self.frame_ready.emit(self.cvt2Qimage(frame))
                except Exception as e :

                    print(f"failed due to {e}")
                    self.mode = 0 

            elif self.mode == 0 :
                for box in results:
                    x1, y1, x2, y2 = box
                    h,w,_=frame.shape
                    face_crop = frame[max(0,y1-10):min(h,y2+10), max(0,x1-10):min(w,x2+10)]

                    if face_crop.size == 0:
                        continue

                    emb = embedder.get_embedding(face_crop)

                    if emb is not None : 
                        name = self.recognize_face(emb, self.db, threshold=0.6)
                    else : name = "unknown"
                    color = (0,255,0) if name != "unknown" else (0,0,255)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                self.frame_ready.emit(self.cvt2Qimage(frame))
        cap.release()
    @Slot()                
    def stop(self) : 
            self.active = False