import numpy as np
import cv2
from PySide6.QtCore import Signal,  Slot , QObject
from PySide6.QtGui import QImage
from ai_models.FaceNetOnnx import FaceNetEmbedderOnnx
from ai_models.InsightOnnx import ScrfdOnnx
import time
from databaseSection.DatabaseManager import DatabaseManager

   


class AiWorker(QObject) :
    frame_ready = Signal(QImage)
    status_update = Signal(str)
    registration_status= Signal(str)
    
    def __init__(self,db:DatabaseManager) : 
        
        super().__init__()
        self.detector = ScrfdOnnx()
        self.embedder = FaceNetEmbedderOnnx()
        self.db = db
        self.student_dict = self.rearrange_db(self.db.students_embeddings)
        self.active = True
         
    
    
    
    def rearrange_db(self,students_embs:list[tuple[str]])  : 
         result: dict[str, list[np.ndarray]] = {}
         for name,lastname,blob in students_embs :
              key = f"{name} {lastname}" 
              if key not in result : 
                   result[key] = []
                
              embedding = np.frombuffer(blob,dtype=np.float32)
              result[key].append(embedding)

         return result 


              
               
         
    def recognize_face(self,emb, database, threshold=0.8):
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
        for name, embeddings in database.items():
            if not embeddings:
                continue
                
            # 2. Prepare the Database Matrix (D)
            # Stack all embeddings for the current person into a single matrix.
            # Shape becomes (N, 512), where N is the number of samples for that person.
            embeddings_matrix = np.array(embeddings)
            
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
      
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        prev_time = 0
        frame_count = 0 
        inference_every = 3 
        last_boxes , last_names = [] , []
        while self.active:
            success, frame = cap.read()

            if not success:
                self.status_update.emit("Failed to get frames")
                break
            frame_count +=1 
            current_time = time.time()
            fps = 1 /(current_time - prev_time)
            prev_time = current_time
            if frame_count % inference_every == 0 : 
                boxes, kps = self.detector.detect(frame)
          
                names = []
                for box in boxes:
                        x1, y1, x2, y2 = box
                        h,w,_=frame.shape
                        face_crop = frame[max(0,y1-10):min(h,y2+10), max(0,x1-10):min(w,x2+10)]

                        if face_crop.size == 0:
                            names.append("unknown")
                            continue

                        emb = self.embedder.get_embedding(face_crop)

                        if emb is not None : 
                            name = self.recognize_face(emb, self.student_dict, threshold=0.65)
                        else : name = "unknown"
                        names.append(name)
                last_boxes,last_names = boxes,names
            for box , name in zip(last_boxes,last_names) : 
                    x1, y1, x2, y2 = box
                    color = (0,255,0) if name != "unknown" else (0,0,255)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
            cv2.putText(frame,f"{int(fps)} FPS",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1.0, (0,255,0),2)

            self.frame_ready.emit(self.cvt2Qimage(frame))
        cap.release()
    @Slot()                
    def stop(self) : 
            self.active = False