from facenet_pytorch import  InceptionResnetV1 
from ultralytics import YOLO
import torch 
import cv2 
import numpy as np

# detect the device being used

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## face detection model 

yolo = YOLO("yolov8nbest.pt")


# embedding generator model 

facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)



known_faces = {}

## to generate embedding from cropped face 
def get_face_emb(face_crop):
    if face_crop is None or face_crop.size == 0:
        return None

    img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5   

    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad() :
        embedding = facenet(tensor)
    return embedding.cpu().numpy()[0]


def cos_sim(a,b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_face_optimized(emb, database, threshold=0.7):
    """
    Compares the query embedding (emb) against all stored embeddings 
    in the database using vectorized NumPy operations.
    """
    if emb is None: 
        return "unknown"
    
    # 1. Prepare the Query Vector (q)
    # Reshape the 1D query vector (e.g., shape (512,)) into a 
    # 2D row vector (shape (1, 512)) for matrix multiplication.
    emb_vector = emb.reshape(1, -1)
    emb_norm = np.linalg.norm(emb_vector)
    
    # Initialize best score starting at the threshold to avoid unnecessary updates
    best_score, best_name = threshold, "unknown" 

    # We still loop over names, but the comparison for each name is vectorized
    for name, embeddings_list in database.items():
        if not embeddings_list:
            continue
            
        # 2. Prepare the Database Matrix (D)
        # Stack all embeddings for the current person into a single matrix.
        # Shape becomes (N, 512), where N is the number of samples for that person.
        embeddings_matrix = np.array(embeddings_list)
        
        # 3. Calculate Dot Products (Numerator: q * D^T)
        # np.dot(1x512, 512xN) -> result is 1xN array of dot products
        # Note: embeddings_matrix.T performs the transpose (D^T)
        dot_products = np.dot(emb_vector, embeddings_matrix.T)
        
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
## comparing the ebmedding with existing ones 
def recognize_face(emb, database, threshold=0.65):
    if emb is None : 
        return "unknown"
    best_score, best_name = 0, "unknown"

    for name, embeddings in database.items():
        for embedding in embeddings:
            if embedding is None : 
                continue
            sim_score = cos_sim(emb, embedding)
            if sim_score > best_score:
                best_score = sim_score
                best_name = name

    return best_name if best_score >= threshold else "unknown"

# main loop , detection , crop , recognition 

def main():

    cap = cv2.VideoCapture(0)
   
    while True:
        success, frame = cap.read()

        if not success:
            print("Failed getting frames")
            break

        results = yolo(frame, imgsz=416, conf=0.5,verbose=False,show=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            
            face_crop = frame[y1+2:y2+2, x1+2:x2+2]

            if face_crop.size == 0:
                continue

            emb = get_face_emb(face_crop)

            if emb is not None : 
                name = recognize_face_optimized(emb, known_faces, threshold=0.6)
            else : name = "unknown"
            color = (0,255,0) if name != "unknown" else (0,0,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("face Detection & Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        ## adding new face 

        if key == ord('a'):

            if len(results.boxes) == 1:
                print("Adding new face")
                new_person = input("Enter name : ")

                box = results.boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                face_crop = frame[y1:y2, x1:x2]
                emb = get_face_emb(face_crop)
                if face_crop is None :
                    continue
                if new_person not in known_faces:
                    known_faces[new_person] = []
                known_faces[new_person].append(emb)


                print(f"{new_person} added successfully")
                print(f"this person has {len(known_faces[new_person])} embeddings")

            else:
                print("Cannot add multiple faces at once!")
        # to quit 
        
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ =="__main__" : 
    main()
