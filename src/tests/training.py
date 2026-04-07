from ultralytics import YOLO

def train_face_model():
    model = YOLO("yolo26n.pt")

    results = model.train(
        # Use your exact path with forward slashes here
        data="C:/Users/YourName/Documents/innovatiana_faces/data.yaml", 
        epochs=300,
        patience=25,
        imgsz=640,
        batch=16,
        device="0", 
        project="attendance_system",
        name="yolo26_faces",
        save=True
    )

if __name__ == "__main__":
    train_face_model()