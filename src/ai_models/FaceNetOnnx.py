
import cv2 
import numpy as np 
import onnxruntime as ort
from pathlib import Path 

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "assets" / "models"
FACENET_PATH = MODELS_DIR / "facenet.onnx"


class FaceNetEmbedderOnnx() :
    def __init__(self,model_path=FACENET_PATH) :
        
        providers=["CUDAExecutionProvider","CPUExecutionProvider"]
        if not model_path.exists() : 
            raise FileNotFoundError(f"Missing model file at {model_path}")
        
        self.session = ort.InferenceSession(model_path,providers=providers)

    def preprocess(self,face_crop) :

        img = cv2.resize(face_crop,(160,160))

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img =(img.astype(np.float32)-127.5) /128.0 

        img=np.transpose(img,(2,0,1))

        img = np.expand_dims(img,axis=0)


        return img
    

    def get_embedding(self,face_crop) : 

        input_tensor = self.preprocess(face_crop)

        outputs = self.session.run(None,{"input":input_tensor})


        return np.squeeze(outputs)