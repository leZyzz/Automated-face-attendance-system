
import cv2 
import numpy as np 
import onnxruntime as ort


class FaceNetEmbedderOnnx() :
    def __init__(self,model_path="/home/zyzz/py/myenv/assets/models/facenet.onnx") :
        
        providers=["CPUExecutionProvider"]

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