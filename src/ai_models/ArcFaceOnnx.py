import cv2 
import numpy as np 
import onnxruntime as ort


class ArcFaceEmbedderOnnx() :
    def __init__(self,model_path="/home/zyzz/py/myenv/assets/models/arcface.onnx") :
        
        providers=["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path,providers=providers)

        input_config = self.session.get_inputs()[0]

        self.input_name = input_config.name


    def preprocess(self,face_crop) :

        img = cv2.resize(face_crop,(112,112))

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img =(img.astype(np.float32)-127.5) /128.0 

        img=np.transpose(img,(2,0,1))

        img = np.expand_dims(img,axis=0)


        return img
    

    def get_embedding(self,face_crop) : 

        input_tensor = self.preprocess(face_crop)

        outputs = self.session.run(None,{self.input_name:input_tensor})


        return np.squeeze(outputs)
