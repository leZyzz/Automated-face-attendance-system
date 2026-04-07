
import cv2 
import numpy as np 
import onnxruntime as ort

class YoloOnnx() : 
    def __init__(self,model_path="/home/zyzz/py/myenv/assets/models/yolov8nbest.onnx"):
        
        providers=["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path,providers=providers)

        #or simply get the input name from netron.com and hardcode it later

        input_config = self.session.get_inputs()[0]

        self.input_name = input_config.name

       


    def preprocess(self,frame) : 

        img = cv2.resize(frame,(640,640))

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0

        img = np.transpose(img,(2,0,1))

        img = np.expand_dims(img,axis=0)

        return img
    
    def postprocess(self,output,original_shape) :
        
        org_h , org_w =  original_shape


        #outputs here are the bounding box guesses , of shape (1,5,8400)
        # we squeeze the output to remove the batch size we added previously 
        # we use .T to transpose it so we can loop through the rows

        predictions = np.squeeze(output).T
        boxes = []
        scores=[]
        for row in predictions : 
            confidence = row[4]
            if confidence >0.5 : 
                center_x,center_y,w,h,_=row
                #top-left corner
                x_min = int(center_x- w / 2)
                y_min = int(center_y- h / 2)
                boxes.append([x_min,y_min,int(w),int(h)])
                scores.append(float(confidence))
        
        indices = cv2.dnn.NMSBoxes(boxes,scores,score_threshold=0.5,nms_threshold=0.4)
        final_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                x_min, y_min, w, h = boxes[i]
                
                # Math to scale the 640x640 coordinates back up to your true webcam resolution
                x_scale = org_w / 640
                y_scale = org_h / 640
                
                x1 = int(x_min * x_scale)
                y1 = int(y_min * y_scale)
                x2 = int((x_min + w) * x_scale)
                y2 = int((y_min + h) * y_scale)
                
                final_boxes.append([x1, y1, x2, y2])
                
        return final_boxes # Returns clean [x1, y1, x2, y2] coordinates
    def detect(self,frame):

        input_tensor=self.preprocess(frame)
        
        outputs = self.session.run(None,{self.input_name:input_tensor})

        final_boxes = self.postprocess(outputs,frame.shape[:2])
        

        return final_boxes