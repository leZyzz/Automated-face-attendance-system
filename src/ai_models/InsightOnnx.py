import cv2
import numpy as np
import onnxruntime as ort 
from pathlib import Path 

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "assets" / "models"
SCRFD_PATH = MODELS_DIR / "det_500m.onnx"


 

class ScrfdOnnx:
    def __init__(self, model_path=SCRFD_PATH):
        if not model_path.exists() :
            raise FileNotFoundError(f"Missing model file at  {model_path}")
        
        self.session = ort.InferenceSession(str(model_path),providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.feat_strides = [8, 16, 32]
        self.num_anchors = 2

    def preprocess(self, frame):
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 128.0
        return np.expand_dims(np.transpose(img, (2, 0, 1)), 0)

    def _anchors(self, stride):
        f = 640 // stride
        cx = (np.arange(f) * stride).repeat(f * self.num_anchors)
        cy = np.tile(np.arange(f) * stride, f).repeat(self.num_anchors)
        # interleave so anchors match output row order
        cx = cx.reshape(f, f, self.num_anchors).transpose(1,0,2).reshape(-1)
        cy = cy.reshape(f, f, self.num_anchors).transpose(1,0,2).reshape(-1)
        return np.stack([cx, cy], axis=1).astype(np.float32)

    def detect(self, frame, conf=0.5, nms=0.4):
        oh, ow = frame.shape[:2]
        sx, sy = ow / 640, oh / 640
        outputs = self.session.run(None, {self.input_name: self.preprocess(frame)})

        all_boxes, all_scores, all_kps = [], [], []
        for i, stride in enumerate(self.feat_strides):
            scores = outputs[i].reshape(-1)
            bp     = outputs[i + 3].reshape(-1, 4) * stride
            kp     = outputs[i + 6].reshape(-1, 5, 2) * stride
            ac     = self._anchors(stride)

            x1, y1 = ac[:,0] - bp[:,0], ac[:,1] - bp[:,1]
            x2, y2 = ac[:,0] + bp[:,2], ac[:,1] + bp[:,3]
            kp[:,:,0] += ac[:,0:1]; kp[:,:,1] += ac[:,1:2]

            mask = scores >= conf
            if not mask.any(): continue
            all_scores.append(scores[mask])
            all_boxes.append(np.stack([x1,y1,x2,y2], 1)[mask])
            all_kps.append(kp[mask])

        if not all_boxes:
            return [], []

        boxes  = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        kps    = np.concatenate(all_kps)

        xywh = boxes.copy(); xywh[:,2] -= xywh[:,0]; xywh[:,3] -= xywh[:,1]
        idx  = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf, nms)

        final_boxes, final_kps = [], []
        for i in idx.flatten():
            b = boxes[i]
            final_boxes.append([int(b[0]*sx), int(b[1]*sy), int(b[2]*sx), int(b[3]*sy)])
            lm = kps[i].copy(); lm[:,0] *= sx; lm[:,1] *= sy
            final_kps.append(lm.astype(int))
        return final_boxes, final_kps


def draw(frame, boxes, kps):
    for box, lm in zip(boxes, kps):
        x1,y1,x2,y2 = box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(frame, f"{len(boxes)} face(s)", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    return frame
