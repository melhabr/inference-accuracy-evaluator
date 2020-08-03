# Inference code in file was adapted from https://gist.github.com/aallan/6778b5a046aef3a76966e335ca186b7f

import cv2

from edgetpu.detection.engine import DetectionEngine

class EdgeTPUInferencer:

    def __init__(self, model):
        self.engine = DetectionEngine(model)

    def inference(self, img):

        initial_h, initial_w, _ = img.shape
        if (initial_h, initial_w) != (300, 300):
            frame = cv2.resize(img, (300, 300))
        else:
            frame = img
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ans = self.engine.detect_with_input_tensor(frame.flatten(), threshold=0.5, top_k=10)

        # Display result
        results = []
        for obj in ans:
            box = obj.bounding_box.flatten().tolist()
            bbox = [0] * 4
            bbox[0] = box[0] * initial_w
            bbox[1] = box[1] * initial_h
            bbox[2] = (box[2] - box[0]) * initial_w
            bbox[3] = (box[3] - box[1]) * initial_h

            result = (bbox, obj.label_id + 1, obj.score)
            results.append(result)

        return results
