# Inference code in file was adapted from https://gist.github.com/aallan/6778b5a046aef3a76966e335ca186b7f

import cv2

from edgetpu.detection.engine import DetectionEngine

from stopwatch import Stopwatch

class EdgeTPUInferencer:

    def __init__(self, model):
        self.engine = DetectionEngine(model)

        self.watch = Stopwatch()

    def inference(self, img):

        self.watch.start()
        initial_h, initial_w, _ = img.shape
        if (initial_h, initial_w) != (300, 300):
            frame = cv2.resize(img, (300, 300))
        else:
            frame = img
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.watch.stop(Stopwatch.MODE_PREPROCESS)

        self.watch.start()
        ans = self.engine.detect_with_input_tensor(frame.flatten(), threshold=0.5, top_k=10)
        self.watch.stop(Stopwatch.MODE_INFER)

        # Display result
        self.watch.start()
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
        self.watch.stop(Stopwatch.MODE_POSTPROCESS)

        return results
