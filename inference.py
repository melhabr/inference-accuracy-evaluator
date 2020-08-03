import argparse
import os
import cv2
import json

from tf_inferencer import TFInferencer
from edgetpu_inferencer import EdgeTPUInferencer
from tensorrt_inferencer import TensorRTInferencer
from openvino_inferencer import OpenVINOInferencer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", help="Directory of images to use for comparison", required=True)
    parser.add_argument("--model", "-m", help="Path of machine learning model", required=True)
    parser.add_argument("--limit", "-l", help="Maximum number of images to evaluate", default=1000, type=int)
    args = parser.parse_args()

    # CHANGE THE FOLLOWING LINE TO SPECIFY CLIENT TYPE
    inferencer = TFInferencer(args.model)
    # inferencer = EdgeTPUInferencer(args.model)
    # inferencer = TensorRTInferencer(args.model)
    # inferencer = OpenVINOInferencer(args.model, "CPU")
    images = os.listdir(args.dir)

    results = []
    imgIds = []

    for i, image_name in enumerate(images):

        if i >= args.limit:
            break

        img = cv2.imread(os.path.join(args.dir, image_name))
        imgId = int(image_name.split('_')[2].split('.')[0].lstrip('0'))
        imgIds.append(imgId)
        inference = inferencer.inference(img)

        for res in inference:
            entry = {"image_id": imgId,
                     "category_id": int(res[1]),
                     "bbox": [round(float(x), 2) for x in res[0]],
                     "score": round(float(res[2]), 3)}
            results.append(entry)

        print("Inferenced image", i)

    with open("results.json", "w") as f:
        f.write(json.dumps(results))

    with open("imgIds.json", "w") as f:
        f.write(json.dumps(imgIds))


if __name__ == '__main__':
    main()
