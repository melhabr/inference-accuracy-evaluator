import argparse
import os
import cv2
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from tf_inferencer import TFInferencer

# The following variables are used to configure the compression analysis
COMPRESSION_LEVELS = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
ENCODING = "JPG"  # If this is changed, be sure to change the corresponding OpenCV encoding parameter as well


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", help="Directory of images to use for comparison", required=True)
    parser.add_argument("--model", "-m", help="Path of machine learning model", required=True)
    parser.add_argument("--annotations", "-a", help="Path of annotations file", default="annotations.json")
    args = parser.parse_args()

    inferencer = TFInferencer(args.model)
    images = os.listdir(args.dir)

    imgIds = []
    for image_name in images:
        imgId = int(image_name.split('_')[2].split('.')[0].lstrip('0'))
        imgIds.append(imgId)

    results = []

    for comp_level in COMPRESSION_LEVELS:

        for i, image_name in enumerate(images):

            img = cv2.imread(os.path.join(args.dir, image_name))
            _, img = cv2.imencode("." + ENCODING, img, [cv2.IMWRITE_JPEG_QUALITY, comp_level])
            img = cv2.imdecode(img, 1)
            imgId = int(image_name.split('_')[2].split('.')[0].lstrip('0'))
            inference = inferencer.inference(img)

            for res in inference:
                entry = {"image_id": imgId,
                         "category_id": int(res[1]),
                         "bbox": [round(float(x), 2) for x in res[0]],
                         "score": round(float(res[2]), 3)}
                results.append(entry)

            print("Inferenced image", i, "| Compression level", comp_level)

        with open("results/results{}.json".format(comp_level), "w") as f:
            f.write(json.dumps(results))

    inferencer.sess.close()

    cocoGT = COCO(args.annotations)

    for comp_level in COMPRESSION_LEVELS:
        print("COMPRESSION LEVEL", comp_level)

        cocoDT = cocoGT.loadRes("results/results{}.json".format(comp_level))

        eval = COCOeval(cocoGT, cocoDT, 'bbox')

        eval.params.imgIds = imgIds
        eval.evaluate()
        eval.accumulate()
        eval.summarize()


if __name__ == '__main__':
    main()
