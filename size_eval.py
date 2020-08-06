import argparse
import os
import cv2
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from tf_inferencer import TFInferencer

# The following variables are used to configure the compression analysis
IMAGE_SIZES = [(300, 300), (250, 250), (200, 200), (150, 150), (100, 100), (50, 50)]


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

    for size in IMAGE_SIZES:

        for i, image_name in enumerate(images):

            img = cv2.imread(os.path.join(args.dir, image_name))
            ih, iw = img.shape[:-1]
            img = cv2.resize(img, size)
            imgId = int(image_name.split('_')[2].split('.')[0].lstrip('0'))
            inference = inferencer.inference(img)

            for res in inference:

                res[0][0] = res[0][0] * iw / size[0]
                res[0][1] = res[0][1] * ih / size[1]
                res[0][2] = res[0][2] * iw / size[0]
                res[0][3] = res[0][3] * ih / size[1]

                entry = {"image_id": imgId,
                         "category_id": int(res[1]),
                         "bbox": [round(float(x), 2) for x in res[0]],
                         "score": round(float(res[2]), 3)}
                results.append(entry)

            print("Inferenced image", i, "| Image size", size)

        with open("results/results{}.json".format(size), "w") as f:
            f.write(json.dumps(results))

    inferencer.sess.close()

    cocoGT = COCO(args.annotations)

    for size in IMAGE_SIZES:
        print("SIZE", size)

        cocoDT = cocoGT.loadRes("results/results{}.json".format(size))

        eval = COCOeval(cocoGT, cocoDT, 'bbox')

        eval.params.imgIds = imgIds
        eval.evaluate()
        eval.accumulate()
        eval.summarize()


if __name__ == '__main__':
    main()
