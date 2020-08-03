import argparse
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", "-a", help="Path of annotations file", default="annotations.json")
    parser.add_argument("--results", "-r", help="Path of results file", default="results.json")
    parser.add_argument("--imgIds", "-i", help="Path of imgIds file", default="imgIds.json")
    args = parser.parse_args()

    cocoGT = COCO(args.annotations)
    cocoDT = cocoGT.loadRes(args.results)

    eval = COCOeval(cocoGT, cocoDT, 'bbox')

    with open(args.imgIds, "r") as f:
        imgIds = json.load(f)
    eval.params.imgIds = imgIds
    eval.evaluate()
    eval.accumulate()
    eval.summarize()


if __name__ == '__main__':
    main()
