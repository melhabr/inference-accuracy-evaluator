# Inference Accuracy Evaluator

The Inference Accuracy Evaluator is a lightweight framework for evaluating the accuracy of various object detection
models on the [COCO dataset](https://cocodataset.org/). The programs available provide methods for direct comparison of 
average precision and recall across differing model versions and formats, for an arbitrary set of COCO training/cross-validation 
pictures. The framework supports a number of inference platforms, including standard tensorflow model execution,
 [Google's EdgeTPU inference framework](https://cloud.google.com/edge-tpu),
Nvidia's CUDA-optimized [TensorRT framework](https://developer.nvidia.com/tensorrt), and [Intel's OpenVINO inference
platform](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html). Also included is a tool for
measuring the performance of a particular model when subjected to different types of compression.

## Requirements

See the [requirements file](REQUIREMENTS.md) for details on required packages and installations.

## Contents

This section details a brief summary of available programs. For in-depth, documentation, please consult the *docs* folder.

`inference.py` - Given an object detection model and directory of images, runs inference and yields an output file for
evaluation 

`eval.py` - Given a COCO annotations file, an inference results JSON file, and a list of image IDs to use, evaluates 
model accuracy

`tf_inferencer.py`, `edgetpu_inferencer.py`, `tensorrt_inferencer.py`, `openvino_inferencer.py` - 
Client inference programs for respective inference platforms

`compression_eval.py` - Additional utility for evaluating a set of images using varying image compression algorithms