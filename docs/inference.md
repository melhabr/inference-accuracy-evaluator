# inference.py 

This program can be used to run inference on a set of images and produce a file detailing the results. By default, the
program is configured to execute inference in the standard TensorFlow environment, but can be readily changed to use a
different inferencing platform. Please see the source code for details. 

### Usage

To use, run the python script with the desired arguments specified. The script will produce two JSON files with the
results: `results.json`, which has the inference results, and `imgIds.json`, which is a list of the image IDs used. 

The list of image IDs needs to specified because the COCO evaluator assumes it should evaluate the model over all images
by default, which may not be the case for the provided set of images.  

### Arguments

`--dir (-d)` - Specifies the directory containing the images.

`--model (-m)` Path to machine learning model. See [inference_clients](inference_clients.md) for more information. 

`--limit (-l)` Maximum number of images to be used for inference. Default 1000