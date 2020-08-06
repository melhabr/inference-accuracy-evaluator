# compression_eval.py

This program evaluates the performance of a given model on a set of images while varying the image size. The image
is resized using OpenCV. 

### Arguments

`--dir (-d)` - The directory containing the images to be inferenced

`--model (-m)` - Path to object detection model

`--annotations (-a)` - Path to annotations file. COCO annotations can be downloaded [here](https://cocodataset.org/#download)
