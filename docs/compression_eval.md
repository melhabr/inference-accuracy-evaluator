# compression_eval.py

This program evaluates the performance of a given model on a set of images while varying the compression level. The image
is encoded/decoded using OpenCV. The current compression algorithm is JPG, and the program is currently configured to use
10 different levels of compression. This can easily be changed, see the source code for more details. 

### Arguments

`--dir (-d)` - The directory containing the images to be inferenced

`--model (-m)` - Path to object detection model

`--annotations (-a)` - Path to annotations file. COCO annotations can be downloaded [here](https://cocodataset.org/#download)
