# Inference Client Interfaces

The following details describe how to use the inference APIs for the classes `TFInferencer`, `EdgeTPUInferencer`, 
`TensorRTInferencer`, and `OpenVINOInferencer`, provided by `tf_inferencer.py`, `edgetpu_inferencer.py`,
`tensorrt_inferencer.py`, and `openvino_inferencer.py`. The classes provide the same functionality and make use of the same 
API for their respective platforms, except where noted otherwise. 

### API Usage

`__init__(self, model)` - Initializes a new inference environment for the respective platform, where `model` is the path
to the model. 

For each platform, the model file should be of the following type:
- TensorFlow - A frozen Tensorflow protobuffer file
- EdgeTPU - An [EdgeTPU-compiled](https://coral.ai/docs/edgetpu/compiler/) tflite graph file
- TensorRT - A [CUDA-compiled](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/uff/uff.html) TensorRT model binary file
- OpenVINO - An [OpenVINO IR model](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html) XML file, with a corresponding `.bin` file of the same name in the same directory

If an instance if `OpenVINOInferencer` is being initialized, the function is `__init__(self, model, device)`, where 
`device` is the device to be used. `device` can be one of `CPU`, `GPU`, `HDDL`, `MYRIAD`, `FPGA`, or other compatible 
devices. See [here](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html)
 for more information.  

`inference(self, img)` - Runs inference on the image `img`, and returns an array of proposals of the form 
`(bbox, class, confidence)`, where `bbox` is the bounding box of the form `(topleftx, toplefty, width, height)`, `class`
is the class id, and `confidence` is the confidence of the proposal. The list may include any number of proposals, and
may be empty.