# Inference code significantly adapted from JK Jung's repository. MIT Licence
# See https://github.com/jkjung-avt/tensorrt_demos

import numpy as np
import cv2

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from stopwatch import Stopwatch

class TensorRTInferencer:

    def __init__(self, model):

        # Initialize TRT environment
        self.input_shape = (300, 300)
        trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(trt_logger, '')
        with open(model, 'rb') as f, trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        self.context = engine.create_execution_context()

        self.watch = Stopwatch()

    def inference(self, img):

        self.watch.start()
        ih, iw = img.shape[:-1]
        if (iw, ih) != self.input_shape:
            img = cv2.resize(img, self.input_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img *= (2.0 / 255.0)
        img -= 1.0
        self.watch.stop(Stopwatch.MODE_PREPROCESS)

        self.watch.start()
        np.copyto(self.host_inputs[0], img.ravel())
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(batch_size=1, bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        self.watch.stop(Stopwatch.MODE_INFER)

        self.watch.start()
        output = self.host_outputs[0]
        results = []
        for prefix in range(0, len(output), 7):

            conf = float(output[prefix + 2])
            if conf < 0.5:
                continue
            x1 = output[prefix + 3] * iw
            y1 = output[prefix + 4] * ih
            x2 = (output[prefix + 5] - output[prefix + 3]) * iw
            y2 = (output[prefix + 6] - output[prefix + 4]) * ih
            cls = int(output[prefix + 1])
            results.append(((x1, y1, x2, y2), cls, conf))
        self.watch.stop(Stopwatch.MODE_POSTPROCESS)

        return results
