#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

# This script is a substantially modified version of the one here:
# https://github.com/openvinotoolkit/openvino/tree/master/inference-engine/ie_bridges/python/sample/object_detection_sample_ssd

import os
import numpy as np
import cv2

from openvino.inference_engine import IECore, IENetwork

class OpenVINOInferencer:

    def __init__(self, model, device):

        ie = IECore()
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        net = IENetwork(model=model_xml, weights=model_bin)

        for input_key in net.inputs:
            if len(net.inputs[input_key].layout) == 4:
                self.n, self.c, self.h, self.w = net.inputs[input_key].shape

        assert (len(net.inputs.keys()) == 1 or len(
            net.inputs.keys()) == 2), "Sample supports topologies only with 1 or 2 inputs"
        self.out_blob = next(iter(net.outputs))
        self.input_name, input_info_name = "", ""

        for input_key in net.inputs:
            if len(net.inputs[input_key].layout) == 4:
                self.input_name = input_key
                net.inputs[input_key].precision = 'U8'
            elif len(net.inputs[input_key].layout) == 2:
                input_info_name = input_key
                net.inputs[input_key].precision = 'FP32'
                if net.inputs[input_key].shape[1] != 3 and net.inputs[input_key].shape[1] != 6 or \
                        net.inputs[input_key].shape[0] != 1:
                    print('Invalid input info. Should be 3 or 6 values length.')

        self.data = {}

        if input_info_name != "":
            infos = np.ndarray(shape=(self.n, self.c), dtype=float)
            for i in range(self.n):
                infos[i, 0] = self.h
                infos[i, 1] = self.w
                infos[i, 2] = 1.0
            self.data[input_info_name] = infos

        # ---------------------------------------- 4. Prepare output blobs ------------------------------------

        output_name, output_info = "", net.outputs[next(iter(net.outputs.keys()))]
        for output_key in net.outputs:
            if net.layers[output_key].type == "DetectionOutput":
                output_name, output_info = output_key, net.outputs[output_key]

        if output_name == "":
            print("Can't find a DetectionOutput layer in the topology")

        output_dims = output_info.shape
        if len(output_dims) != 4:
            print("Incorrect output dimensions for SSD model")
        max_proposal_count, object_size = output_dims[2], output_dims[3]

        if object_size != 7:
            print("Output item should have 7 as a last dimension")

        output_info.precision = "FP32"

        self.exec_net = ie.load_network(network=net, device_name=device)

    def inference(self, img):

        ih, iw = img.shape[:-1]
        if (ih, iw) != (self.h, self.w):
            img = cv2.resize(img, (self.w, self.h))
        img = img.transpose((2, 0, 1))  # Change data layout from HWC to CHW

        self.data[self.input_name] = img
        res = self.exec_net.infer(inputs=self.data)
        res = res[self.out_blob]
        res = res[0][0]

        results = []
        for proposal in res:

            if int(proposal[1]) == 0:
                continue
            if proposal[2] < 0.5:
                continue

            box = [0] * 4

            box[0] = iw * proposal[3]
            box[1] = ih * proposal[4]
            box[2] = iw * (proposal[5] - proposal[3])
            box[3] = ih * (proposal[6] - proposal[4])

            result = (box, int(proposal[1]), proposal[2])
            results.append(result)

        return results
