# The input shape and tensor names are currently configured for mobilenet_v2
import numpy as np
import tensorflow as tf
import cv2


class TFInferencer:

    def __init__(self, model):

        with tf.gfile.GFile(model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            self.input_node = tf.placeholder(np.uint8, shape=[None, 300, 300, 3], name='input')
            tf.import_graph_def(graph_def, name='', input_map={'image_tensor': self.input_node})

        self.sess = tf.Session(graph=graph)

    def inference(self, img):

        (ih, iw) = img.shape[:-1]
        if (iw, ih) != (300, 300):
            img = cv2.resize(img, (300, 300))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img.reshape((1, 300, 300, 3))

        out = self.sess.run(['detection_classes:0', 'detection_scores:0', 'detection_boxes:0'],
                            feed_dict={self.input_node: img})

        results = []
        for i in range(len(out[0][0])):

            if out[1][0][i] == 0:
                continue

            res = out[2][0][i]
            boxes = [0] * 4
            boxes[0] = res[1] * iw
            boxes[1] = res[0] * ih
            boxes[2] = (res[3] - res[1]) * iw
            boxes[3] = (res[2] - res[0]) * ih

            res = (boxes, out[0][0][i], out[1][0][i])

            results.append(res)

        return results
