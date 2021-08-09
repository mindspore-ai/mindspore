# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""postprocess for 310 inference"""
import os
import argparse
import numpy as np
import cv2

import mindspore.nn as nn


parser = argparse.ArgumentParser("unet quant postprocess")
parser.add_argument("--result_path", type=str, required=True, help="path to inference results.")
parser.add_argument("--label_path", type=str, required=True, help="path to label.npy.")
parser.add_argument("--input_path", type=str, required=True, help="path to input data.")

args, _ = parser.parse_known_args()

class dice_coeff(nn.Metric):
    """Unet Metric, return dice coefficient and IOU."""

    def __init__(self, print_res=True, show_eval=False):
        super(dice_coeff, self).__init__()
        self.clear()
        self.show_eval = show_eval
        self.print_res = print_res
        self.img_num = 0
        # network config
        self.include_background = True
        self.eval_resize = False
        self.num_classes = 2

    def clear(self):
        self._dice_coeff_sum = 0
        self._iou_sum = 0
        self._samples_num = 0
        self.img_num = 0

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Need 2 inputs (y_predict, y), but got {}'.format(len(inputs)))
        y = self._convert_data(inputs[1])
        self._samples_num += y.shape[0]
        y = y.transpose(0, 2, 3, 1)
        b, h, w, c = y.shape
        if b != 1:
            raise ValueError('Batch size should be 1 when in evaluation.')
        y = y.reshape((h, w, c))
        start_index = 0
        if not self.include_background:
            y = y[:, :, 1:]
            start_index = 1

        y_softmax = np.squeeze(self._convert_data(inputs[0]), axis=0)
        if self.eval_resize:
            y_pred = []
            for i in range(start_index, self.num_classes):
                y_pred.append(cv2.resize(np.uint8(y_softmax[:, :, i] * 255), (w, h)) / 255)
            y_pred = np.stack(y_pred, axis=-1)
        else:
            y_pred = y_softmax
            if not self.include_background:
                y_pred = y_softmax[:, :, start_index:]

        y_pred = y_pred.astype(np.float32)
        inter = np.dot(y_pred.flatten(), y.flatten())
        union = np.dot(y_pred.flatten(), y_pred.flatten()) + np.dot(y.flatten(), y.flatten())

        single_dice_coeff = 2 * float(inter) / float(union + 1e-6)
        single_iou = single_dice_coeff / (2 - single_dice_coeff)
        if self.print_res:
            print("single dice coeff is: {}, IOU is: {}".format(single_dice_coeff, single_iou))
        self._dice_coeff_sum += single_dice_coeff
        self._iou_sum += single_iou

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return (self._dice_coeff_sum / float(self._samples_num), self._iou_sum / float(self._samples_num))


if __name__ == '__main__':
    metrics = dice_coeff()
    # eval_activate = "softmax"

    label_list = np.load(args.label_path)
    for j in range(len(os.listdir(args.input_path))):
        file_name = os.path.join(args.result_path, "ISBI_test_bs_1_" + str(j) + "_output_0.bin")
        rst_out = np.fromfile(file_name, np.float32).reshape(1, 388, 388, 2)
        label = label_list[j]
        metrics.update(rst_out, label)
    eval_score = metrics.eval()
    print("==================== Cross valid dice coeff is:", eval_score[0])
    print("==================== Cross valid dice IOU is:", eval_score[1])
