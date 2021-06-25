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
"""unet 310 infer."""
import os
import cv2
import numpy as np

from src.model_utils.config import config

class dice_coeff():
    def __init__(self):
        self.clear()
    def clear(self):
        self._dice_coeff_sum = 0
        self._iou_sum = 0
        self._samples_num = 0

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Need 2 inputs ((y_softmax, y_argmax), y), but got {}'.format(len(inputs)))
        y = np.array(inputs[1])
        self._samples_num += y.shape[0]
        y = y.transpose(0, 2, 3, 1)
        b, h, w, c = y.shape
        if b != 1:
            raise ValueError('Batch size should be 1 when in evaluation.')
        y = y.reshape((h, w, c))
        if config.eval_activate.lower() == "softmax":
            y_softmax = np.squeeze(inputs[0][0], axis=0)
            if config.eval_resize:
                y_pred = []
                for m in range(config.num_classes):
                    y_pred.append(cv2.resize(np.uint8(y_softmax[:, :, m] * 255), (w, h)) / 255)
                y_pred = np.stack(y_pred, axis=-1)
            else:
                y_pred = y_softmax
        elif config.eval_activate.lower() == "argmax":
            y_argmax = np.squeeze(inputs[0][1], axis=0)
            y_pred = []
            for n in range(config.num_classes):
                if config.eval_resize:
                    y_pred.append(cv2.resize(np.uint8(y_argmax == n), (w, h), interpolation=cv2.INTER_NEAREST))
                else:
                    y_pred.append(np.float32(y_argmax == n))
            y_pred = np.stack(y_pred, axis=-1)
        else:
            raise ValueError('config eval_activate should be softmax or argmax.')
        y_pred = y_pred.astype(np.float32)
        inter = np.dot(y_pred.flatten(), y.flatten())
        union = np.dot(y_pred.flatten(), y_pred.flatten()) + np.dot(y.flatten(), y.flatten())

        single_dice_coeff = 2*float(inter)/float(union+1e-6)
        single_iou = single_dice_coeff / (2 - single_dice_coeff)
        print("single dice coeff is: {}, IOU is: {}".format(single_dice_coeff, single_iou))
        self._dice_coeff_sum += single_dice_coeff
        self._iou_sum += single_iou

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return (self._dice_coeff_sum / float(self._samples_num), self._iou_sum / float(self._samples_num))


if __name__ == '__main__':
    rst_path = config.rst_path
    metrics = dice_coeff()

    if hasattr(config, "dataset") and config.dataset == "Cell_nuclei":
        img_size = tuple(config.image_size)
        for i, bin_name in enumerate(os.listdir('./preprocess_Result/')):
            f = bin_name.replace(".png", "")
            bin_name_softmax = f + "_0.bin"
            bin_name_argmax = f + "_1.bin"
            file_name_sof = rst_path + bin_name_softmax
            file_name_arg = rst_path + bin_name_argmax
            softmax_out = np.fromfile(file_name_sof, np.float32).reshape(1, 96, 96, 2)
            argmax_out = np.fromfile(file_name_arg, np.float32).reshape(1, 96, 96)
            mask = cv2.imread(os.path.join(config.data_path, f, "mask.png"), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size)
            mask = mask.astype(np.float32) / 255
            mask = (mask > 0.5).astype(np.int)
            mask = (np.arange(2) == mask[..., None]).astype(int)
            mask = mask.transpose(2, 0, 1).astype(np.float32)
            label = mask.reshape(1, 2, 96, 96)
            metrics.update((softmax_out, argmax_out), label)
    else:
        label_list = np.load('label.npy')
        for j in range(len(os.listdir('./preprocess_Result/'))):
            file_name_sof = rst_path + "ISBI_test_bs_1_" + str(j) + "_0" + ".bin"
            file_name_arg = rst_path + "ISBI_test_bs_1_" + str(j) + "_1" + ".bin"
            softmax_out = np.fromfile(file_name_sof, np.float32).reshape(1, 576, 576, 2)
            argmax_out = np.fromfile(file_name_arg, np.float32).reshape(1, 576, 576)
            label = label_list[j]
            metrics.update((softmax_out, argmax_out), label)

    eval_score = metrics.eval()
    print("============== Cross valid dice coeff is:", eval_score[0])
    print("============== Cross valid IOU is:", eval_score[1])
