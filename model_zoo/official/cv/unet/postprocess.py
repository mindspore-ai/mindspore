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
import argparse
import cv2
import numpy as np

from src.data_loader import create_dataset, create_cell_nuclei_dataset
from src.config import cfg_unet

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
        if cfg_unet["eval_activate"].lower() == "softmax":
            y_softmax = np.squeeze(inputs[0][0], axis=0)
            if cfg_unet["eval_resize"]:
                y_pred = []
                for m in range(cfg_unet["num_classes"]):
                    y_pred.append(cv2.resize(np.uint8(y_softmax[:, :, m] * 255), (w, h)) / 255)
                y_pred = np.stack(y_pred, axis=-1)
            else:
                y_pred = y_softmax
        elif cfg_unet["eval_activate"].lower() == "argmax":
            y_argmax = np.squeeze(inputs[0][1], axis=0)
            y_pred = []
            for n in range(cfg_unet["num_classes"]):
                if cfg_unet["eval_resize"]:
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


def test_net(data_dir,
             cross_valid_ind=1,
             cfg=None):

    if 'dataset' in cfg and cfg['dataset'] == "Cell_nuclei":
        valid_dataset = create_cell_nuclei_dataset(data_dir, cfg['img_size'], 1, 1, is_train=False,
                                                   eval_resize=cfg["eval_resize"], split=0.8)
    else:
        _, valid_dataset = create_dataset(data_dir, 1, 1, False, cross_valid_ind, False, do_crop=cfg['crop'],
                                          img_size=cfg['img_size'])
    labels_list = []

    for data in valid_dataset:
        labels_list.append(data[1].asnumpy())

    return labels_list


def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_url', dest='data_url', type=str, default='data/',
                        help='data directory')
    parser.add_argument('-p', '--rst_path', dest='rst_path', type=str, default='./result_Files/',
                        help='infer result path')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    label_list = test_net(data_dir=args.data_url, cross_valid_ind=cfg_unet['cross_valid_ind'], cfg=cfg_unet)
    rst_path = args.rst_path
    metrics = dice_coeff()

    if 'dataset' in cfg_unet and cfg_unet['dataset'] == "Cell_nuclei":
        for i, bin_name in enumerate(os.listdir('./preprocess_Result/')):
            bin_name_softmax = bin_name.replace(".png", "") + "_0.bin"
            bin_name_argmax = bin_name.replace(".png", "") + "_1.bin"
            file_name_sof = rst_path + bin_name_softmax
            file_name_arg = rst_path + bin_name_argmax
            softmax_out = np.fromfile(file_name_sof, np.float32).reshape(1, 96, 96, 2)
            argmax_out = np.fromfile(file_name_arg, np.float32).reshape(1, 96, 96)
            label = label_list[i]
            metrics.update((softmax_out, argmax_out), label)
    else:
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
