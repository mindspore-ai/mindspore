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
import numpy as np

from src.data_loader import create_dataset
from src.config import cfg_unet
from scipy.special import softmax


class dice_coeff():
    def __init__(self):
        self.clear()

    def clear(self):
        self._dice_coeff_sum = 0
        self._samples_num = 0

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Mean dice coefficient need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))

        y_pred = inputs[0]
        y = np.array(inputs[1])

        self._samples_num += y.shape[0]
        y_pred = y_pred.transpose(0, 2, 3, 1)
        y = y.transpose(0, 2, 3, 1)
        y_pred = softmax(y_pred, axis=3)

        inter = np.dot(y_pred.flatten(), y.flatten())
        union = np.dot(y_pred.flatten(), y_pred.flatten()) + np.dot(y.flatten(), y.flatten())

        single_dice_coeff = 2*float(inter)/float(union+1e-6)
        print("single dice coeff is:", single_dice_coeff)
        self._dice_coeff_sum += single_dice_coeff

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')

        return self._dice_coeff_sum / float(self._samples_num)


def test_net(data_dir,
             cross_valid_ind=1,
             cfg=None):

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

    for j in range(len(os.listdir(rst_path))):
        file_name = rst_path + "ISBI_test_bs_1_" + str(j) + "_0" + ".bin"
        output = np.fromfile(file_name, np.float32).reshape(1, 2, 576, 576)
        label = label_list[j]
        metrics.update(output, label)

    print("Cross valid dice coeff is: ", metrics.eval())
