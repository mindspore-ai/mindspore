# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import argparse
import logging
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as F
from mindspore import context, Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.loss.loss import _Loss

from src.data_loader import create_dataset
from src.unet import UNet
from src.config import cfg_unet

from scipy.special import softmax

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=device_id)




class CrossEntropyWithLogits(_Loss):
    def __init__(self):
        super(CrossEntropyWithLogits, self).__init__()
        self.transpose_fn = F.Transpose()
        self.reshape_fn = F.Reshape()
        self.softmax_cross_entropy_loss = nn.SoftmaxCrossEntropyWithLogits()
        self.cast = F.Cast()
    def construct(self, logits, label):
        # NCHW->NHWC
        logits = self.transpose_fn(logits, (0, 2, 3, 1))
        logits = self.cast(logits, mindspore.float32)
        label = self.transpose_fn(label, (0, 2, 3, 1))

        loss = self.reduce_mean(self.softmax_cross_entropy_loss(self.reshape_fn(logits, (-1, 2)),
                                                                self.reshape_fn(label, (-1, 2))))
        return self.get_loss(loss)


class dice_coeff(nn.Metric):
    def __init__(self):
        super(dice_coeff, self).__init__()
        self.clear()
    def clear(self):
        self._dice_coeff_sum = 0
        self._samples_num = 0

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Mean dice coeffcient need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))

        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
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
             ckpt_path,
             cross_valid_ind=1,
             cfg=None):

    net = UNet(n_channels=cfg['num_channels'], n_classes=cfg['num_classes'])
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)

    criterion = CrossEntropyWithLogits()
    _, valid_dataset = create_dataset(data_dir, 1, 1, False, cross_valid_ind, False)
    model = Model(net, loss_fn=criterion, metrics={"dice_coeff": dice_coeff()})

    print("============== Starting Evaluating ============")
    dice_score = model.eval(valid_dataset, dataset_sink_mode=False)
    print("============== Cross valid dice coeff is:", dice_score)


def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_url', dest='data_url', type=str, default='data/',
                        help='data directory')
    parser.add_argument('-p', '--ckpt_path', dest='ckpt_path', type=str, default='ckpt_unet_medical_adam-1_600.ckpt',
                        help='checkpoint path')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    print("Testing setting:", args)
    test_net(data_dir=args.data_url,
             ckpt_path=args.ckpt_path,
             cross_valid_ind=cfg_unet['cross_valid_ind'],
             cfg=cfg_unet)
