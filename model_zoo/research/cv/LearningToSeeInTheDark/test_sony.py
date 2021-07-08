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
"""test"""
from __future__ import division
import argparse as arg
import os
import glob
from PIL import Image
import h5py
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor, dtype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.unet_parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Cell):
    """ Unet """

    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.outc = OutConv(32, n_classes)

    def construct(self, x):
        """Unet construct"""

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits


def pack_raw(raw):
    """ pack sony raw data into 4 channels """

    im = np.maximum(raw - 512, 0) / (16383 - 512)  # subtract the black level
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


def get_test_data(input_dir1, gt_dir1, test_ids1):
    """ trans input raw data into arrays then pack into a list """

    final_test_inputs = []
    for test_id in test_ids1:
        in_files = glob.glob(input_dir1 + '%05d_00*.hdf5' % test_id)

        gt_files = glob.glob(gt_dir1 + '%05d_00*.hdf5' % test_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        gt_exposure = float(gt_fn[9: -6])

        for in_path in in_files:

            in_fn = os.path.basename(in_path)
            in_exposure = float(in_fn[9: -6])
            ratio = min(gt_exposure / in_exposure, 300.0)
            ima = h5py.File(in_path, 'r')
            in_rawed = ima.get('in')[:]
            input_image = np.expand_dims(pack_raw(in_rawed), axis=0) * ratio
            input_image = np.minimum(input_image, 1.0)
            input_image = input_image.transpose([0, 3, 1, 2])
            input_image = np.float32(input_image)
            final_test_inputs.append(input_image)
    return final_test_inputs


if __name__ == '__main__':
    parser = arg.ArgumentParser(description='Mindspore SID Eval')
    parser.add_argument('--device_target', default='Ascend',
                        help='device where the code will be implemented')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data')
    parser.add_argument('--checkpoint_path', required=True, default=None, help='ckpt file path')
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    local_data_path = args.data_url
    input_dir = os.path.join(local_data_path, 'short/')
    gt_dir = os.path.join(local_data_path, 'long/')
    test_fns = glob.glob(gt_dir + '1*.hdf5')
    test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]
    ckpt_dir = args.checkpoint_path
    param_dict = load_checkpoint(ckpt_dir)
    net = UNet(4, 12)
    load_param_into_net(net, param_dict)

    in_ims = get_test_data(input_dir, gt_dir, test_ids)
    i = 0
    for in_im in in_ims:
        output = net(Tensor(in_im, dtype.float32))
        output = output.asnumpy()
        output = np.minimum(np.maximum(output, 0), 1)
        output = np.trunc(output[0] * 255)
        output = output.astype(np.int8)
        output = output.transpose([1, 2, 0])
        image_out = Image.fromarray(output, 'RGB')
        image_out.save('output_%d.png' % i)
        i += 1
