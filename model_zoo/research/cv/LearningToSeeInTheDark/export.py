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
""" export MINDIR"""
import argparse as arg
import numpy as np
import mindspore as ms
from mindspore import context, Tensor, export, load_checkpoint
import mindspore.nn as nn
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


if __name__ == '__main__':
    parser = arg.ArgumentParser(description='SID export')
    parser.add_argument('--device_target', type=str, choices=['Ascend', 'GPU', 'CPU'], default='Ascend',
                        help='device where the code will be implemented')
    parser.add_argument('--device_id', type=int, default=0, help='device id')
    parser.add_argument('--file_format', type=str, choices=['AIR', 'MINDIR'], default='MINDIR',
                        help='file format')
    parser.add_argument('--checkpoint_path', required=True, default=None, help='ckpt file path')
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == 'Ascend':
        context.set_context(device_id=args.device_id)

    ckpt_dir = args.checkpoint_path
    net = UNet(4, 12)
    load_checkpoint(ckpt_dir, net=net)
    net.set_train(False)

    input_data = Tensor(np.zeros([1, 4, 1424, 2128]), ms.float32)
    export(net, input_data, file_name='sid', file_format=args.file_format)
