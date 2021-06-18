#!/usr/bin/env python3
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

import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.common.initializer import HeUniform


class DnCNN(nn.Cell):
    def __init__(self, channels, num_of_layers=20):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(channels, out_channels=features, kernel_size=kernel_size, \
                      pad_mode='pad', padding=padding, has_bias=False, weight_init=HeUniform()))
        layers.append(nn.ReLU())

        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(features, out_channels=features, kernel_size=kernel_size, \
                          pad_mode='pad', padding=padding, has_bias=False, weight_init=HeUniform()))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU())

        layers.append(nn.Conv2d(features, out_channels=channels, kernel_size=kernel_size, \
                      pad_mode='pad', padding=padding, has_bias=False, weight_init=HeUniform()))
        self.dncnn = nn.SequentialCell(layers)

    def construct(self, x):
        out = self.dncnn(x)
        return out

if __name__ == "__main__":
    #for test
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = DnCNN(1, num_of_layers=17)
    a = Tensor(np.ones((2, 1, 40, 40)), mindspore.float32)
    output = net(a)
    print(output)
    print(type(output))
    np_out = output.asnumpy()
    print(type(np_out))
