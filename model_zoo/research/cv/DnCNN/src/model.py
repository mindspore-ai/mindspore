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

"""architecture of DnCNN"""

import mindspore.nn as nn


class DnCNN(nn.Cell):
    """architecture of DnCNN"""
    def __init__(self, depth=17, n_channels=64, image_channels=1, kernel_size=3):
        super(DnCNN, self).__init__()
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels,
                                kernel_size=kernel_size, stride=1, pad_mode="pad", padding=1, has_bias=False))
        layers.append(nn.ReLU())
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                                    kernel_size=kernel_size, pad_mode='pad', padding=padding, has_bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.05))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels,
                                kernel_size=kernel_size, pad_mode='pad', padding=padding, has_bias=False))
        self.dncnn = nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.dncnn(x)
        return x

if __name__ == '__main__':
    from mindspore import Tensor, context
    import numpy as np

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=0)
    net = DnCNN(image_channels=1, depth=17)
    x_test = Tensor(np.ones([2, 1, 180, 180]).astype(np.float32))
    y = net(x_test)
    print(y.shape)
