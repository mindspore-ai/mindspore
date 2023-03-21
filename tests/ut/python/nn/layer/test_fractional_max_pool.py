# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
test fractional maxpooling api
"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import _cell_graph_executor
import mindspore.common.dtype as mstype


class FractionalMaxPool2dNet(nn.Cell):
    """FractionalMaxPool2d"""

    def __init__(self):
        super(FractionalMaxPool2dNet, self).__init__()
        _random_samples = Tensor(np.array([[[0.8, 0.8]]]), mstype.float32)
        self.pool1 = nn.FractionalMaxPool2d(kernel_size=2, output_size=(2, 2), _random_samples=_random_samples,
                                            return_indices=True)
        self.pool2 = nn.FractionalMaxPool2d(kernel_size=2, output_ratio=(0.5, 0.5), _random_samples=_random_samples,
                                            return_indices=True)

    def construct(self, x):
        output1 = self.pool1(x)
        output2 = self.pool2(x)
        return output1, output2


def test_compile_fractional_maxpool2d():
    """
    Feature: Test FractioanlMaxPool2d
    Description: Test the functionality of FractionalMaxPool2d
    Expectation: Success
    """
    input_x = Tensor(np.array([0.3220, 0.9545, 0.7879, 0.0975, 0.3698,
                               0.5135, 0.5740, 0.3435, 0.1895, 0.8764,
                               0.9581, 0.4760, 0.9014, 0.8522, 0.3664,
                               0.4980, 0.9673, 0.9879, 0.6988, 0.9022,
                               0.9304, 0.1558, 0.0153, 0.1559, 0.9852]).reshape([1, 1, 5, 5]), mstype.float32)
    net = FractionalMaxPool2dNet()
    _cell_graph_executor.compile(net, input_x)


class FractionalMaxPool3dNet(nn.Cell):
    """FractionalMaxPool3d"""

    def __init__(self):
        super(FractionalMaxPool3dNet, self).__init__()
        _random_samples = Tensor(np.array([0.7, 0.7, 0.7]).reshape([1, 1, 3]), mstype.float32)
        self.pool1 = nn.FractionalMaxPool3d(kernel_size=(1, 1, 1), output_size=(1, 1, 2),
                                            _random_samples=_random_samples, return_indices=True)
        self.pool2 = nn.FractionalMaxPool3d(kernel_size=(1, 1, 1), output_ratio=(0.5, 0.5, 0.5),
                                            _random_samples=_random_samples, return_indices=True)

    def construct(self, x):
        output1 = self.pool1(x)
        output2 = self.pool2(x)
        return output1, output2


def test_compile_fractional_maxpool3d():
    """
    Feature: Test FractioanlMaxPool3d
    Description: Test the functionality of FractionalMaxPool3d
    Expectation: Success
    """
    input_x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
                     .reshape([1, 1, 2, 2, 4]), mstype.float32)
    net = FractionalMaxPool3dNet()
    _cell_graph_executor.compile(net, input_x)
