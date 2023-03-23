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
test fractional maxpooling ops
"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops
from mindspore.common.api import _cell_graph_executor
import mindspore.common.dtype as mstype


class FractionalMaxPool2dNet(nn.Cell):
    """fractional_max_pool2d"""

    def construct(self, x, _random_samples):
        output1 = ops.fractional_max_pool2d(x, kernel_size=2, output_size=(2, 2), _random_samples=_random_samples,
                                            return_indices=True)
        output2 = ops.fractional_max_pool2d(x, kernel_size=2, output_ratio=(0.5, 0.5), _random_samples=_random_samples,
                                            return_indices=True)
        return output1, output2


def test_compile_fractional_maxpool2d():
    """
    Feature: Test fractional_max_pool2d
    Description: Test the functionality of fractional_max_pool2d
    Expectation: Success
    """
    input_x = Tensor(np.array([0.3220, 0.9545, 0.7879, 0.0975, 0.3698,
                               0.5135, 0.5740, 0.3435, 0.1895, 0.8764,
                               0.9581, 0.4760, 0.9014, 0.8522, 0.3664,
                               0.4980, 0.9673, 0.9879, 0.6988, 0.9022,
                               0.9304, 0.1558, 0.0153, 0.1559, 0.9852]).reshape([1, 1, 5, 5]), mstype.float32)
    _random_samples = Tensor(np.array([[[0.0, 0.0]]]), mstype.float32)
    net = FractionalMaxPool2dNet()
    _cell_graph_executor.compile(net, input_x, _random_samples)


class FractionalMaxPool3dNet(nn.Cell):
    """fractional_max_pool3d"""

    def construct(self, x, _random_samples):
        output1 = ops.fractional_max_pool3d(x, kernel_size=(1, 1, 1), output_size=(1, 1, 2),
                                            _random_samples=_random_samples, return_indices=True)
        output2 = ops.fractional_max_pool3d(x, kernel_size=(1, 1, 1), output_ratio=(0.5, 0.5, 0.5),
                                            _random_samples=_random_samples, return_indices=True)
        return output1, output2


def test_compile_fractional_maxpool3d():
    """
    Feature: Test fractional_max_pool3d
    Description: Test the functionality of fractional_max_pool3d
    Expectation: Success
    """
    input_x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
                     .reshape([1, 1, 2, 2, 4]), mstype.float32)
    _random_samples = Tensor(np.array([0.0, 0.0, 0.0]).reshape([1, 1, 3]), mstype.float32)
    net = FractionalMaxPool3dNet()
    _cell_graph_executor.compile(net, input_x, _random_samples)
