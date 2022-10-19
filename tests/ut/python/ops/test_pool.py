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
test pooling api
"""
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.api import _cell_graph_executor


class LPPool1d(nn.Cell):
    """LPPool1d"""

    def construct(self, x):
        output = ops.lp_pool1d(x, norm_type=1, kernel_size=3, stride=1)
        return output


def test_compile_lpool1d():
    """
    Feature: Test LPPool1d
    Description: Test the functionality of LPPool1d
    Expectation: Success
    """
    net = LPPool1d()
    x = ms.Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)), dtype=ms.float32)
    y = ms.Tensor(np.arange(3 * 4).reshape((3, 4)), dtype=ms.float32)
    _cell_graph_executor.compile(net, x)
    _cell_graph_executor.compile(net, y)


class LPPool2d(nn.Cell):
    """LPPool2d"""

    def construct(self, x):
        out = ops.lp_pool2d(x, norm_type=1, kernel_size=3, stride=1)
        return out


def test_compile_lppool2d():
    """
    Feature: Test LPPool2d
    Description: Test the functionality of LPPool2d
    Expectation: Success
    """
    net = LPPool2d()
    x = ms.Tensor(np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5)), dtype=ms.float32)
    _cell_graph_executor.compile(net, x)
