# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import pytest
import numpy as np
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import nn_ops


class PadV3Net(nn.Cell):
    def __init__(self, mode, paddings_contiguous=True):
        super(PadV3Net, self).__init__()
        self.ops = nn_ops.PadV3(mode, paddings_contiguous)
        self.mode = mode

    def construct(self, x, paddings, value):
        out = self.ops(x, paddings, value)
        return out


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('x_data_type', [np.int16, np.float32])
@pytest.mark.parametrize('mode', ["constant", "reflect", "edge"])
@pytest.mark.parametrize('ms_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_padv3_constant_shape_3d(x_data_type, mode, ms_mode):
    """
    Feature: test padv3 x and const shape paddings
    Description: test padv3 with const shape paddings
    Expectation: Success
    """
    context.set_context(mode=ms_mode, device_target="Ascend")
    x = Tensor(np.arange(9).reshape(1, 3, 3).astype(x_data_type))
    paddings = (1, 2)
    value = None
    if mode == "constant":
        value = 99 if x_data_type == np.int16 else 99.0
    net = PadV3Net(mode)
    out = net(x, paddings, value)
    if mode == "constant":
        expect = np.array([[[99, 0, 1, 2, 99, 99],
                            [99, 3, 4, 5, 99, 99],
                            [99, 6, 7, 8, 99, 99]]]).astype(x_data_type)
    elif mode == "reflect":
        expect = np.array([[[1, 0, 1, 2, 1, 0],
                            [4, 3, 4, 5, 4, 3],
                            [7, 6, 7, 8, 7, 6]]]).astype(x_data_type)
    else:
        expect = np.array([[[0, 0, 1, 2, 2, 2],
                            [3, 3, 4, 5, 5, 5],
                            [6, 6, 7, 8, 8, 8]]]).astype(x_data_type)
    np.testing.assert_almost_equal(expect, out.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('x_data_type', [np.int16, np.float32])
@pytest.mark.parametrize('mode', ["constant", "reflect", "edge"])
@pytest.mark.parametrize('ms_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_padv3_constant_shape_4d(x_data_type, mode, ms_mode):
    """
    Feature: test padv3 x and const shape paddings
    Description: test padv3 with const shape paddings
    Expectation: Success
    """
    context.set_context(mode=ms_mode, device_target="Ascend")
    x = Tensor(np.arange(36).reshape(1, 3, 4, 3).astype(x_data_type))
    paddings = (1, 2, 2, 3)
    value = None
    if mode == "constant":
        value = 99 if x_data_type == np.int16 else 99.0
    net = PadV3Net(mode)
    out = net(x, paddings, value)
    if mode == "constant":
        expect = np.array([[[[99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99],
                             [99, 0, 1, 2, 99, 99],
                             [99, 3, 4, 5, 99, 99],
                             [99, 6, 7, 8, 99, 99],
                             [99, 9, 10, 11, 99, 99],
                             [99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99]],
                            [[99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99],
                             [99, 12, 13, 14, 99, 99],
                             [99, 15, 16, 17, 99, 99],
                             [99, 18, 19, 20, 99, 99],
                             [99, 21, 22, 23, 99, 99],
                             [99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99]],
                            [[99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99],
                             [99, 24, 25, 26, 99, 99],
                             [99, 27, 28, 29, 99, 99],
                             [99, 30, 31, 32, 99, 99],
                             [99, 33, 34, 35, 99, 99],
                             [99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99],
                             [99, 99, 99, 99, 99, 99]]]]).astype(x_data_type)
    elif mode == "reflect":
        expect = np.array([[[[7, 6, 7, 8, 7, 6],
                             [4, 3, 4, 5, 4, 3],
                             [1, 0, 1, 2, 1, 0],
                             [4, 3, 4, 5, 4, 3],
                             [7, 6, 7, 8, 7, 6],
                             [10, 9, 10, 11, 10, 9],
                             [7, 6, 7, 8, 7, 6],
                             [4, 3, 4, 5, 4, 3],
                             [1, 0, 1, 2, 1, 0]],
                            [[19, 18, 19, 20, 19, 18],
                             [16, 15, 16, 17, 16, 15],
                             [13, 12, 13, 14, 13, 12],
                             [16, 15, 16, 17, 16, 15],
                             [19, 18, 19, 20, 19, 18],
                             [22, 21, 22, 23, 22, 21],
                             [19, 18, 19, 20, 19, 18],
                             [16, 15, 16, 17, 16, 15],
                             [13, 12, 13, 14, 13, 12]],
                            [[31, 30, 31, 32, 31, 30],
                             [28, 27, 28, 29, 28, 27],
                             [25, 24, 25, 26, 25, 24],
                             [28, 27, 28, 29, 28, 27],
                             [31, 30, 31, 32, 31, 30],
                             [34, 33, 34, 35, 34, 33],
                             [31, 30, 31, 32, 31, 30],
                             [28, 27, 28, 29, 28, 27],
                             [25, 24, 25, 26, 25, 24]]]]).astype(x_data_type)
    else:
        expect = np.array([[[[0, 0, 1, 2, 2, 2],
                             [0, 0, 1, 2, 2, 2],
                             [0, 0, 1, 2, 2, 2],
                             [3, 3, 4, 5, 5, 5],
                             [6, 6, 7, 8, 8, 8],
                             [9, 9, 10, 11, 11, 11],
                             [9, 9, 10, 11, 11, 11],
                             [9, 9, 10, 11, 11, 11],
                             [9, 9, 10, 11, 11, 11]],
                            [[12, 12, 13, 14, 14, 14],
                             [12, 12, 13, 14, 14, 14],
                             [12, 12, 13, 14, 14, 14],
                             [15, 15, 16, 17, 17, 17],
                             [18, 18, 19, 20, 20, 20],
                             [21, 21, 22, 23, 23, 23],
                             [21, 21, 22, 23, 23, 23],
                             [21, 21, 22, 23, 23, 23],
                             [21, 21, 22, 23, 23, 23]],
                            [[24, 24, 25, 26, 26, 26],
                             [24, 24, 25, 26, 26, 26],
                             [24, 24, 25, 26, 26, 26],
                             [27, 27, 28, 29, 29, 29],
                             [30, 30, 31, 32, 32, 32],
                             [33, 33, 34, 35, 35, 35],
                             [33, 33, 34, 35, 35, 35],
                             [33, 33, 34, 35, 35, 35],
                             [33, 33, 34, 35, 35, 35]]]]).astype(x_data_type)
    np.testing.assert_almost_equal(expect, out.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('x_data_type', [np.int16, np.float32])
@pytest.mark.parametrize('mode', ["constant", "edge"])
@pytest.mark.parametrize('ms_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('is_dyn_paddings', [True, False])
def test_padv3_constant_shape_5d(x_data_type, mode, ms_mode, is_dyn_paddings):
    """
    Feature: test padv3 x and const shape paddings
    Description: test padv3 with const shape paddings
    Expectation: Success
    """
    context.set_context(mode=ms_mode, device_target="Ascend")
    x = Tensor(np.arange(18).reshape(1, 1, 2, 3, 3).astype(x_data_type))
    paddings = (1, 2, 1, 1, 0, 1)
    if is_dyn_paddings:
        paddings = Tensor(paddings, dtype=ms.int64)
    value = None
    if mode == "constant":
        value = 99 if x_data_type == np.int16 else 99.0
    net = PadV3Net(mode)
    out = net(x, paddings, value)
    if mode == "constant":
        expect = np.array([[[[[99, 99, 99, 99, 99, 99],
                              [99, 0, 1, 2, 99, 99],
                              [99, 3, 4, 5, 99, 99],
                              [99, 6, 7, 8, 99, 99],
                              [99, 99, 99, 99, 99, 99]],
                             [[99, 99, 99, 99, 99, 99],
                              [99, 9, 10, 11, 99, 99],
                              [99, 12, 13, 14, 99, 99],
                              [99, 15, 16, 17, 99, 99],
                              [99, 99, 99, 99, 99, 99]],
                             [[99, 99, 99, 99, 99, 99],
                              [99, 99, 99, 99, 99, 99],
                              [99, 99, 99, 99, 99, 99],
                              [99, 99, 99, 99, 99, 99],
                              [99, 99, 99, 99, 99, 99]]]]]).astype(x_data_type)
    else:
        expect = np.array([[[[[0, 0, 1, 2, 2, 2],
                              [0, 0, 1, 2, 2, 2],
                              [3, 3, 4, 5, 5, 5],
                              [6, 6, 7, 8, 8, 8],
                              [6, 6, 7, 8, 8, 8]],
                             [[9, 9, 10, 11, 11, 11],
                              [9, 9, 10, 11, 11, 11],
                              [12, 12, 13, 14, 14, 14],
                              [15, 15, 16, 17, 17, 17],
                              [15, 15, 16, 17, 17, 17]],
                             [[9, 9, 10, 11, 11, 11],
                              [9, 9, 10, 11, 11, 11],
                              [12, 12, 13, 14, 14, 14],
                              [15, 15, 16, 17, 17, 17],
                              [15, 15, 16, 17, 17, 17]]]]]).astype(x_data_type)
    np.testing.assert_almost_equal(expect, out.asnumpy())
