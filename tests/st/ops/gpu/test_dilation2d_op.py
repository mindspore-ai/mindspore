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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
import mindspore.ops.operations.nn_ops as ops


class NetDilation2D(nn.Cell):
    def __init__(self):
        super(NetDilation2D, self).__init__()
        self.dilation2d = ops.Dilation2D(stride=1, dilation=2, pad_mode='SAME', data_format='NCHW')

    def construct(self, x, fil):
        return self.dilation2d(x, fil)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dilation2d_graph():
    """
    Feature: dilation2d
    Description: Test of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    types = [np.float16, np.float32, np.float64, np.int32, np.int64,
             np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16]
    for type_i in types:
        x = Tensor(np.ones([1, 1, 4, 4]).astype(type_i))
        fil = Tensor(np.ones([1, 2, 2]).astype(type_i))
        dilation2d = NetDilation2D()
        output = dilation2d(x, fil).transpose(0, 2, 3, 1)
        expect_output = np.array([[[[2.], [2.], [2.], [2.]],
                                   [[2.], [2.], [2.], [2.]],
                                   [[2.], [2.], [2.], [2.]],
                                   [[2.], [2.], [2.], [2.]]]]).astype(type_i)
        assert np.allclose(output.asnumpy(), expect_output)
        assert output.shape == expect_output.shape


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dilation2d_pynative():
    """
    Feature: dilation2d
    Description: Test of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    types = [np.float16, np.float32, np.float64, np.int32, np.int64,
             np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16]
    for type_i in types:
        x = Tensor(np.ones([1, 1, 4, 4]).astype(type_i))
        fil = Tensor(np.ones([1, 2, 2]).astype(type_i))
        dilation2d = NetDilation2D()
        output = dilation2d(x, fil).transpose(0, 2, 3, 1)
        expect_output = np.array([[[[2.], [2.], [2.], [2.]],
                                   [[2.], [2.], [2.], [2.]],
                                   [[2.], [2.], [2.], [2.]],
                                   [[2.], [2.], [2.], [2.]]]]).astype(type_i)
        assert np.allclose(output.asnumpy(), expect_output)
        assert output.shape == expect_output.shape
