# Copyright 2020 Huawei Technologies Co., Ltd
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

from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner
import mindspore.nn as nn
import mindspore.context as context

class DynamicShapeNet(nn.Cell):
    def __init__(self):
        super(DynamicShapeNet, self).__init__()
        self.convert_to_dynamic_shape_op = inner.GpuConvertToDynamicShape()
        self.dynamic_shape_op = P.Shape()

    def construct(self, x):
        x_dynamic_shape = self.convert_to_dynamic_shape_op(x)
        return self.dynamic_shape_op(x_dynamic_shape)


def dynamic_shape(np_type):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    dynamic_shape_net = DynamicShapeNet()

    shape = (1,)
    x = Tensor(np.zeros(shape).astype(np_type))
    ms_out = dynamic_shape_net(x)
    expected = np.array(shape)
    np.testing.assert_array_equal(ms_out, expected)

    shape = (7,)
    x = Tensor(np.zeros(shape).astype(np_type))
    ms_out = dynamic_shape_net(x)
    expected = np.array(shape)
    np.testing.assert_array_equal(ms_out, expected)

    shape = (1, 1)
    x = Tensor(np.zeros(shape).astype(np_type))
    ms_out = dynamic_shape_net(x)
    expected = np.array(shape)
    np.testing.assert_array_equal(ms_out, expected)

    shape = (1, 7)
    x = Tensor(np.zeros(shape).astype(np_type))
    ms_out = dynamic_shape_net(x)
    expected = np.array(shape)
    np.testing.assert_array_equal(ms_out, expected)

    shape = (3, 1)
    x = Tensor(np.zeros(shape).astype(np_type))
    ms_out = dynamic_shape_net(x)
    expected = np.array(shape)
    np.testing.assert_array_equal(ms_out, expected)

    shape = (2, 4)
    x = Tensor(np.zeros(shape).astype(np_type))
    ms_out = dynamic_shape_net(x)
    expected = np.array(shape)
    np.testing.assert_array_equal(ms_out, expected)

    shape = (1, 1, 1)
    x = Tensor(np.zeros(shape).astype(np_type))
    ms_out = dynamic_shape_net(x)
    expected = np.array(shape)
    np.testing.assert_array_equal(ms_out, expected)

    shape = (1, 5, 3)
    x = Tensor(np.zeros(shape).astype(np_type))
    ms_out = dynamic_shape_net(x)
    expected = np.array(shape)
    np.testing.assert_array_equal(ms_out, expected)

    shape = (2, 3, 1, 3, 1)
    x = Tensor(np.zeros(shape).astype(np_type))
    ms_out = dynamic_shape_net(x)
    expected = np.array(shape)
    np.testing.assert_array_equal(ms_out, expected)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_int32():
    dynamic_shape(np.int32)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_float16():
    dynamic_shape(np.float16)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_float32():
    dynamic_shape(np.float32)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_bool():
    dynamic_shape(np.bool)
