# Copyright 2024 Huawei Technologies Co., Ltd
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

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import auto_generate as ops
from mindspore import context
from mindspore.common import mutable

context.set_context(mode=context.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.tuple_to_tensor = ops.TupleToTensor()
        self.scalar_to_tensor = ops.ScalarToTensor()

    def construct(self, x, y):
        return self.tuple_to_tensor(x), self.scalar_to_tensor(y)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_seq_to_tensor0():
    """
    Feature: test xxToTensor.
    Description: inputs is dynamic sequence or scalar; DType=None
    Expectation: the result match with numpy result
    """
    x0 = mutable((1, 2, 3), True)
    y0 = mutable(3)
    expect_x0 = np.array([1, 2, 3], dtype=np.int64)
    expect_y0 = np.array(3, dtype=np.int64)
    net = Net()
    res_x, res_y = net(x0, y0)
    assert np.allclose(res_x.asnumpy(), expect_x0, 1.e-4, 1.e-4, equal_nan=True)
    assert np.allclose(res_y.asnumpy(), expect_y0, 1.e-4, 1.e-4, equal_nan=True)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_seq_to_tensor1():
    """
    Feature: test xxToTensor.
    Description: inputs is dynamic sequence or scalar; DType=None
    Expectation: the result match with numpy result
    """
    x0 = mutable((1.1, 2.1, 3.1), True)
    y0 = mutable(3.1)
    expect_x0 = np.array([1.1, 2.1, 3.1], dtype=np.float32)
    expect_y0 = np.array(3.1, dtype=np.float32)
    net = Net()
    res_x, res_y = net(x0, y0)
    assert np.allclose(res_x.asnumpy(), expect_x0, 1.e-4, 1.e-4, equal_nan=True)
    assert np.allclose(res_y.asnumpy(), expect_y0, 1.e-4, 1.e-4, equal_nan=True)

class Net1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.tuple_to_tensor = ops.TupleToTensor()
        self.scalar_to_tensor = ops.ScalarToTensor()

    def construct(self, x, y):
        return self.tuple_to_tensor(x, mstype.int64), self.scalar_to_tensor(y, mstype.int64)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_seq_to_tensor2():
    """
    Feature: test xxToTensor.
    Description: inputs is dynamic sequence or scalar; DType=int64
    Expectation: the result match with numpy result
    """
    x0 = mutable((1, 2, 3), True)
    y0 = mutable(3)
    expect_x0 = np.array([1, 2, 3], dtype=np.int64)
    expect_y0 = np.array(3, dtype=np.int64)
    net = Net1()
    res_x, res_y = net(x0, y0)
    assert np.allclose(res_x.asnumpy(), expect_x0, 1.e-4, 1.e-4, equal_nan=True)
    assert np.allclose(res_y.asnumpy(), expect_y0, 1.e-4, 1.e-4, equal_nan=True)

class Net2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.tuple_to_tensor = ops.TupleToTensor()
        self.scalar_to_tensor = ops.ScalarToTensor()

    def construct(self, x, y):
        return self.tuple_to_tensor(x, mstype.float32), self.scalar_to_tensor(y, mstype.float32)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_seq_to_tensor3():
    """
    Feature: test xxToTensor.
    Description: inputs is dynamic sequence or scalar; DType=float32
    Expectation: the result match with numpy result
    """
    x0 = mutable((1.1, 2.1, 3.1), True)
    y0 = mutable(3.1)
    expect_x0 = np.array([1.1, 2.1, 3.1], dtype=np.float32)
    expect_y0 = np.array(3.1, dtype=np.float32)
    net = Net2()
    res_x, res_y = net(x0, y0)
    assert np.allclose(res_x.asnumpy(), expect_x0, 1.e-4, 1.e-4, equal_nan=True)
    assert np.allclose(res_y.asnumpy(), expect_y0, 1.e-4, 1.e-4, equal_nan=True)
