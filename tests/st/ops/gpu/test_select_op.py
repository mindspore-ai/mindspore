# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, jit
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.select = P.Select()

    def construct(self, cond_op, input_x, input_y):
        return self.select(cond_op, input_x, input_y)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("dtype", [np.bool_, np.int8, np.int16, np.int32, np.int64,
                                   np.uint8, np.uint16, np.uint32, np.uint64,
                                   np.float16, np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_select(dtype, mode):
    """
    Feature: ALL To ALL
    Description: test cases for Select
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="GPU")
    select = Net()
    cond = np.array([[True, False], [True, False]]).astype(np.bool)
    error_tol = np.ones(shape=[2, 2]) * 1.0e-3

    x = np.array([[1, 0], [1, 0]]).astype(dtype)
    y = np.array([[0, 0], [1, 1]]).astype(dtype)
    output = select(Tensor(cond), Tensor(x), Tensor(y))
    expect = np.array([[1, 0], [1, 1]]).astype(dtype)
    assert np.allclose(output.numpy(), expect, error_tol, error_tol)


def test_functional_select_scalar():
    """
    Feature: Test functional select operator. Support x or y is a int/float.
    Description: Operator select's input `x` is a Tensor with int32 type, input `y` is a int.
    Expectation: Assert result.
    """
    context.set_context(device_target="GPU")
    cond = np.array([[True, False], [True, False]]).astype(np.bool)
    x = np.array([[12, 1], [1, 0]]).astype(np.int32)
    y = 2
    output = ops.select(Tensor(cond), Tensor(x), y)
    expect = [[12, 2], [1, 2]]
    error = np.ones(shape=[2, 2]) * 1.0e-6
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@jit
def select_tensor_fn(condition, x, y):
    return x.select(condition, y)


@jit
def select_ops_fn(condition, x, y):
    return ops.select(condition, x, y)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tensor_select_input_tensor():
    """
    Feature: Test tensor select interface.
    Description: Operator select's input `y` is a Tensor.
    Expectation: Assert result.
    """
    cond = Tensor([True, False])
    x = Tensor([2, 3], mindspore.int32)
    y = Tensor([1, 2], mindspore.int32)
    output1 = x.select(cond, y)
    output2 = select_tensor_fn(cond, x, y)
    assert np.all(output1.asnumpy() == np.array([2, 2]))
    assert np.all(output2.asnumpy() == np.array([2, 2]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tensor_select_input_number():
    """
    Feature: Test tensor select interface.
    Description: Operator select's input `y` is a float number.
    Expectation: Assert result.
    """
    cond = Tensor([True, False])
    x = Tensor([2, 3], mindspore.int32)
    y = 5
    output1 = x.select(cond, y)
    output2 = select_tensor_fn(cond, x, y)
    assert np.all(output1.asnumpy() == np.array([2, 5]))
    assert np.all(output2.asnumpy() == np.array([2, 5]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_select_fn_vmap():
    """
    Feature: Test select function interface with interface.
    Description: Use select functional interface in vmap.
    Expectation: Assert result.
    """
    select_vmap_1 = ops.vmap(select_ops_fn, (0, 0, 0))
    select_vmap_2 = ops.vmap(select_ops_fn, (0, 0, None))
    condition = Tensor([[True, False], [True, True]])
    x = Tensor([[2, 3], [3, 1]], mindspore.int32)
    y = Tensor([[1, 2], [5, 6]], mindspore.int32)
    output1 = select_vmap_1(condition, x, y)
    output2 = select_vmap_2(condition, x, 5)
    assert np.all(output1.asnumpy() == np.array([[2, 2], [3, 1]]))
    assert np.all(output2.asnumpy() == np.array([[2, 5], [3, 1]]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_select_tensor_vmap():
    """
    Feature: Test select tensor interface with interface.
    Description: Use select functional interface in vmap.
    Expectation: Assert result.
    """
    select_vmap_1 = ops.vmap(select_tensor_fn, (0, 0, 0))
    select_vmap_2 = ops.vmap(select_tensor_fn, (0, 0, None))
    condition = Tensor([[True, False], [True, True]])
    x = Tensor([[2, 3], [3, 1]], mindspore.int32)
    y = Tensor([[1, 2], [5, 6]], mindspore.int32)
    output1 = select_vmap_1(condition, x, y)
    output2 = select_vmap_2(condition, x, 5)
    assert np.all(output1.asnumpy() == np.array([[2, 2], [3, 1]]))
    assert np.all(output2.asnumpy() == np.array([[2, 5], [3, 1]]))
