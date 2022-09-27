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

import numpy as np
import pytest

import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, ms_function
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.select = P.Select()

    def construct(self, cond_op, input_x, input_y):
        return self.select(cond_op, input_x, input_y)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_select():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    select = Net()
    cond = np.array([[True, False], [True, False]]).astype(np.bool)
    x = np.array([[1.2, 1], [1, 0]]).astype(np.float32)
    y = np.array([[1, 2], [3, 4.0]]).astype(np.float32)
    output = select(Tensor(cond), Tensor(x), Tensor(y))
    expect = [[1.2, 2], [1, 4.0]]
    error = np.ones(shape=[2, 2]) * 1.0e-6
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([[1, 0], [1, 0]]).astype(np.bool)
    y = np.array([[0, 0], [1, 1]]).astype(np.bool)
    output = select(Tensor(cond), Tensor(x), Tensor(y))
    expect = np.array([[1, 0], [1, 1]]).astype(np.bool)
    assert np.all(output.asnumpy() == expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.array([[1, 0], [1, 0]]).astype(np.bool)
    y = np.array([[0, 0], [1, 1]]).astype(np.bool)
    output = select(Tensor(cond), Tensor(x), Tensor(y))
    expect = np.array([[1, 0], [1, 1]]).astype(np.bool)
    assert np.all(output.asnumpy() == expect)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@ms_function
def select_tensor_fn(condition, x, y):
    return x.select(condition, y)


@ms_function
def select_ops_fn(condition, x, y):
    return ops.select(condition, x, y)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
