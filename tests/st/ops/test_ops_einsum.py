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
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops


class Net(nn.Cell):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation

    def construct(self, *operands):
        return ops.einsum(self.equation, *operands)


class NetSublist(nn.Cell):
    """Test ops.einsum in sublist format."""

    def construct(self, x, y):
        return ops.einsum(x, [..., 0, 1], y, [..., 1, 2], [..., 0, 2])


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_einsum(mode):
    """
    Feature: ops.einsum
    Description: Verify the result of einsum
    Expectation: success
    """
    ms.set_context(mode=mode)

    x = Tensor(np.array([1.0, 2.0, 4.0]), ms.float32)
    equation = "i->"
    net = Net(equation)
    output = net(x)
    expect_output = [7.]
    assert np.allclose(output.asnumpy(), expect_output)

    x = Tensor(np.array([1.0, 2.0, 4.0]), ms.float32)
    y = Tensor(np.array([2.0, 4.0, 3.0]), ms.float32)
    equation = "i,i->i"
    net = Net(equation)
    output = net(x, y)
    expect_output = [2., 8., 12.]
    assert np.allclose(output.asnumpy(), expect_output)

    x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), ms.float32)
    y = Tensor(np.array([[2.0, 3.0], [1.0, 2.0], [4.0, 5.0]]), ms.float32)
    equation = "ij,jk->ik"
    net = Net(equation)
    output = net(x, y)
    expect_output = [[16., 22.],
                     [37., 52.]]
    assert np.allclose(output.asnumpy(), expect_output)

    x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), ms.float32)
    equation = "ij->ji"
    net = Net(equation)
    output = net(x)
    expect_output = [[1., 4.],
                     [2., 5.],
                     [3., 6.]]
    assert np.allclose(output.asnumpy(), expect_output)

    x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), ms.float32)
    equation = "ij->j"
    net = Net(equation)
    output = net(x)
    expect_output = [5., 7., 9.]
    assert np.allclose(output.asnumpy(), expect_output)

    x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), ms.float32)
    equation = "...->"
    net = Net(equation)
    output = net(x)
    expect_output = [21.]
    assert np.allclose(output.asnumpy(), expect_output)

    x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = Tensor(np.array([2.0, 4.0, 1.0]), ms.float32)
    equation = "j,i->ji"
    net = Net(equation)
    output = net(x, y)
    expect_output = [[2., 4., 1.],
                     [4., 8., 2.],
                     [6., 12., 3.]]
    assert np.allclose(output.asnumpy(), expect_output)

    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ms.float32)
    y = Tensor([[2.0, 3.0], [1.0, 2.0], [4.0, 5.0]], ms.float32)
    net = NetSublist()
    output = net(x, y)
    expect_output = [[16., 22.],
                     [37., 52.]]
    assert np.allclose(output.asnumpy(), expect_output)
