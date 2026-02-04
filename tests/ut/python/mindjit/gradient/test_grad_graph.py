# Copyright 2021 Huawei Technologies Co., Ltd
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
"""test function grad in graph mode"""
import numpy as np
import pytest
from mindspore import Tensor, ops, nn, context
from mindspore.ops.functional import grad
from mindspore.common import dtype
from mindspore.graph.api import _pynative_executor

context.set_context(mode=context.GRAPH_MODE)


class SingleInputSingleOutputNet(nn.Cell):
    def construct(self, x):
        return x**3


class MultipleInputsMultipleOutputsNet(nn.Cell):
    def construct(self, x, y, z):
        return x**2 + y**2 + z**2, x*y*z


def function(x, y, z):
    return x**2 + y**2 + z**2, x*y*z


def test_grad_single_input_single_output_cell_graph():
    """
    Features: Function grad.
    Description: Test F.grad with single input and single output net in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputSingleOutputNet()
    grad(net)(x)


def test_grad_multiple_inputs_multiple_outputs_cell_graph():
    """
    Features: Function grad.
    Description: Test F.grad with multiple inputs and multiple outputs net in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    net = MultipleInputsMultipleOutputsNet()
    grad(net, grad_position=(1, 2))(x, y, z)


class NetMul(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul = ops.Mul()

    def construct(self, x, y):
        out = self.mul(x, y)
        return out


@pytest.mark.parametrize(
    "grad_position_error",
    ((-1, ValueError), (2, (IndexError, RuntimeError)), (1.0, TypeError),
    ((0, 0, 1), ValueError), ((0, -1), ValueError), ((0.0, 1.0), TypeError),
    ((0, 2), (IndexError, RuntimeError)), ([0, 1], TypeError),
))
def test_grad_invalid_position(grad_position_error):
    """
    Features: Function grad.
    Description: Test F.grad with invalid grad position in graph mode.
    Expectation: No exception.
    """
    net = NetMul()
    with pytest.raises(grad_position_error[1]):
        grad_net = grad(net, grad_position=grad_position_error[0])
        x = Tensor([1, 2, 3], dtype.float32)
        y = Tensor([1, 2, 3], dtype.float32)
        grad_net(x, y)
        _pynative_executor.sync()
