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
import pytest
import numpy as np
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.ops import composite as C
from mindspore import context, Tensor
from mindspore.common.api import jit

grad_all = C.GradOperation(get_all=True)


def var_hook_function(grad_out):
    print("grad:", grad_out)


class GraphVarHook(nn.Cell):
    def __init__(self):
        super(GraphVarHook, self).__init__()
        self.relu = nn.ReLU()
        self.hook = P.HookBackward(var_hook_function)

    def construct(self, x):
        x = x + x
        x = x * x
        x = self.hook(x)
        x = self.relu(x)
        return x


class MsFuncVarHook(nn.Cell):
    def __init__(self):
        super(MsFuncVarHook, self).__init__()
        self.relu = nn.ReLU()
        self.hook = P.HookBackward(var_hook_function)

    @jit
    def construct(self, x):
        x = x + x
        x = x * x
        x = self.hook(x)
        x = self.relu(x)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_var_hook_forward():
    input_x = Tensor(np.random.randn(2, 2).astype(np.float32))
    context.set_context(mode=context.PYNATIVE_MODE)
    net1 = MsFuncVarHook()
    out1 = net1(input_x)
    context.set_context(mode=context.GRAPH_MODE)
    net2 = GraphVarHook()
    out2 = net2(input_x)
    assert np.allclose(out1.asnumpy(), out2.asnumpy(), 0.00001, 0.00001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_var_hook_grad():
    input_x = Tensor(np.random.randn(2, 2).astype(np.float32))
    context.set_context(mode=context.PYNATIVE_MODE)
    net1 = MsFuncVarHook()
    grad_out1 = grad_all(net1)(input_x)
    context.set_context(mode=context.GRAPH_MODE)
    net2 = GraphVarHook()
    grad_out2 = grad_all(net2)(input_x)
    assert np.allclose(grad_out1[0].asnumpy(), grad_out2[0].asnumpy(), 0.00001, 0.00001)


def cell_hook_function(cell_id, grad_input, grad_output):
    print("cell id:", cell_id)
    print("grad input:", grad_input)
    print("grad output:", grad_output)


class GraphCellHook(nn.Cell):
    def __init__(self):
        super(GraphCellHook, self).__init__()
        self.relu = nn.ReLU()
        self.relu.register_backward_hook(cell_hook_function)

    def construct(self, x):
        x = x + x
        x = x * x
        x = self.relu(x)
        return x


class MsFuncCellHook(nn.Cell):
    def __init__(self):
        super(MsFuncCellHook, self).__init__()
        self.relu = nn.ReLU()
        self.relu.register_backward_hook(cell_hook_function)

    @jit
    def construct(self, x):
        x = x + x
        x = x * x
        x = self.relu(x)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cell_hook_forward():
    input_x = Tensor(np.random.randn(2, 2).astype(np.float32))
    context.set_context(mode=context.PYNATIVE_MODE)
    net1 = MsFuncCellHook()
    out1 = net1(input_x)
    context.set_context(mode=context.GRAPH_MODE)
    net2 = GraphCellHook()
    out2 = net2(input_x)
    assert np.allclose(out1.asnumpy(), out2.asnumpy(), 0.00001, 0.00001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cell_hook_grad():
    input_x = Tensor(np.random.randn(2, 2).astype(np.float32))
    context.set_context(mode=context.PYNATIVE_MODE)
    net1 = MsFuncCellHook()
    grad_out1 = grad_all(net1)(input_x)
    context.set_context(mode=context.GRAPH_MODE)
    net2 = GraphCellHook()
    grad_out2 = grad_all(net2)(input_x)
    assert np.allclose(grad_out1[0].asnumpy(), grad_out2[0].asnumpy(), 0.00001, 0.00001)
