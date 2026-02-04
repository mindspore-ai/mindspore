# Copyright 2025 Huawei Technologies Co., Ltd
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
from mindspore import ops, nn, jit, context
from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfFirstInput, GradOfAllInputs


def hook_fn_add(grads):
    return tuple([grad + 1.0 for grad in grads])


def hook_fn_return_single(grads):
    return grads[0] + 1.0


def hook_fn_return_none(unused):
    return


def hook_fn_return_int(unused):
    return 2


def hook_fn_duplicate_grads(grads):
    return grads + grads


class HookNet(nn.Cell):
    def __init__(self, hook):
        super(HookNet, self).__init__()
        self.hook = ops.HookBackward(hook)

    def construct(self, x):
        x = x + x
        x = self.hook(x)
        x = x * x
        return x


class MultiInputHookNet(nn.Cell):
    def __init__(self, hook):
        super(MultiInputHookNet, self).__init__()
        self.hook = ops.HookBackward(hook)

    def construct(self, x, y):
        x1 = x + x
        y1 = y * y
        x1, y1 = self.hook(x1, y1)
        return x1 + y1


def test_hook_backward_ops_op_output():
    """
    Feature: Hook backward ops
    Description: Test hook backward ops for op output.
    Expectation: Success
    """
    input_x = ms.Tensor(np.random.rand(5, 5), dtype=ms.float32)
    expect_grad = (input_x * 4.0 + 1.0) * 2.0

    net = HookNet(hook_fn_add)
    grad_x = GradOfFirstInput(net, sens_param=False)(input_x)
    assert np.allclose(grad_x.asnumpy(), expect_grad.asnumpy(), 0.00001, 0.00001)

    net = HookNet(hook_fn_return_single)
    grad_x = GradOfFirstInput(net, sens_param=False)(input_x)
    assert np.allclose(grad_x.asnumpy(), expect_grad.asnumpy(), 0.00001, 0.00001)

    net = HookNet(hook_fn_return_none)
    grad_x = GradOfFirstInput(net, sens_param=False)(input_x)
    assert np.allclose(grad_x.asnumpy(), (input_x * 8.0).asnumpy(), 0.00001, 0.00001)


def test_hook_backward_ops_multi_input():
    """
    Feature: Hook backward ops
    Description: Test hook backward ops for multi input.
    Expectation: Success
    """

    input_x = ms.Tensor(np.random.rand(3, 3), dtype=ms.float32)
    input_y = ms.Tensor(np.random.rand(3, 3), dtype=ms.float32)

    net = MultiInputHookNet(hook_fn_add)
    grad_x, grad_y = GradOfAllInputs(net, sens_param=False)(input_x, input_y)
    assert np.allclose(grad_x.asnumpy(), np.ones_like(input_x) * 4, 0.00001, 0.00001)
    assert np.allclose(grad_y.asnumpy(), (input_y * 4).asnumpy(), 0.00001, 0.00001)


def var_hook_function(grad_out):
    print("grad:", grad_out)


class GraphVarHook(nn.Cell):
    def __init__(self):
        super(GraphVarHook, self).__init__()
        self.relu = nn.ReLU()
        self.hook = ops.HookBackward(var_hook_function)

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
        self.hook = ops.HookBackward(var_hook_function)

    @jit
    def construct(self, x):
        x = x + x
        x = x * x
        x = self.hook(x)
        x = self.relu(x)
        return x


def test_hook_backward_ops_graph_and_jit():
    """
    Feature: Hook backward ops
    Description: Test hook backward ops in graph mode and jit.
    Expectation: Success
    """
    input_x = ms.Tensor(np.random.randn(2, 2).astype(np.float32))
    context.set_context(mode=context.PYNATIVE_MODE)
    net1 = MsFuncVarHook()
    out1, grad_out1 = ms.value_and_grad(net1, grad_position=0)(input_x)
    context.set_context(mode=context.GRAPH_MODE)
    net2 = GraphVarHook()
    out2, grad_out2 = ms.value_and_grad(net2, grad_position=0)(input_x)
    assert np.allclose(out1.asnumpy(), out2.asnumpy(), 0.00001, 0.00001)
    assert np.allclose(grad_out1.asnumpy(), grad_out2.asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_hook_backward_ops():
    """
    Feature: Hook backward ops
    Description: Test suite for hook backward ops
    Expectation: Success
    """
    test_hook_backward_ops_op_output()
    test_hook_backward_ops_multi_input()
    test_hook_backward_ops_graph_and_jit()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_hook_backward_ops_hook_return_error():
    """
    Feature: Hook backward ops
    Description: Hook function return error.
    Expectation: Raise correct error.
    """
    input_x = ms.Tensor(np.random.rand(5, 5), dtype=ms.float32)
    input_y = ms.Tensor(np.random.rand(5, 5), dtype=ms.float32)

    with pytest.raises(TypeError):
        net = HookNet(hook_fn_return_int)
        GradOfFirstInput(net, sens_param=False)(input_x)

    with pytest.raises(ValueError):
        net = MultiInputHookNet(hook_fn_duplicate_grads)
        GradOfFirstInput(net, sens_param=False)(input_x, input_y)
