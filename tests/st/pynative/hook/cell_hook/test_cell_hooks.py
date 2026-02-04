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
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from tests.mark_utils import arg_mark
from tests.st.pynative.hook.cell_hook.common import assert_equal

def forward_pre_hook_fn1(cell, inputs):
    pass

def forward_pre_hook_fn2(cell, inputs):
    return inputs[0] + 2, inputs[1] + 3

def forward_pre_hook_fn3(cell, inputs):
    return inputs[0] * 2, inputs[1] * 3

def forward_hook_fn1(cell, inputs, outputs):
    pass

def forward_hook_fn2(cell, inputs, outputs):
    return (inputs[0] + outputs[0], inputs[1] + outputs[1], outputs[2])

def forward_hook_fn3(cell, inputs, outputs):
    return (inputs[0] * outputs[0], inputs[1] * outputs[1], outputs[2])

def backward_pre_hook_fn1(cell, grad_output):
    pass

def backward_pre_hook_fn2(cell, grad_output):
    return (grad_output[0] * 2, grad_output[1] * 3, grad_output[2] * 4)

def backward_pre_hook_fn3(cell, grad_output):
    return (grad_output[0] + 2, grad_output[1] + 3, grad_output[2] + 4)

def backward_hook_fn1(cell, grad_input, grad_output):
    pass

def backward_hook_fn2(cell, grad_input, grad_output):
    return (grad_input[0] * grad_output[0], grad_input[1] * grad_output[1] * grad_output[2])

def backward_hook_fn3(cell, grad_input, grad_output):
    return (grad_input[0] + grad_output[0], grad_input[1] + grad_output[1] + grad_output[2])

class HookNet(nn.Cell):
    def __init__(self):
        super(HookNet, self).__init__()
        self.mul = ops.Mul()
        self.w1 = ms.Parameter(ms.Tensor(10.0, ms.float32), name="w1")
        self.w2 = ms.Parameter(ms.Tensor(20.0, ms.float32), name="w2")

    def construct(self, x, y):
        o1 = self.mul(x, y)
        o2 = o1 * self.w1
        o3 = o2 * self.w2
        return o1, o2, o3

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.hook_net = HookNet()

    @ms.jit
    def construct(self, x, y):
        return self.hook_net(x, y)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('reg_inner_net', [True])
def test_cell_hooks(mode, reg_inner_net):
    """
    Feature: Cell Hooks
    Description: Test all cell hooks all in one
    Expectation: success
    """
    ms.set_context(mode=mode)

    x = ms.Tensor(np.arange(2, 5).astype(np.float32))
    y = ms.Tensor(np.arange(5, 8).astype(np.float32))

    net = Net()
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())

    expected_out1 = (
        ms.Tensor([10, 18, 28], ms.float32),
        ms.Tensor([100, 180, 280], ms.float32),
        ms.Tensor([2000, 3600, 5600], ms.float32)
    )

    expected_grad1 = (
        (ms.Tensor([1055, 1266, 1477], ms.float32), ms.Tensor([422, 633, 844], ms.float32)),
        (ms.Tensor(1176, ms.float32), ms.Tensor(560, ms.float32))
    )

    py_out = net(x, y)
    py_grad = grad_net(x, y)
    assert_equal(py_out, expected_out1)
    assert_equal(py_grad, expected_grad1)


    if reg_inner_net:
        reg_net = net.hook_net
    else:
        reg_net = net

    handle11 = reg_net.register_forward_pre_hook(forward_pre_hook_fn1)
    handle12 = reg_net.register_forward_pre_hook(forward_pre_hook_fn2)
    handle13 = reg_net.register_forward_pre_hook(forward_pre_hook_fn3)
    handle21 = reg_net.register_forward_hook(forward_hook_fn1)
    handle22 = reg_net.register_forward_hook(forward_hook_fn2)
    handle23 = reg_net.register_forward_hook(forward_hook_fn3)
    handle31 = reg_net.register_backward_pre_hook(backward_pre_hook_fn1)
    handle32 = reg_net.register_backward_pre_hook(backward_pre_hook_fn2)
    handle33 = reg_net.register_backward_pre_hook(backward_pre_hook_fn3)
    handle41 = reg_net.register_backward_hook(backward_hook_fn1)
    handle42 = reg_net.register_backward_hook(backward_hook_fn2)
    handle43 = reg_net.register_backward_hook(backward_hook_fn3)

    py_out = net(x, y)
    py_grad = grad_net(x, y)

    expected_out2 = (
        ms.Tensor([1600, 2800, 4464], ms.float32),
        ms.Tensor([46656, 73629, 108900], ms.float32),
        ms.Tensor([38400, 54000, 72000], ms.float32)
    )
    expected_grad2 = (
        (ms.Tensor([596488, 713448, 839816], ms.float32), ms.Tensor([5239338, 7073898, 9120426], ms.float32)),
        (ms.Tensor(267708, ms.float32), ms.Tensor(65760, ms.float32))
    )

    assert_equal(py_out, expected_out2)
    assert_equal(py_grad, expected_grad2)

    handle11.remove()
    handle12.remove()
    handle13.remove()
    handle21.remove()
    handle22.remove()
    handle23.remove()
    handle31.remove()
    handle32.remove()
    handle33.remove()
    handle41.remove()
    handle42.remove()
    handle43.remove()

    py_out = net(x, y)
    py_grad = grad_net(x, y)

    assert_equal(py_out, expected_out1)
    assert_equal(py_grad, expected_grad1)
