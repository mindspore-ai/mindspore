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
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import GradOperation
from mindspore.common import ParameterTuple


def forward_pre_hook_fn_add(cell_id, inp):
    x = inp[0] + inp[0]
    return x


def forward_pre_hook_fn_mul(cell_id, inp):
    x = inp[0] * inp[0]
    return x


def forward_hook_fn_relu(cell_id, inp, outp):
    out = nn.ReLU()(outp)
    return out


def forward_hook_fn_add(cell_id, inp, outp):
    out = outp + outp
    return out


def backward_hook_fn(cell_id, grad_inp, grad_outp):
    return Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32))


def backward_hook_fn2(cell_id, grad_inp, grad_outp):
    return Tensor(np.ones([1]).astype(np.float32) * 2), Tensor(np.ones([1]).astype(np.float32) * 3)


def backward_hook_fn3(cell_id, grad_inp, grad_outp):
    return Tensor(np.ones([1]).astype(np.float32) * 5), Tensor(np.ones([1]).astype(np.float32) * 6)


def backward_hook_fn4(cell_id, grad_inp, grad_outp):
    return (Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 10),)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = nn.MatMul()
        self.handle = self.mul.register_backward_hook(backward_hook_fn)

    def construct(self, x, y):
        x = self.mul(x, y)
        x = x + x
        return x


class SingleNet(nn.Cell):
    def __init__(self):
        super(SingleNet, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CmpNet(nn.Cell):
    def __init__(self):
        super(CmpNet, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CmpNetPreHook(nn.Cell):
    def __init__(self):
        super(CmpNetPreHook, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")

    def construct(self, x):
        x = x + x
        x = x * x
        x = self.conv(x)
        x = self.bn(x)
        return x


class CmpNetFWHook(nn.Cell):
    def __init__(self):
        super(CmpNetFWHook, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x + x
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_backward_hook():
    """
    Feature: PyNative hook function.
    Description: Test PyNative backward hook function.
    Expectation: The calculation result is correct.
    """

    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.ones([1]).astype(np.float32))
    input_y = Tensor(np.ones([1]).astype(np.float32))
    grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
    # case 1: register hook function in __init__ function.
    net = Net()
    grad = grad_op(net)(input_x, input_y)
    assert len(grad) == 2
    assert np.allclose(grad[0].asnumpy(), input_x.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), input_x.asnumpy(), 0.000001, 0.000001)
    # case 2: remove hook function by handle.
    net.handle.remove()
    net.handle.remove()
    grad = grad_op(net)(input_x, input_y)
    assert len(grad) == 2
    expect_grad = Tensor(np.ones([1]).astype(np.float32) * 2)
    assert np.allclose(grad[0].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)
    # case 3: register hook function by handle
    net = Net()
    net.mul.register_backward_hook(backward_hook_fn2)
    handle3 = net.mul.register_backward_hook(backward_hook_fn3)
    grad = grad_op(net)(input_x, input_y)
    assert len(grad) == 2
    expect_gradx = Tensor(np.ones([1]).astype(np.float32) * 5)
    expect_grady = Tensor(np.ones([1]).astype(np.float32) * 6)
    assert np.allclose(grad[0].asnumpy(), expect_gradx.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grady.asnumpy(), 0.000001, 0.000001)
    # case 5: remove hook function by handle.
    handle3.remove()
    grad = grad_op(net)(input_x, input_y)
    assert len(grad) == 2
    expect_gradx = Tensor(np.ones([1]).astype(np.float32) * 2)
    expect_grady = Tensor(np.ones([1]).astype(np.float32) * 3)
    assert np.allclose(grad[0].asnumpy(), expect_gradx.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grady.asnumpy(), 0.000001, 0.000001)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_hook_base_line():
    """
    Feature: PyNative hook function.
    Description: The base line case for PyNative hook function.
    Expectation: The calculation result is correct.
    """

    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 2)
    grad_op = GradOperation(get_all=True, get_by_list=True, sens_param=False)
    # register pre forward hook.
    net = SingleNet()
    handle1 = net.conv.register_forward_pre_hook(forward_pre_hook_fn_add)
    handle2 = net.conv.register_forward_pre_hook(forward_pre_hook_fn_mul)
    out = net(input_x)
    cmp_net_pre_hook = CmpNetPreHook()
    expect_out = cmp_net_pre_hook(input_x)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(input_x)
    expect_grad = grad_op(cmp_net_pre_hook, ParameterTuple(cmp_net_pre_hook.trainable_params()))(input_x)
    assert len(grad) == len(expect_grad)
    assert np.allclose(grad[0][0].asnumpy(), expect_grad[0][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][1].asnumpy(), expect_grad[1][1].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][2].asnumpy(), expect_grad[1][2].asnumpy(), 0.000001, 0.000001)
    # register forward hook.
    handle1.remove()
    handle2.remove()
    handlea = net.bn.register_forward_hook(forward_hook_fn_relu)
    handleb = net.bn.register_forward_hook(forward_hook_fn_add)
    out = net(input_x)
    cmp_net_fw_hook = CmpNetFWHook()
    expect_out = cmp_net_fw_hook(input_x)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(input_x)
    expect_grad = grad_op(cmp_net_fw_hook, ParameterTuple(cmp_net_fw_hook.trainable_params()))(input_x)
    assert len(grad) == len(expect_grad)
    assert np.allclose(grad[0][0].asnumpy(), expect_grad[0][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][1].asnumpy(), expect_grad[1][1].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][2].asnumpy(), expect_grad[1][2].asnumpy(), 0.000001, 0.000001)
    # register backward hook.
    handlea.remove()
    handleb.remove()
    net.conv.register_backward_hook(backward_hook_fn4)
    out = net(input_x)
    compare_net = CmpNet()
    expect_out = compare_net(input_x)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(input_x)
    expect_grad = grad_op(compare_net, ParameterTuple(compare_net.trainable_params()))(input_x)
    assert len(grad) == len(expect_grad)
    expect_gradx = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 10)
    assert np.allclose(grad[0][0].asnumpy(), expect_gradx.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][1].asnumpy(), expect_grad[1][1].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][2].asnumpy(), expect_grad[1][2].asnumpy(), 0.000001, 0.000001)
