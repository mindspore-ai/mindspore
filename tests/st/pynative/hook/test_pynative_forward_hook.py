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
from mindspore.common.api import jit
from mindspore import ops as P


def forward_pre_hook_fn_bn(cell_id, inp):
    out = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")(inp[0])
    return out


def forward_pre_hook_fn_add(cell_id, inp):
    out = inp[0] + inp[0]
    return out


def forward_pre_hook_fn_mul(cell_id, inp):
    out = inp[0] * inp[0]
    return out


def forward_pre_hook_fn_multi_relu(cell_id, inp):
    out = nn.ReLU()(inp[0])
    return out, inp[1]


def forward_pre_hook_fn_multi_add(cell_id, inp):
    x = inp[0] + inp[1]
    y = inp[0] * inp[1]
    return x, y


def forward_hook_fn_conv(cell_id, inp, outp):
    out = P.Log()(outp)
    return out


def forward_hook_fn_add(cell_id, inp, outp):
    out = outp + outp
    return out


def forward_hook_fn_mul(cell_id, inp, outp):
    out = outp * outp
    return out


@jit
def forward_hook_fn_with_ms_func(cell_id, inp, outp):
    return outp


def backward_hook_fn(cell_id, grad_inp, grad_outp):
    print("Enter backward hook function.")
    return grad_outp


def backward_hook_fn_inner(cell_id, grad_inp, grad_outp):
    print("Enter backward hook function inner.")
    return grad_outp


class SingleNet(nn.Cell):
    def __init__(self):
        super(SingleNet, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.relu = nn.ReLU()
        self.handle1 = self.conv.register_forward_hook(forward_hook_fn_add)
        self.handle2 = self.conv.register_forward_pre_hook(forward_pre_hook_fn_add)
        self.handle3 = self.relu.register_forward_hook(forward_hook_fn_add)
        self.handle4 = self.relu.register_forward_pre_hook(forward_pre_hook_fn_bn)
        self.handle1.remove()
        self.handle1.remove()
        self.handle2.remove()
        self.handle2.remove()

    def construct(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class SingleNetInConstruct(nn.Cell):
    def __init__(self):
        super(SingleNetInConstruct, self).__init__()
        self.handle1 = None
        self.handle2 = None
        self.handle3 = None
        self.handle4 = None
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.relu = nn.ReLU()

    def construct(self, x):
        self.handle1 = self.conv.register_forward_hook(forward_hook_fn_add)
        self.handle2 = self.conv.register_forward_pre_hook(forward_pre_hook_fn_add)
        self.handle3 = self.relu.register_forward_hook(forward_hook_fn_add)
        self.handle4 = self.relu.register_forward_pre_hook(forward_pre_hook_fn_bn)
        self.handle1.remove()
        self.handle1.remove()
        self.handle2.remove()
        self.handle2.remove()
        x = self.conv(x)
        x = self.relu(x)
        return x


class SingleNetMsFuncInner(nn.Cell):
    def __init__(self):
        super(SingleNetMsFuncInner, self).__init__()
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.bn.register_forward_pre_hook(forward_pre_hook_fn_add)
        self.bn.register_forward_pre_hook(forward_pre_hook_fn_mul)
        self.bn.register_forward_hook(forward_hook_fn_add)
        self.bn.register_forward_hook(forward_hook_fn_mul)
        self.bn.register_backward_hook(backward_hook_fn_inner)
        self.bn.register_backward_hook(backward_hook_fn_inner)
        self.relu = nn.ReLU()

    @jit
    def construct(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return x


class SingleNetMsFunc(nn.Cell):
    def __init__(self):
        super(SingleNetMsFunc, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.inner = SingleNetMsFuncInner()
        self.inner.register_forward_pre_hook(forward_pre_hook_fn_add)
        self.inner.register_forward_hook(forward_hook_fn_add)
        self.inner.register_forward_hook(forward_hook_fn_mul)
        self.inner.register_backward_hook(backward_hook_fn)
        self.inner.register_backward_hook(backward_hook_fn)

    def construct(self, x):
        x = self.conv(x)
        x = self.inner(x)
        x = x + x
        return x


class CompareSingleNet1(nn.Cell):
    def __init__(self):
        super(CompareSingleNet1, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x + x
        return x


class CompareSingleNet2(nn.Cell):
    def __init__(self):
        super(CompareSingleNet2, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.relu = nn.ReLU()

    def construct(self, x):
        x = x + x
        x = x * x
        x = self.conv(x)
        x = x + x
        x = self.bn(x)
        x = x + x
        x = self.relu(x)
        x = x + x
        x = x * x
        return x


class CompareSingleNet3(nn.Cell):
    def __init__(self):
        super(CompareSingleNet3, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.relu = nn.ReLU()

    def construct(self, x):
        x = x * x
        x = self.conv(x)
        x = x + x
        x = x + x
        x = self.relu(x)
        x = x + x
        return x


class CompareSingleNet4(nn.Cell):
    def __init__(self):
        super(CompareSingleNet4, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class CompareSingleNet5(nn.Cell):
    def __init__(self):
        super(CompareSingleNet5, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = x + x
        x = self.bn(x)
        x = self.relu(x)
        x = x + x
        x = x * x
        x = x + x
        return x


class MultiNet(nn.Cell):
    def __init__(self):
        super(MultiNet, self).__init__()
        self.mul1 = nn.MatMul()
        self.handle1 = self.mul1.register_forward_pre_hook(forward_pre_hook_fn_multi_add)
        self.handle2 = self.mul1.register_forward_hook(forward_hook_fn_conv)
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.mul2 = nn.MatMul()
        self.handle3 = self.mul2.register_forward_pre_hook(forward_pre_hook_fn_multi_relu)
        self.handle4 = self.mul2.register_forward_hook(forward_hook_fn_add)

    def construct(self, x, y):
        x = self.mul1(x, y)
        x = self.bn(x)
        x = self.mul2(x, x)
        return x


class CompareMultiNet1(nn.Cell):
    def __init__(self):
        super(CompareMultiNet1, self).__init__()
        self.mul = nn.MatMul()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.relu = nn.ReLU()
        self.log = P.Log()

    def construct(self, x, y):
        x = self.mul(x + x, x * y)
        x = self.log(x)
        x = self.bn(x)
        y = self.relu(x)
        x = self.mul(y, x)
        x = x + x
        return x


class CompareMultiNet2(nn.Cell):
    def __init__(self):
        super(CompareMultiNet2, self).__init__()
        self.mul = nn.MatMul()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.relu = nn.ReLU()
        self.log = P.Log()

    def construct(self, x, y):
        x = self.mul(x, y)
        x = self.log(x)
        x = self.bn(x)
        x = self.mul(x, x)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_forward_hook():
    """
    Feature: PyNative hook function.
    Description: Test PyNative forward hook function and forward pre hook function with single input.
    Expectation: The calculation result is correct.
    """

    context.set_context(mode=context.PYNATIVE_MODE)
    inputs = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 2)
    grad_op = GradOperation(get_all=True, get_by_list=True, sens_param=False)
    # case 1: calling remove() of handle to remove some hook function.
    net = SingleNet()
    out = net(inputs)
    compare_single_net1 = CompareSingleNet1()
    expect_out = compare_single_net1(inputs)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(inputs)
    expect_grad = grad_op(compare_single_net1, ParameterTuple(compare_single_net1.trainable_params()))(inputs)
    assert len(grad) == len(expect_grad)
    assert np.allclose(grad[0][0].asnumpy(), expect_grad[0][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][0].asnumpy(), 0.000001, 0.000001)
    # case 2: register new hook function.
    handle1 = net.conv.register_forward_pre_hook(forward_pre_hook_fn_add)
    net.conv.register_forward_pre_hook(forward_pre_hook_fn_mul)
    net.conv.register_forward_hook(forward_hook_fn_add)
    net.relu.register_forward_pre_hook(forward_pre_hook_fn_add)
    handle2 = net.relu.register_forward_hook(forward_hook_fn_mul)
    out = net(inputs)
    compare_single_net2 = CompareSingleNet2()
    expect_out = compare_single_net2(inputs)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(inputs)
    expect_grad = grad_op(compare_single_net2, ParameterTuple(compare_single_net2.trainable_params()))(inputs)
    assert len(grad) == len(expect_grad)
    assert np.allclose(grad[0][0].asnumpy(), expect_grad[0][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][0].asnumpy(), 0.000001, 0.000001)
    # case 3: remove some hook function.
    handle1.remove()
    net.handle4.remove()
    handle2.remove()
    out = net(inputs)
    compare_single_net3 = CompareSingleNet3()
    expect_out = compare_single_net3(inputs)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(inputs)
    expect_grad = grad_op(compare_single_net3, ParameterTuple(compare_single_net3.trainable_params()))(inputs)
    assert len(grad) == len(expect_grad)
    assert np.allclose(grad[0][0].asnumpy(), expect_grad[0][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][0].asnumpy(), 0.000001, 0.000001)


@pytest.mark.level2
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_forward_hook_multi_inp():
    """
    Feature: PyNative hook function.
    Description: Test PyNative forward hook function and forward pre hook function with multi input.
    Expectation: The calculation result is correct.
    """

    context.set_context(mode=context.PYNATIVE_MODE)
    inputs = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 3)
    grad_op = GradOperation(get_all=True, get_by_list=True, sens_param=False)
    # case 1: register hook function for multi-input op.
    multi_net = MultiNet()
    out = multi_net(inputs, inputs)
    compare_multi_net1 = CompareMultiNet1()
    expect_out = compare_multi_net1(inputs, inputs)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    grad = grad_op(multi_net, ParameterTuple(multi_net.trainable_params()))(inputs, inputs)
    expect_grad = grad_op(compare_multi_net1, ParameterTuple(compare_multi_net1.trainable_params()))(inputs, inputs)
    assert len(grad) == len(expect_grad)
    assert np.allclose(grad[0][0].asnumpy(), expect_grad[0][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[0][1].asnumpy(), expect_grad[0][1].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][1].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][1].asnumpy(), expect_grad[1][2].asnumpy(), 0.000001, 0.000001)
    # case 2: remove some hook function for multi-input op.
    multi_net.handle1.remove()
    multi_net.handle3.remove()
    multi_net.handle4.remove()
    out = multi_net(inputs, inputs)
    compare_multi_net2 = CompareMultiNet2()
    expect_out = compare_multi_net2(inputs, inputs)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    grad = grad_op(multi_net, ParameterTuple(multi_net.trainable_params()))(inputs, inputs)
    expect_grad = grad_op(compare_multi_net2, ParameterTuple(compare_multi_net2.trainable_params()))(inputs, inputs)
    assert len(grad) == len(expect_grad)
    assert np.allclose(grad[0][0].asnumpy(), expect_grad[0][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[0][1].asnumpy(), expect_grad[0][1].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][1].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][1].asnumpy(), expect_grad[1][2].asnumpy(), 0.000001, 0.000001)
    # case 4: register hook function in construct.
    net = SingleNetInConstruct()
    compare_net = CompareSingleNet1()
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(inputs)
    expect_grad = grad_op(compare_net, ParameterTuple(compare_net.trainable_params()))(inputs)
    assert len(grad) == len(expect_grad)
    assert np.allclose(grad[0][0].asnumpy(), expect_grad[0][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][0].asnumpy(), 0.000001, 0.000001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_forward_hook_exception():
    """
    Feature: PyNative hook function.
    Description: Test PyNative forward hook function and forward pre hook function in exception case.
    Expectation: Raises exception.
    """

    context.set_context(mode=context.PYNATIVE_MODE)
    net = SingleNet()
    with pytest.raises(TypeError):
        net.relu.register_forward_pre_hook("Test")
    with pytest.raises(TypeError):
        net.conv.register_forward_pre_hook(forward_hook_fn_with_ms_func)
    with pytest.raises(TypeError):
        net.conv.register_forward_hook(forward_hook_fn_with_ms_func)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_forward_hook_with_ms_func():
    """
    Feature: PyNative hook function.
    Description: Test PyNative forward hook function and forward pre hook function with @jit decorated function.
    Expectation: The calculation result is correct.
    """

    inputs = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 2)
    grad_op = GradOperation(get_all=True, get_by_list=True, sens_param=False)
    # case: ms_funciton in pynative mode.
    context.set_context(mode=context.PYNATIVE_MODE)
    single_net_msfunc = SingleNetMsFunc()
    out = single_net_msfunc(inputs)
    compare_single_net5 = CompareSingleNet5()
    expect_out = compare_single_net5(inputs)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    grad = grad_op(single_net_msfunc, ParameterTuple(single_net_msfunc.trainable_params()))(inputs)
    expect_grad = grad_op(compare_single_net5, ParameterTuple(compare_single_net5.trainable_params()))(inputs)
    assert len(grad) == len(expect_grad)
    assert np.allclose(grad[0][0].asnumpy(), expect_grad[0][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][1].asnumpy(), expect_grad[1][1].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][2].asnumpy(), expect_grad[1][2].asnumpy(), 0.000001, 0.000001)
    # case: ms_funciton in graph mode.
    context.set_context(mode=context.GRAPH_MODE)
    out = single_net_msfunc(inputs)
    compare_single_net1 = CompareSingleNet1()
    expect_out = compare_single_net1(inputs)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    grad = grad_op(single_net_msfunc, ParameterTuple(single_net_msfunc.trainable_params()))(inputs)
    expect_grad = grad_op(compare_single_net1, ParameterTuple(compare_single_net1.trainable_params()))(inputs)
    assert len(grad) == len(expect_grad)
    assert np.allclose(grad[0][0].asnumpy(), expect_grad[0][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][1].asnumpy(), expect_grad[1][1].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][2].asnumpy(), expect_grad[1][2].asnumpy(), 0.000001, 0.000001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_forward_hook_in_graph_mode():
    """
    Feature: PyNative hook function.
    Description: Test PyNative forward hook function and forward pre hook function in graph mode.
    Expectation: The calculation result is correct.
    """

    context.set_context(mode=context.GRAPH_MODE)
    inputs = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 3)
    grad_op = GradOperation(get_all=True, get_by_list=True, sens_param=False)
    net = SingleNet()
    out = net(inputs)
    compare_net = CompareSingleNet4()
    expect_out = compare_net(inputs)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(inputs)
    expect_grad = grad_op(compare_net, ParameterTuple(compare_net.trainable_params()))(inputs)
    assert len(grad) == len(expect_grad)
    assert np.allclose(grad[0][0].asnumpy(), expect_grad[0][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][0].asnumpy(), 0.000001, 0.000001)
