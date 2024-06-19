# Copyright 2023 Huawei Technologies Co., Ltd
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
""" test_hook_grad_with_torch """
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore.ops import GradOperation
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from tests.st.pynative.utils import GradOfAllInputs
from tests.st.pynative.utils import GradOfFirstInput, GradOfAllInputsAndParams
import torch
import torch.nn as pynn


class MyMul(pynn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        out = inputs * inputs
        return out

    def my_hook(self, module, grad_input, grad_output):
        grad_input = grad_input[0] * 2
        grad_input = tuple(
            [grad_input, grad_input])
        return grad_input


class MyMean(pynn.Module):
    def forward(self, inputs):
        out = inputs / 2
        return out


def tensor_hook(grad):
    print('tensor hook')
    print('grad:', grad)
    return grad


class MyNet(pynn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.f1 = pynn.Linear(2, 1, bias=True)
        self.f2 = MyMean()
        self.weight_init()
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        output = self.f1(inputs)
        output = self.f2(output)
        return output

    def weight_init(self):
        self.f1.weight.data.fill_(1.0)
        self.f1.bias.data.fill_(0.0)


class MyNet2(pynn.Module):
    def __init__(self):
        super(MyNet2, self).__init__()
        self.f1 = MyNet()
        self.f2 = MyMul()

    def forward(self, inputs):
        out = self.f1(inputs)
        out = self.f2(out)
        return out


class MEMul(nn.Cell):
    def construct(self, x):
        out = x * x
        return out


class MEMul1(nn.Cell):
    def __init__(self):
        super(MEMul1, self).__init__()
        self.f = MEMul()
        self.f.set_grad()
        self.grad = GradOfAllInputs(self.f, sens_param=False)

    def construct(self, x):
        out = self.f(x)
        return out

    def bprop(self, x, out, dout):
        grads = self.grad(x)
        grads = grads[0] * 2
        return (grads,)


class CustomNetWithParam(nn.Cell):
    def __init__(self):
        super(CustomNetWithParam, self).__init__()
        self.w = Parameter(Tensor(np.array([2.0], dtype=np.float32)), name='weight')
        self.grad = GradOperation(get_all=True, get_by_list=True, sens_param=True)
        self.internal_params = [self.w]

    def construct(self, x):
        output = self.w * x
        return output

    def bprop(self, *args):
        return (self.w * args[-1],), {self.w: args[0] * args[-1]}


class NetWithParam(nn.Cell):
    def __init__(self):
        super(NetWithParam, self).__init__()
        self.w = Parameter(Tensor(np.array([2.0], dtype=np.float32)), name='weight')
        self.grad = GradOperation(get_all=True, get_by_list=True, sens_param=True)
        self.internal_params = [self.w]

    def construct(self, x):
        output = self.w * x
        return output


class MEMean(nn.Cell):
    def construct(self, x):
        out = x / 2
        return out


class MENet1(nn.Cell):
    def __init__(self):
        super(MENet1, self).__init__()
        self.f1 = nn.Dense(2, 1, weight_init="ones", bias_init="zeros", has_bias=True, activation=None)
        self.f2 = MEMean()

    def construct(self, x):
        output = self.f1(x)
        output = self.f2(output)
        return output


class MENet2(nn.Cell):
    def __init__(self):
        super(MENet2, self).__init__()
        self.f1 = MENet1()
        self.f2 = MEMul1()

    def construct(self, x):
        output = self.f1(x)
        output = self.f2(output)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bprop_compara_with_pytorch():
    """
    Feature: Test custom bprop nested grad feature
    Description: Test custom bprop nested grad
    Expectation: Success
    """
    net = MyNet2()
    net.register_backward_hook(net.f2.my_hook)
    netme = MENet2()
    grad_net = GradOfFirstInput(netme)
    grad_net.set_train()

    for _ in range(0, 3):
        output_np = np.ones([2, 1]).astype(dtype=np.float32)
        input_np = np.random.randn(2, 2).astype(dtype=np.float32)

        inputs = torch.from_numpy(input_np.copy().astype(np.float32))
        output = torch.from_numpy(output_np.copy().astype(np.float32))
        inputs.requires_grad = True
        inputs.register_hook(tensor_hook)
        result = net(inputs)
        result.backward(output)

        input_me = Tensor(input_np.copy().astype(np.float32))
        output_me = Tensor(output_np.copy().astype(np.float32))
        input_grad = grad_net(input_me, output_me)
        assert np.allclose(inputs.grad, input_grad.asnumpy(), 0.001, 0.001)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bprop_with_weight():
    """
    Feature: Test custom bprop with weight feature
    Description: Test custom bprop with weight
    Expectation: Success
    """

    input1 = Tensor(np.ones(1).astype(dtype=np.float32))
    sens_param = Tensor(np.ones(1).astype(dtype=np.float32))
    net = NetWithParam()
    grad_net = GradOfAllInputsAndParams(net)
    grad1 = grad_net(input1, sens_param)

    custom_net = CustomNetWithParam()
    grad_custom_net = GradOfAllInputsAndParams(custom_net)
    grad2 = grad_custom_net(input1, sens_param)

    assert np.allclose(grad1[0][0].asnumpy(), grad2[0][0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(grad1[1][0].asnumpy(), grad2[1][0].asnumpy(), 0.0001, 0.0001)


class MEMul1WithUsedMap(nn.Cell):
    def __init__(self):
        super(MEMul1WithUsedMap, self).__init__()
        self.f = MEMul()
        self.used_bprop_inputs = [0]

    def construct(self, x):
        out = self.f(x)
        return out

    def bprop(self, *args):
        grads = args[0] * 2
        return (grads,)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bprop_used_map():
    """
    Feature: Test custom bprop with used map
    Description: Test custom bprop with used map
    Expectation: Success
    """
    input1 = Tensor(np.ones(1).astype(dtype=np.float32))
    output = Tensor(np.ones(1).astype(dtype=np.float32))
    net = MEMul1WithUsedMap()
    grad_net = GradOfFirstInput(net)
    input_grad = grad_net(input1, output)
    assert np.allclose(input_grad.asnumpy(), np.array([2], dtype=np.float32), 0.0001, 0.0001)
