# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
""" test_bprop """
import numpy as np
import torch
from torch.nn.parameter import Parameter as tParameter
import mindspore as ms
from mindspore import nn
from mindspore import context
from mindspore.common import Tensor
from mindspore.common.parameter import Parameter
from mindspore import ops
from mindspore.ops import operations as P
from tests.st.pynative.utils import GradOfAllParams, GradOfAllInputs, HighGrad
from tests.mark_utils import arg_mark


def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight1 = Parameter(Tensor(np.array([2.0, 2.0, 2.0]), ms.float32), name="weight1")
        self.weight2 = Parameter(Tensor(np.array([3.0, 3.0, 3.0]), ms.float32), name="weight2")

    def construct(self, x):
        x = x / self.weight1
        x = x * self.weight2
        return x


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_network_pipeline_set_grad():
    """
    Feature: Test pipeline_grad
    Description: Net is grad by pipeline
    Expectation: Success
    """
    net = Net()
    net.set_grad()
    ms_grad = GradOfAllParams(net, False)

    for _ in range(2):
        input_1 = Tensor(np.array([2.0, 4.0, 6.0]), ms.float32)
        input_2 = Tensor(np.array([6.0, 10.0, 12.0]), ms.float32)
        input_3 = Tensor(np.array([14.0, 16.0, 18.0]), ms.float32)

        net(input_1)
        net(input_2)
        net(input_3)

        output1 = ms_grad(input_1)
        output2 = ms_grad(input_2)
        output3 = ms_grad(input_3)

        assert np.allclose(output1[0].asnumpy(), Tensor(np.array([-1.5, -3.0, -4.5])).astype(np.float32).asnumpy(),
                           0.001, 0.001)
        assert np.allclose(output1[1].asnumpy(), Tensor(np.array([1, 2, 3])).astype(np.float32).asnumpy(), 0.001, 0.001)

        assert np.allclose(output2[0].asnumpy(), Tensor(np.array([-4.5, -7.5, -9.0])).astype(np.float32).asnumpy(),
                           0.001, 0.001)
        assert np.allclose(output2[1].asnumpy(), Tensor(np.array([3, 5, 6])).astype(np.float32).asnumpy(), 0.001, 0.001)

        assert np.allclose(output3[0].asnumpy(),
                           Tensor(np.array([-1.05e+01, -1.2e+01, -1.35e+01])).astype(np.float32).asnumpy(), 0.001,
                           0.001)
        assert np.allclose(output3[1].asnumpy(), Tensor(np.array([7, 8, 9])).astype(np.float32).asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_network_pipeline_set_grad_mix_order():
    """
    Feature: Test pipeline_grad
    Description: Net is grad by pipeline
    Expectation: Success
    """
    net = Net()
    net.set_grad()
    ms_grad = GradOfAllParams(net, False)

    for _ in range(2):
        input_1 = Tensor(np.array([2.0, 4.0, 6.0]), ms.float32)
        input_2 = Tensor(np.array([6.0, 10.0, 12.0]), ms.float32)
        input_3 = Tensor(np.array([14.0, 16.0, 18.0]), ms.float32)

        net(input_1)
        net(input_2)
        net(input_3)

        output2 = ms_grad(input_2)  # order change
        output1 = ms_grad(input_1)
        output3 = ms_grad(input_3)

        assert np.allclose(output1[0].asnumpy(), Tensor(np.array([-1.5, -3.0, -4.5])).astype(np.float32).asnumpy(),
                           0.001, 0.001)
        assert np.allclose(output1[1].asnumpy(), Tensor(np.array([1, 2, 3])).astype(np.float32).asnumpy(), 0.001, 0.001)

        assert np.allclose(output2[0].asnumpy(), Tensor(np.array([-4.5, -7.5, -9.0])).astype(np.float32).asnumpy(),
                           0.001, 0.001)
        assert np.allclose(output2[1].asnumpy(), Tensor(np.array([3, 5, 6])).astype(np.float32).asnumpy(), 0.001, 0.001)

        assert np.allclose(output3[0].asnumpy(),
                           Tensor(np.array([-1.05e+01, -1.2e+01, -1.35e+01])).astype(np.float32).asnumpy(), 0.001,
                           0.001)
        assert np.allclose(output3[1].asnumpy(), Tensor(np.array([7, 8, 9])).astype(np.float32).asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_network_pipeline_grad_first():
    """
    Feature: Test pipeline_grad
    Description: Net is grad by pipeline
    Expectation: Success
    """
    net = Net()
    net.set_grad()
    ms_grad = GradOfAllParams(net, False)

    for _ in range(2):
        input_1 = Tensor(np.array([2.0, 4.0, 6.0]), ms.float32)
        input_2 = Tensor(np.array([6.0, 10.0, 12.0]), ms.float32)
        input_3 = Tensor(np.array([14.0, 16.0, 18.0]), ms.float32)

        output1 = ms_grad(input_1)
        output2 = ms_grad(input_2)
        output3 = ms_grad(input_3)

        assert np.allclose(output1[0].asnumpy(), Tensor(np.array([-1.5, -3.0, -4.5])).astype(np.float32).asnumpy(),
                           0.001, 0.001)
        assert np.allclose(output1[1].asnumpy(), Tensor(np.array([1, 2, 3])).astype(np.float32).asnumpy(), 0.001, 0.001)

        assert np.allclose(output2[0].asnumpy(), Tensor(np.array([-4.5, -7.5, -9.0])).astype(np.float32).asnumpy(),
                           0.001, 0.001)
        assert np.allclose(output2[1].asnumpy(), Tensor(np.array([3, 5, 6])).astype(np.float32).asnumpy(), 0.001, 0.001)

        assert np.allclose(output3[0].asnumpy(),
                           Tensor(np.array([-1.05e+01, -1.2e+01, -1.35e+01])).astype(np.float32).asnumpy(), 0.001,
                           0.001)
        assert np.allclose(output3[1].asnumpy(), Tensor(np.array([7, 8, 9])).astype(np.float32).asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_network_pipeline_with_grad_first_high_grad():
    """
    Feature: Test pipeline_grad
    Description: Net is grad by pipeline
    Expectation: Success
    """
    net = Net()
    net.set_grad()
    ms_grad = HighGrad(net, [GradOfAllParams, GradOfAllParams])

    for _ in range(2):
        input_1 = Tensor(np.array([2.0, 4.0, 6.0]), ms.float32)
        input_2 = Tensor(np.array([6.0, 10.0, 12.0]), ms.float32)
        input_3 = Tensor(np.array([14.0, 16.0, 18.0]), ms.float32)

        net(input_1)
        net(input_2)
        net(input_3)

        output1 = ms_grad(input_1)
        output2 = ms_grad(input_2)
        output3 = ms_grad(input_3)

        assert np.allclose(output1[0].asnumpy(), Tensor(np.array([1, 2, 3])).astype(np.float32).asnumpy(), 0.001, 0.001)
        assert np.allclose(output1[1].asnumpy(), Tensor(np.array([-5e-01, -1, -1.5])).astype(np.float32).asnumpy(),
                           0.001, 0.001)

        assert np.allclose(output2[0].asnumpy(), Tensor(np.array([3, 5, 6])).astype(np.float32).asnumpy(), 0.001, 0.001)
        assert np.allclose(output2[1].asnumpy(), Tensor(np.array([-1.5, -2.5, -3])).astype(np.float32).asnumpy(), 0.001,
                           0.001)

        assert np.allclose(output3[0].asnumpy(), Tensor(np.array([7, 8, 9])).astype(np.float32).asnumpy(), 0.001, 0.001)
        assert np.allclose(output3[1].asnumpy(), Tensor(np.array([-3.5, -4, -4.5])).astype(np.float32).asnumpy(), 0.001,
                           0.001)


class HighNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight1 = Parameter(Tensor(np.array([2.0, 2.0, 2.0]), ms.float32), name="weight1")
        self.weight2 = Parameter(Tensor(np.array([3.0, 3.0, 3.0]), ms.float32), name="weight2")
        self.net = Net()
        self.ms_grad = GradOfAllParams(self.net, False)

    def construct(self, x):
        x = x / self.weight1
        x = x * self.weight2
        x = self.ms_grad(x)
        return x


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_network_pipeline_with_set_grad_high_grad():
    """
    Feature: Test pipeline_grad with high grad
    Description: Net is grad by pipeline
    Expectation: Success
    """
    net = HighNet()
    net.set_grad()
    ms_grad = GradOfAllInputs(net, False)

    for _ in range(2):
        input_1 = Tensor(np.array([2.0, 4.0, 6.0]), ms.float32)
        input_2 = Tensor(np.array([6.0, 10.0, 12.0]), ms.float32)
        input_3 = Tensor(np.array([14.0, 16.0, 18.0]), ms.float32)

        net(input_1)
        net(input_2)

        output1 = ms_grad(input_1)
        net(input_3)
        output2 = ms_grad(input_2)
        output3 = ms_grad(input_3)

        assert np.allclose(output1[0].asnumpy(),
                           Tensor(np.array([-3.75e-1, -3.75e-1, -3.75e-1])).astype(np.float32).asnumpy(), 0.001, 0.001)

        assert np.allclose(output2[0].asnumpy(),
                           Tensor(np.array([-3.75e-1, -3.75e-1, -3.75e-1])).astype(np.float32).asnumpy(), 0.001, 0.001)

        assert np.allclose(output3[0].asnumpy(),
                           Tensor(np.array([-3.75e-1, -3.75e-1, -3.75e-1])).astype(np.float32).asnumpy(), 0.001, 0.001)


class OneInputBprop(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = P.ReLU()

    def construct(self, x):
        return self.op(x)

    def bprop(self, x, out, dout):
        return (5 * x,)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_network_pipeline_with_bprop():
    """
    Feature: Test pipeline_grad
    Description: Net is grad by pipeline
    Expectation: Success
    """
    net = OneInputBprop()
    net.set_grad()
    ms_grad = GradOfAllInputs(net, False)

    for _ in range(2):
        input_1 = Tensor(np.array([2.0, 4.0, 6.0]), ms.float32)
        input_2 = Tensor(np.array([6.0, 10.0, 12.0]), ms.float32)
        input_3 = Tensor(np.array([14.0, 16.0, 18.0]), ms.float32)

        net(input_1)
        net(input_2)
        net(input_3)

        output1 = ms_grad(input_1)
        output2 = ms_grad(input_2)
        output3 = ms_grad(input_3)

        assert np.allclose(output1[0].asnumpy(),
                           Tensor(np.array([1.0e+01, 2.0e+01, 3.0e+01])).astype(np.float32).asnumpy(), 0.001, 0.001)
        assert np.allclose(output2[0].asnumpy(),
                           Tensor(np.array([3.0e+01, 5.0e+01, 6.0e+01])).astype(np.float32).asnumpy(), 0.001, 0.001)
        assert np.allclose(output3[0].asnumpy(),
                           Tensor(np.array([7.0e+01, 8.0e+01, 9.0e+01])).astype(np.float32).asnumpy(), 0.001, 0.001)


class MEMul1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.f = Net()
        self.f.set_grad()
        self.grad = GradOfAllInputs(self.f, sens_param=False)

    def construct(self, x):
        out = self.f(x)
        return out

    def bprop(self, x, out, dout):
        grads = list(self.grad(x))
        grads[0] = grads[0] * 2
        return tuple(grads)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_network_pipeline_with_bprop_high_grad():
    """
    Feature: Test pipeline_grad
    Description: Net is grad by pipeline
    Expectation: Success
    """
    net = MEMul1()
    net.set_grad()
    ms_grad = GradOfAllInputs(net, sens_param=False)

    for _ in range(2):
        input_1 = Tensor(np.array([2.0, 4.0, 6.0]), ms.float32)
        input_2 = Tensor(np.array([6.0, 10.0, 12.0]), ms.float32)
        input_3 = Tensor(np.array([14.0, 16.0, 18.0]), ms.float32)

        net(input_1)
        net(input_2)
        net(input_3)

        output1 = ms_grad(input_1)
        output2 = ms_grad(input_2)
        output3 = ms_grad(input_3)

        assert np.allclose(output1[0].asnumpy(), Tensor(np.array([3.0, 3.0, 3.0])).astype(np.float32).asnumpy(), 0.001,
                           0.001)
        assert np.allclose(output2[0].asnumpy(), Tensor(np.array([3.0, 3.0, 3.0])).astype(np.float32).asnumpy(), 0.001,
                           0.001)
        assert np.allclose(output3[0].asnumpy(), Tensor(np.array([3.0, 3.0, 3.0])).astype(np.float32).asnumpy(), 0.001,
                           0.001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_network_pipeline_mix_other_grad_bprop():
    """
    Feature: Test pipeline_grad
    Description: Net is grad by pipeline
    Expectation: Success
    """
    net = Net()
    net.set_grad()
    ms_grad = GradOfAllParams(net, False)

    net_bprop = OneInputBprop()
    net_bprop.set_grad()
    ms_grad_bprop = GradOfAllInputs(net_bprop, sens_param=False)

    for _ in range(2):
        input_1 = Tensor(np.array([2.0, 4.0, 6.0]), ms.float32)
        input_2 = Tensor(np.array([6.0, 10.0, 12.0]), ms.float32)
        input_3 = Tensor(np.array([14.0, 16.0, 18.0]), ms.float32)

        net(input_1)
        bprop_1 = ms_grad_bprop(Tensor(np.array([2, 2]).astype(np.float32)))
        assert np.allclose(bprop_1[0].asnumpy(), Tensor(np.array([1.0e+01, 1.0e+01])).astype(np.float32).asnumpy(),
                           0.001, 0.001)

        net(input_2)
        bprop_2 = ms_grad_bprop(Tensor(np.array([5, 5]).astype(np.float32)))
        assert np.allclose(bprop_2[0].asnumpy(), Tensor(np.array([2.5e+01, 2.5e+01])).astype(np.float32).asnumpy(),
                           0.001, 0.001)

        net(input_3)

        output1 = ms_grad(input_1)
        output2 = ms_grad(input_2)
        output3 = ms_grad(input_3)

        assert np.allclose(output1[0].asnumpy(), Tensor(np.array([-1.5, -3.0, -4.5])).astype(np.float32).asnumpy(),
                           0.001, 0.001)
        assert np.allclose(output1[1].asnumpy(), Tensor(np.array([1, 2, 3])).astype(np.float32).asnumpy(), 0.001, 0.001)

        assert np.allclose(output2[0].asnumpy(), Tensor(np.array([-4.5, -7.5, -9.0])).astype(np.float32).asnumpy(),
                           0.001, 0.001)
        assert np.allclose(output2[1].asnumpy(), Tensor(np.array([3, 5, 6])).astype(np.float32).asnumpy(), 0.001, 0.001)

        assert np.allclose(output3[0].asnumpy(),
                           Tensor(np.array([-1.05e+01, -1.2e+01, -1.35e+01])).astype(np.float32).asnumpy(), 0.001,
                           0.001)
        assert np.allclose(output3[1].asnumpy(), Tensor(np.array([7, 8, 9])).astype(np.float32).asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_network_pipeline_mix_other_forward_and_grad():
    """
    Feature: Test pipeline_grad
    Description: Net is grad by pipeline
    Expectation: Success
    """
    net = Net()
    net.set_grad()
    ms_grad = GradOfAllParams(net, False)

    for _ in range(2):
        input_1 = Tensor(np.array([2.0, 4.0, 6.0]), ms.float32)
        input_2 = Tensor(np.array([6.0, 10.0, 12.0]), ms.float32)
        input_3 = Tensor(np.array([14.0, 16.0, 18.0]), ms.float32)
        input_4 = Tensor(np.array([6.0, 10.0, 12.0]), ms.float32)

        net(input_1)
        net(input_2)
        output1 = ms_grad(input_1)
        assert np.allclose(output1[0].asnumpy(), Tensor(np.array([-1.5, -3.0, -4.5])).astype(np.float32).asnumpy(),
                           0.001, 0.001)
        assert np.allclose(output1[1].asnumpy(), Tensor(np.array([1, 2, 3])).astype(np.float32).asnumpy(), 0.001, 0.001)

        net(input_3)
        output2 = ms_grad(input_2)
        assert np.allclose(output2[0].asnumpy(), Tensor(np.array([-4.5, -7.5, -9.0])).astype(np.float32).asnumpy(),
                           0.001, 0.001)
        assert np.allclose(output2[1].asnumpy(), Tensor(np.array([3, 5, 6])).astype(np.float32).asnumpy(), 0.001, 0.001)

        net(input_4)
        output3 = ms_grad(input_3)
        assert np.allclose(output3[0].asnumpy(),
                           Tensor(np.array([-1.05e+01, -1.2e+01, -1.35e+01])).astype(np.float32).asnumpy(), 0.001,
                           0.001)
        assert np.allclose(output3[1].asnumpy(), Tensor(np.array([7, 8, 9])).astype(np.float32).asnumpy(), 0.001, 0.001)

        output4 = ms_grad(input_4)
        assert np.allclose(output4[0].asnumpy(), Tensor(np.array([-4.5, -7.5, -9.0])).astype(np.float32).asnumpy(),
                           0.001, 0.001)
        assert np.allclose(output4[1].asnumpy(), Tensor(np.array([3, 5, 6])).astype(np.float32).asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_network_pipeline_same_input():
    """
    Feature: Test pipeline_grad
    Description: Net is grad by pipeline
    Expectation: Success
    """
    net = Net()
    net.set_grad()
    ms_grad = GradOfAllParams(net, False)

    input_1 = Tensor(np.array([2.0, 4.0, 6.0]), ms.float32)
    input_2 = Tensor(np.array([6.0, 10.0, 12.0]), ms.float32)
    input_3 = Tensor(np.array([14.0, 16.0, 18.0]), ms.float32)

    for _ in range(2):
        net(input_1)
        net(input_2)
        net(input_3)

        output1 = ms_grad(input_1)
        output2 = ms_grad(input_2)
        output3 = ms_grad(input_3)

        assert np.allclose(output1[0].asnumpy(), Tensor(np.array([-1.5, -3.0, -4.5])).astype(np.float32).asnumpy(),
                           0.001, 0.001)
        assert np.allclose(output1[1].asnumpy(), Tensor(np.array([1, 2, 3])).astype(np.float32).asnumpy(), 0.001, 0.001)

        assert np.allclose(output2[0].asnumpy(), Tensor(np.array([-4.5, -7.5, -9.0])).astype(np.float32).asnumpy(),
                           0.001, 0.001)
        assert np.allclose(output2[1].asnumpy(), Tensor(np.array([3, 5, 6])).astype(np.float32).asnumpy(), 0.001, 0.001)

        assert np.allclose(output3[0].asnumpy(),
                           Tensor(np.array([-1.05e+01, -1.2e+01, -1.35e+01])).astype(np.float32).asnumpy(), 0.001,
                           0.001)
        assert np.allclose(output3[1].asnumpy(), Tensor(np.array([7, 8, 9])).astype(np.float32).asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_network_pipeline_forward_and_backward_with_different_input():
    """
    Feature: Test pipeline_grad
    Description: Net is grad by pipeline
    Expectation: Success
    """
    net = Net()
    net.set_grad()
    ms_grad = GradOfAllParams(net, False)

    for _ in range(2):
        net(Tensor(np.array([2.0, 4.0, 6.0]), ms.float32))
        net(Tensor(np.array([6.0, 10.0, 12.0]), ms.float32))
        net(Tensor(np.array([14.0, 16.0, 18.0]), ms.float32))

        output1 = ms_grad(Tensor(np.array([2.0, 4.0, 6.0]), ms.float32))
        output2 = ms_grad(Tensor(np.array([6.0, 10.0, 12.0]), ms.float32))
        output3 = ms_grad(Tensor(np.array([14.0, 16.0, 18.0]), ms.float32))

        assert np.allclose(output1[0].asnumpy(), Tensor(np.array([-1.5, -3.0, -4.5])).astype(np.float32).asnumpy(),
                           0.001, 0.001)
        assert np.allclose(output1[1].asnumpy(), Tensor(np.array([1, 2, 3])).astype(np.float32).asnumpy(), 0.001, 0.001)

        assert np.allclose(output2[0].asnumpy(), Tensor(np.array([-4.5, -7.5, -9.0])).astype(np.float32).asnumpy(),
                           0.001, 0.001)
        assert np.allclose(output2[1].asnumpy(), Tensor(np.array([3, 5, 6])).astype(np.float32).asnumpy(), 0.001, 0.001)

        assert np.allclose(output3[0].asnumpy(),
                           Tensor(np.array([-1.05e+01, -1.2e+01, -1.35e+01])).astype(np.float32).asnumpy(), 0.001,
                           0.001)
        assert np.allclose(output3[1].asnumpy(), Tensor(np.array([7, 8, 9])).astype(np.float32).asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_pipeline_simulate_machine_01():
    """
    Feature: Test pipeline_grad
    Description: Compare with torch grad
    Expectation: Success
    """
    def get_data(input_shape, weight_shape, bias_shape, input_count):
        input_np1 = np.random.randn(*input_shape).astype(np.float32)
        input_np2 = np.random.randn(*input_shape).astype(np.float32)
        input_np3 = np.random.randn(*input_shape).astype(np.float32)
        input_np4 = np.random.randn(*input_shape).astype(np.float32)
        input_np5 = np.random.randn(*input_shape).astype(np.float32)
        input_np6 = np.random.randn(*input_shape).astype(np.float32)
        input_np7 = np.random.randn(*input_shape).astype(np.float32)
        input_np8 = np.random.randn(*input_shape).astype(np.float32)
        weight_np = np.random.randn(*weight_shape).astype(np.float32)
        bias_np = np.random.randn(*bias_shape).astype(np.float32)
        if input_count == 3:
            return input_np1, input_np2, input_np3, weight_np, bias_np
        if input_count == 4:
            return input_np1, input_np2, input_np3, input_np4, weight_np, bias_np
        if input_count == 5:
            return input_np1, input_np2, input_np3, input_np4, input_np5, weight_np, bias_np
        if input_count == 6:
            return input_np1, input_np2, input_np3, input_np4, input_np5, input_np6, weight_np, bias_np
        if input_count == 7:
            return input_np1, input_np2, input_np3, input_np4, input_np5, input_np6, input_np7, weight_np, \
                   bias_np
        return input_np1, input_np2, input_np3, input_np4, input_np5, input_np6, input_np7, input_np8, \
               weight_np, bias_np

    def get_pt_input(numpy_list):
        if len(numpy_list) == 2:
            return torch.from_numpy(numpy_list[0]), torch.from_numpy(numpy_list[1])
        if len(numpy_list) == 3:
            return torch.from_numpy(numpy_list[0]), torch.from_numpy(numpy_list[1]), torch.from_numpy(
                numpy_list[2])
        if len(numpy_list) == 4:
            return torch.from_numpy(numpy_list[0]), torch.from_numpy(numpy_list[1]), torch.from_numpy(
                numpy_list[2]), torch.from_numpy(numpy_list[3])
        if len(numpy_list) == 5:
            return torch.from_numpy(numpy_list[0]), torch.from_numpy(numpy_list[1]), torch.from_numpy(
                numpy_list[2]), torch.from_numpy(numpy_list[3]), torch.from_numpy(numpy_list[4])
        if len(numpy_list) == 6:
            return torch.from_numpy(numpy_list[0]), torch.from_numpy(numpy_list[1]), torch.from_numpy(
                numpy_list[2]), torch.from_numpy(numpy_list[3]), torch.from_numpy(
                numpy_list[4]), torch.from_numpy(numpy_list[5])
        if len(numpy_list) == 7:
            return torch.from_numpy(numpy_list[0]), torch.from_numpy(numpy_list[1]), torch.from_numpy(
                numpy_list[2]), torch.from_numpy(numpy_list[3]), torch.from_numpy(
                numpy_list[4]), torch.from_numpy(numpy_list[5]), torch.from_numpy(numpy_list[6])
        return torch.from_numpy(numpy_list[0]), torch.from_numpy(numpy_list[1]), torch.from_numpy(
            numpy_list[2]), torch.from_numpy(numpy_list[3]), torch.from_numpy(
            numpy_list[4]), torch.from_numpy(numpy_list[5]), torch.from_numpy(
            numpy_list[6]), torch.from_numpy(numpy_list[7])

    def get_ms_input(numpy_list):
        if len(numpy_list) == 2:
            return Tensor(numpy_list[0]), Tensor(numpy_list[1])
        if len(numpy_list) == 3:
            return Tensor(numpy_list[0]), Tensor(numpy_list[1]), Tensor(numpy_list[2])
        if len(numpy_list) == 4:
            return Tensor(numpy_list[0]), Tensor(numpy_list[1]), Tensor(numpy_list[2]), Tensor(
                numpy_list[3])
        if len(numpy_list) == 5:
            return Tensor(numpy_list[0]), Tensor(numpy_list[1]), Tensor(numpy_list[2]), Tensor(
                numpy_list[3]), Tensor(numpy_list[4])
        if len(numpy_list) == 6:
            return Tensor(numpy_list[0]), Tensor(numpy_list[1]), Tensor(numpy_list[2]), Tensor(
                numpy_list[3]), Tensor(numpy_list[4]), Tensor(numpy_list[5])
        if len(numpy_list) == 7:
            return Tensor(numpy_list[0]), Tensor(numpy_list[1]), Tensor(numpy_list[2]), Tensor(
                numpy_list[3]), Tensor(numpy_list[4]), Tensor(numpy_list[5]), \
                   Tensor(numpy_list[6])
        return Tensor(numpy_list[0]), Tensor(numpy_list[1]), Tensor(numpy_list[2]), Tensor(
            numpy_list[3]), Tensor(numpy_list[4]), Tensor(numpy_list[5]), Tensor(
            numpy_list[6]), Tensor(numpy_list[7])

    def _count_unequal_element(data_expected, data_me, rtol, atol):
        assert data_expected.shape == data_me.shape
        total_count = len(data_expected.flatten())
        error = np.abs(data_expected - data_me)
        greater = np.greater(error, atol + np.abs(data_me) * rtol)
        nan_diff = np.not_equal(np.isnan(data_expected), np.isnan(data_me))
        inf_diff = np.not_equal(np.isinf(data_expected), np.isinf(data_me))
        # ICKTGQ
        if data_expected.dtype in ('complex64', 'complex128'):
            greater = greater + nan_diff + inf_diff
        else:
            neginf_diff = np.not_equal(np.isneginf(data_expected), np.isneginf(data_me))
            greater = greater + nan_diff + inf_diff + neginf_diff
        loss_count = np.count_nonzero(greater)
        assert (loss_count / total_count) < rtol, \
            "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
                format(data_expected[greater], data_me[greater], error[greater])

    def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
        if not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
            _count_unequal_element(data_expected, data_me, rtol, atol)
        else:
            assert np.array(data_expected).shape == np.array(data_me).shape

    class MsNet(nn.Cell):
        def __init__(self, weight, bias):
            super().__init__()
            self.count = 0
            self.weight = Parameter(weight, name="weight")
            self.bias = Parameter(bias, name="bias")

        def construct(self, x):
            y = ops.matmul(x, self.weight)
            out = ops.add(y, self.bias)
            self.count += 1
            return out

    class TorchNet(torch.nn.Module):
        def __init__(self, weight, bias):
            super().__init__()
            self.weight = tParameter(weight)
            self.bias = tParameter(bias)

        def forward(self, x):
            y = torch.matmul(x, self.weight)
            out = torch.add(y, self.bias)
            return out

    def get_torch_grad_compare(input_pt, pt_net, sens, flag="AllParams", requires_grad=True,
                               ms_out_list=None, rtol=0.001, atol=0.001):
        if flag == "AllParams":
            input_pt.requires_grad = True
            pt_net.zero_grad()
            pt_out = pt_net(input_pt)
            pt_out.backward(gradient=sens)
            if requires_grad:
                pt_weight = pt_net.weight.grad.detach().numpy()
                pt_bias = pt_net.bias.grad.detach().numpy()
                allclose_nparray(ms_out_list[0].asnumpy(), pt_weight, rtol, atol)
                allclose_nparray(ms_out_list[1].asnumpy(), pt_bias, rtol, atol)
            else:
                pt_bias = pt_net.bias.grad.detach().numpy()
                allclose_nparray(ms_out_list[0].asnumpy(), pt_bias, rtol, atol)

        elif flag == "FirstInput":
            input_pt.requires_grad = True
            pt_net.zero_grad()
            pt_out = pt_net(input_pt)
            pt_out.backward(gradient=sens)
            allclose_nparray(ms_out_list[0].asnumpy(), input_pt.grad.detach().numpy(), rtol, atol)

    input_np1, input_np2, input_np3, input_np4, input_np5, input_np6, input_np7, input_np8, weight_np, bias_np \
        = get_data(input_shape=(2, 2), weight_shape=(2, 2), bias_shape=(2, 2), input_count=8)
    weight_pt, bias_pt = get_pt_input([weight_np, bias_np])
    weight_ms, bias_ms = get_ms_input([weight_np, bias_np])
    ms_net = MsNet(weight_ms, bias_ms)
    pt_net = TorchNet(weight_pt, bias_pt)
    ms_net.set_grad()
    grad_ms_net = GradOfAllParams(ms_net, False)
    sens = torch.from_numpy(np.ones((2, 2)).astype(np.float32))
    for _ in range(2):
        input_ms_1, input_ms_2, input_ms_3, input_ms_4, input_ms_5, input_ms_6, input_ms_7, input_ms_8 = get_ms_input(
            [input_np1, input_np2, input_np3, input_np4, input_np5, input_np6, input_np7,
             input_np8])
        input_pt_1, input_pt_2, input_pt_3, input_pt_4, input_pt_5, input_pt_6, input_pt_7, input_pt_8 = get_pt_input(
            [input_np1, input_np2, input_np3, input_np4, input_np5, input_np6, input_np7,
             input_np8])
        ms_net(input_ms_1)
        ms_net(input_ms_2)
        ms_net(input_ms_3)
        ms_net(input_ms_4)
        ms_weight_1, ms_bias_1 = grad_ms_net(input_ms_1)
        ms_net(input_ms_5)
        ms_weight_2, ms_bias_2 = grad_ms_net(input_ms_2)
        ms_net(input_ms_6)
        ms_weight_3, ms_bias_3 = grad_ms_net(input_ms_3)
        ms_net(input_ms_7)
        ms_weight_4, ms_bias_4 = grad_ms_net(input_ms_4)
        ms_net(input_ms_8)
        ms_weight_5, ms_bias_5 = grad_ms_net(input_ms_5)
        pt_net(input_pt_1)
        pt_net(input_pt_2)
        pt_net(input_pt_3)
        pt_net(input_pt_4)
        # torch第一次反向权重求导
        get_torch_grad_compare(input_pt_1, pt_net, sens, ms_out_list=[ms_weight_1, ms_bias_1])
        pt_net(input_pt_5)
        # torch第二次反向权重求导
        get_torch_grad_compare(input_pt_2, pt_net, sens, ms_out_list=[ms_weight_2, ms_bias_2])
        pt_net(input_pt_6)
        # torch第三次反向权重求导
        get_torch_grad_compare(input_pt_3, pt_net, sens, ms_out_list=[ms_weight_3, ms_bias_3])
        pt_net(input_pt_7)
        # torch第四次反向权重求导
        get_torch_grad_compare(input_pt_4, pt_net, sens, ms_out_list=[ms_weight_4, ms_bias_4])
        pt_net(input_pt_8)
        # torch第五次反向权重求导
        get_torch_grad_compare(input_pt_5, pt_net, sens, ms_out_list=[ms_weight_5, ms_bias_5])
    assert ms_net.count == 16
