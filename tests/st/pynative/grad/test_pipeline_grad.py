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
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.common import Tensor
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from tests.st.pynative.utils import GradOfAllParams, GradOfAllInputs, HighGrad
from tests.mark_utils import arg_mark


def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
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
def test_network_pipeline_with_high_grad():
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
        super(MEMul1, self).__init__()
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
