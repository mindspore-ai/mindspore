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
# ==============================================================================
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore import Tensor, context
from mindspore.common.parameter import ParameterTuple, Parameter
from mindspore.common.initializer import initializer


class FullyConnectedNet(nn.Cell):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnectedNet, self).__init__(auto_prefix=False)
        self.linear1 = nn.Dense(input_size, hidden_size, weight_init="XavierUniform")
        self.linear2 = nn.Dense(hidden_size, output_size, weight_init="XavierUniform")
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class EmaUpdate(nn.Cell):
    def __init__(self, policy_net, target_net, tau, period):
        super(EmaUpdate, self).__init__()
        self.tau = tau
        self.period = period
        # Use CellList manage parameters of multiple cells
        self.cell_list = nn.CellList()
        self.cell_list.append(policy_net)
        self.cell_list.append(target_net)
        self.policy_param = ParameterTuple(self.cell_list[0].get_parameters())
        self.target_param = ParameterTuple(self.cell_list[1].get_parameters())
        self.step = Parameter(initializer(0, [1]), name='step', requires_grad=False)
        self.hyper_map = C.HyperMap()
        self.assignadd = P.AssignAdd()

    def ema(self, tau, policy_param, target_param):
        new_param = (1 - tau) * target_param + tau * policy_param
        P.Assign()(target_param, new_param)
        return target_param

    def construct(self):
        if self.step % self.period == 0:
            self.hyper_map(F.partial(self.ema, self.tau), self.policy_param, self.target_param)
        self.assignadd(self.step, 1)
        return self.step


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_target_update(mode):
    """
    Feature: manage parameters with CellList.
    Description: Check the name of parameter in CellList.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    policy_net = FullyConnectedNet(4, 100, 2)
    target_net = FullyConnectedNet(4, 100, 2)
    tau = 0.2
    tau_tensor = Tensor(np.array([tau], dtype=np.float32))
    ema_update = EmaUpdate(policy_net, target_net, tau_tensor, period=1)
    res = ema_update()
    assert res == 1
    assert ema_update.step.name == "step"
    assert ema_update.policy_param[0].name == "0.linear1.weight"
    assert ema_update.policy_param[1].name == "0.linear1.bias"
    assert ema_update.policy_param[2].name == "0.linear2.weight"
    assert ema_update.policy_param[3].name == "0.linear2.bias"
    assert ema_update.target_param[0].name == "1.linear1.weight"
    assert ema_update.target_param[1].name == "1.linear1.bias"
    assert ema_update.target_param[2].name == "1.linear2.weight"
    assert ema_update.target_param[3].name == "1.linear2.bias"


class DenseNet(nn.Cell):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.fc1 = nn.Dense(16, 16)
        self.fc2 = nn.Dense(16, 16)

    def construct(self, x):
        out = self.fc2(self.fc1(x))
        return out


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_two_dense_net(mode):
    """
    Feature: Check the name of parameter .
    Description: Check the name of parameter in two network.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.random.randn(4, 16).astype(np.float32))
    net = DenseNet()
    net(x)
    assert net.fc1._params['weight'].name == "fc1.weight"
    assert net.fc1._params['bias'].name == "fc1.bias"
    assert net.fc2._params['weight'].name == "fc2.weight"
    assert net.fc2._params['bias'].name == "fc2.bias"


class InnerNet(nn.Cell):
    def __init__(self):
        super(InnerNet, self).__init__()
        self.param = Parameter(Tensor([1], ms.float32), name="name_a")

    def construct(self, x):
        return x + self.param


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_two_net(mode):
    """
    Feature: Check the name of parameter .
    Description: Check the name of parameter in two network.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    net1 = InnerNet()
    net2 = InnerNet()
    res1 = net1(Tensor([1], ms.float32))
    res2 = net2(Tensor([1], ms.float32))
    assert res1 == res2 == 2
    assert net1.param.name == net1.param.name == "name_a"


class OutNet1(nn.Cell):
    def __init__(self, net1, net2):
        super(OutNet1, self).__init__()
        self.param1 = ParameterTuple(net1.get_parameters())
        self.param2 = ParameterTuple(net2.get_parameters())

    def construct(self, x):
        return x + self.param1[0] + self.param2[0]


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_inner_out_net_1(mode):
    """
    Feature: Check the name of parameter .
    Description: Check the name of parameter in two network.
    Expectation: No exception.
    """
    with pytest.raises(RuntimeError, match="its name 'name_a' already exists."):
        context.set_context(mode=mode)
        net1 = InnerNet()
        net2 = InnerNet()
        out_net = OutNet1(net1, net2)
        res = out_net(Tensor([1], ms.float32))
        print("res:", res)


class OutNet2(nn.Cell):
    def __init__(self, net1, net2):
        super(OutNet2, self).__init__()
        self.cell_list = nn.CellList()
        self.cell_list.append(net1)
        self.cell_list.append(net2)
        self.param1 = ParameterTuple(self.cell_list[0].get_parameters())
        self.param2 = ParameterTuple(self.cell_list[1].get_parameters())

    def construct(self, x):
        return x + self.param1[0] + self.param2[0]


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_inner_out_net_2(mode):
    """
    Feature: Check the name of parameter .
    Description: Check the name of parameter in two network.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    net1 = InnerNet()
    net2 = InnerNet()
    out_net = OutNet2(net1, net2)
    res = out_net(Tensor([1], ms.float32))
    assert res == 3
    assert out_net.param1[0].name == "0.param"
    assert out_net.param2[0].name == "1.param"
