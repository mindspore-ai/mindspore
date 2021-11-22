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
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore import Tensor, context
from mindspore.common.parameter import ParameterTuple, Parameter
from mindspore.common.initializer import initializer

context.set_context(mode=context.GRAPH_MODE)


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
        out = P.Assign()(target_param, new_param)
        return out

    def construct(self):
        if self.step % self.period == 0:
            self.hyper_map(F.partial(self.ema, self.tau), self.policy_param, self.target_param)
        return self.assignadd(self.step, 1)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_target_update():
    """
    Feature: manage parameters with CellList.
    Description: Check the name of parameter in CellList.
    Expectation: No exception.
    """
    policy_net = FullyConnectedNet(4, 100, 2)
    target_net = FullyConnectedNet(4, 100, 2)
    tau = 0.2
    tau_tensor = Tensor(np.array([tau], dtype=np.float32))
    ema_update = EmaUpdate(policy_net, target_net, tau_tensor, period=1)
    ema_update()
