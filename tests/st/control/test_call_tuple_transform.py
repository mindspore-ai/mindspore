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
import mindspore as ms
from mindspore import context
from mindspore.ops import operations as P
from mindspore.common.api import jit
from mindspore.common.tensor import Tensor
import mindspore.nn as nn

import numpy as np
import pytest


class MAPPOCriticNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.linear1_actor = nn.Dense(54,  # input local obs shape
                                      64,
                                      weight_init='XavierUniform',
                                      # paper uses orthogonal with gain 5/3 for every dense123
                                      has_bias=False,
                                      activation=nn.Tanh())

    def construct(self, x):
        # Feature Extraction
        x = self.linear1_actor(x)

        return x


class MAPPOActor(nn.Cell):

    def __init__(self, actor_net):
        super().__init__()
        self.actor_net = actor_net

    def construct(self, inputs_data):
        _, global_obs = inputs_data
        out = self.actor_net(global_obs)

        return out


class TestClass(nn.Cell):
    def __init__(self, actor_list):
        super().__init__()
        self.zero = Tensor(0, ms.int32)
        self.actor_list = actor_list
        self.less = P.Less()
        self.zeros = P.Zeros()

    def train(self):
        state = Tensor(np.random.random((3, 128, 18)), ms.float32)
        init_global_obs = self.zeros((128, 54), ms.float32)
        out = self.test(state, init_global_obs)
        return out

    @jit
    def test(self, state, init_global_obs):
        num_agent = self.zero
        while self.less(num_agent, 3):
            samples = (state[num_agent], init_global_obs)
            self.actor_list[num_agent](samples)
            num_agent += 1

        return num_agent


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net():
    """
    Feature: Tuple arg transform.
    Description: Test the pass: transform tuple arg to tensor arg.
    Expectation: Compile done without error.
    """
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, save_graphs_path="./graph_ir")
    actor_list = nn.CellList()
    for _ in range(3):
        net = MAPPOCriticNet()
        actor = MAPPOActor(net)
        actor_list.append(actor)
    test = TestClass(actor_list)
    graph_out = test.train()

    assert np.allclose(graph_out.asnumpy(), graph_out.asnumpy(), 0.0001, 0.0001)
