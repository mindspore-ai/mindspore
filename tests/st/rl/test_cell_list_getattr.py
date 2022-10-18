# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
""" test a list of cell, and getattr by its item """
import pytest
import numpy as np
from mindspore import context, nn, dtype, Tensor, jit, jit_class
from mindspore.ops import operations as P
from mindspore.ops import composite as C


class Actor(nn.Cell):
    def act(self, x, y):
        return x + y


@jit_class
class Actor2:
    def act(self, x, y):
        return x + y


class Trainer(nn.Cell):
    def __init__(self, net_list):
        super(Trainer, self).__init__()
        self.net_list = net_list

    def construct(self, x, y):
        return self.net_list[0].act(x, y)


class GradNet(nn.Cell):
    def __init__(self, network, get_all=False, get_by_list=False, sens_param=False):
        super(GradNet, self).__init__()
        self.network = network
        self.grad = C.GradOperation(get_all, get_by_list, sens_param)

    def construct(self, *inputs):
        grads = self.grad(self.network)(*inputs)
        return grads


def verify_list_item_getattr(trainer, expect_res, expect_grad_res):
    x = Tensor([3], dtype=dtype.float32)
    y = Tensor([6], dtype=dtype.float32)
    res = trainer(x, y)
    assert np.array_equal(res.asnumpy(), expect_res.asnumpy())

    grad_net = GradNet(trainer)
    res2 = grad_net(x, y)
    assert np.array_equal(res2.asnumpy(), expect_grad_res.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_list_item_getattr():
    """
    Feature: getattr by the item from list of cell.
    Description: Support RL use method in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    actor_list = [Actor()]
    trainer = Trainer(actor_list)
    expect_res = Tensor([9], dtype=dtype.float32)
    expect_grad_res = Tensor([1], dtype=dtype.float32)
    verify_list_item_getattr(trainer, expect_res, expect_grad_res)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cell_list_getattr():
    """
    Feature: getattr by the item from nn.CellList.
    Description: Support RL use method in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    actor_list = nn.CellList()
    for _ in range(3):
        actor_list.append(Actor())
    trainer = Trainer(actor_list)
    expect_res = Tensor([9], dtype=dtype.float32)
    expect_grad_res = Tensor([1], dtype=dtype.float32)
    verify_list_item_getattr(trainer, expect_res, expect_grad_res)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_msclass_list_getattr():
    """
    Feature: getattr by the item from list of ms_class.
    Description: Support RL use method in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    actor_list = [Actor2()]
    trainer = Trainer(actor_list)
    expect_res = Tensor([9], dtype=dtype.float32)
    expect_grad_res = Tensor([1], dtype=dtype.float32)
    verify_list_item_getattr(trainer, expect_res, expect_grad_res)


class Trainer2(nn.Cell):
    def __init__(self, net_list):
        super(Trainer2, self).__init__()
        self.net_list = net_list
        self.less = P.Less()
        self.zero_float = Tensor(0, dtype=dtype.float32)

    def construct(self, x, y):
        sum_value = self.zero_float
        num_actor = 0
        while num_actor < 3:
            sum_value += self.net_list[num_actor].act(x, y)
            num_actor += 1
        return sum_value


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_list_item_getattr2():
    """
    Feature: getattr by the item from list of cell with a Tensor variable.
    Description: Support RL use method in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    actor_list = [Actor(), Actor(), Actor()]
    trainer = Trainer2(actor_list)
    expect_res = Tensor([27], dtype=dtype.float32)
    expect_grad_res = Tensor([3], dtype=dtype.float32)
    verify_list_item_getattr(trainer, expect_res, expect_grad_res)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cell_list_getattr2():
    """
    Feature: getattr by the item from nn.CellList.
    Description: Support RL use method in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    actor_list = nn.CellList()
    for _ in range(3):
        actor_list.append(Actor())
    trainer = Trainer2(actor_list)
    expect_res = Tensor([27], dtype=dtype.float32)
    expect_grad_res = Tensor([3], dtype=dtype.float32)
    verify_list_item_getattr(trainer, expect_res, expect_grad_res)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_msclass_list_getattr2():
    """
    Feature: getattr by the item from list of ms_class with a Tensor variable.
    Description: Support RL use method in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    actor_list = [Actor2(), Actor2(), Actor2()]
    trainer = Trainer2(actor_list)
    expect_res = Tensor([27], dtype=dtype.float32)
    expect_grad_res = Tensor([3], dtype=dtype.float32)
    verify_list_item_getattr(trainer, expect_res, expect_grad_res)


class MSRL(nn.Cell):
    def __init__(self, agent):
        super(MSRL, self).__init__()
        self.agent = agent


class Agent(nn.Cell):
    def __init__(self, actor):
        super(Agent, self).__init__()
        self.actor = actor

    def act(self, x, y):
        out = self.actor.act(x, y)
        return out


class Trainer3(nn.Cell):
    def __init__(self, msrl):
        super(Trainer3, self).__init__()
        self.msrl = msrl

    @jit
    def test(self, x, y):
        num_actor = 0
        output = 0
        while num_actor < 3:
            output += self.msrl.agent[num_actor].act(x, y)
            num_actor += 1
        return output


def verify_list_item_getattr2(trainer, expect_res, expect_grad_res):
    x = Tensor([2], dtype=dtype.int32)
    y = Tensor([3], dtype=dtype.int32)
    res = trainer.test(x, y)
    assert np.array_equal(res.asnumpy(), expect_res.asnumpy())

    grad_net = GradNet(trainer)
    res2 = grad_net(x, y)
    assert np.array_equal(res2.asnumpy(), expect_grad_res.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_list_item_getattr3():
    """
    Feature: getattr by the item from list of cell.
    Description: Support RL use method in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    agent_list = []
    for _ in range(3):
        actor = Actor()
        agent_list.append(Agent(actor))
    msrl = MSRL(agent_list)
    trainer = Trainer3(msrl)
    expect_res = Tensor([15], dtype=dtype.int32)
    expect_grad_res = Tensor([3], dtype=dtype.int32)
    verify_list_item_getattr2(trainer, expect_res, expect_grad_res)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cell_list_getattr3():
    """
    Feature: getattr by the item from list of cell.
    Description: Support RL use method in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    agent_list = nn.CellList()
    for _ in range(3):
        actor = Actor()
        agent_list.append(Agent(actor))
    msrl = MSRL(agent_list)
    trainer = Trainer3(msrl)
    expect_res = Tensor([15], dtype=dtype.int32)
    expect_grad_res = Tensor([3], dtype=dtype.int32)
    verify_list_item_getattr2(trainer, expect_res, expect_grad_res)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_msclass_list_getattr3():
    """
    Feature: getattr by the item from list of ms_class.
    Description: Support RL use method in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    agent_list = []
    for _ in range(3):
        actor = Actor2()
        agent_list.append(Agent(actor))
    msrl = MSRL(agent_list)
    trainer = Trainer3(msrl)
    expect_res = Tensor([15], dtype=dtype.int32)
    expect_grad_res = Tensor([3], dtype=dtype.int32)
    verify_list_item_getattr2(trainer, expect_res, expect_grad_res)
