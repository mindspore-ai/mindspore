# Copyright 2021 Huawei Technologies Co., Ltd
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
import numpy as np
from mindspore import context, nn, dtype, Tensor
from mindspore.ops import operations as P


class Actor(nn.Cell):
    def act(self, x, y):
        return x + y


class Trainer(nn.Cell):
    def __init__(self, net_list):
        super(Trainer, self).__init__()
        self.net_list = net_list

    def construct(self, x, y):
        return self.net_list[0].act(x, y)


def test_list_item_getattr():
    """
    Feature: getattr by the item from list of cell.
    Description: Support RL use method in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    actor_list = [Actor()]
    trainer = Trainer(actor_list)
    x = Tensor([3], dtype=dtype.float32)
    y = Tensor([6], dtype=dtype.float32)
    res = trainer(x, y)
    print(f'res: {res}')
    expect_res = Tensor([9], dtype=dtype.float32)
    assert np.array_equal(res.asnumpy(), expect_res.asnumpy())


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


def test_list_item_getattr2():
    """
    Feature: getattr by the item from list of cell with a Tensor variable.
    Description: Support RL use method in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    actor_list = [Actor(), Actor(), Actor()]
    trainer = Trainer2(actor_list)
    x = Tensor([3], dtype=dtype.float32)
    y = Tensor([6], dtype=dtype.float32)
    res = trainer(x, y)
    print(f'res: {res}')
    expect_res = Tensor([27], dtype=dtype.float32)
    assert np.array_equal(res.asnumpy(), expect_res.asnumpy())
