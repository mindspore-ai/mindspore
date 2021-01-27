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
""" test call inner net attr"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.ops import composite as C

context.set_context(mode=context.GRAPH_MODE, save_graphs=True)


class InnerInNet(nn.Cell):
    def __init__(self, init_data, const):
        super(InnerInNet, self).__init__()
        self.weight = Parameter(init_data, name="weight_s")
        self.t = init_data
        self.const = const

    def construct(self, input_x):
        if self.const:
            return input_x * self.t
        return input_x * self.weight


class InnerNet(nn.Cell):
    def __init__(self, init_data, const):
        super(InnerNet, self).__init__()
        self.inner_in_net = InnerInNet(init_data, const)
        self.t = init_data
        self.const = const

    def construct(self, input_x):
        if self.const:
            return self.inner_in_net.t / self.inner_in_net(input_x)
        return self.inner_in_net.weight / self.inner_in_net(input_x)


class Net(nn.Cell):
    def __init__(self, init_data, const):
        super(Net, self).__init__()
        self.inner_net = InnerNet(init_data, const)
        self.x = Tensor(np.ones((2, 3)) * 5)
        self.y = Tensor(np.ones((2, 3)) * 6)
        self.const = const
        self.weight = Parameter(init_data, name="weight_s")

    def construct(self, input_x, input_y):
        if self.const:
            return self.inner_net.t + self.inner_net(self.x) - self.y
        return self.inner_net.t + self.inner_net(input_x) - input_y


class OuterMostNet(nn.Cell):
    def __init__(self, init_data, const):
        super(OuterMostNet, self).__init__()
        self.net = Net(init_data, const)

    def construct(self, input_x, input_y):
        return self.net.inner_net.inner_in_net.t


class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.forward_net = net
        self.sens = Tensor(np.ones((2, 2), np.float32) * 5)
        self.grad_all = C.GradOperation(get_all=True)

    def construct(self, input_x, input_y):
        return self.grad_all(self.forward_net)(input_x, input_y)


def test_inner_net_attr():
    input_x = Tensor(np.ones((2, 3)) * 2)
    input_y = Tensor(np.ones((2, 3)) * 3)
    init_data = Tensor(np.ones((2, 3)) * 4)

    test_var_net = Net(init_data, False)
    test_var_net(input_x, input_y)

    grad_net = GradNet(test_var_net)
    grad_net(input_x, input_y)

    test_const_net = Net(init_data, True)
    ret = test_const_net(input_x, input_y)
    expect = -1.8 * np.ones((2, 3))
    assert np.allclose(ret.asnumpy(), expect)

    test_outer_net = OuterMostNet(init_data, True)
    ret = test_outer_net(input_x, input_y)
    assert np.allclose(ret.asnumpy(), init_data.asnumpy())
