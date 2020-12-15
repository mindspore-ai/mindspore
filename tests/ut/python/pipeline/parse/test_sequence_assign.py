# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test enumerate"""

import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P
from mindspore.ops import composite as C

context.set_context(mode=context.GRAPH_MODE)


def test_list_index_1D():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self):
            list_ = [[1], [2, 2], [3, 3, 3]]
            list_[0] = [100]
            return list_

    net = Net()
    out = net()
    assert out[0] == [100]
    assert out[1] == [2, 2]
    assert out[2] == [3, 3, 3]


def test_list_neg_index_1D():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self):
            list_ = [[1], [2, 2], [3, 3, 3]]
            list_[-3] = [100]
            return list_

    net = Net()
    out = net()
    assert out[0] == [100]
    assert out[1] == [2, 2]
    assert out[2] == [3, 3, 3]


def test_list_index_2D():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self):
            list_ = [[1], [2, 2], [3, 3, 3]]
            list_[1][0] = 200
            list_[1][1] = 201
            return list_

    net = Net()
    out = net()
    assert out[0] == [1]
    assert out[1] == [200, 201]
    assert out[2] == [3, 3, 3]


def test_list_neg_index_2D():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self):
            list_ = [[1], [2, 2], [3, 3, 3]]
            list_[1][-2] = 200
            list_[1][-1] = 201
            return list_

    net = Net()
    out = net()
    assert out[0] == [1]
    assert out[1] == [200, 201]
    assert out[2] == [3, 3, 3]


def test_list_index_3D():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self):
            list_ = [[1], [2, 2], [[3, 3, 3]]]
            list_[2][0][0] = 300
            list_[2][0][1] = 301
            list_[2][0][2] = 302
            return list_

    net = Net()
    out = net()
    assert out[0] == [1]
    assert out[1] == [2, 2]
    assert out[2] == [[300, 301, 302]]


def test_list_neg_index_3D():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self):
            list_ = [[1], [2, 2], [[3, 3, 3]]]
            list_[2][0][-3] = 300
            list_[2][0][-2] = 301
            list_[2][0][-1] = 302
            return list_

    net = Net()
    out = net()
    assert out[0] == [1]
    assert out[1] == [2, 2]
    assert out[2] == [[300, 301, 302]]


def test_list_index_1D_parameter():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            list_ = [x]
            list_[0] = 100
            return list_

    net = Net()
    net(Tensor(0))


def test_list_index_2D_parameter():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            list_ = [[x, x]]
            list_[0][0] = 100
            return list_

    net = Net()
    net(Tensor(0))


def test_list_index_3D_parameter():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            list_ = [[[x, x]]]
            list_[0][0][0] = 100
            return list_

    net = Net()
    net(Tensor(0))


def test_const_list_index_3D_bprop():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [[1], [2, 2], [[3, 3], [3, 3]]]
            self.relu = P.ReLU()

        def construct(self, input_x):
            list_x = self.value
            list_x[2][0][1] = input_x
            return self.relu(list_x[2][0][1])

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_all_with_sens = C.GradOperation(get_all=True, sens_param=True)

        def construct(self, x, sens):
            return self.grad_all_with_sens(self.net)(x, sens)

    net = Net()
    grad_net = GradNet(net)
    x = Tensor(np.arange(2 * 3).reshape(2, 3))
    sens = Tensor(np.arange(2 * 3).reshape(2, 3))
    grad_net(x, sens)


def test_parameter_list_index_3D_bprop():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [[1], [2, 2], [[3, 3], [3, 3]]]
            self.relu = P.ReLU()

        def construct(self, x, value):
            list_value = [[x], [x, x], [[x, x], [x, x]]]
            list_value[2][0][1] = value
            return self.relu(list_value[2][0][1])

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_all_with_sens = C.GradOperation(get_all=True, sens_param=True)

        def construct(self, x, value, sens):
            return self.grad_all_with_sens(self.net)(x, value, sens)

    net = Net()
    grad_net = GradNet(net)
    x = Tensor(np.arange(2 * 3).reshape(2, 3))
    value = Tensor(np.ones((2, 3), np.int64))
    sens = Tensor(np.arange(2 * 3).reshape(2, 3))
    grad_net(x, value, sens)
