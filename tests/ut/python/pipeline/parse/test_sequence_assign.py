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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context

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
