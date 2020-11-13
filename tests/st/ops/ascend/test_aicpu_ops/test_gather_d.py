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
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
class Net(nn.Cell):
    def __init__(self, dim=0):
        super(Net, self).__init__()
        self.op = P.GatherD()
        self.dim = dim

    def construct(self, x, index):
        return self.op(x, self.dim, index)


class NetGrad(nn.Cell):
    def __init__(self, dim=0, shape=None):
        super(NetGrad, self).__init__()
        self.op = G.GatherDGrad(dim, shape)

    def construct(self, index, x):
        return self.op(index, x)


def test_net():
    x = Tensor(np.array([[772, 231, 508, 545, 615, 249],
                         [923, 210, 480, 696, 482, 761],
                         [465, 904, 521, 824, 607, 669],
                         [156, 539, 56, 159, 916, 566],
                         [122, 676, 714, 261, 19, 936]]), mindspore.int32)
    index = Tensor(np.array([[0, 0, 0, 1, 1],
                             [0, 0, 0, 1, 4],
                             [0, 0, 0, 1, -1],
                             [1, 1, 1, 0, 0]]), mindspore.int32)
    dim = 0
    net = Net(dim)
    out = net(x, index)
    print(out.asnumpy())

    expect_out = np.array([[772, 231, 508, 696, 482],
                           [772, 231, 508, 696, 19],
                           [772, 231, 508, 696, 19],
                           [923, 210, 480, 545, 615]])
    assert np.array_equal(out.asnumpy(), expect_out)


def test_net_bool():
    x = Tensor(np.array([[0, 1, 0, 0, 1, 0],
                         [0, 1, 0, 0, 1, 0],
                         [0, 0, 1, 1, 0, 1],
                         [1, 0, 1, 1, 0, 0],
                         [1, 1, 1, 1, 0, 0]]), mindspore.bool_)
    index = Tensor(np.array([[0, 0, 0, 1, 1],
                             [0, 0, 0, 1, 4],
                             [0, 0, 0, 1, -1],
                             [1, 1, 1, 0, 0]]), mindspore.int32)
    dim = 0
    net = Net(dim)
    out = net(x, index)
    print(out.asnumpy())

    expect_out = np.array([[0, 1, 0, 0, 1],
                           [0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 1]]).astype(np.bool)
    assert np.array_equal(out.asnumpy(), expect_out)


def test_net_grad():
    index = Tensor(np.array([[0, 1, 2, 0, 0],
                             [2, 0, 0, 1, -1]]), mindspore.int32)
    x = Tensor(np.array([[772, 231, 508, 615, 249],
                         [122, 676, 714, 261, 936]]), mindspore.int32)
    net = NetGrad(dim=0, shape=(3, 5))
    out = net(index, x)
    print(out.asnumpy())

    expect_out = np.array([[772, 676, 714, 615, 249],
                           [0, 231, 0, 261, 0],
                           [122, 0, 508, 0, 936]])
    assert np.array_equal(out.asnumpy(), expect_out)
