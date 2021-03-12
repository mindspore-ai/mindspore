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
""" test tuple index by negative number"""
import numpy as np
import pytest

from mindspore import nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P
from mindspore.ops import composite as C

context.set_context(mode=context.GRAPH_MODE)


def test_tuple_index_by_negative_number():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.index = -1
            self.split = P.Split(axis=0, output_num=4)

        def construct(self, x):
            out = self.split(x)
            ret = [out[-1], out[-2], out[-3], out[-4]]
            ret[-1] = 100
            return ret

    class GradNet(nn.Cell):
        def __init__(self, net, get_all):
            super(GradNet, self).__init__()
            self.forward_net = net
            self.sens = Tensor(np.ones((2, 2), np.float32) * 5)
            self.grad_all = C.GradOperation(get_all=get_all)

        def construct(self, x):
            return self.grad_all(self.forward_net)(x)

    net = Net()
    grad_net = GradNet(net, True)
    x = Tensor(np.ones((4, 2, 3)))
    net(x)
    grad_net(x)


def test_tuple_index_by_negative_number_out_bound():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.index = -1
            self.split = P.Split(axis=0, output_num=2)

        def construct(self, x):
            out = self.split(x)
            return out[-1], out[-2], out[-3]

    net = Net()
    x = Tensor(np.ones((2, 2, 3)))
    with pytest.raises(IndexError) as err:
        net(x)
    assert "TupleGetItem evaluator index should be in range[-2, 2), but got -3" in str(err.value)
