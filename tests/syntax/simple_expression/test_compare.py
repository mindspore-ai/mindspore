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
""" test syntax for logic expression """

import mindspore.nn as nn
import mindspore
from mindspore import context
from mindspore.common.tensor import Tensor

context.set_context(mode=context.GRAPH_MODE)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.m = 1

    def construct(self, x, y):
        return x > y


def test_compare_bool_vs_bool():
    net = Net()
    ret = net(True, True)
    print(ret)


def test_compare_bool_vs_int():
    net = Net()
    ret = net(True, 1)
    print(ret)


def test_compare_tensor_int_vs_tensor_float():
    x = Tensor(1, mindspore.int32)
    y = Tensor(1.5, mindspore.float64)
    net = Net()
    ret = net(x, y)
    print(ret)
