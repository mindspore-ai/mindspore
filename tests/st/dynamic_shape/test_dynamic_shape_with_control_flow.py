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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore import Parameter
from tests.mark_utils import arg_mark


class UniqueIf(nn.Cell):
    def __init__(self):
        super(UniqueIf, self).__init__()
        self.unique = P.Unique()
        self.shape = P.DynamicShape()

    def construct(self, x, index):
        x_unique = self.unique(x)[0]
        if index > 3:
            x_unique = x_unique + 2
        else:
            x_unique = x_unique - 3
        return self.shape(x_unique)


class UniqueWhile(nn.Cell):
    def __init__(self):
        super(UniqueWhile, self).__init__()
        self.unique = P.Unique()
        self.shape = P.DynamicShape()
        self.mod = P.Mod()

    def construct(self, x, y, index):
        while index < 3:
            x = self.mod(x, y[index])
            x = self.unique(x)[0]
            index = index + 1
        return self.shape(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_unique_if():
    """
    Feature: Dynamic shape for control flow.
    Description: If scene.
    Expectation: No exception.
    """
    x = Tensor(np.array([4, 5, 1, 2, 3, 3, 4, 5]).astype(np.int32))
    index = Tensor([0], mstype.int32)
    context.set_context(mode=context.GRAPH_MODE)
    net = UniqueIf()
    x_shape = net(x, index)
    assert x_shape == 5


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_unique_while():
    """
    Feature: Dynamic shape for control flow.
    Description: While scene.
    Expectation: No exception.
    """
    x = Tensor(np.array([12406268, 4962722, 720966, 75948, 6776, 960, 67, 8]).astype(np.int32))
    y = Tensor(np.array([957, 67, 7]).astype(np.int32))
    index = Tensor([0], mstype.int32)
    context.set_context(mode=context.GRAPH_MODE)
    net = UniqueWhile()
    x_shape = net(x, y, index)
    assert x_shape == 3


class DynShapeJointNet(nn.Cell):
    def __init__(self):
        super(DynShapeJointNet, self).__init__()
        self.cond = Parameter(Tensor([True]))

    def construct(self, x, y):
        if self.cond:
            out = x
        else:
            out = y
        return out
