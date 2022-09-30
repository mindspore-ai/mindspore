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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_diff_size_join_as_dyn_rank():
    """
    Feature: Dynamic shape join for control flow.
    Description: Two different size shapes joined as dynamic rank shape.
    Expectation: No exception.
    """
    x = np.arange(2 * 3 * 2).reshape((2, 3, 2))
    y = np.arange(88, 2 * 3 * 2 * 2 + 88).reshape((2, 3, 2, 2))
    input_x_dyn = Tensor(shape=[2, None, 2], dtype=mstype.float32)
    input_y_dyn = Tensor(shape=[2, 3, None, None], dtype=mstype.float32)
    dyn_net = DynShapeJointNet()
    dyn_net.set_inputs(input_x_dyn, input_y_dyn)
    dyn_net(Tensor(x.astype(np.float32)), Tensor(y.astype(np.float32)))
