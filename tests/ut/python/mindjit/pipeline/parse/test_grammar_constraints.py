# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
"""
test mindspore grammar constraints
1. function must have return statement
2. raise statement can not be used
"""
# pylint: disable=R1705, R1710, W0223
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE)


def test_nest_branch_with_return():
    class NetBranchWithReturn(nn.Cell):
        def construct(self, x, y, z):
            if x == 1:
                return 10
            else:
                return 5

    net = NetBranchWithReturn()
    x = Tensor(0, mstype.int32)
    y = Tensor(5, mstype.int32)
    z = Tensor(2, mstype.int32)
    net(x, y, z)


def test_any_with_no_return():
    class NetAnyNoReturn(nn.Cell):
        def construct(self, inp):
            result = inp.any()
            if result:
                return 6

    np_input = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.bool_)
    tensor = Tensor(np_input)
    net = NetAnyNoReturn()
    net(tensor)


def test_missing_construct():
    class NetMissConstruct(nn.Cell):
        def construct1(self, inp):
            return 5

    np_input = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.bool_)
    tensor = Tensor(np_input)
    net = NetMissConstruct()
    with pytest.raises(AttributeError) as info:
        net(tensor)
    assert "construct" in str(info.value)
    assert "not defined" in str(info.value)
