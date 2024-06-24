# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import pytest

import mindspore.nn as nn
from mindspore import context
from mindspore.common import mutable
from sequence_help import context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


class Net(nn.Cell):
    def construct(self, x, y):
        return int(x), float(y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_scalar_cast():
    """
    Feature: test sequence_add op
    Description: two inputs are dynamic sequence
    Expectation: the result match with tuple result
    """
    x = mutable(2.1)
    y = mutable(3)
    expectx = 2
    expecty = 3.0
    net = Net()
    res = net(x, y)
    assert res[0] == expectx
    assert res[1] == expecty
