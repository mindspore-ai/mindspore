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
import pytest
from mindspore import Tensor, context
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_in_asnumpy():
    """
    Feature: Support in.
    Description: Support in in fallback runtime.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, y):
            input_x = Tensor([1, 2], dtype=mstype.int32).asnumpy()
            return y in input_x

    net = Net()
    res = net(2)
    assert res
