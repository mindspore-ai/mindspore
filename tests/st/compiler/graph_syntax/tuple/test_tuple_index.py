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
""" test tuple index """

import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.skip(reason="No support yet.")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tuple_index_is_tensor_in_control_flow():
    """
    Feature: Support tuple while the index is Tensor.
    Description: Support tuple while the index is Tensor.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            y = (1, 2, 3, 4)
            index = x[0] + 1
            if x[index] > 0:
                return y[index]
            return y[index] * 2

    x = ms.Tensor([-1], ms.int32)
    net = Net()
    ret = net(x)
    assert ret == 2
