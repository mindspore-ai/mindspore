# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore import ops

class InnerNet():
    def __init__(self):
        self.x = [Tensor([1]), Tensor([2]), Tensor([3])]

class Net(Cell):
    "Fallback network."
    def __init__(self):
        super(Net, self).__init__()
        obj = InnerNet()
        self.x = obj.x

    def construct(self, y):
        self.x.extend((y,))
        return ops.addn(self.x)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_tuple_input():
    """
    Feature: Test acl call with graph mode and dynamic shape.
    Description: Add 1+2+3+5 result.
    Expectation: output equal with 11.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level='O0')
    y = Tensor([5])
    net = Net()
    output = net(y)
    assert output == 11
