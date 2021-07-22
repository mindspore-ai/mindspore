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
import pytest
from mindspore.ops import composite as C
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.context as context
from mindspore.common.tensor import Tensor

class Net(nn.Cell):
    def construct(self, x, y):
        while x < y:
            x = x * x + 1
        return x


class GradNet(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad_op = C.GradOperation(get_all=True)

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_grad():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    x = Tensor([2.0], dtype=mstype.float32)
    y = Tensor([2.0], dtype=mstype.float32)
    GradNet(Net())(x, y)
