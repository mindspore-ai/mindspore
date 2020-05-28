# Copyright 2020 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import composite as C
from cus_add3 import CusAdd3
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class Net(nn.Cell):
    """Net definition"""

    def __init__(self):
        super(Net, self).__init__()
        self.add3 = CusAdd3(1.0)

    def construct(self, input1, input2):
        return self.add3(input1, input2)

@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_net():
    input1 = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    input2 = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    add3_net = Net()
    output = add3_net(Tensor(input1), Tensor(input2))
    expect = np.array([3.0, 7.0, 13.0]).astype(np.float32)
    assert (output.asnumpy() == expect).all()