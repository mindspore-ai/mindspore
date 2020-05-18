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
import numpy as np
import pytest
from cus_square import CusSquare

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    """Net definition"""

    def __init__(self):
        super(Net, self).__init__()
        self.square = CusSquare()

    def construct(self, data):
        return self.square(data)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_net():
    x = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    square = Net()
    output = square(Tensor(x))
    print(x)
    print(output.asnumpy())
    expect = np.array([1.0, 16.0, 81.0]).astype(np.float32)
    assert (output.asnumpy() == expect).all()
