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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.pad_and_shift = P.PadAndShift()
        self.shift_idx = 1

    def construct(self, x, y):
        return self.pad_and_shift(x, y, self.shift_idx)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_pad_and_shift_cpu():
    x = Tensor(np.array([9, 13, -1, -1, -1, -1, -1, -1]), mstype.int32)
    y = Tensor(np.array([0, 3, 5]), mstype.int32)
    net = Net()
    output = net(x, y)
    expect = np.array([-1, -1, -1, 9, 13])
    assert (output.asnumpy() == expect).all()
