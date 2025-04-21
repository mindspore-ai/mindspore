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
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import function as F

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.sigmoid = F.sigmoid

    def construct(self, x):
        return self.sigmoid(x)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_sigmod_bfloat16(mode):
    """
    Feature: test Sigmoid forward.
    Description: test bfloat16 inputs.
    Expectation: compare the result with exception value.
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), mstype.bfloat16)
    output = net(x).float().asnumpy()
    expect_output = np.array([0.73046875, 0.87890625, 0.953125, 0.98046875, 0.9921875]).astype(np.float32)
    np.testing.assert_allclose(output, expect_output, 1e-3, 1e-3)
