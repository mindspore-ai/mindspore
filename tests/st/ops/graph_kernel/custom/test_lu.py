# Copyright 2022 Huawei Technologies Co., Ltd
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
import mindspore.context as context
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.scipy.ops import LU


class NetLU(nn.Cell):
    def __init__(self):
        super(NetLU, self).__init__()
        self.ops = LU()

    def construct(self, a):
        return self.ops(a)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_lu():
    """
    Feature: lu op use custom compile test on graph mode.
    Description: test lu op on graph mode
    Expectation: the result equal to expect.
    """
    num = 32
    one_tensor = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            one_tensor[i, j] = min(min(i, j), 8) + 1
    upper_matrix = np.triu(one_tensor).astype(np.float32)
    lower_matrix = np.tril(np.ones((num, num))).astype(np.float32)
    input1 = np.dot(lower_matrix, upper_matrix)
    expect = upper_matrix + lower_matrix - np.eye(num)
    real_input = Tensor(input1, mstype.float32)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = NetLU()
    result = net(real_input)
    rtol = 0.001
    atol = 0.001
    assert np.allclose(result[0].asnumpy(), expect, rtol, atol, equal_nan=True)
