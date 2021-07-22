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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Netnan(nn.Cell):
    def __init__(self):
        super(Netnan, self).__init__()
        self.isnan = P.IsNan()

    def construct(self, x):
        return self.isnan(x)


x1 = np.array([[1.2, 2, np.nan, 88]]).astype(np.float32)
x2 = np.array([[np.inf, 1, 88.0, 0]]).astype(np.float32)
x3 = np.array([[1, 2], [3, 4], [5.0, 88.0]]).astype(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_nan():
    ms_isnan = Netnan()
    output1 = ms_isnan(Tensor(x1))
    expect1 = [[False, False, True, False]]
    assert (output1.asnumpy() == expect1).all()

    output2 = ms_isnan(Tensor(x2))
    expect2 = [[False, False, False, False]]
    assert (output2.asnumpy() == expect2).all()

    output3 = ms_isnan(Tensor(x3))
    expect3 = [[False, False], [False, False], [False, False]]
    assert (output3.asnumpy() == expect3).all()
