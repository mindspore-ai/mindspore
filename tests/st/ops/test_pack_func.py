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

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore.ops._packfunc import pack


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
def test_pack_basic_cell():
    """
    Feature: pack of cell
    Description: Verify the result of pack
    Expectation: success
    """
    class Net(nn.Cell):
        @pack
        def construct(self, x, y):
            z = x + y
            return z * z
    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    x = ms.Tensor([1, 2, 3, 4])
    y = ms.Tensor([4, 5, 6, 7])
    output = net(x, y)
    expect = np.array([25, 49, 81, 121])
    assert np.allclose(output.asnumpy(), expect)
