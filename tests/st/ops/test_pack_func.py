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
from mindspore import nn, Tensor
from mindspore.ops._packfunc import pack
from mindspore.ops import functional as F


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
            if F.is_sequence_shape_unknown(x.shape):
                z = x + y
                return z * z
            z = x + y + 1
            return z * z
    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    x = Tensor([1, 2, 3, 4])
    y = Tensor([4, 5, 6, 7], ms.float64)
    output = net(x, y)
    expect = np.array([36., 64., 100., 144.])
    assert np.allclose(output.asnumpy(), expect)
    assert output.dtype == ms.float64
