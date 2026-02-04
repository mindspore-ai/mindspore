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

import numpy as np
import pytest
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class FracNet(nn.Cell):
    def construct(self, x):
        return x.frac()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential',
          mem_peak=4.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_frac(mode):
    """
    Feature: Tensor.frac
    Description: Verify the result of Tensor.frac
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode)
    x = Tensor([2.129, 4.097, 1.3211, 0.12, 0.9], ms.float32)
    net = FracNet()
    output = net(x)
    expect_output = Tensor([0.129, 0.097, 0.3211, 0.12, 0.9], ms.float32)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
