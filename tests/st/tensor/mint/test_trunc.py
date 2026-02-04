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

import mindspore.common.dtype as mstype
import mindspore as ms
import mindspore.nn as nn

class TruncNet(nn.Cell):
    def construct(self, x):
        return x.trunc()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_f_trunc(mode):
    """
    Feature: tensor.trunc()
    Description: Verify the result of tensor.trunc
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = TruncNet()
    x = ms.Tensor([3.4732, 0.5466, -0.8008, -3.9079], dtype=mstype.float32)
    output = net(x)
    expected = np.array([3, 0, 0, -3], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)
