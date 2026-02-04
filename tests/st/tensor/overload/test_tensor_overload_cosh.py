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

import numpy as np
import pytest
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class CoshNet(nn.Cell):
    def construct(self, x):
        return x.cosh()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cosh(mode):
    """
    Feature: test Tensor.cosh
    Description: Verify the result of Tensor.cosh
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x_np = np.array([[1., 2., 3.], [4., 5., 6.]]).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    net = CoshNet()
    ms_output = net(x)
    expect_output = np.array([[1.5430806, 3.7621953, 10.067661], [27.308233, 74.20995, 201.71564]])
    assert np.allclose(ms_output.asnumpy(), expect_output, rtol=1e-3)
