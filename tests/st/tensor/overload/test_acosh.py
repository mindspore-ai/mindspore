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


class AcoshNet(nn.Cell):
    def construct(self, x):
        return x.acosh()


def generate_random_input(shape, dtype):
    return np.random.uniform(1, 5000, shape).astype(dtype)


def generate_expect_forward_output(x):
    return np.arccosh(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_acosh(mode):
    """
    Feature: test Tensor.acosh
    Description: Verify the result of Tensor.acosh
    Expectation: expect correct forward result
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = generate_random_input((2, 3, 4), np.float32)
    net = AcoshNet()
    ms_output = net(Tensor(x, ms.float32))
    expect_output = generate_expect_forward_output(x)
    assert np.allclose(ms_output.asnumpy(), expect_output, rtol=1e-4, equal_nan=True)
