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


class Net(nn.Cell):
    def construct(self, x):
        return x.sqrt()


def create_random_tensor(shape=(10,), low=0.0, high=1.0):
    """
    Randomly create a 1*10 Tensor
    """
    random_array = np.random.uniform(low, high, shape)
    random_tensor = ms.Tensor(random_array, ms.float32)
    return random_array, random_tensor


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_sqrt(mode):
    """
    Feature: tensor.sqrt
    Description: Verify the result of sqrt
    Expectation: success
    """
    ms.set_context(mode=mode)
    x_np, x_ms = create_random_tensor()
    net = Net()
    output = net(x_ms)
    expect_output = np.sqrt(x_np)
    assert np.allclose(output.asnumpy(), expect_output, rtol=5e-3, atol=1e-4)
