# Copyright 2025 Huawei Technologies Co., Ltd
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


class Net(nn.Cell):
    def construct(self, x):
        x.log_()
        return x


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_log(mode):
    """
    Feature: tensor.log
    Description: Verify the result of tensor.log
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    input_np = np.array([0.1, 1.0, 3.0])
    net = Net()
    inputs = Tensor(input_np, ms.float32)
    outputs = net(inputs)
    expect_output = np.log(input_np)
    atol = 1e-4
    assert np.allclose(outputs.asnumpy(), expect_output, atol)
