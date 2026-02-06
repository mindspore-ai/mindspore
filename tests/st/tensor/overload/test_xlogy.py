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


class Net(nn.Cell):
    def construct(self, x, other):
        return x.xlogy(other)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_xlogy(mode):
    """
    Feature: Tensor.logaddexp
    Description: Verify the result of Tensor.logaddexp
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    x = Tensor(np.array([[-5, 0, 4]]), ms.float32)
    other_scalar = 2
    output_scalar = net(x, other_scalar)
    other_tensor = Tensor(2)
    output_tensor = net(x, other_tensor)
    expect_output = np.array([-3.46573591, 0., 2.77258873])
    assert np.allclose(output_scalar.asnumpy(), expect_output)
    assert np.allclose(output_tensor.asnumpy(), expect_output)
