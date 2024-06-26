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


class Net(nn.Cell):
    def construct(self, x):
        return x.sinc()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_sinc(mode):
    """
    Feature: tensor.sinc
    Description: Verify the result of sinc
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([0.62, 0.28, 0.43, 0]), ms.float32)
    net = Net()
    output = net(x)
    expect_output = np.array([0.47735006, 0.8759357, 0.7224278, 1.], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output, rtol=5e-3, atol=1e-4)
