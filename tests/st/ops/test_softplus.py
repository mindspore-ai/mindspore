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
from tests.mark_utils import arg_mark
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, nn, ops


class Net(nn.Cell):
    def construct(self, x, beta=1, threshold=20):
        return ops.softplus(x, beta, threshold)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_softplus(mode):
    """
    Feature: test ops.softplus
    Description: verify the result of softplus
    Expectation: success
    """
    ms.set_context(mode=mode)
    softplus = Net()
    x = Tensor(np.array([0.1, 0.2, 30, 25]), ms.float32)

    output = softplus(x)
    expect_output = np.array([0.7443967, 0.7981389, 30.0000000, 25.0000000])
    assert np.allclose(output.asnumpy(), expect_output)

    output = softplus(x, 0.3, 100)
    expect_output = np.array([2.3608654, 2.4119902, 30.0004082, 25.0018444])
    assert np.allclose(output.asnumpy(), expect_output)
