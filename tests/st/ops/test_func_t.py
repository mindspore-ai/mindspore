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
import mindspore.nn as nn
from mindspore import Tensor, ops


class Net(nn.Cell):
    def construct(self, x):
        return ops.t(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_t(mode):
    """
    Feature: ops.t
    Description: Verify the result of t
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[1, 2, 3], [2, 3, 4]], ms.float32)
    net = Net()
    output = net(x)
    expect_output = [[1, 2], [2, 3], [3, 4]]
    assert np.allclose(output.asnumpy(), expect_output)
