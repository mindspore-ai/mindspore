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
from mindspore import Tensor
from mindspore import ops


class Net(nn.Cell):
    def construct(self, x, diagonal=0):
        return ops.tril(x, diagonal)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tril(mode):
    """
    Feature: tril
    Description: Verify the result of tril
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[-1.8297, -0.8474, 1.0292], [-1.2167, 0.5574, -0.6753], [-0.6702, 0.2276, 1.2421]])
    net = Net()
    output = net(x)
    expect_output = np.array([[-1.8297, 0., 0.], [-1.2167, 0.5574, 0.], [-0.6702, 0.2276, 1.2421]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
