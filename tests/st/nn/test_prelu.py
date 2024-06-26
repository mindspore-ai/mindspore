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

import mindspore as ms
import mindspore.nn as nn
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.PReLU(channel=2, w=-0.25)

    def construct(self, x):
        out = self.pool(x)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_prelu_normal(mode):
    """
    Feature: PReLU
    Description: Verify the result of PReLU
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([[[0.9192, -0.1487],
                    [-0.3999, -0.6840]],

                   [[0.4745, -0.6271],
                    [-0.6547, -0.5856]],

                   [[-0.2572, -0.8412],
                    [0.1918, -0.6117]]])
    net = Net()
    out = net(x)
    expect_out = np.array([[[0.9192, 0.037175],
                            [0.099975, 0.171]],

                           [[0.4745, 0.156775],
                            [0.163675, 0.1464]],

                           [[0.0643, 0.2103],
                            [0.1918, 0.152925]]])
    assert np.allclose(out.asnumpy().astype(np.float16), expect_out.astype(np.float16))
