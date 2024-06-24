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
    def construct(self, x):
        return ops.threshold(x, thr=0.0, value=20)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_threshold(mode):
    """
    Feature: ops.threshold
    Description: Verify the result of threshold
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[-0.90507276, -0.17371726, 0.94101539, -1.69267641, -0.4937978],
                [-2.17424723, -0.4541659, 1.28097345, -0.56799973, 0.26738557],
                [-1.07365091, 0.26963376, 0.34930261, -0.22567234, 0.69921238]], ms.float32)
    net = Net()
    output = net(x)
    expect_output = [[2.00000000e+01, 2.00000000e+01, 9.41015363e-01, 2.00000000e+01, 2.00000000e+01],
                     [2.00000000e+01, 2.00000000e+01, 1.28097343e+00, 2.00000000e+01, 2.67385572e-01],
                     [2.00000000e+01, 2.69633770e-01, 3.49302620e-01, 2.00000000e+01, 6.99212372e-01]]
    assert np.allclose(output.asnumpy(), expect_output)
