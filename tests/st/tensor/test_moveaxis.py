# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, nn


class Net(nn.Cell):
    def construct(self, x):
        return x.moveaxis(1, 0)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_net(mode):
    """
    Feature: test moveaxis
    Description: verify the result of moveaxis
    Expectation: assertion success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([[[-0.3362], [-0.8437]], [[-0.9627], [0.1727]], [[0.5173], [-0.1398]]]), ms.float32)
    moveaxis = Net()
    output = moveaxis(x)
    np_out = np.array([[[-0.3362], [-0.9627], [0.5173]], [[-0.8437], [0.1727], [-0.1398]]])
    assert np.allclose(output.asnumpy(), np_out)
