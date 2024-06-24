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
from mindspore import ops


class Net(nn.Cell):
    def construct(self, x):
        return ops.digamma(x)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_digamma(mode):
    """
    Feature: ops.digamma
    Description: Verify the result of digamma
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([-0.5, 0.5, 10, 1.0], ms.float32)
    net = Net()
    output = net(x)
    expect_output = [0.03648978, -1.9635109, 2.2517526, -0.5772159]
    assert np.allclose(output.asnumpy(), expect_output)
