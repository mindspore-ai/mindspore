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
        return ops.view_as_real(x)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_view_as_real(mode):
    """
    Feature: ops.view_as_real
    Description: Verify the result of view_as_real
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([2 + 1j, 2 + 3j, 2 - 1j, 2], ms.complex64)
    net = Net()
    output = net(x)
    expect_output = [[2., 1.],
                     [2., 3.],
                     [2., -1.],
                     [2., 0.]]
    assert np.allclose(output.asnumpy(), expect_output)
