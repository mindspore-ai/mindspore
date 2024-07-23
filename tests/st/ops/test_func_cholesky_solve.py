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
from tests.mark_utils import arg_mark
import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, ops
import mindspore.context as context
from mindspore.common import dtype as mstype


class CholeskySolveOpsNet(nn.Cell):
    def construct(self, input1, input2, upper):
        return ops.cholesky_solve(input1, input2, upper)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_cholesky_solve(mode):
    """
    Feature: Test cholesky_solve ops api.
    Description: Test cholesky_solve ops api for Graph and PyNative modes.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    u = Tensor([[9.99999940e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                [1.19209304e-07, 9.99999940e-01, 0.00000000e+00, 0.00000000e+00],
                [0.00000000e+00, -2.98023259e-08, 9.99999940e-01, 0.00000000e+00],
                [1.19209304e-07, 2.98023100e-08, 5.96046519e-08, 1.00000000e+00]], mstype.float32)
    b = Tensor([[-7.49088705e-01, -5.73587477e-01],
                [-1.29632413e+00, 2.38306046e+00],
                [4.38669994e-02, -7.10199354e-03],
                [-5.40791094e-01, -2.93331075e+00]], mstype.float32)
    net = CholeskySolveOpsNet()
    output = net(b, u, upper=False)
    expect_output = [[-7.49088585e-01, -5.73587537e-01],
                     [-1.29632425e+00, 2.38306093e+00],
                     [4.38670032e-02, -7.10174954e-03],
                     [-5.40790975e-01, -2.93331075e+00]]
    assert np.allclose(output.asnumpy(), expect_output)
