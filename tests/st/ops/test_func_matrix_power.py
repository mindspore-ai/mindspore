
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
import mindspore as ms
from mindspore import ops
import mindspore.nn as nn


class NetMatrixPower(nn.Cell):
    def construct(self, x, n):
        return ops.matrix_power(x, n)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.skip(reason="I6BZ6M")
def test_matrix_power(mode):
    """
    Feature: matrix_power
    Description: Verify the result of matrix_power
    Expectation: success.
    """
    ms.set_context(mode=mode)
    arrs = [
        np.random.rand(1, 2, 2).astype('float32'),
        np.random.rand(2, 3, 3).astype('float32'),
        np.random.rand(3, 4, 4).astype('float32'),
    ]
    net_matrix_power = NetMatrixPower()

    for arr in arrs:
        for n in range(0, 4):
            expect_out = np.linalg.matrix_power(arr, n)
            out = net_matrix_power(ms.Tensor(arr), n)
            assert np.allclose(out.asnumpy(), expect_out, rtol=1e-4, atol=1e-4)
