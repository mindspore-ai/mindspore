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
import numpy as np
import pytest

import mindspore as ms
from mindspore import ops
import mindspore.nn as nn
from mindspore.common import dtype as mstype


class Net(nn.Cell):
    def construct(self, x, y, dim):
        output = ops.vecdot(x, y, axis=dim)
        return output


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_vecdot_real_number(mode):
    """
    Feature: ops.vecdot
    Description: Verify the result of vecdot
    Expectation: success
    """
    ms.set_context(mode=mode)

    x = ms.Tensor([[1., 3.], [5., 7.], [9., 8.]], dtype=mstype.float32)
    y = ms.Tensor([[4., 5.], [6., 7.], [3., 2.]], dtype=mstype.float32)
    dim = -1

    net = Net()
    output = net(x, y, dim)
    expect_output = [19., 79., 43.]
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_vecdot_complex_number(mode):
    """
    Feature: ops.vecdot
    Description: Verify the result of vecdot
    Expectation: success
    """
    ms.set_context(mode=mode)

    x = ms.Tensor([[1+2j, 3+4j], [5-6j, 7+3j]], dtype=mstype.complex128)
    y = ms.Tensor([[5+6j, 7+8j], [10-4j, 9+2j]], dtype=mstype.complex128)
    dim = -1

    net = Net()
    output = net(x, y, dim)
    expect_output = [70-8j, 143+27j]
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_vecdot_broadcast_case(mode):
    """
    Feature: ops.vecdot
    Description: Verify the result of vecdot
    Expectation: success
    """
    ms.set_context(mode=mode)

    x = ms.Tensor([[[0.7748, 0.2545],
                    [1.5837, -1.3298],
                    [-0.1745, 0.3521]],
                   [[0.9076, -1.0319],
                    [-0.9841, 1.3651],
                    [0.6060, 0.3432]]], dtype=mstype.float32)
    y = ms.Tensor([[[-0.7813, 0.1177]],
                   [[-0.5985, 0.3956]]], dtype=mstype.float32)
    dim = 1

    net = Net()
    output = net(x, y, dim)
    expect_output = [[-1.7064, -0.0851],
                     [-0.3169, 0.2676]]
    expect_output_shape = [2, 2]
    rtol = 1e-04
    atol = 1e-04
    assert np.allclose(output.asnumpy(), expect_output, rtol, atol)
    assert np.allclose(ms.Tensor(output.shape).asnumpy(), expect_output_shape)
