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
import mindspore.nn as nn


class Net(nn.Cell):

    def __init__(self, norm_ord):
        super().__init__()
        self.norm_ord = norm_ord

    def construct(self, x, axis=(-2, -1), keepdims=True):
        output = ms.ops.matrix_norm(x, ord=self.norm_ord, axis=axis, keepdims=keepdims)
        return output


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='allcards', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_matrix_norm(mode):
    """
    Feature: ops.matrix_norm
    Description: Verify the result of matrix_norm
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.ops.arange(0, 12, dtype=ms.float32) - 6
    x = x.reshape(2, 2, 3)

    net = Net('fro')
    output = net(x)
    expect_output = [[[9.53939]],
                     [[7.41620]]]
    assert np.allclose(output.asnumpy(), expect_output)

    net = Net('nuc')
    output = net(x)
    expect_output = [[[10.28090]],
                     [[8.34847]]]
    assert np.allclose(output.asnumpy(), expect_output)

    net = Net(float('inf'))
    output = net(x)
    expect_output = [[[15.]],
                     [[12.]]]
    assert np.allclose(output.asnumpy(), expect_output)

    net = Net(float('-inf'))
    output = net(x)
    expect_output = [[[6.]],
                     [[3.]]]
    assert np.allclose(output.asnumpy(), expect_output)

    net = Net(1)
    output = net(x)
    expect_output = [[[9.]],
                     [[7.]]]
    assert np.allclose(output.asnumpy(), expect_output)

    net = Net(-1)
    output = net(x)
    expect_output = [[[5.]],
                     [[3.]]]
    assert np.allclose(output.asnumpy(), expect_output)

    net = Net(2)
    output = net(x)
    expect_output = [[[9.508033]],
                     [[7.348468]]]
    assert np.allclose(output.asnumpy(), expect_output)

    net = Net(-2)
    output = net(x)
    expect_output = [[[0.77287]],
                     [[1.00000]]]
    assert np.allclose(output.asnumpy(), expect_output)
