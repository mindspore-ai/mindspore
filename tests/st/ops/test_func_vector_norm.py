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
    def construct(self, x, norm_ord=2, axis=None, keepdims=True):
        output = ms.ops.vector_norm(x, ord=norm_ord, axis=axis, keepdims=keepdims)
        return output


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_vector_norm_default_axis_case(mode):
    """
    Feature: ops.vector_norm
    Description: Verify the result of vector_norm
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.ops.arange(0, 12, dtype=ms.float32) - 6
    x = x.reshape(3, 4)

    net = Net()

    output = net(x, norm_ord=2)
    expect_output = 12.0830
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, norm_ord=float('inf'))
    expect_output = 6.
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, norm_ord=float('-inf'))
    expect_output = 0.
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, norm_ord=0)
    expect_output = 11.
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, norm_ord=4.5)
    expect_output = 7.2244
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_vector_norm_int_axis_case(mode):
    """
    Feature: ops.vector_norm
    Description: Verify the result of vector_norm
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.ops.arange(0, 12, dtype=ms.float32) - 6
    x = x.reshape(3, 4)

    net = Net()

    output = net(x, norm_ord=2, axis=0, keepdims=True)
    expect_output = [[6.6332, 5.9161, 5.6569, 5.9161]]
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, norm_ord=float('inf'), axis=0, keepdims=True)
    expect_output = [[6., 5., 4., 5.]]
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, norm_ord=float('-inf'), axis=0, keepdims=True)
    expect_output = [[2., 1., 0., 1.]]
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, norm_ord=0, axis=0, keepdims=True)
    expect_output = [[3., 3., 2., 3.]]
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, norm_ord=4.5, axis=0, keepdims=True)
    expect_output = [[6.0189, 5.1082, 4.6661, 5.1082]]
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_vector_norm_tuple_axis_case(mode):
    """
    Feature: ops.vector_norm
    Description: Verify the result of vector_norm
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.ops.arange(0, 12, dtype=ms.float32) - 6
    x = x.reshape(3, 4)

    net = Net()

    output = net(x, norm_ord=2, axis=(0, 1), keepdims=True)
    expect_output = [[12.0830]]
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, norm_ord=0, axis=(0, 1), keepdims=True)
    expect_output = [[11.]]
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, norm_ord=4.5, axis=(0, 1), keepdims=True)
    expect_output = [[7.2244]]
    assert np.allclose(output.asnumpy(), expect_output)

    x = ms.ops.arange(0, 24, dtype=ms.float32) - 6
    x = x.reshape(2, 2, 2, 3)

    output = net(x, norm_ord=float('inf'), axis=(0, 1, 2), keepdims=True)
    expect_output = [[[[15., 16., 17.]]]]
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, norm_ord=float('inf'), axis=(0, 1, 2), keepdims=False)
    expect_output = [15., 16., 17.]
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, norm_ord=float('-inf'), axis=(0, 1, 2), keepdims=True)
    expect_output = [[[[0., 1., 1.]]]]
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, norm_ord=float('-inf'), axis=(0, 1, 2), keepdims=False)
    expect_output = [0., 1., 1.]
    assert np.allclose(output.asnumpy(), expect_output)
