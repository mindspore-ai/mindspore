# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore.nn import Cell
from mindspore.ops import gelu

rtol = 1e-3


class GeluCell(Cell):
    def __init__(self):
        super().__init__()
        self.gelu = gelu

    def construct(self, x, approximate):
        return self.gelu(x, approximate)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('approximate', ['tanh', 'none'])
def test_ops_forward(context_mode, approximate):
    """
    Feature: test gelu forward
    Description: test gelu forward
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)

    gelu_cell = GeluCell()

    # 2 x 2
    x = np.array([[0.1, 0.2], [0.3, 0.4]], np.float32)

    output = gelu_cell(ms.tensor(x), approximate).asnumpy()
    if approximate:
        expect = np.array([[0.0540, 0.1159],
                           [0.1854, 0.2622]])
    else:
        expect = np.array([[0.0540, 0.1159],
                           [0.1854, 0.2622]])

    np.testing.assert_allclose(output, expect, rtol=rtol)

    gelu_cell.set_inputs(ms.tensor(shape=[None, None], dtype=ms.float32), approximate)

    # 3 x 3
    x = np.array([[0.1, 0.2], [0.3, 0.4]], np.float32)

    output = gelu_cell(ms.tensor(x), approximate).asnumpy()
    if approximate:
        expect = np.array([[0.0540, 0.1159],
                           [0.1854, 0.2622]])
    else:
        expect = np.array([[0.0540, 0.1159],
                           [0.1854, 0.2622]])

    np.testing.assert_allclose(output, expect, rtol=rtol)

    x = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], np.float32)

    output = gelu_cell(ms.tensor(x), approximate).asnumpy()
    if approximate:
        expect = np.array([[0.0540, 0.1159, 0.1854],
                           [0.2622, 0.3457, 0.4354],
                           [0.5306, 0.6304, 0.7342]])
    else:
        expect = np.array([[0.0540, 0.1159, 0.1854],
                           [0.2622, 0.3457, 0.4354],
                           [0.5306, 0.6305, 0.7343]])

    np.testing.assert_allclose(output, expect, rtol=rtol)

    gelu_cell.set_inputs(ms.tensor(shape=None, dtype=ms.float32), approximate)

    # 2 x 2 x 2
    x = np.array([[0.1, 0.2], [0.3, 0.4]], np.float32)

    output = gelu_cell(ms.tensor(x), approximate).asnumpy()
    if approximate:
        expect = np.array([[0.0540, 0.1159],
                           [0.1854, 0.2622]])
    else:
        expect = np.array([[0.0540, 0.1159],
                           [0.1854, 0.2622]])

    np.testing.assert_allclose(output, expect, rtol=rtol)

    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], np.float32)

    output = gelu_cell(ms.tensor(x), approximate).asnumpy()
    if approximate:
        expect = np.array([[[0.0540, 0.1159],
                            [0.1854, 0.2622]],
                           [[0.3457, 0.4354],
                            [0.5306, 0.6304]]])
    else:
        expect = np.array([[[0.0540, 0.1159],
                            [0.1854, 0.2622]],
                           [[0.3457, 0.4354],
                            [0.5306, 0.6305]]])

    np.testing.assert_allclose(output, expect, rtol=rtol)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('approximate', ['tanh', 'none'])
def test_ops_backward(context_mode, approximate):
    """
    Feature: test gelu backward
    Description: test gelu backward
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)

    gelu_cell = GeluCell()

    # 2 x 2
    x = np.array([[0.1, 0.2], [0.3, 0.4]], np.float32)

    output = ops.grad(gelu_cell, (0))(ms.tensor(x), approximate).asnumpy()
    if approximate:
        expect = np.array([[0.5795, 0.6575],
                           [0.7323, 0.8027]])
    else:
        expect = np.array([[0.5795, 0.6575],
                           [0.7323, 0.8027]])

    np.testing.assert_allclose(output, expect, rtol=rtol)
