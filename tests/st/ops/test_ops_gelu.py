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
"""test ops gelu"""
import numpy as np
import pytest

import mindspore as ms
from mindspore.nn import Cell
from mindspore.ops import gelu
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase

rtol = 1e-3


class GeluCell(Cell):
    def __init__(self):
        super().__init__()
        self.gelu = gelu

    def construct(self, x, approximate):
        return self.gelu(x, approximate)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.0)
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

    output = ms.grad(gelu_cell, (0))(ms.tensor(x), approximate).asnumpy()
    if approximate:
        expect = np.array([[0.5795, 0.6575],
                           [0.7323, 0.8027]])
    else:
        expect = np.array([[0.5795, 0.6575],
                           [0.7323, 0.8027]])

    np.testing.assert_allclose(output, expect, rtol=rtol)


def ops_gelu_binary_compare(input_binary_data, output_binary_data, approximate='none', is_bf16=False):

    if is_bf16:
        inputx = ms.Tensor(input_binary_data[0], ms.bfloat16)
        loss = 4e-3
    else:
        inputx = ms.Tensor(input_binary_data[0])
        loss = 1e-4
    output = GeluCell()(inputx, approximate)
    if is_bf16:
        assert np.allclose(output.float().asnumpy(), output_binary_data[0], loss, loss)
    else:
        assert np.allclose(output.asnumpy(), output_binary_data[0], loss, loss)

    grads = ms.grad(GeluCell(), (0))(inputx, approximate)
    if is_bf16:
        assert np.allclose(grads.float().asnumpy(), output_binary_data[1], loss, loss)
    else:
        assert np.allclose(grads.asnumpy(), output_binary_data[1], loss, loss)


@ops_binary_cases(OpsBinaryCase(input_info=[((1, 1, 1544, 20480), np.float32)],
                                output_info=[((1, 1, 1544, 20480), np.float32), ((1, 1, 1544, 20480), np.float32)],
                                extra_info='pgmoe'))
def ops_gelu_binary_case1(input_binary_data=None, output_binary_data=None):
    ops_gelu_binary_compare(input_binary_data, output_binary_data, 'none', True)


@pytest.mark.parametrize("mode", ['pynative', 'kbk', 'ge'])
def test_ops_gelu_binary_cases(mode):
    """
    Feature: ops test
    Description: test gelu with binary data
    Expectation: success
    """
    if mode == "kbk":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    elif mode == 'ge':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O2')
    else:
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    # ops_gelu_binary_case1()
