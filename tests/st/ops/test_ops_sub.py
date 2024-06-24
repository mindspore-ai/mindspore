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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore as ms
from mindspore import ops, JitConfig
from mindspore.nn import Cell
from mindspore.ops.extend import sub
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP

rtol = 1e-3


class SubCell(Cell):
    def __init__(self):
        super().__init__()
        self.sub = sub

    def construct(self, x, y, alpha):
        return self.sub(x, y, alpha)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_forward(context_mode):
    """
    Feature: test sub forward
    Description: test sub forward
    Expectation: success
    """
    ms.context.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)

    sub_cell = SubCell()

    x = np.random.randn(64, 32, 57344).astype(np.float32)
    y = np.random.randn(64, 32, 1).astype(np.float32)
    alpha = 2.0

    output = sub_cell(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = x - y * alpha

    np.testing.assert_allclose(output, expect, rtol=rtol)

    sub_cell.set_inputs(ms.tensor(shape=[None, None], dtype=ms.float32),
                        ms.tensor(shape=[None, None], dtype=ms.float32), alpha)

    x = np.random.randn(64, 1536).astype(np.float32)
    y = 1.
    alpha = 2.0

    output = sub_cell(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = x - y * alpha

    np.testing.assert_allclose(output, expect, rtol=rtol)

    sub_cell.set_inputs(ms.tensor(shape=[None, None, None, None], dtype=ms.float32),
                        ms.tensor(shape=[None, None, None, None], dtype=ms.float32), alpha)

    x = np.random.randn(3, 4, 64, 64).astype(np.float32)
    y = np.random.randn(3, 4, 64, 64).astype(np.float32)

    output = sub_cell(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = x - y * alpha

    np.testing.assert_allclose(output, expect, rtol=rtol)

    sub_cell.set_inputs(ms.tensor(shape=None, dtype=ms.float32),
                        ms.tensor(shape=None, dtype=ms.float32), alpha)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ops_dynamic():
    """
    Feature: ops.extend.sub
    Description: dynamic shape and rank
    Expectation: success
    """
    x1 = ms.Tensor(np.array([[1, 2], [3, 4]], np.float32))
    y1 = ms.Tensor(np.array([[5, 6], [7, 8]], np.float32))
    x2 = ms.Tensor(np.array([[1, 2, 3]], np.float32))
    y2 = ms.Tensor(np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], np.float32))

    TEST_OP(sub, [[x1, y1, 1.], [x2, y2, 2.]], 'sub_ext', disable_input_check=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_backward(context_mode):
    """
    Feature: test sub backward
    Description: test sub backward
    Expectation: success
    """
    ms.context.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)

    sub_cell = SubCell()

    # 2 x 2
    x = np.array([[1, 2], [3, 4]], np.float32)
    y = np.array([[1, 2], [3, 4]], np.float32)
    alpha = 2.0

    output = ops.grad(sub_cell, (0))(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = np.ones_like(y)

    np.testing.assert_allclose(output, expect, rtol=rtol)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_bf16(context_mode):
    """
    Feature: ops.extend.sub
    Description: bf16
    Expectation: success
    """
    ms.context.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)

    sub_cell = SubCell()

    x_np_32 = np.random.randn(1).astype(np.float32)
    y_np_32 = np.random.randn(1, 4096, 4096).astype(np.float32)
    x = ms.tensor(x_np_32, ms.bfloat16)
    y = ms.tensor(y_np_32, ms.bfloat16)
    x_np_bf16 = x.float().asnumpy()
    y_np_bf16 = y.float().asnumpy()
    alpha = 2.0

    output = sub_cell(x, y, alpha).float().asnumpy()
    expect = x_np_bf16 - y_np_bf16 * alpha

    np.testing.assert_allclose(output, expect, rtol=4e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_bool(context_mode):
    """
    Feature: test sub backward
    Description: test sub backward
    Expectation: success
    """
    ms.context.set_context(mode=context_mode)

    sub_cell = SubCell()
    sub_cell.set_jit_config(JitConfig(jit_level='O0'))

    # 2 x 2
    x = np.array([[True, True], [False, False]], np.bool_)
    y = np.array([[True, False], [True, False]], np.bool_)
    alpha = True

    output = sub_cell(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = x ^ (y & alpha)

    np.testing.assert_allclose(output, expect, rtol=rtol)
