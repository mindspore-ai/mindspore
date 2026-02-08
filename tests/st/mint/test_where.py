# Copyright 2025 Huawei Technologies Co., Ltd
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


"""test where"""
import numpy as np
import pytest
import mindspore as ms
from mindspore.mint import where
from mindspore import Tensor, jit, JitConfig, context
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return Tensor(np.random.randn(*shape).astype(dtype))


def generate_expect_forward_output(condition, x, y):
    return np.where(condition, x, y)


def generate_expect_backward_output(condition):
    return np.zeros(np.shape(condition), dtype=np.bool_), \
        np.where(condition, 1, 0), np.where(condition, 0, 1)


@test_utils.run_with_cell
def where_forward_func(condition, x, y):
    return where(condition, x, y)


@test_utils.run_with_cell
def where_backward_func(condition, x, y):
    return ms.grad(where_forward_func, (0, 1, 2))(condition, x, y)


@test_utils.run_with_cell
def where_overload_forward_func(condition):
    return where(condition)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_where_overload(mode):
    """
    Feature: ops.where(condition)
    Description: Verify the result of ops.where(condition)
    Expectation: success
    """
    context.set_context(mode=mode, jit_level="O0")
    x = Tensor([[0, 1], [2, 3]], dtype=ms.float32)
    indices1, indices2 = where_overload_forward_func(x)
    expected_indices1 = np.array([0, 1, 1], dtype=np.int64)
    expected_indices2 = np.array([1, 0, 1], dtype=np.int64)
    assert np.allclose(indices1.asnumpy(), expected_indices1)
    assert np.allclose(indices2.asnumpy(), expected_indices2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_where_normal(mode):
    """
    Feature: Test where with static shape in graph and pynative mode.
    Description: call ops.where with valid input and index.
    Expectation: return the correct value.
    """
    x = generate_random_input((2, 3, 4, 5), np.float32)
    y = generate_random_input((2, 3, 4, 5), np.float32)
    cond = x > 0

    if mode == 'pynative':
        ms_out = where_forward_func(cond, x, y)
    else:
        ms_out = (jit(where_forward_func, jit_config=JitConfig(jit_level="O0")))(
            cond, x, y)

    expect = generate_expect_forward_output(
        cond.asnumpy(), x.asnumpy(), y.asnumpy())
    assert np.allclose(ms_out.asnumpy(), expect, rtol=1e-4)

    # auto grad
    x = generate_random_input((2, 3, 4, 5), np.float32)
    y = generate_random_input((2, 3, 4, 5), np.float32)
    cond = x > 0

    if mode == 'pynative':
        ms_cond, ms_x, ms_y = where_backward_func(cond, x, y)
    else:
        ms_cond, ms_x, ms_y = (
            jit(where_backward_func, jit_config=JitConfig(jit_level="O0")))(cond, x, y)
    expect_cond, expect_x, expect_y = generate_expect_backward_output(
        cond.asnumpy())
    assert np.allclose(ms_cond.asnumpy(), expect_cond, rtol=1e-4)
    assert np.allclose(ms_x.asnumpy(), expect_x, rtol=1e-4)
    assert np.allclose(ms_y.asnumpy(), expect_y, rtol=1e-4)
