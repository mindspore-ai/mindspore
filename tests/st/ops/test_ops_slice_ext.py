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
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, jit
from mindspore.ops.auto_generate import SliceExt
import tests.st.utils.test_utils as test_utils
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, dim, start, end, step):
    condition = np.zeros(x.shape[dim])
    if start < 0:
        start += x.shape[dim]
    condition[start:end:step] = 1
    return np.compress(condition, x, axis=dim)


@test_utils.run_with_cell
def slice_ext_forward_func(x, dim, start, end, step):
    return SliceExt()(x, dim, start, end, step)


@test_utils.run_with_cell
def slice_ext_backward_func(x, dim, start, end, step):
    return ops.grad(slice_ext_forward_func, (0))(x, dim, start, end, step) # pylint: disable=not-callable


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential',
        mem_peak=2048.00)
@pytest.mark.parametrize('context_mode', ['pynative', 'KBK'])
def test_ops_slice_ext(context_mode):
    """
    Feature: pyboost function.
    Description: test function slice_ext forward.
    Expectation: expect correct result.
    """
    x = generate_random_input((3, 960, 64, 64), np.float16)
    dim = 2
    start = 0
    end = 64
    step = 1
    expect_forward = generate_expect_forward_output(x, dim, start, end, step)
    expect_grad = np.zeros_like(x)
    expect_grad[:, :, start:end:step, :] = 1

    if context_mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = slice_ext_forward_func(ms.Tensor(x), dim, start, end, step)
        output_grad = slice_ext_backward_func(ms.Tensor(x), dim, start, end, step)
    else:
        output_forward = \
            (jit(slice_ext_forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), dim, start, end, step)
        output_grad = \
            (jit(slice_ext_backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), dim, start, end, step)

    np.testing.assert_allclose(output_forward.asnumpy(), expect_forward, rtol=1e-3)
    np.allclose(output_grad.asnumpy(), expect_grad, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize('context_mode', ['pynative', 'KBK'])
def test_ops_slice_ext_case01(context_mode):
    """
    Feature: pyboost function.
    Description: test function slice_ext forward.
    Expectation: expect correct result.
    """
    x = generate_random_input((3, 1920, 32, 32), np.float16)
    dim = 1
    start = 1280
    end = 1920
    step = 2
    expect_forward = generate_expect_forward_output(x, dim, start, end, step)
    expect_grad = np.zeros_like(x)
    expect_grad[:, start:end:step, :, :] = 1

    if context_mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = slice_ext_forward_func(ms.Tensor(x), dim, start, end, step)
        output_grad = slice_ext_backward_func(ms.Tensor(x), dim, start, end, step)
    else:
        output_forward = \
            (jit(slice_ext_forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), dim, start, end, step)
        output_grad = \
            (jit(slice_ext_backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), dim, start, end, step)

    np.testing.assert_allclose(output_forward.asnumpy(), expect_forward, rtol=1e-3)
    np.allclose(output_grad.asnumpy(), expect_grad, rtol=1e-3)
