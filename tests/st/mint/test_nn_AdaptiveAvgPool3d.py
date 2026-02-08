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
from mindspore import ops, jit, JitConfig
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_std_input(output_size):
    x = np.full(shape=(2, 2, 4, 3), fill_value=[0.3, 0.4, 0.5], dtype=np.float32)
    if output_size == 1:
        output = np.full(shape=(2, 1, 1, 1), fill_value=0.4000, dtype=np.float32)
        output_grad = np.full(shape=(2, 2, 4, 3), fill_value=[0.0417, 0.0417, 0.0417], dtype=np.float32)
    else:
        output = np.full(shape=(2, 2, 2, 2), fill_value=[0.3500, 0.4500], dtype=np.float32)
        output_grad = np.full(shape=(2, 2, 4, 3), fill_value=[0.2500, 0.5000, 0.2500], dtype=np.float32)
    return x, output, output_grad


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def adaptive_avg_pool3d_forward_func(x, output_size):
    return ms.mint.nn.functional.adaptive_avg_pool3d(x, output_size)


def adaptive_avg_pool3d_forward_dyn_mean(x):
    return ms.mint.nn.functional.adaptive_avg_pool3d(x, output_size=(1, 1, 1))


def adaptive_avg_pool3d_forward_dyn_adaptive(x):
    return ms.mint.nn.functional.adaptive_avg_pool3d(x, output_size=(2, 2, 2))


def adaptive_avg_pool3d_backward_func(x, output_size):
    return ops.grad(adaptive_avg_pool3d_forward_func, (0,))(x, output_size)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=36.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_adaptive_avg_pool3d_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function adaptive_avg_pool3d.
    Expectation: expect correct result.
    """
    output_sizes = [1, (2, 2, 2)]
    for output_size in output_sizes:
        x, expect, expect_grad = generate_std_input(output_size)
        if mode == 'pynative':
            ms.context.set_context(mode=ms.PYNATIVE_MODE)
            output = adaptive_avg_pool3d_forward_func(ms.Tensor(x, dtype=ms.float32), output_size)
            output_grad = adaptive_avg_pool3d_backward_func(ms.Tensor(x, dtype=ms.float32), output_size)
        else:
            output = (jit(adaptive_avg_pool3d_forward_func, jit_config=JitConfig(jit_level='O0')))(
                ms.Tensor(x, dtype=ms.float32), output_size)
            output_grad = (jit(adaptive_avg_pool3d_backward_func, jit_config=JitConfig(jit_level='O0')))(
                ms.Tensor(x, dtype=ms.float32), output_size)
        np.allclose(output.asnumpy(), expect, rtol=1e-5, equal_nan=True)
        np.allclose(output_grad.asnumpy(), expect_grad, rtol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=36.0)
def test_adaptive_avg_pool3d_dynamic():
    """
    Feature: dynamic shape forward, backward features.
    Description: test adaptive_avg_pool3d with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = generate_random_input((3, 4, 5, 6), np.float32)
    x2 = generate_random_input((3, 7, 8, 3, 5), np.float32)
    TEST_OP(adaptive_avg_pool3d_forward_dyn_mean, [[ms.Tensor(x1)], [ms.Tensor(x2)]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'])
    TEST_OP(adaptive_avg_pool3d_forward_dyn_adaptive, [[ms.Tensor(x1)], [ms.Tensor(x2)]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'])
    TEST_OP(adaptive_avg_pool3d_forward_func, [[ms.Tensor(x1), (1, 1, 1)], [ms.Tensor(x2), (2, 2, 2)]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'],
            case_config={'disable_input_check': True})
    TEST_OP(adaptive_avg_pool3d_forward_func, [[ms.Tensor(x1), 1], [ms.Tensor(x2), 2]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'],
            case_config={'disable_input_check': True})


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=36.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_adaptive_avg_pool3d_bfloat16(mode):
    """
    Feature: standard forward, backward features for bfloat16.
    Description: test function adaptive_avg_pool3d.
    Expectation: expect correct result.
    """
    output_sizes = [1, (2, 2, 2)]
    for output_size in output_sizes:
        x, expect, expect_grad = generate_std_input(output_size)
        if mode == 'pynative':
            ms.context.set_context(mode=ms.PYNATIVE_MODE)
            output = adaptive_avg_pool3d_forward_func(ms.Tensor(x, dtype=ms.bfloat16), output_size)
            output_grad = adaptive_avg_pool3d_backward_func(ms.Tensor(x, dtype=ms.bfloat16), output_size)
        else:
            output = (jit(adaptive_avg_pool3d_forward_func, jit_config=JitConfig(jit_level='O0')))(
                ms.Tensor(x, dtype=ms.bfloat16), output_size)
            output_grad = (jit(adaptive_avg_pool3d_backward_func, jit_config=JitConfig(jit_level='O0')))(
                ms.Tensor(x, dtype=ms.bfloat16), output_size)
        np.allclose(output.asnumpy(), expect, rtol=4e-3, equal_nan=True)
        np.allclose(output_grad.asnumpy(), expect_grad, rtol=4e-3, equal_nan=True)
