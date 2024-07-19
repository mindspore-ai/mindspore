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
from mindspore import ops
from mindspore.mint.nn.functional import hardswish
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def hardswish_expect_forward_func(x):
    return np.where(x <= -3, 0, np.where(x >= 3, x, x * (x + 3) / 6))


@test_utils.run_with_cell
def hardswish_forward_func(x):
    return hardswish(x)


@test_utils.run_with_cell
def hardswish_backward_func(x):
    return ops.grad(hardswish_forward_func, (0))(x)


@test_utils.run_with_cell
def hardswish_vmap_func(x):
    return ops.vmap(hardswish_forward_func)(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows',
                      'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_hardswish_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function hardswish forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    # forward
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output_f = hardswish_forward_func(ms.Tensor(x))
    expect_f = hardswish_expect_forward_func(x)
    np.testing.assert_allclose(output_f.asnumpy(), expect_f, rtol=1e-3)

    # backward
    x2 = np.array([-1, -2, 0, 2, 1]).astype('float32')
    output_b = hardswish_backward_func(ms.Tensor(x2))
    expect_b = np.array([0.16666667, -0.16666667, 0.5, 1.1666666, 0.8333333])
    np.testing.assert_allclose(output_b.asnumpy(), expect_b, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_hardswish_bf16(context_mode):
    """
    Feature: pyboost function.
    Description: test function hardswish forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    input_bf16 = ms.Tensor([-4.2, 2.4, -0.3, -1.2, 5.1], dtype=ms.bfloat16)
    output_f = hardswish_forward_func(input_bf16)
    expect_f = np.array([-0.0000, 2.1719, -0.1357, -0.3594, 5.0938])
    np.testing.assert_allclose(output_f.float().asnumpy(), expect_f, rtol=4e-3, atol=4e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows',
                      'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_hardswish_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function hardswish vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = hardswish_vmap_func(ms.Tensor(x))
    expect = hardswish_expect_forward_func(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows',
                      'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hardswish_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function hardswish  dynamic feature.
    Expectation: expect correct result.
    """
    ms_data1 = generate_random_input((2, 3, 4, 5), np.float32)
    ms_data2 = generate_random_input((3, 4, 5, 6, 7), np.float32)
    TEST_OP(hardswish_forward_func
            , [[ms.Tensor(ms_data1)], [ms.Tensor(ms_data2)]]
            , 'hswish')
