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
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.ops import sqrt
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.random(shape).astype(dtype) + 1


def generate_expect_forward_output(x):
    return np.sqrt(x)


def generate_expect_backward_output(x):
    return 0.5 / np.sqrt(x)


@test_utils.run_with_cell
def sqrt_forward_func(x):
    return sqrt(x)


@test_utils.run_with_cell
def sqrt_backward_func(x):
    return ops.grad(sqrt_forward_func, (0))(x)


@test_utils.run_with_cell
def sqrt_vmap_func(x):
    return ops.vmap(sqrt_forward_func, in_axes=0, out_axes=0)(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_sqrt_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function sqrt forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((64, 32, 1), np.float32)
    output = sqrt_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((2, 3, 4, 5), np.float32)
    output2 = sqrt_backward_func(ms.Tensor(x2))
    expect2 = generate_expect_backward_output(x2)
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=1e-3)



@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_sqrt_forward_case01(context_mode):
    """
    Feature: pyboost function.
    Description: test function sqrt forward add cases.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((1), np.float32)
    output = sqrt_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_sqrt_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function sqrt vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = sqrt_vmap_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_ops_sqrt_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function sqrt with dynamic shape and dynamic rank.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    x2 = generate_random_input((3, 4, 5, 6, 7), np.float32)

    TEST_OP(sqrt_forward_func,
            [[ms.Tensor(x1)], [ms.Tensor(x2)]], 'sqrt')
