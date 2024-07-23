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
import mindspore.common.dtype as mstype
from mindspore import ops, Tensor, jit, JitConfig
from mindspore.ops import square
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return np.square(x)


def generate_expect_backward_output(x):
    return 2 * x


@test_utils.run_with_cell
def square_forward_func(x):
    return square(x)


@test_utils.run_with_cell
def square_backward_func(x):
    return ops.grad(square_forward_func, (0))(x)


@test_utils.run_with_cell
def square_vmap_func(x, in_axes=0):
    return ops.vmap(square_forward_func, in_axes, out_axes=0)(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
def test_square_normal(mode):
    """
    Feature: Test square with static shape in graph and pynative mode.
    Description: call ops.square with valid input and index.
    Expectation: return the correct value.
    """
    x = generate_random_input((8192,), np.float32)

    if mode == 'pynative':
        output = square_forward_func(Tensor(x))
        output1 = square_backward_func(Tensor(x))
    elif mode == 'KBK':
        output = (jit(square_forward_func, jit_config=JitConfig(jit_level="O0")))(Tensor(x))
        output1 = (jit(square_backward_func, jit_config=JitConfig(jit_level="O0")))(Tensor(x))
    else:
        output = (jit(square_forward_func, jit_config=JitConfig(jit_level="O2")))(Tensor(x))
        output1 = (jit(square_backward_func, jit_config=JitConfig(jit_level="O2")))(Tensor(x))

    expect = generate_expect_forward_output(x)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4)
    expect1 = generate_expect_backward_output(x)
    assert np.allclose(output1.asnumpy(), expect1, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_square_bfloat16(context_mode):
    """
    Feature: pyboost function.
    Description: test function square forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((64, 32, 57344), np.float32)
    output = square_forward_func(Tensor(x, mstype.bfloat16))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.float().asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_square_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function square vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((7168, 8192), np.float32)
    output = square_vmap_func(Tensor(x), 0)
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_square_dynamic_shape_testop():
    """
    Feature: Test square with dynamic shape in graph mode using TEST_OP.
    Description: call ops.square with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((64, 32, 3584), np.float32)
    x2 = generate_random_input((3, 512, 64, 64), np.float32)

    TEST_OP(square_forward_func, [[Tensor(x1)], [Tensor(x2)]], 'square')
