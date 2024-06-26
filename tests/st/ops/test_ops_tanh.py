# Copyright 2023 Huawei Technotanhies Co., Ltd
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
from mindspore import context, Tensor
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(x, dtype):
    return np.tanh(x).astype(dtype)

def generate_expect_backward_output(x, dtype):
    output = 1 - np.power(np.tanh(x), 2)
    return  output.astype(dtype)

@test_utils.run_with_cell
def tanh_forward_func(x):
    return ms.ops.tanh(x)

@test_utils.run_with_cell
def tanh_backward_func(x):
    return ms.ops.grad(tanh_forward_func, (0))(x)

@test_utils.run_with_cell
def tanh_vmap_func(x):
    return ms.ops.vmap(tanh_forward_func, in_axes=0, out_axes=0)(x)




@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_forward(mode):
    """
    Feature: test tanh operator
    Description: test tanh run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)
    x_np = generate_random_input((2, 3, 4), np.float32)
    x_tensor = Tensor(x_np, ms.float32)
    output = tanh_forward_func(x_tensor)
    expect = generate_expect_forward_output(x_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_backward(mode):
    """
    Feature: test tanh operator
    Description: test tanh run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)
    x_np = generate_random_input((2, 3, 4), np.float32)
    x_tensor = Tensor(x_np, ms.float32)
    output = tanh_backward_func(x_tensor)
    expect = generate_expect_backward_output(x_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_vmap(mode):
    """
    Feature: pyboost function.
    Description: test function tanh vmap feature.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    x_np = generate_random_input((2, 3, 4, 5), np.float32)
    x_tensor = Tensor(x_np, ms.float32)
    output = tanh_vmap_func(x_tensor)
    expect = generate_expect_forward_output(x_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tanh_dynamic_shape_testop():
    """
    Feature: Test tanh with dynamic shape in graph mode using TEST_OP.
    Description: call ops.tanh with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((3, 4, 5), np.float32)
    x2 = generate_random_input((3, 7, 8, 3), np.float32)

    TEST_OP(ms.ops.tanh, [[ms.Tensor(x1)], [ms.Tensor(x2)]], 'tanh')
