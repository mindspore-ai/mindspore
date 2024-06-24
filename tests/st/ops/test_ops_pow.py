# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from mindspore import ops
import mindspore as ms

def generate_random_input(shape, dtype):
    return np.random.uniform(0.9, 1.0, size=shape).astype(dtype), np.random.uniform(0.9, 1.0, size=shape).astype(dtype)

@test_utils.run_with_cell
def pow_forward_func(x, y):
    return ops.pow(x, y)


@test_utils.run_with_cell
def pow_backward_func(x, y):
    return ops.grad(pow_forward_func, (0, 1))(x, y)


@test_utils.run_with_cell
def pow_vmap_func(x, y):
    return ops.vmap(pow_forward_func, in_axes=0, out_axes=0)(x, y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_with_On
def test_pow_op_forward(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op pow forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_np, y_np = generate_random_input((2, 3, 4, 5), dtype=data_type)

    x = ms.Tensor(x_np)
    y = ms.Tensor(y_np)
    out = pow_forward_func(x, y)
    expect_out = np.power(x_np, y_np)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_with_On
def test_pow_op_backward(context_mode, data_type):
    """
    Feature: Auto grad.
    Description: test auto grad of op pow.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1, 2, 4]).astype(data_type))
    y = ms.Tensor(np.array([2, 4, 3]).astype(data_type))
    grads = pow_backward_func(x, y)
    expect_out = np.array([[2., 32., 48.], [0., 11.0903549, 88.7228394]]).astype(np.float32)
    np.testing.assert_allclose(grads[0].asnumpy(), expect_out[0], rtol=1e-3)
    np.testing.assert_allclose(grads[1].asnumpy(), expect_out[1], rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_with_On
def test_pow_op_vmap(context_mode, data_type):
    """
    Feature: test vmap function.
    Description: test pow op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_np, y_np = generate_random_input((2, 3, 4, 5), dtype=data_type)

    x = ms.Tensor(x_np)
    y = ms.Tensor(y_np)
    out = pow_vmap_func(x, y)
    expect_out = np.power(x_np, y_np)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pow_dynamic_shape():
    """
    Feature: Test pow with dynamic shape in graph mode.
    Description: call ops.pow with valid input and index.
    Expectation: return the correct value.
    """
    x1, other1 = generate_random_input((2, 3, 4), np.float32)
    x2, other2 = generate_random_input((3, 4, 5, 6), np.float32)

    TEST_OP(pow_forward_func,
            [[ms.Tensor(x1), ms.Tensor(other1)], [ms.Tensor(x2), ms.Tensor(other2)]], 'pow')
