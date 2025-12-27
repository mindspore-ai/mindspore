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

"""Mock tests for op plugin kernels."""

import pytest
import numpy as np

from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.op_plugin.mock_build import build_and_setup_mock_plugin # pylint: disable=unused-import

import mindspore as ms
from mindspore import Tensor
from mindspore import mint
from mindspore.ops.auto_generate.gen_ops_prim import expand_dims_view_op


ms.set_context(device_target="CPU")

def set_mode(mode):
    if mode == "kbk":
        ms.context.set_context(mode=ms.GRAPH_MODE,
                               jit_config={"jit_level": "O0"})
    else:
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

@test_utils.run_with_cell
def logical_and_forward_func(x, y):
    return mint.logical_and(x, y)

@test_utils.run_with_cell
def cumsum_ext_forward_func(x, dim):
    return mint.cumsum(x, dim)

@test_utils.run_with_cell
def inplace_relu_forward_func(x):
    mint.nn.functional.relu_(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['kbk', 'pynative'])
def test_normal_op(mode):
    """
    Feature: op_plugin kernel
    Description: Test op_plugin kernel
    Expectation: Correct result.
    """
    set_mode(mode)
    x = Tensor([1, 2, 3, 4], ms.int64)
    dim = 0
    expect = np.cumsum(x.asnumpy(), dim)
    output = cumsum_ext_forward_func(x, dim)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['kbk', 'pynative'])
def test_op_with_existing_cpu_kernelmod(mode):
    """
    Feature: op_plugin kernel
    Description: Test op_plugin kernel when normal cpu kernelmod exists
    Expectation: Correct result.
    """
    set_mode(mode)
    x = Tensor([True, False, True], ms.bool_)
    y = Tensor([True, True, False], ms.bool_)
    # there is a normal logical_and cpu kernelmod,
    # so mock logical_and op is implemented as logical_or
    # to ensure op plugin is used
    expect = np.logical_or(x.asnumpy(), y.asnumpy())
    output = logical_and_forward_func(x, y)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['kbk', 'pynative'])
def test_inplace_op(mode):
    """
    Feature: op_plugin kernel
    Description: Test op_plugin kernel for inplace op
    Expectation: Correct result.
    """
    set_mode(mode)
    x = Tensor([-1.0, 4.0, -8.0, 2.0, -5.0, 9.0], ms.float32)
    expect = np.maximum(x.asnumpy(), 0.0)
    inplace_relu_forward_func(x)
    assert np.allclose(x.asnumpy(), expect)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_view_op(mode):
    """
    Feature: op_plugin kernel
    Description: Test op_plugin kernel for view feature. Disabled for now
    Expectation: Correct result.
    """

    @test_utils.run_with_cell
    def view_func(x):
        out = expand_dims_view_op(x, 1)
        mint.nn.functional.relu_(out)
        return out
    set_mode(mode)
    x = Tensor([-1.0, 4.0, -8.0, 2.0, -5.0, 9.0], ms.float32)
    expected_x_after_inplace_relu = np.maximum(x.asnumpy(), 0.0)
    expect_view = expected_x_after_inplace_relu.reshape(6, 1)
    view = view_func(x)
    assert np.allclose(x.asnumpy(), expected_x_after_inplace_relu)
    assert np.allclose(view.asnumpy(), expect_view)
    assert expect_view.shape == view.shape

    @test_utils.run_with_cell
    def view_grad_func(x):
        x_copy = x * 1
        out = expand_dims_view_op(x_copy, 1)
        mint.nn.functional.relu_(out)
        return out
    grad_fn = ms.ops.grad(view_grad_func, (0,))
    output_grad = grad_fn(x).asnumpy()
    assert np.allclose(output_grad, [0., 1., 0., 1., 0., 1.])

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_noncontiguous_input_op(mode):
    """
    Feature: op_plugin kernel
    Description: Test op_plugin kernel for noncontiguous input op
    Expectation: Correct result.
    """
    set_mode(mode)
    orig_x = np.random.randint(0, 2, size=(4, 4)) == 1
    orig_y = np.random.randint(0, 2, size=(4, 4)) == 1
    x_np = orig_x[1:, ::2]
    y_np = orig_y[1:, ::2]
    x_noncontiguous = Tensor(orig_x, ms.bool_)[1:, ::2]
    y_noncontiguous = Tensor(orig_y, ms.bool_)[1:, ::2]
    expect = np.logical_or(x_np, y_np)
    output = logical_and_forward_func(x_noncontiguous, y_noncontiguous)
    assert np.allclose(output.asnumpy(), expect)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_scalar_tuple_input(mode):
    """
    Feature: op_plugin kernel
    Description: Test op_plugin kernel for scalar tuple input
    Expectation: Correct result.
    """
    @test_utils.run_with_cell
    def randn_forward_func(shape):
        return mint.randn(shape, dtype=ms.float32)
    set_mode(mode)
    shape = (2, 3)
    expect = np.arange(0, shape[0] * shape[1]).reshape(shape)
    output = randn_forward_func(shape)
    assert np.allclose(output.asnumpy(), expect)

    @test_utils.run_with_cell
    def sum_ext_func(x): # tuple input as the second argument
        return mint.sum(x, [0, 1])
    x = Tensor([[1, 2, 3], [4, 5, 6]], dtype=ms.float32)
    sum_ext_func(x) # should not raise any exception

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_tensor_tuple_input(mode):
    """
    Feature: op_plugin kernel
    Description: Test op_plugin kernel for tensor tuple input
    Expectation: Correct result.
    """
    @test_utils.run_with_cell
    def stack_func(inputs):
        return mint.stack(inputs, 0)

    set_mode(mode)
    inputs = (Tensor([1, 2, 3], dtype=ms.float32), Tensor([4, 5, 6], dtype=ms.float32))
    expect = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    output = stack_func(inputs)
    assert np.allclose(output.asnumpy(), expect)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_py_method_dispatch(mode):
    """
    Feature: op_plugin kernel
    Description: Test op_plugin kernel for py_method interface
    Expectation: Correct result.
    """
    @test_utils.run_with_cell
    def add_scalar_func(x, alpha):
        return mint.add(x, alpha)

    def add_scalar_tensor(x, alpha):
        return x.add(alpha)

    set_mode(mode)
    x = Tensor([1, 2, 3], dtype=ms.float32)
    alpha = 1.0
    expect = np.array([-1, -1, -1], dtype=np.float32)
    output_func = add_scalar_func(x, alpha)
    output_tensor = add_scalar_tensor(x, alpha)
    assert np.allclose(output_func.asnumpy(), expect)
    assert np.allclose(output_tensor.asnumpy(), expect)
