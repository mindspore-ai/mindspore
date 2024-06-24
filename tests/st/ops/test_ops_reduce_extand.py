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
import mindspore.ops.function as F
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore.ops.function import prod
from mindspore.ops.function.math_func import mean_ext as mean
from mindspore.ops.composite import GradOperation
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(name, x, axis=None, keep_dims=False, dtype=None):
    if name == "mean":
        return np.mean(x, axis=axis, dtype=dtype, keepdims=keep_dims)
    if name == "prod":
        return np.prod(x, axis=axis, dtype=dtype, keepdims=keep_dims)
    if name == "sum":
        return np.sum(x, axis=axis, dtype=dtype, keepdims=keep_dims)
    return None


class ProdNet(nn.Cell):
    def __init__(self, axis=None, dtype=None):
        super().__init__()
        self.axis = axis
        self.dtype = dtype

    def construct(self, x, keep_dims=False):
        return prod(x, self.axis, keep_dims, self.dtype)


class ProdGradNet(nn.Cell):
    def __init__(self, net):
        super(ProdGradNet, self).__init__()
        self.grad = GradOperation(sens_param=False)
        self.net = net

    def construct(self, x, keep_dims=False):
        return self.grad(self.net)(x, keep_dims)


def mean_func(x, axis=None, keep_dims=False, dtype=None):
    return mean(x, axis, keep_dims, dtype)


def sum_func(x, axis=None, keep_dims=False, dtype=None):
    return F.sum(x, axis, keep_dims, dtype=dtype)


def prod_func(x, axis=None, keep_dims=False, dtype=None):
    return prod(x, axis, keep_dims, dtype)


@test_utils.run_with_cell
def mean_forward_func(x, axis=None, keep_dims=False, dtype=None):
    return mean_func(x, axis, keep_dims, dtype)


@test_utils.run_with_cell
def sum_forward_func(x, axis=None, keep_dims=False, dtype=None):
    return sum_func(x, axis, keep_dims, dtype=dtype)


@test_utils.run_with_cell
def mean_backward_func(x, axis=None, keep_dims=False, dtype=None):
    return ops.grad(mean_forward_func, (0))(x, axis, keep_dims, dtype)


@test_utils.run_with_cell
def sum_backward_func(x, axis=None, keep_dims=False, dtype=None):
    return ops.grad(sum_forward_func, (0))(x, axis, keep_dims, dtype)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('keep_dims', [False, True])
@pytest.mark.parametrize('in_dtype', [mstype.float16])
@pytest.mark.parametrize('out_dtype', [mstype.float32])
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mean_normal(keep_dims, in_dtype, out_dtype, context_mode):
    """
    Feature: pyboost function.
    Description: test function mean forward and backward on ascend with different datatype.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    axis = (0, 1, 2, 3)
    x = generate_random_input((64, 4, 64, 64), mstype.dtype_to_nptype(in_dtype))
    output = mean_forward_func(Tensor(x), axis, keep_dims, out_dtype)
    expect = generate_expect_forward_output("mean", x, axis, keep_dims, mstype.dtype_to_nptype(out_dtype))
    np.testing.assert_equal(output.dtype, out_dtype)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    axis = (0, -1)
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(mstype.dtype_to_nptype(in_dtype))
    grads = mean_backward_func(Tensor(x), axis, False, out_dtype)
    expect = np.full((2, 3, 4), 1 / (2 * 4), mstype.dtype_to_nptype(in_dtype))
    np.testing.assert_equal(grads.dtype, in_dtype)
    np.testing.assert_allclose(grads.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mean_default(context_mode):
    """
    Feature: pyboost function.
    Description: test function mean forward and backward on ascend with default args.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = mean_forward_func(Tensor(x))
    expect = generate_expect_forward_output("mean", x)
    np.testing.assert_equal(output.dtype, mstype.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x1 = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
    grads = mean_backward_func(Tensor(x1))
    expect = np.full((2, 3, 4), 1 / (2 * 3 * 4), np.float32)
    np.testing.assert_equal(grads.dtype, mstype.float32)
    np.testing.assert_allclose(grads.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mean_dynamic():
    """
    Feature: pyboost function.
    Description: test function mean with dynamic shape and rank.
    Expectation: expect correct result.
    """
    input1 = Tensor(generate_random_input((2, 3, 4), np.float32))
    axis1 = (0, -1)
    keep_dims1 = False
    input2 = Tensor(generate_random_input((3, 3, 4, 4), np.float32))
    axis2 = (0, 1)
    keep_dims2 = True
    TEST_OP(mean_func, [[input1, axis1, keep_dims1], [input2, axis2, keep_dims2]], '', disable_yaml_check=True)

    input3 = Tensor(generate_random_input((2, 3, 4), np.float32))
    input4 = Tensor(generate_random_input((2, 3), np.float32))
    TEST_OP(mean_func, [[input3], [input4]], '', disable_yaml_check=True)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('keep_dims', [False, True])
@pytest.mark.parametrize('in_dtype', [mstype.float16])
@pytest.mark.parametrize('out_dtype', [mstype.float32])
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sum_normal(keep_dims, in_dtype, out_dtype, context_mode):
    """
    Feature: pyboost function.
    Description: test function sum forward on ascend with different datatype.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    axis = (0, -1)
    x = generate_random_input((2, 3, 4, 5), mstype.dtype_to_nptype(in_dtype))
    output = sum_forward_func(Tensor(x), axis, keep_dims, out_dtype)
    expect = generate_expect_forward_output("sum", x, axis, keep_dims, mstype.dtype_to_nptype(out_dtype))
    np.testing.assert_equal(output.dtype, out_dtype)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    axis = (0, -1)
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(mstype.dtype_to_nptype(in_dtype))
    grads = sum_backward_func(Tensor(x), axis, False, out_dtype)
    expect = np.ones((2, 3, 4), mstype.dtype_to_nptype(in_dtype))
    np.testing.assert_equal(grads.dtype, in_dtype)
    np.testing.assert_allclose(grads.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sum_default(context_mode):
    """
    Feature: pyboost function.
    Description: test function sum on ascend with default args.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = sum_forward_func(Tensor(x))
    expect = generate_expect_forward_output("sum", x)
    np.testing.assert_equal(output.dtype, mstype.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x1 = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
    grads = sum_backward_func(Tensor(x1))
    expect = np.ones((2, 3, 4), np.float32)
    np.testing.assert_equal(grads.dtype, mstype.float32)
    np.testing.assert_allclose(grads.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sum_dynamic():
    """
    Feature: pyboost function.
    Description: test function sum with dynamic shape and rank.
    Expectation: expect correct result.
    """
    input1 = Tensor(generate_random_input((2, 3, 4), np.float32))
    axis1 = (0, -1)
    keep_dims1 = False
    input2 = Tensor(generate_random_input((3, 3, 4, 4), np.float32))
    axis2 = (0, 1)
    keep_dims2 = True
    TEST_OP(sum_func, [[input1, axis1, keep_dims1], [input2, axis2, keep_dims2]], '', disable_yaml_check=True)

    input3 = Tensor(generate_random_input((2, 3, 4), np.float32))
    input4 = Tensor(generate_random_input((2, 3), np.float32))
    TEST_OP(sum_func, [[input3], [input4]], '', disable_yaml_check=True)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('axis', [(-1), ()])
@pytest.mark.parametrize('in_dtype', [mstype.float16])
@pytest.mark.parametrize('out_dtype', [mstype.float32, mstype.int8, mstype.uint8, mstype.complex128])
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sum_vaild_dtype(axis, in_dtype, out_dtype, context_mode):
    """
    Feature: pyboost function.
    Description: test function sum forward on ascend with different datatype.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4), mstype.dtype_to_nptype(in_dtype))
    output = sum_forward_func(Tensor(x), axis, False, out_dtype)
    np.testing.assert_equal(output.dtype, out_dtype)

    x1 = generate_random_input((3, 4, 5), mstype.dtype_to_nptype(in_dtype))
    grads = sum_backward_func(Tensor(x1), axis, False, out_dtype)
    np.testing.assert_equal(grads.dtype, in_dtype)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('axis', [(-1), ()])
@pytest.mark.parametrize('in_dtype', [mstype.bool_, mstype.int8, mstype.int16, mstype.int32, mstype.uint8])
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sum_default_dtype(axis, in_dtype, context_mode):
    """
    Feature: pyboost function.
    Description: test function sum forward on ascend with different datatype.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4), mstype.dtype_to_nptype(in_dtype))
    output = sum_forward_func(Tensor(x), axis, False, None)
    np.testing.assert_equal(output.dtype, mstype.int64)

    x1 = generate_random_input((3, 4, 5), mstype.dtype_to_nptype(in_dtype))
    grads = sum_backward_func(Tensor(x1), axis, False, None)
    np.testing.assert_equal(grads.dtype, in_dtype)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('keep_dims', [False, True])
@pytest.mark.parametrize('in_dtype', [mstype.float32])
@pytest.mark.parametrize('out_dtype', [mstype.float32])
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_prod_normal(keep_dims, in_dtype, out_dtype, context_mode):
    """
    Feature: pyboost function.
    Description: test function prod forward on ascend with different datatype.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    axis = 0
    x = generate_random_input((2, 3, 4, 5), mstype.dtype_to_nptype(in_dtype))
    output = ProdNet(axis, out_dtype)(Tensor(x), keep_dims)
    expect = generate_expect_forward_output("prod", x, axis, keep_dims, mstype.dtype_to_nptype(out_dtype))
    np.testing.assert_equal(output.dtype, out_dtype)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    axis = -1
    x = np.array([[1, 2, 3], [4, 5, 6]]).astype(mstype.dtype_to_nptype(in_dtype))
    grads = ProdGradNet(ProdNet(axis, out_dtype))(Tensor(x), False)
    expect = np.array([[6, 3, 2], [30, 24, 20]]).astype(mstype.dtype_to_nptype(in_dtype))
    np.testing.assert_equal(grads.dtype, in_dtype)
    np.testing.assert_allclose(grads.asnumpy(), expect, rtol=1e-3)

    axis = -1
    x = np.array([[0, 1, 2], [3, 4, 5]]).astype(mstype.dtype_to_nptype(in_dtype))
    grads = ProdGradNet(ProdNet(axis, out_dtype))(Tensor(x), False)
    expect = np.array([[2, 0, 0], [20, 15, 12]]).astype(mstype.dtype_to_nptype(in_dtype))
    np.testing.assert_equal(grads.dtype, in_dtype)
    np.testing.assert_allclose(grads.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('axis', [0, -1])
@pytest.mark.parametrize('keep_dims', [False, True])
@pytest.mark.parametrize('in_dtype', [mstype.float16])
@pytest.mark.parametrize('out_dtype', [mstype.float32])
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_prod_normal_1d(axis, keep_dims, in_dtype, out_dtype, context_mode):
    """
    Feature: pyboost function.
    Description: test function prod forward on ascend with different datatype.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = np.random.randn(5).astype(mstype.dtype_to_nptype(in_dtype))
    output = ProdNet(axis, out_dtype)(Tensor(x), keep_dims)
    expect = generate_expect_forward_output("prod", x, axis, keep_dims, mstype.dtype_to_nptype(out_dtype))
    np.testing.assert_equal(output.dtype, out_dtype)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-2)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_prod_default(context_mode):
    """
    Feature: pyboost function.
    Description: test function prod on ascend with default args.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4), np.float32)
    output = ProdNet()(Tensor(x))
    expect = generate_expect_forward_output("prod", x)
    np.testing.assert_equal(output.dtype, mstype.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x1 = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    grads = ProdGradNet(ProdNet())(Tensor(x1))
    expect = np.array([[720, 360, 240], [180, 144, 120]]).astype(np.float32)
    np.testing.assert_equal(grads.dtype, mstype.float32)
    np.testing.assert_allclose(grads.asnumpy(), expect, rtol=1e-3)

    x1 = np.array([[0, 1, 2], [3, 4, 5]]).astype(np.float32)
    grads = ProdGradNet(ProdNet())(Tensor(x1))
    expect = np.array([[120, 0, 0], [0, 0, 0]]).astype(np.float32)
    np.testing.assert_equal(grads.dtype, mstype.float32)
    np.testing.assert_allclose(grads.asnumpy(), expect, rtol=1e-3)

    x1 = np.array([[0, 1, 2], [3, 4, 0]]).astype(np.float32)
    grads = ProdGradNet(ProdNet())(Tensor(x1))
    expect = np.array([[0, 0, 0], [0, 0, 0]]).astype(np.float32)
    np.testing.assert_equal(grads.dtype, mstype.float32)
    np.testing.assert_allclose(grads.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_prod_dynamic():
    """
    Feature: pyboost function.
    Description: test function prod with dynamic shape and rank.
    Expectation: expect correct result.
    """
    input1 = Tensor(generate_random_input((2, 3, 4), np.float32))
    axis1 = -1
    keep_dims1 = False
    input2 = Tensor(generate_random_input((3, 3, 4, 4), np.float32))
    axis2 = 1
    keep_dims2 = True
    TEST_OP(prod_func, [[input1, axis1, keep_dims1], [input2, axis2, keep_dims2]], '', disable_yaml_check=True)

    input3 = Tensor(generate_random_input((2, 3, 4), np.float32))
    keep_dims3 = False
    input4 = Tensor(generate_random_input((2, 3), np.float32))
    keep_dims4 = False
    TEST_OP(ProdNet(), [[input3, keep_dims3], [input4, keep_dims4]], '', disable_input_check=True,
            disable_yaml_check=True)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('axis', [-1, None])
@pytest.mark.parametrize('in_dtype', [mstype.float16])
@pytest.mark.parametrize('out_dtype', [mstype.float32, mstype.int8, mstype.uint8])
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_prod_vaild_dtype(axis, in_dtype, out_dtype, context_mode):
    """
    Feature: pyboost function.
    Description: test function prod forward on ascend with different datatype.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4), mstype.dtype_to_nptype(in_dtype))
    output = ProdNet(axis, out_dtype)(Tensor(x), False)
    np.testing.assert_equal(output.dtype, out_dtype)

    x1 = generate_random_input((3, 4, 5), mstype.dtype_to_nptype(in_dtype))
    grads = ProdGradNet(ProdNet(axis, out_dtype))(Tensor(x1), False)
    np.testing.assert_equal(grads.dtype, in_dtype)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('axis', [-1, None])
@pytest.mark.parametrize('in_dtype', [mstype.int8, mstype.int32, mstype.uint8])
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_prod_default_dtype(axis, in_dtype, context_mode):
    """
    Feature: pyboost function.
    Description: test function prod forward on ascend with different datatype.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4), mstype.dtype_to_nptype(in_dtype))
    output = ProdNet(axis, None)(Tensor(x), False)
    np.testing.assert_equal(output.dtype, mstype.int64)

    x1 = generate_random_input((3, 4, 5), mstype.dtype_to_nptype(in_dtype))
    grads = ProdGradNet(ProdNet(axis, None))(Tensor(x1), False)
    np.testing.assert_equal(grads.dtype, in_dtype)
