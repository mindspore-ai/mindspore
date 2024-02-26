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
import mindspore.ops.function as F
import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore import Tensor
from mindspore.ops.function import mean, prod
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
def prod_forward_func(x, axis=None, keep_dims=False, dtype=None):
    return prod(x, axis, keep_dims, dtype)


@test_utils.run_with_cell
def mean_backward_func(x, axis=None, keep_dims=False, dtype=None):
    return ops.grad(mean_forward_func, (0))(x, axis, keep_dims, dtype)


@test_utils.run_with_cell
def sum_backward_func(x, axis=None, keep_dims=False, dtype=None):
    return ops.grad(sum_forward_func, (0))(x, axis, keep_dims, dtype)


@test_utils.run_with_cell
def prod_backward_func(x, axis=None, keep_dims=False, dtype=None):
    return ops.grad(prod_forward_func, (0))(x, axis, keep_dims, dtype)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
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
    axis = (0, -1)
    x = generate_random_input((2, 3, 4, 5), mstype.dtype_to_nptype(in_dtype))
    output = mean_forward_func(ms.Tensor(x), axis, keep_dims, out_dtype)
    expect = generate_expect_forward_output("mean", x, axis, keep_dims, mstype.dtype_to_nptype(out_dtype))
    np.testing.assert_equal(output.dtype, out_dtype)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    axis = (0, -1)
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(mstype.dtype_to_nptype(in_dtype))
    grads = mean_backward_func(ms.Tensor(x), axis, False, out_dtype)
    expect = np.full((2, 3, 4), 1 / (2 * 4), mstype.dtype_to_nptype(in_dtype))
    np.testing.assert_equal(grads.dtype, in_dtype)
    np.testing.assert_allclose(grads.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mean_default(context_mode):
    """
    Feature: pyboost function.
    Description: test function mean forward and backward on ascend with default args.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = mean_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output("mean", x)
    np.testing.assert_equal(output.dtype, mstype.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x1 = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
    grads = mean_backward_func(ms.Tensor(x1))
    expect = np.full((2, 3, 4), 1 / (2 * 3 * 4), np.float32)
    np.testing.assert_equal(grads.dtype, mstype.float32)
    np.testing.assert_allclose(grads.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mean_dynamic(context_mode):
    """
    Feature: pyboost function.
    Description: test function mean with dynamic shape and rank.
    Expectation: expect correct result.
    """
    input1 = Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    axis1 = (0, -1)
    input2 = Tensor(generate_random_input((3, 3, 4, 4), np.float32))
    axis2 = (0, -1)
    TEST_OP(mean_func, [[input1, axis1], [input2, axis2]], mode=context_mode, grad=True)

    input3 = Tensor(generate_random_input((3, 4, 5), np.float16))
    axis3 = ()
    keep_dims3 = False
    dtype3 = mstype.float32
    input4 = Tensor(generate_random_input((3, 4), np.float16))
    axis4 = ()
    keep_dims4 = False
    dtype4 = mstype.float32
    TEST_OP(mean_func, [[input3, axis3, keep_dims3, dtype3], [input4, axis4, keep_dims4, dtype4]],
            nontensor_dynamic_type='None', mode=context_mode, grad=True, test_resize=False)

    input5 = Tensor(generate_random_input((2, 3, 4), np.float32))
    input6 = Tensor(generate_random_input((2, 3), np.float32))
    TEST_OP(mean_func, [[input5], [input6]], mode=context_mode, grad=True)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
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
    output = sum_forward_func(ms.Tensor(x), axis, keep_dims, out_dtype)
    expect = generate_expect_forward_output("sum", x, axis, keep_dims, mstype.dtype_to_nptype(out_dtype))
    np.testing.assert_equal(output.dtype, out_dtype)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    axis = (0, -1)
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(mstype.dtype_to_nptype(in_dtype))
    grads = sum_backward_func(ms.Tensor(x), axis, False, out_dtype)
    expect = np.ones((2, 3, 4), mstype.dtype_to_nptype(in_dtype))
    np.testing.assert_equal(grads.dtype, in_dtype)
    np.testing.assert_allclose(grads.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sum_default(context_mode):
    """
    Feature: pyboost function.
    Description: test function sum on ascend with default args.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = sum_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output("sum", x)
    np.testing.assert_equal(output.dtype, mstype.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x1 = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
    grads = sum_backward_func(ms.Tensor(x1))
    expect = np.ones((2, 3, 4), np.float32)
    np.testing.assert_equal(grads.dtype, mstype.float32)
    np.testing.assert_allclose(grads.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sum_dynamic(context_mode):
    """
    Feature: pyboost function.
    Description: test function sum with dynamic shape and rank.
    Expectation: expect correct result.
    """
    input1 = Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    axis1 = (0, -1)
    input2 = Tensor(generate_random_input((3, 3, 4, 4), np.float32))
    axis2 = (0, -1)
    TEST_OP(sum_func, [[input1, axis1], [input2, axis2]], mode=context_mode, grad=True)

    input3 = Tensor(generate_random_input((3, 4, 5), np.float32))
    axis3 = ()
    keep_dims3 = False
    dtype3 = mstype.int32
    input4 = Tensor(generate_random_input((3, 4), np.float32))
    axis4 = ()
    keep_dims4 = False
    dtype4 = mstype.int64
    TEST_OP(sum_func, [[input3, axis3, keep_dims3, dtype3], [input4, axis4, keep_dims4, dtype4]],
            nontensor_dynamic_type='None', mode=context_mode, grad=True, test_resize=False)

    input5 = Tensor(generate_random_input((2, 3, 4), np.float32))
    input6 = Tensor(generate_random_input((2, 3), np.float32))
    TEST_OP(sum_func, [[input5], [input6]], mode=context_mode, grad=True)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
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
    output = sum_forward_func(ms.Tensor(x), axis, False, out_dtype)
    np.testing.assert_equal(output.dtype, out_dtype)

    x1 = generate_random_input((3, 4, 5), mstype.dtype_to_nptype(in_dtype))
    grads = sum_backward_func(ms.Tensor(x1), axis, False, out_dtype)
    np.testing.assert_equal(grads.dtype, in_dtype)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
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
    output = sum_forward_func(ms.Tensor(x), axis, False, None)
    np.testing.assert_equal(output.dtype, mstype.int64)

    x1 = generate_random_input((3, 4, 5), mstype.dtype_to_nptype(in_dtype))
    grads = sum_backward_func(ms.Tensor(x1), axis, False, None)
    np.testing.assert_equal(grads.dtype, in_dtype)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('keep_dims', [False, True])
@pytest.mark.parametrize('in_dtype', [mstype.float16])
@pytest.mark.parametrize('out_dtype', [mstype.float32])
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.skip(reason="No support yet")
def test_prod_normal(keep_dims, in_dtype, out_dtype, context_mode):
    """
    Feature: pyboost function.
    Description: test function prod forward on ascend with different datatype.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    axis = 0
    x = generate_random_input((2, 3, 4, 5), mstype.dtype_to_nptype(in_dtype))
    output = prod_forward_func(ms.Tensor(x), axis, keep_dims, out_dtype)
    expect = generate_expect_forward_output("prod", x, axis, keep_dims, mstype.dtype_to_nptype(out_dtype))
    np.testing.assert_equal(output.dtype, out_dtype)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    axis = -1
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(mstype.dtype_to_nptype(in_dtype))
    grads = prod_backward_func(ms.Tensor(x), axis, False, out_dtype)
    expect = np.array([[[6.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                        [2.1000e+02, 1.6800e+02, 1.4000e+02, 1.2000e+02],
                        [9.9000e+02, 8.8000e+02, 7.9200e+02, 7.2000e+02]],
                       [[2.7300e+03, 2.5200e+03, 2.3400e+03, 2.1840e+03],
                        [5.8140e+03, 5.4720e+03, 5.1680e+03, 4.8960e+03],
                        [1.0626e+04, 1.0120e+04, 9.6600e+03, 9.2400e+03]]]).astype(mstype.dtype_to_nptype(in_dtype))
    np.testing.assert_equal(grads.dtype, in_dtype)
    np.testing.assert_allclose(grads.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('axis', [0, -1])
@pytest.mark.parametrize('keep_dims', [False, True])
@pytest.mark.parametrize('in_dtype', [mstype.float16])
@pytest.mark.parametrize('out_dtype', [mstype.float32])
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.skip(reason="No support yet")
def test_prod_normal_1d(axis, keep_dims, in_dtype, out_dtype, context_mode):
    """
    Feature: pyboost function.
    Description: test function prod forward on ascend with different datatype.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = np.random.randn(5).astype(mstype.dtype_to_nptype(in_dtype))
    output = prod_forward_func(ms.Tensor(x), axis, keep_dims, out_dtype)
    expect = generate_expect_forward_output("prod", x, axis, keep_dims, mstype.dtype_to_nptype(out_dtype))
    np.testing.assert_equal(output.dtype, out_dtype)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-2)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.skip(reason="No support yet")
def test_prod_default(context_mode):
    """
    Feature: pyboost function.
    Description: test function prod on ascend with default args.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = prod_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output("prod", x)
    np.testing.assert_equal(output.dtype, mstype.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x1 = np.arange(2 * 3).reshape(2, 3).astype(np.float32)
    grads = prod_backward_func(ms.Tensor(x1))
    expect = np.array([[120, 0, 0], [0, 0, 0]]).astype(np.float32)
    np.testing.assert_equal(grads.dtype, mstype.float32)
    np.testing.assert_allclose(grads.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.skip(reason="No support yet")
def test_prod_dynamic(context_mode):
    """
    Feature: pyboost function.
    Description: test function prod with dynamic shape and rank.
    Expectation: expect correct result.
    """
    input1 = Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    axis1 = -1
    input2 = Tensor(generate_random_input((3, 3, 4, 4), np.float32))
    axis2 = -1
    TEST_OP(prod_func, [[input1, axis1], [input2, axis2]], mode=context_mode, grad=True)

    input3 = Tensor(generate_random_input((3, 4, 5), np.float32))
    axis3 = 0
    keep_dims3 = False
    dtype3 = mstype.int32
    input4 = Tensor(generate_random_input((3, 4), np.float32))
    axis4 = 0
    keep_dims4 = False
    dtype4 = mstype.int64
    TEST_OP(prod_func, [[input3, axis3, keep_dims3, dtype3], [input4, axis4, keep_dims4, dtype4]],
            nontensor_dynamic_type='None', mode=context_mode, grad=True, test_resize=False)

    input5 = Tensor(generate_random_input((2, 3, 4), np.float32))
    input6 = Tensor(generate_random_input((2, 3), np.float32))
    TEST_OP(prod_func, [[input5], [input6]], mode=context_mode, grad=True)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('axis', [-1, None])
@pytest.mark.parametrize('in_dtype', [mstype.float16])
@pytest.mark.parametrize('out_dtype', [mstype.float32, mstype.int8, mstype.uint8])
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.skip(reason="No support yet")
def test_prod_vaild_dtype(axis, in_dtype, out_dtype, context_mode):
    """
    Feature: pyboost function.
    Description: test function prod forward on ascend with different datatype.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4), mstype.dtype_to_nptype(in_dtype))
    output = prod_forward_func(ms.Tensor(x), axis, False, out_dtype)
    np.testing.assert_equal(output.dtype, out_dtype)

    x1 = generate_random_input((3, 4, 5), mstype.dtype_to_nptype(in_dtype))
    grads = prod_backward_func(ms.Tensor(x1), axis, False, out_dtype)
    np.testing.assert_equal(grads.dtype, in_dtype)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('axis', [-1, None])
@pytest.mark.parametrize('in_dtype', [mstype.int8, mstype.int16, mstype.int32, mstype.uint8])
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.skip(reason="No support yet")
def test_prod_default_dtype(axis, in_dtype, context_mode):
    """
    Feature: pyboost function.
    Description: test function prod forward on ascend with different datatype.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4), mstype.dtype_to_nptype(in_dtype))
    output = prod_forward_func(ms.Tensor(x), axis, False, None)
    np.testing.assert_equal(output.dtype, mstype.int64)

    x1 = generate_random_input((3, 4, 5), mstype.dtype_to_nptype(in_dtype))
    grads = prod_backward_func(ms.Tensor(x1), axis, False, None)
    np.testing.assert_equal(grads.dtype, in_dtype)
