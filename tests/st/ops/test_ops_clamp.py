# Copyright 2024 Huawei Technocasties Co., Ltd
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
from mindspore.ops import clamp
import mindspore as ms
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randint(1, 10, size=shape).astype(dtype)


def generate_expect_forward_output(x, min_, max_):
    return np.clip(x, min_, max_)


def generate_expect_backward_output(x, min_, max_):
    out = np.ones_like(x)
    if min_ is not None:
        out = np.select([x >= min_], [out], 0)
    if max_ is not None:
        out = np.select([x <= max_], [out], 0)
    return out


@test_utils.run_with_cell
def clamp_forward_func(x, min_, max_):
    return clamp(x, min_, max_)


@test_utils.run_with_cell
def clamp_backward_func(x, min_, max_):
    return ms.grad(clamp_forward_func, (0, 1, 2))(x, min_, max_)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', mem_peak=1024.02)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_clamp_normal0(context_mode):
    """
    Feature: pyboost function.
    Description: test function clamp forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    # min & max
    output = clamp_forward_func(ms.Tensor(x), 2, 7)
    expect = generate_expect_forward_output(x, 2, 7)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (2, 3, 4, 5)

    x = generate_random_input((2, 3, 4), np.float32)
    # min & max
    output = clamp_backward_func(ms.Tensor(x), 2, 7)
    expect = generate_expect_backward_output(x, 2, 7)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (2, 3, 4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', mem_peak=1024.02)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_clamp_normal1(context_mode):
    """
    Feature: pyboost function.
    Description: test function clamp forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    # min
    output = clamp_forward_func(ms.Tensor(x), 2, None)
    expect = generate_expect_forward_output(x, 2, None)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (2, 3, 4, 5)

    x = generate_random_input((2, 3, 4), np.float32)
    # min
    output = clamp_backward_func(ms.Tensor(x), 2, None)
    expect = generate_expect_backward_output(x, 2, None)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (2, 3, 4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', mem_peak=1024.02)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_clamp_normal2(context_mode):
    """
    Feature: pyboost function.
    Description: test function clamp forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    # max
    output = clamp_forward_func(ms.Tensor(x), None, 7)
    expect = generate_expect_forward_output(x, None, 7)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (2, 3, 4, 5)

    x = generate_random_input((2, 3, 4), np.float32)
    # max
    output = clamp_backward_func(ms.Tensor(x), None, 7)
    expect = generate_expect_backward_output(x, None, 7)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (2, 3, 4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', mem_peak=2048.0)
def test_ops_clamp_min_max_tensor_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function clamp forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))
    min1 = ms.Tensor(generate_random_input((3, 4, 1), np.float32))
    max1 = ms.Tensor(generate_random_input((3, 1, 1), np.float32))

    x2 = ms.Tensor(generate_random_input((3, 4, 5, 6), np.float32))
    min2 = ms.Tensor(generate_random_input((3, 4, 5, 1), np.float32))
    max2 = ms.Tensor(generate_random_input((3, 4, 1, 6), np.float32))

    test_cell = test_utils.to_cell_obj(clamp_forward_func)
    TEST_OP(test_cell, [[x1, min1, max1], [x2, min2, max2]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'all_dim_zero': True})


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', mem_peak=2048.00)
def test_ops_clamp_min_max_scalar_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function clamp forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))
    min1 = 2
    max1 = 7

    x2 = ms.Tensor(generate_random_input((3, 4, 5, 6), np.float32))
    min2 = 3
    max2 = 8

    test_cell = test_utils.to_cell_obj(clamp_forward_func)
    TEST_OP(test_cell, [[x1, min1, max1], [x2, min2, max2]], disable_mode=['GRAPH_MODE_GE'])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', mem_peak=1024.02)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_clamp_type_promotion(context_mode):
    """
    Feature: type promotion.
    Description: test clamp type promotion in different scenarios.
    Expectation: verify ClampScalar and ClampTensor type promotion behaviors.
    """
    ms.context.set_context(mode=context_mode)
    
    # Case 1: int64 + float scalar → float32 output (cross-level promotion)
    x = ms.Tensor([1, 2, 3], dtype=ms.int64)
    result = clamp(x, min=1.5, max=2.5)
    assert result.dtype == ms.float32, f"Case 1: Expected float32, got {result.dtype}"
    expect = np.array([1.5, 2.0, 2.5], dtype=np.float32)
    np.testing.assert_allclose(result.asnumpy(), expect, rtol=1e-3)
    
    # Case 2: float16 + float32 scalar → float16 output (ClampScalar conservative strategy)
    x = ms.Tensor([1.0, 2.0], dtype=ms.float16)
    result = clamp(x, max=2.0)
    assert result.dtype == ms.float16, f"Case 2: Expected float16, got {result.dtype}"
    expect = np.array([1.0, 2.0], dtype=np.float16)
    np.testing.assert_allclose(result.asnumpy(), expect, rtol=1e-3)
    
    # Case 3: float16 + float32 tensor → float32 output (ClampTensor precise promotion)
    x = ms.Tensor([1.0, 2.0], dtype=ms.float16)
    result = clamp(x, max=ms.Tensor([2.0], dtype=ms.float32))
    assert result.dtype == ms.float32, f"Case 3: Expected float32, got {result.dtype}"
    expect = np.array([1.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(result.asnumpy(), expect, rtol=1e-3)
    
    # Case 4: verify scalar vs tensor path difference
    x_fp16 = ms.Tensor([1.5], dtype=ms.float16)
    scalar_result = clamp(x_fp16, max=2.0)
    tensor_result = clamp(x_fp16, max=ms.Tensor([2.0], dtype=ms.float32))
    assert scalar_result.dtype == ms.float16, f"Case 4 scalar: expected float16, got {scalar_result.dtype}"
    assert tensor_result.dtype == ms.float32, f"Case 4 tensor: expected float32, got {tensor_result.dtype}"
    assert scalar_result.dtype != tensor_result.dtype, \
        "Case 4: Scalar and tensor paths should produce different dtypes"
