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
from tests.mark_utils import arg_mark
import pytest
import numpy as np
from mindspore import ops
from mindspore.ops import clamp
import mindspore as ms
import tests.st.utils.test_utils as test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


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
    return ops.grad(clamp_forward_func, (0, 1, 2))(x, min_, max_)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_clamp_forward0(context_mode):
    """
    Feature: pyboost function.
    Description: test function clamp forward.
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_clamp_forward1(context_mode):
    """
    Feature: pyboost function.
    Description: test function clamp forward.
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_clamp_forward2(context_mode):
    """
    Feature: pyboost function.
    Description: test function clamp forward.
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_clamp_backward0(context_mode):
    """
    Feature: pyboost function.
    Description: test function clamp backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4), np.float32)
    # min & max
    output = clamp_backward_func(ms.Tensor(x), 2, 7)
    expect = generate_expect_backward_output(x, 2, 7)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (2, 3, 4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_clamp_backward1(context_mode):
    """
    Feature: pyboost function.
    Description: test function clamp backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4), np.float32)
    # min
    output = clamp_backward_func(ms.Tensor(x), 2, None)
    expect = generate_expect_backward_output(x, 2, None)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (2, 3, 4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_clamp_backward2(context_mode):
    """
    Feature: pyboost function.
    Description: test function clamp backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4), np.float32)
    # max
    output = clamp_backward_func(ms.Tensor(x), None, 7)
    expect = generate_expect_backward_output(x, None, 7)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (2, 3, 4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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
    TEST_OP(test_cell, [[x1, min1, max1], [x2, min2, max2]], '', disable_yaml_check=True, disable_mode=['GRAPH_MODE'])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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
    TEST_OP(test_cell, [[x1, min1, max1], [x2, min2, max2]], '', disable_yaml_check=True, disable_mode=['GRAPH_MODE'])
