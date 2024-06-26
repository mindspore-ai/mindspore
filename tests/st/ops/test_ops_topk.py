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
from mindspore import ops
from mindspore.mint import topk
import mindspore as ms
import tests.st.utils.test_utils as test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randint(1, 10, size=shape).astype(dtype)

def generate_expect_forward_output(x, k, dim):
    index = np.argsort(-x, axis=dim)
    index = index.take(np.arange(k), axis=dim)
    value = abs(np.sort(-x, axis=dim))
    value = value.take(np.arange(k), axis=dim)
    return value, index

def generate_expect_backward_output(x, k, dim):
    values, indices = generate_expect_forward_output(x, k, dim)
    ones = np.ones_like(values)
    zeros = np.zeros_like(x)
    np.put_along_axis(zeros, indices, ones, dim)
    return zeros


@test_utils.run_with_cell
def topk_forward_func(x, k, dim=-1, largest=True, issorted=True):
    return topk(x, k, dim, largest, issorted)


@test_utils.run_with_cell
def topk_backward_func(x, k, dim=-1, largest=True, issorted=True):
    return ops.grad(topk_forward_func, (0, 1, 2, 3, 4))(x, k, dim, largest, issorted)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE])
def test_ops_topk_ext_forward0(context_mode):
    """
    Feature: pyboost function.
    Description: test function topk_ext forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((7, 8, 9), np.float32)

    output0, output1 = topk_forward_func(ms.Tensor(x), 3, 2)
    expect0, expect1 = generate_expect_forward_output(x, 3, 2)
    np.testing.assert_allclose(output0.asnumpy(), expect0, rtol=1e-3)
    np.testing.assert_allclose(output1.asnumpy(), expect1, rtol=1e-3)
    assert output0.asnumpy().dtype == 'float32'
    assert output0.asnumpy().shape == (7, 8, 3)
    assert output1.asnumpy().dtype == 'int64'
    assert output1.asnumpy().shape == (7, 8, 3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE])
def test_ops_topk_ext_forward1(context_mode):
    """
    Feature: pyboost function.
    Description: test function topk_ext forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((7, 8, 9), np.float32)

    output0, output1 = topk_forward_func(ms.Tensor(x), 3, 1)
    expect0, expect1 = generate_expect_forward_output(x, 3, 1)
    np.testing.assert_allclose(output0.asnumpy(), expect0, rtol=1e-3)
    np.testing.assert_allclose(output1.asnumpy(), expect1, rtol=1e-3)
    assert output0.asnumpy().dtype == 'float32'
    assert output0.asnumpy().shape == (7, 3, 9)
    assert output1.asnumpy().dtype == 'int64'
    assert output1.asnumpy().shape == (7, 3, 9)



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE])
def test_ops_topk_ext_forward2(context_mode):
    """
    Feature: pyboost function.
    Description: test function topk_ext forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((7, 8, 9), np.float32)

    output0, output1 = topk_forward_func(ms.Tensor(x), 3, 0)
    expect0, expect1 = generate_expect_forward_output(x, 3, 0)
    np.testing.assert_allclose(output0.asnumpy(), expect0, rtol=1e-3)
    np.testing.assert_allclose(output1.asnumpy(), expect1, rtol=1e-3)
    assert output0.asnumpy().dtype == 'float32'
    assert output0.asnumpy().shape == (3, 8, 9)
    assert output1.asnumpy().dtype == 'int64'
    assert output1.asnumpy().shape == (3, 8, 9)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE])
def test_ops_topk_ext_backward0(context_mode):
    """
    Feature: pyboost function.
    Description: test function topk_ext backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((7, 8, 9), np.float32)
    # min & max
    output = topk_backward_func(ms.Tensor(x), 3, 0)
    expect = generate_expect_backward_output(x, 3, 0)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (7, 8, 9)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE])
def test_ops_topk_ext_backward1(context_mode):
    """
    Feature: pyboost function.
    Description: test function topk_ext backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((7, 8, 9), np.float32)
    # min
    output = topk_backward_func(ms.Tensor(x), 3, 1)
    expect = generate_expect_backward_output(x, 3, 1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (7, 8, 9)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE])
def test_ops_topk_ext_backward2(context_mode):
    """
    Feature: pyboost function.
    Description: test function topk_ext backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((7, 8, 9), np.float32)
    # max
    output = topk_backward_func(ms.Tensor(x), 3, 2)
    expect = generate_expect_backward_output(x, 3, 2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (7, 8, 9)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ops_topk_ext_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function topk_ext forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = ms.Tensor(generate_random_input((7, 8), np.float32))
    k1 = 3


    x2 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    k2 = 4


    test_cell = test_utils.to_cell_obj(topk_forward_func)
    TEST_OP(test_cell, [[x1, k1], [x2, k2]], '', disable_yaml_check=True, disable_mode=['GRAPH_MODE'])
