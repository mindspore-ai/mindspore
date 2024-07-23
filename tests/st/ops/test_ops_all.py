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
from mindspore import ops, context
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, axis, keep_dims):
    return np.all(x, axis, keepdims=keep_dims)

def generate_expect_backward_output(x):
    return np.zeros_like(x)


def generate_expect_forward_vmap_output(x, axis, keep_dims, in_axes):
    x_shape = x.shape
    slices = []
    for i in range(x_shape[in_axes]):
        # Create a list of slice(None) for all dimensions
        idx = [slice(None)] * len(x_shape)
        # Replace the slice(None) at the specified axis with the current index
        idx[in_axes] = i
        slices.append(np.all(x[tuple(idx)], axis, keepdims=keep_dims))

    return np.stack(slices)



@test_utils.run_with_cell
def all_forward_func(x, axis, keep_dims):
    return ms.ops.all(x, axis, keep_dims)


@test_utils.run_with_cell
def all_backward_func(x, axis, keep_dims):
    return ms.ops.grad(all_forward_func, (0))(x, axis, keep_dims)

@test_utils.run_with_cell
def all_vmap_func(x, axis, keep_dims):
    return ops.vmap(all_forward_func, in_axes=(0, None, None), out_axes=0)(x, axis, keep_dims)



@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_all_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function all forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    axis = None
    keep_dims = True
    output = all_forward_func(ms.Tensor(x), axis, keep_dims)
    expect = generate_expect_forward_output(x, axis, keep_dims)
    np.testing.assert_equal(output.asnumpy(), expect)




@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_all_forward_llama(context_mode):
    """
    Feature: pyboost function.
    Description: test function all forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode)
    x = generate_random_input((7168, 8981), np.float32)
    axis = ()
    keep_dims = False
    output = all_forward_func(ms.Tensor(x), axis, keep_dims)
    expect = generate_expect_forward_output(x, axis, keep_dims)
    np.testing.assert_equal(output.asnumpy(), expect)



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_all_forward_SDv1(context_mode):
    """
    Feature: pyboost function.
    Description: test function all forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode)
    x = np.random.choice(a=[True, False])
    axis = ()
    keep_dims = True
    output = all_forward_func(ms.Tensor(x), axis, keep_dims)
    expect = generate_expect_forward_output(x, axis, keep_dims)
    np.testing.assert_equal(output.asnumpy(), expect)



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_all_backward(context_mode):
    """
    Feature: pyboost function.
    Description: test function all backward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4), np.float32)
    axis = ()
    keep_dims = False
    output = all_backward_func(ms.Tensor(x), axis, keep_dims)
    expect = generate_expect_backward_output(x)
    np.testing.assert_equal(output.asnumpy(), expect)



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_all_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function all vmap feature.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.bool)
    axis = (1, 2)
    keep_dims = True
    output = all_vmap_func(ms.Tensor(x), axis, keep_dims)
    expect = generate_expect_forward_vmap_output(x, axis, keep_dims, 0)
    np.testing.assert_equal(output.asnumpy(), expect)



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_all_dynamic_shape_testop():
    """
    Feature: Test all with dynamic shape in graph mode using TEST_OP.
    Description: call ops.all with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((3, 4, 5), np.float32)
    x2 = generate_random_input((3, 7, 8, 3), np.float32)
    axis1 = (0, 1)
    axis2 = (1, 2)
    keep_dims1 = True
    keep_dims2 = False

    TEST_OP(all_forward_func, [[ms.Tensor(x1), axis1, keep_dims1], [ms.Tensor(x2), axis2, keep_dims2]], 'reduce_all',
            disable_input_check=True, disable_grad=True)
