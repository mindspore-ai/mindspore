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
from mindspore.mint import unique
import mindspore as ms
import tests.st.utils.test_utils as test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape):
    return np.random.randint(0, 10, shape)

def forward_expect_func(inputx, return_inverse=False, return_counts=False):
    return np.unique(inputx, False, return_inverse, return_counts)

@test_utils.run_with_cell
def unique_forward_func(inputx, is_sorted=True, return_inverse=False, return_counts=False, dim=None):
    return unique(inputx, is_sorted, return_inverse, return_counts, dim)

@test_utils.run_with_cell
def unique_forward_func_dynamic(inputx, is_sorted=True, dim=1):
    return unique(inputx, is_sorted, True, True, dim)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_ops_unique_forward_dim_None(mode):
    """
    Feature: pyboost function.
    Description: test function unique forward dim None.
    Expectation: expect correct result.
    """
    inputx_np = generate_random_input((5, 6, 7))
    inputx = ms.Tensor(inputx_np)

    expect_out1 = forward_expect_func(inputx_np)
    expect_out2, expect_inverse2 = forward_expect_func(inputx_np, True, False)
    expect_inverse2 = expect_inverse2.reshape(5, 6, 7)
    expect_out3, expect_counts3 = forward_expect_func(inputx_np, False, True)
    expect_out4, expect_inverse4, expect_counts4 = forward_expect_func(inputx_np, True, True)
    expect_inverse4 = expect_inverse4.reshape(5, 6, 7)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        out1 = unique_forward_func(inputx)
        out2, inverse2 = unique_forward_func(inputx, True, True, False, None)
        out3, counts3 = unique_forward_func(inputx, True, False, True, None)
        out4, inverse4, counts4 = unique_forward_func(inputx, True, True, True, None)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op = ms.jit(unique_forward_func, jit_config=ms.JitConfig(jit_level="O0"))
        out1 = op(inputx)
        out2, inverse2 = op(inputx, True, True, False, None)
        out3, counts3 = op(inputx, True, False, True, None)
        out4, inverse4, counts4 = op(inputx, True, True, True, None)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        out1 = unique_forward_func(inputx)
        out2, inverse2 = unique_forward_func(inputx, True, True, False, None)
        out3, counts3 = unique_forward_func(inputx, True, False, True, None)
        out4, inverse4, counts4 = unique_forward_func(inputx, True, True, True, None)


    np.testing.assert_allclose(out1.asnumpy(), expect_out1, rtol=1e-3)
    np.testing.assert_allclose(out2.asnumpy(), expect_out2, rtol=1e-3)
    np.testing.assert_allclose(inverse2.asnumpy(), expect_inverse2, rtol=1e-3)
    np.testing.assert_allclose(out3.asnumpy(), expect_out3, rtol=1e-3)
    np.testing.assert_allclose(counts3.asnumpy(), expect_counts3, rtol=1e-3)
    np.testing.assert_allclose(out4.asnumpy(), expect_out4, rtol=1e-3)
    np.testing.assert_allclose(inverse4.asnumpy(), expect_inverse4, rtol=1e-3)
    np.testing.assert_allclose(counts4.asnumpy(), expect_counts4, rtol=1e-3)

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_ops_unique_forward_with_dim(mode):
    """
    Feature: pyboost function.
    Description: test function unique forward dim not none.
    Expectation: expect correct result.
    """
    inputx = ms.Tensor([[1, 3, 2, 3, 4, 5, 3, 2], [2, 4, 3, 2, 2, 5, 2, 5], [1, 3, 2, 3, 4, 5, 3, 2]])



    expect_out1 = np.array([[1, 2, 2, 3, 3, 4, 5], [2, 3, 5, 2, 4, 2, 5], [1, 2, 2, 3, 3, 4, 5]])
    expect_inverse1 = np.array([0, 4, 1, 3, 5, 6, 3, 2])
    expect_counts1 = np.array([1, 1, 1, 2, 1, 1, 1])

    expect_out2 = np.array([[1, 3, 2, 3, 4, 5, 3, 2], [2, 4, 3, 2, 2, 5, 2, 5]])
    expect_inverse2 = np.array([0, 1, 0])
    expect_counts2 = np.array([2, 1])


    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        out1, inverse1, counts1 = unique_forward_func(inputx, True, True, True, 1)
        out2, inverse2, counts2 = unique_forward_func(inputx, True, True, True, 0)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op = ms.jit(unique_forward_func, jit_config=ms.JitConfig(jit_level="O0"))
        out1, inverse1, counts1 = op(inputx, True, True, True, 1)
        out2, inverse2, counts2 = op(inputx, True, True, True, 0)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        out1, inverse1, counts1 = unique_forward_func(inputx, True, True, True, 1)
        out2, inverse2, counts2 = unique_forward_func(inputx, True, True, True, 0)

    np.testing.assert_allclose(out1.asnumpy(), expect_out1, rtol=1e-3)
    np.testing.assert_allclose(inverse1.asnumpy(), expect_inverse1, rtol=1e-3)
    np.testing.assert_allclose(counts1.asnumpy(), expect_counts1, rtol=1e-3)
    np.testing.assert_allclose(out2.asnumpy(), expect_out2, rtol=1e-3)
    np.testing.assert_allclose(inverse2.asnumpy(), expect_inverse2, rtol=1e-3)
    np.testing.assert_allclose(counts2.asnumpy(), expect_counts2, rtol=1e-3)



@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
def test_ops_unique_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function unique forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = ms.Tensor(generate_random_input((7, 8, 9)))
    sorted1 = True
    dim1 = 0

    x2 = ms.Tensor(generate_random_input((8, 9)))
    sorted2 = False
    dim2 = 1

    test_cell = test_utils.to_cell_obj(unique_forward_func_dynamic)
    TEST_OP(test_cell, [[x1, sorted1, dim1], [x2, sorted2, dim2]], "", disable_grad=True, disable_mode=["GRAPH_MODE"],
            disable_yaml_check=True)
