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
from mindspore import Tensor, mint, jit
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def select_ext_forward_func(x, dim, index):
    return mint.select(x, dim, index)


@test_utils.run_with_cell
def select_ext_backward_func(x, dim, index):
    return ms.grad(select_ext_forward_func)(x, dim, index)

def GenInputData(np_data_type, shape=(3, 4, 5)):
    """GenInputData"""
    size = 1
    for s in shape:
        size *= s
    data = np.arange(size).reshape(*shape).astype(np_data_type)
    return Tensor(data)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', ['pynative'])
def test_ops_select_ext(mode):
    """
    Feature: pyboost function.
    Description: test function select forward.
    Expectation: expect correct result.
    """
    input_tensor_list = [
        np.array([[2, 3, 5, 1], [3, 1, 2, 6]], dtype=np.float16)
    ]
    expect_list = [
        np.array([3, 1], dtype=np.float16)
    ]
    expect_grad_list = [
        np.array([[0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.float16)
    ]
    dim_list = [1]
    index_list = [1]

    for i in range(len(input_tensor_list)):
        x = input_tensor_list[i]
        dim = dim_list[i]
        index = index_list[i]
        if mode == 'pynative':
            ms.context.set_context(mode=ms.PYNATIVE_MODE)
            output = select_ext_forward_func(ms.Tensor(x), dim, index)
            out_grad = select_ext_backward_func(ms.Tensor(x), dim, index)
        elif mode == 'KBK':
            ms.context.set_context(mode=ms.GRAPH_MODE)
            output = (jit(select_ext_forward_func, jit_level="O0"))(ms.Tensor(x), dim, index)
            out_grad = (jit(select_ext_backward_func, jit_level="O0"))(ms.Tensor(x), dim, index)
        else:
            ms.context.set_context(mode=ms.GRAPH_MODE)
            output = (jit(select_ext_forward_func, backend="GE"))(ms.Tensor(x), dim, index)
            out_grad = (jit(select_ext_backward_func, backend="GE"))(ms.Tensor(x), dim, index)
        expect = expect_list[i]
        expect_grad = expect_grad_list[i]
        np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
        np.testing.assert_allclose(out_grad.asnumpy(), expect_grad, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
def test_select_ext_dynamic_shape():
    """
    Feature: Test select with dynamic shape in pynative mode.
    Description: call mint.select with valid input, dim and index.
    Expectation: return the correct value.
    """
    ms_data1 = ms.Tensor(np.array([[2, 3, 5, 1], [3, 1, 2, 6]], dtype=np.float16))
    dim1 = 0
    index1 = 0

    ms_data2 = ms.Tensor(np.array([[[2, 3, 5], [3, 1, 2]], [[2, 3, 5], [3, 1, 2]]], dtype=np.float16))
    dim2 = 1
    index2 = 1
    TEST_OP(select_ext_forward_func, [[ms_data1, dim1, index1], [ms_data2, dim2, index2]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            disable_case=['EmptyTensor', 'ScalarTensor'])
