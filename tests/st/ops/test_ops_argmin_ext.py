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
from mindspore import ops, Tensor, mint, jit, JitConfig
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, dim=None, keepdim=False):
    return np.argmin(x, axis=dim)


@test_utils.run_with_cell
def argmin_ext_forward_func(x, dim=None, keepdim=False):
    return mint.argmin(x, dim=dim, keepdim=keepdim)


@test_utils.run_with_cell
def argmin_ext_backward_func(x, dim=None, keepdim=False):
    return ops.grad(argmin_ext_forward_func)(x, dim, keepdim)

def GenInputData(np_data_type, shape=(3, 4, 5)):
    """GenInputData"""
    size = 1
    for s in shape:
        size *= s
    data = np.arange(size).reshape(*shape).astype(np_data_type)
    return Tensor(data)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_ops_argmin_ext(mode):
    """
    Feature: pyboost function.
    Description: test function argmin forward.
    Expectation: expect correct result.
    """
    input_tensor_list = [
        generate_random_input((2, 3, 4, 5), np.float32)
    ]
    dim_list = [0]
    keepdim_list = [False]

    for i in range(len(input_tensor_list)):
        x = input_tensor_list[i]
        dim = dim_list[i]
        keepdim = keepdim_list[i]
        if mode == 'pynative':
            ms.context.set_context(mode=ms.PYNATIVE_MODE)
            output = argmin_ext_forward_func(ms.Tensor(x), dim, keepdim)
            out_grad = argmin_ext_backward_func(ms.Tensor(x), dim, keepdim)
        elif mode == 'KBK':
            output = (jit(argmin_ext_forward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x), dim, keepdim)
            out_grad = (jit(argmin_ext_backward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x), dim, keepdim)
        else:
            output = (jit(argmin_ext_forward_func, jit_config=JitConfig(jit_level="O2")))(ms.Tensor(x), dim, keepdim)
            out_grad = (jit(argmin_ext_backward_func, jit_config=JitConfig(jit_level="O2")))(ms.Tensor(x), dim, keepdim)
        expect = generate_expect_forward_output(x, dim, keepdim)
        np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
        np.testing.assert_allclose(out_grad.asnumpy(), 0, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_argmax_ext_dynamic_shape():
    """
    Feature: Test argmin with dynamic shape in pynative mode and KBK mode.
    Description: call mint.argmin with valid input, dim and keepdim.
    Expectation: return the correct value.
    """
    ms_data1 = ms.Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    dim1 = 0
    keepdim1 = True

    ms_data2 = ms.Tensor(generate_random_input((5, 8, 7), np.float32))
    dim2 = 1
    keepdim2 = False
    TEST_OP(argmin_ext_forward_func, [[ms_data1, dim1, keepdim1], [ms_data2, dim2, keepdim2]],
            'argmin_ext', disable_mode=['GRAPH_MODE'])
