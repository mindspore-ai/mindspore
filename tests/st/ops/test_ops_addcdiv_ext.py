# Copyright 2024 Huawei Technocasties Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,ipi
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, jit, JitConfig, mint
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

ms.context.set_context(device_target="Ascend")

def generate_random_input(shape1, shape2, shape3, dtype):
    x1 = np.random.normal(0, 10, size=shape1).astype(dtype)
    x2 = np.random.normal(0, 10, size=shape2).astype(dtype)
    x3 = np.random.normal(0, 10, size=shape3).astype(dtype)
    return x1, x2, x3

def generate_expect_forward_output(x1, x2, x3, value=1):
    return np.add(x1, np.multiply(np.divide(x2, x3), value))

@test_utils.run_with_cell
def addcdiv_ext_forward_func(x1, x2, x3, value=1):
    return mint.addcdiv(x1, x2, x3, value=value)

@test_utils.run_with_cell
def addcdiv_ext_backward_func(x1, x2, x3, value=1):
    return ms.grad(addcdiv_ext_forward_func)(x1, x2, x3, value=value)

@test_utils.run_with_cell
def addcdiv_ext_vmap_func(x):
    return ops.vmap(addcdiv_ext_forward_func, in_axes=0, out_axed=0)(x)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_ops_addcdiv_ext_normal(mode):
    """
    Feature: pyboost function.
    Description: test function argmin forward.
    Expectation: expect correct result.
    """
    x1, x2, x3 = generate_random_input([1, 3], [3, 1], [1, 3], np.float32)
    value = 1.0
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = addcdiv_ext_forward_func(ms.Tensor(x1), ms.Tensor(x2), ms.Tensor(x3), value=value)
        output_grad = addcdiv_ext_backward_func(ms.Tensor(x1), ms.Tensor(x2), ms.Tensor(x3), value=value)
    else:
        output = (
            jit(addcdiv_ext_forward_func, backend="ms_backend", jit_config=JitConfig(jit_level="O0"))
        )(ms.Tensor(x1), ms.Tensor(x2), ms.Tensor(x3), value=value)
        output_grad = (
            jit(addcdiv_ext_backward_func, backend="ms_backend", jit_config=JitConfig(jit_level="O0"))
        )(ms.Tensor(x1), ms.Tensor(x2), ms.Tensor(x3), value=value)
    expect = generate_expect_forward_output(x1, x2, x3, value)
    print(output_grad)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=1024.00)
def test_ops_addcdiv_ext_forwad_dynamic_shape():
    """
    Feature: Test argmin with dynamic shape in pynative mode and KBK mode.
    Description: call mint.argmin with valid input, dim and keepdim.
    Expectation: return the correct value.
    """
    input1_x1, input1_x2, input1_x3 = generate_random_input([1, 3], [3, 1], [1, 3], np.float32)
    input2_x1, input2_x2, input2_x3 = generate_random_input([3, 1, 1], [3, 4, 1], [3, 1, 1], np.float32)
    TEST_OP(addcdiv_ext_forward_func,
            [[ms.Tensor(input1_x1), ms.Tensor(input1_x2), ms.Tensor(input1_x3), 1.0],
             [ms.Tensor(input2_x1), ms.Tensor(input2_x2), ms.Tensor(input2_x3), 2.0]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            case_config={'all_dim_zero': True})
