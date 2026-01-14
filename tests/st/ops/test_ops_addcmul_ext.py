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
from mindspore import mint, jit
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

ms.context.set_context(device_target="Ascend")

def generate_random_input(shape1, shape2, shape3, dtype):
    x = np.random.normal(0, 10, shape1).astype(dtype)
    x1 = np.random.normal(0, 10, shape2).astype(dtype)
    x2 = np.random.normal(0, 10, shape3).astype(dtype)
    return x, x1, x2

def generate_expect_forward_output(x, x1, x2, value=1):
    return np.add(x, np.multiply(np.multiply(x1, x2), value))


@test_utils.run_with_cell
def addcmul_ext_forward_func(x, x1, x2, value=1):
    return mint.addcmul(x, x1, x2, value=value)


@test_utils.run_with_cell
def addcmul_ext_backward_func(x, x1, x2, value=1):
    return ms.grad(addcmul_ext_forward_func)(x, x1, x2, value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_ops_addcmul_ext_normal(mode):
    """
    Feature: pyboost function.
    Description: test function addcmul_ext forward and backward.
    Expectation: expect correct result.
    """
    x, x1, x2 = generate_random_input([2, 1], [2, 3], [2, 1], np.float32)
    v = 2
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = addcmul_ext_forward_func(ms.Tensor(x), ms.Tensor(x1), ms.Tensor(x2), value=v)
        output_grad = addcmul_ext_backward_func(ms.Tensor(x), ms.Tensor(x1), ms.Tensor(x2), value=v)
    else:
        output = (
            jit(addcmul_ext_forward_func, backend="ms_backend")
        )(ms.Tensor(x), ms.Tensor(x1), ms.Tensor(x2), v)
        output_grad = (
            jit(addcmul_ext_backward_func, backend="ms_backend")
        )(ms.Tensor(x), ms.Tensor(x1), ms.Tensor(x2), v)
    expect = generate_expect_forward_output(x, x1, x2, value=v)
    print("output_grad:", output_grad)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_ops_addcmul_ext_forward_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function addcmul_ext forward with dynamic shape.
    Expectation: expect correct result.
    """
    data1_x, data1_x1, data1_x2 = generate_random_input([3, 1], [3, 4], [3, 1], np.float32)
    data1_v = 2.0
    data2_x, data2_x1, data2_x2 = generate_random_input([3, 1, 1], [3, 4, 1], [3, 1, 1], np.float32)
    data2_v = 3.0
    TEST_OP(
        addcmul_ext_forward_func,
        [
            [ms.Tensor(data1_x), ms.Tensor(data1_x1), ms.Tensor(data1_x2), data1_v],
            [ms.Tensor(data2_x), ms.Tensor(data2_x1), ms.Tensor(data2_x2), data2_v]
        ],
        disable_mode=['GRAPH_MODE_GE'],
        case_config={'all_dim_zero': True}
    )
