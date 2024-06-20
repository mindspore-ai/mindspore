# Copyright 2024 Huawei Technoelu_grad_exties Co., Ltd
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
from mindspore import context, Tensor
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP



elu_grad_ext = ms.ops.auto_generate.EluGradExt()

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(dy, y_x, alpha, dtype):
    return (dy * np.where(y_x > 0, 1, alpha*np.exp(y_x * 1))).astype(dtype)


@test_utils.run_with_cell
def elu_grad_ext_forward_func(dy, y, alpha):
    return elu_grad_ext(dy, y, alpha)

@test_utils.run_with_cell
def elu_grad_ext_vmap_func(dy, y, alpha):
    return ms.ops.vmap(elu_grad_ext_forward_func, in_axes=(0, 0, None), out_axes=0)(dy, y, alpha)



@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_elu_grad_ext_forward(mode):
    """
    Feature: test elu_grad_ext operator
    Description: test elu_grad_ext run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)

    alpha = np.random.uniform(0.5, 2)
    y_np = generate_random_input((2, 3, 4), np.float32)
    dy_np = generate_random_input((2, 3, 4), np.float32)
    y_tensor = Tensor(y_np, ms.float32)
    dy_tensor = Tensor(dy_np, ms.float32)
    output = elu_grad_ext_forward_func(dy_tensor, y_tensor, alpha)
    expect = generate_expect_forward_output(dy_np, y_np, alpha, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)




@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_elu_grad_ext_dynamic_shape_testop():
    """
    Feature: Test elu_grad_ext with dynamic shape in graph mode using TEST_OP.
    Description: call ops.elu_grad_ext with valid input and index.
    Expectation: return the correct value.
    """

    alpha1 = np.random.uniform(0.5, 2)
    alpha2 = np.random.uniform(0.5, 2)
    y1 = generate_random_input((3, 4, 5), np.float32)
    dy1 = generate_random_input((3, 4, 5), np.float32)
    y2 = generate_random_input((3, 7, 8, 3), np.float32)
    dy2 = generate_random_input((3, 7, 8, 3), np.float32)

    TEST_OP(elu_grad_ext, [[ms.Tensor(dy1), ms.Tensor(y1), alpha1],
                           [ms.Tensor(dy2), ms.Tensor(y2), alpha2]],
            'elu_grad_ext', disable_grad=True, disable_input_check=True)



@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_elu_grad_ext_vmap(mode):
    """
    Feature: pyboost function.
    Description: test function elu_grad_ext vmap feature.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)

    alpha = np.random.uniform(0.5, 2)
    y_np = generate_random_input((2, 3, 4), np.float32)
    dy_np = generate_random_input((2, 3, 4), np.float32)
    y_tensor = Tensor(y_np, ms.float32)
    dy_tensor = Tensor(dy_np, ms.float32)
    output = elu_grad_ext_vmap_func(dy_tensor, y_tensor, alpha)
    expect = generate_expect_forward_output(dy_np, y_np, alpha, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)
