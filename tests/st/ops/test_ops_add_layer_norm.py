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

"""test where"""
import numpy as np
import pytest
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import add_layer_norm



def generate_random_input(shape, dtype):
    np.random.seed(0)
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x1, x2, gamma, beta, eps=1e-5):
    res = x1 + x2
    meanOut = res.mean(1).reshape(2, 1)
    rstdOut = np.power((res.var(1).reshape(2, 1) + eps), -0.5)
    y = rstdOut * (res - meanOut) * gamma + beta
    return y, meanOut, rstdOut, res


def generate_expect_backward_output(addtion_out):
    if addtion_out:
        grad_x_out = np.ones([2, 3]).astype(np.float32)
    else:
        grad_x_out = np.zeros([2, 3]).astype(np.float32)
    grad_gamma_out = np.asarray([2.1147, -0.5852, -1.5295]).astype(np.float32)
    grad_beta_out = np.asarray([2, 2, 2]).astype(np.float32)
    return grad_x_out, grad_gamma_out, grad_beta_out


def add_layer_norm_forward_func(x1, x2, gamma, beta, epsilon, additionalOut):
    return add_layer_norm(x1, x2, gamma, beta, epsilon, additionalOut)


def add_layer_norm_backward_func(x1, x2, gamma, beta, epsilon, additionalOut):
    return ms.grad(add_layer_norm_forward_func, (0, 2, 3))(x1, x2, gamma, beta, epsilon, additionalOut)


@pytest.mark.parametrize('tensor_type', [mstype.float32, mstype.float16, mstype.bfloat16])
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('addtion_out', [True, False])
def test_add_layer_norm(tensor_type, context_mode, addtion_out):
    """
    Feature: test add_layer_norm fusion in kbk mode
    Description: test add_layer_norm.
    Expectation: the result is the same with aclnn version of two ops
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    x1 = generate_random_input((2, 3), np.float32)
    x2 = generate_random_input((2, 3), np.float32)
    gamma = np.ones([3]).astype(np.float32)
    beta = np.zeros([3]).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=tensor_type)
    x2_tensor = Tensor(x2, dtype=tensor_type)
    gamma_tensor = Tensor(gamma, dtype=tensor_type)
    beta_tensor = Tensor(beta, dtype=tensor_type)

    output = add_layer_norm(x1_tensor, x2_tensor, gamma_tensor, beta_tensor, 1e-5, addtion_out)
    output_grad = add_layer_norm_backward_func(x1_tensor, x2_tensor, gamma_tensor, beta_tensor, 1e-5, addtion_out)
    expect = generate_expect_forward_output(x1, x2, gamma, beta)
    expect_grad = generate_expect_backward_output(addtion_out)
    np.testing.assert_allclose(output[0].float().asnumpy(), expect[0], rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(output[1].float().asnumpy(), expect[1], rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(output[2].float().asnumpy(), expect[2], rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(output_grad[0].float().asnumpy(), expect_grad[0], rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(output_grad[1].float().asnumpy(), expect_grad[1], rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(output_grad[2].float().asnumpy(), expect_grad[2], rtol=5e-3, atol=5e-3)
    if addtion_out:
        np.testing.assert_allclose(output[3].float().asnumpy(), expect[3], rtol=5e-3, atol=5e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=40.0)
@pytest.mark.parametrize('addtion_out', [True, False])
def test_add_layer_norm_dynamic_shape(addtion_out):
    """
    Feature: test add_layer_norm fusion with dynamic shape inputs
    Description: test add_layer_norm.
    Expectation: the result is the same with aclnn version of two ops
    """
    x1 = generate_random_input((2, 3), np.float32)
    x2 = generate_random_input((2, 3), np.float32)
    x1_tensor = Tensor(x1, dtype=mstype.float32)
    x2_tensor = Tensor(x2, dtype=mstype.float32)

    x3 = generate_random_input((4, 3), np.float32)
    x4 = generate_random_input((4, 3), np.float32)
    x3_tensor = Tensor(x3, dtype=mstype.float32)
    x4_tensor = Tensor(x4, dtype=mstype.float32)

    gamma = np.ones([3]).astype(np.float32)
    beta = np.zeros([3]).astype(np.float32)
    gamma_tensor = Tensor(gamma, dtype=mstype.float32)
    beta_tensor = Tensor(beta, dtype=mstype.float32)

    test_cell = test_utils.to_cell_obj(add_layer_norm_forward_func)

    case_config = {'disable_input_check': True}
    if not addtion_out:
        case_config = {'disable_input_check': True, 'ignore_output_index': 3}
    # TEST_OP Todo ScalarTensor cause core dump error.
    TEST_OP(test_cell,
            [[x1_tensor, x2_tensor, gamma_tensor, beta_tensor, 1e-5, addtion_out],
             [x3_tensor, x4_tensor, gamma_tensor, beta_tensor, 1e-5, addtion_out]],
            disable_mode=["GRAPH_MODE_GE"],
            disable_case=['EmptyTensor',
                          'ScalarTensor'],
            case_config=case_config)
