# sub_right 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a sub_ of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# pylint: disable=unused-variable
import numpy as np
import mindspore as ms
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfAllInputs

class Net(ms.nn.Cell):
    def construct(self, x, y, alpha=1):
        z = x + 0
        return z.sub_(y, alpha=alpha)


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


def inplace_sub_forward_func(x, y, alpha=1):
    return Net()(x, y, alpha)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=1024.0)
def test_inplace_sub_std():
    """
    Feature: standard forward, backward features.
    Description: test function sub_.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    y = generate_random_input((2, 3, 4), np.float32)
    z = generate_random_input((2, 3, 1), np.float32)  # broadcast

    alpha = 1.5

    expect_y = x - y * alpha
    expect_z = x - z * alpha

    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    output_y = inplace_sub_forward_func(ms.Tensor(x), ms.Tensor(y), alpha)
    output_z = inplace_sub_forward_func(ms.Tensor(x), ms.Tensor(z), alpha)

    np.allclose(output_y.asnumpy(), expect_y, rtol=1e-5, equal_nan=True)
    np.allclose(output_z.asnumpy(), expect_z, rtol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.0)
def test_sub__dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test sub_ forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_x1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_y1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_x2 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))
    tensor_y2 = ms.Tensor(generate_random_input((1, 1, 5), np.float32))  # broadcast
    alpha1 = 1.5
    alpha2 = 2.5

    # TEST_OP(inplace_sub_forward_func, [[tensor_x1, tensor_y1, alpha1], [tensor_x2, tensor_y2, alpha2]],
    #         'inplace_sub_ext', disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'], case_config={'disable_grad': True})
    TEST_OP(inplace_sub_forward_func,
            [[tensor_x1, tensor_y1, alpha1], [tensor_x2, tensor_y2, alpha2]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            case_config={'all_dim_zero': True})


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2.0)
def test_sub__bfloat16():
    """
    Feature: test sub_ functional API.
    Description: testcase for sub_ functional API.
    Expectation: the result match with expected result.
    """
    x = generate_random_input((3, 4, 2), np.float32)
    y = generate_random_input((3, 4, 1), np.float32)

    expect_y = x - y

    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    output = inplace_sub_forward_func(ms.Tensor(x, dtype=ms.bfloat16), ms.Tensor(y, dtype=ms.bfloat16))

    np.allclose(output.float().asnumpy(), expect_y, 0.004, 0.004, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2.0)
def test_sub__bool():
    """
    Feature: test sub_ functional API.
    Description: testcase for sub_ functional API.
    Expectation: the result match with expected result.
    """
    x = generate_random_input((3, 4, 2), np.bool_)
    y = generate_random_input((3, 4, 1), np.bool_)
    alpha = 1
    expect_y = x ^ (y & alpha)

    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    output = inplace_sub_forward_func(ms.Tensor(x, dtype=ms.bfloat16), ms.Tensor(y, dtype=ms.bfloat16))

    np.allclose(output.float().asnumpy(), expect_y, 0.004, 0.004, equal_nan=True)

def generate_expect_backward_output(x, y, alpha):
    x_grad = ms.Tensor(np.ones(x.shape), dtype=x.dtype)
    y_grad = ms.Tensor(np.ones(x.shape), dtype=x.dtype) * (-1) * (alpha)
    return x_grad, y_grad

@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
# @pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_inplace_sub_backward():
    """
    Feature: mint
    Description: Verify the result of sub_
    Expectation: success
    """
    test_shape = (1, 2)
    x = ms.Tensor(generate_random_input(test_shape, np.float32))
    y = ms.Tensor(generate_random_input(test_shape, np.float32))
    alpha = -2
    expect_x_grad, expect_y_grad = generate_expect_backward_output(x, y, alpha)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    test_cell = Net()
    test_cell.set_inputs()
    grad_func = GradOfAllInputs(test_cell, sens_param=False)
    output_x_grad, output_other_grad = grad_func(x, y, alpha)

    np.testing.assert_allclose(output_x_grad.asnumpy(), expect_x_grad.asnumpy(), rtol=1e-5)
    np.testing.assert_allclose(output_other_grad.asnumpy(), expect_y_grad.asnumpy(), rtol=1e-5)

@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=2.0)
# @pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_inplace_sub_backward_other_scalar():
    """
    Feature: mint
    Description: Verify the result of sub_
    Expectation: success
    """
    test_shape = (1, 2)
    x = ms.Tensor(generate_random_input(test_shape, np.float32))
    y = 3
    alpha = -2
    expect_x_grad, expect_y_grad = generate_expect_backward_output(x, y, alpha)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    test_cell = Net()
    test_cell.set_inputs()
    grad_func = GradOfAllInputs(test_cell, sens_param=False)
    output_x_grad = grad_func(x, y, alpha)[0]

    np.testing.assert_allclose(output_x_grad.asnumpy(), expect_x_grad.asnumpy(), rtol=1e-5)
