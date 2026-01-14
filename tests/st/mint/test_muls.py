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
# pylint: disable=unused-variable
import numpy as np
import mindspore as ms
from mindspore import mint
from mindspore import ops
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

class Net(ms.nn.Cell):
    def construct(self, x, y):
        z = mint.mul(x, y)
        return z


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


def mint_muls_forward_func(x, y):
    return Net()(x, y)


def mint_muls_backward_func(x, y):
    net = Net()
    grad = ops.GradOperation(get_all=True)
    return grad(net)(x, y)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
def test_muls_std():
    """
    Feature: standard forward, backward features.
    Description: test function copy.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    y = 1.5

    expect_z = x * y

    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    output_z = mint_muls_forward_func(ms.Tensor(x), y)

    np.allclose(output_z.asnumpy(), expect_z, rtol=1e-5, equal_nan=True)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
def test_muls_grad_std():
    """
    Feature: standard forward, backward features.
    Description: test function copy.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    y = 1.5

    expect_grad_x = np.ones_like(x, dtype=np.float32) * y

    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    output_x_grad = mint_muls_backward_func(ms.Tensor(x), y)

    np.allclose(output_x_grad[0].asnumpy(), expect_grad_x, rtol=1e-5, equal_nan=True)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=1024.00)
def test_muls_dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test copy forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_x1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_x2 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))
    tensor_y1 = 1.5
    tensor_y2 = 2.5

    TEST_OP(mint_muls_forward_func, [[tensor_x1, tensor_y1], [tensor_x2, tensor_y2]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'])


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2.0)
def test_muls_bfloat16():
    """
    Feature: test copy functional API.
    Description: testcase for copy functional API.
    Expectation: the result match with expected result.
    """
    x = generate_random_input((3, 4, 2), np.float32)
    y = 1.5

    expect_y = x * y

    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    output = mint_muls_forward_func(ms.Tensor(x, dtype=ms.bfloat16), y)

    np.allclose(output.float().asnumpy(), expect_y, 0.004, 0.004, equal_nan=True)
