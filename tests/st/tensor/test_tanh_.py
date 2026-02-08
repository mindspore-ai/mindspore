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
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, jit
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


class Net(ms.nn.Cell):
    def construct(self, x):
        x = x + 0
        return x.tanh_()


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def inplace_tanh_forward_func(x):
    return Net()(x)


@jit(backend="ms_backend")
def inplace_tanh_backward_func(x):
    grad = ops.GradOperation(get_all=True)
    return grad(Net())(x)


def np_tanh(x):
    return np.tanh(x)


def np_tanh_grad(x):
    return 1 - np.power(np.tanh(x), 2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inplace_tanh_normal(mode):
    """
    Feature: standard forward, backward features.
    Description: test function inplace tanh.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 2, 3, 4), np.float32)
    expect_output = np_tanh(x)
    expect_grad = np_tanh_grad(x)

    ms.context.set_context(mode=mode, jit_config={"jit_level": "O0"})
    output = inplace_tanh_forward_func(ms.Tensor(x))
    grad = inplace_tanh_backward_func(ms.Tensor(x))

    np.allclose(output.asnumpy(), expect_output, rtol=1e-5, equal_nan=True)
    np.allclose(grad[0].asnumpy(), expect_grad, rtol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_inplace_tanh_dynamic():
    """
    Feature: dynamic shape forward, backward features.
    Description: test inplace tanh forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_x1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_x2 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))

    TEST_OP(inplace_tanh_forward_func, [[tensor_x1], [tensor_x2]],
            disable_mode=['GRAPH_MODE_GE'], inplace_update=True)
