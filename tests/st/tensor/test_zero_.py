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
import pytest
import mindspore as ms
from mindspore import ops
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


class Net(ms.nn.Cell):
    def construct(self, x):
        x = x * 1
        x.zero_()
        return x


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


def zero_forward_func(x):
    return Net()(x)


def zero_backward_func(x):
    grad = ops.GradOperation(get_all=True)
    return grad(Net())(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_zero_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function zero.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 2, 3, 4), np.float32)
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    output_x = zero_forward_func(ms.Tensor(x))
    np.allclose(output_x.asnumpy(), x, rtol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.0)
def test_zero_dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test zero forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_x1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_x2 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))

    TEST_OP(zero_forward_func, [[tensor_x1], [tensor_x2]],
            disable_mode=['GRAPH_MODE_GE'])
