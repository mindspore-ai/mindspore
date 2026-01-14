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
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


class Net(ms.nn.Cell):
    def construct(self, x):
        x.floor_()
        return x


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


def floor_forward_func(x):
    return Net()(x)


def floor_forward_func_grad(x):
    x = x * 1
    return Net()(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.00)
def test_floor_std():
    """
    Feature: standard forward, backward features.
    Description: test function zero.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 2, 3, 4), np.float32)

    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    output_x = floor_forward_func(ms.Tensor(x))

    np.allclose(output_x.asnumpy(), x, rtol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.00)
def test_floor_dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test zero forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_x1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_x2 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))

    TEST_OP(floor_forward_func_grad,
            [[tensor_x1], [tensor_x2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ViewTensor'])
