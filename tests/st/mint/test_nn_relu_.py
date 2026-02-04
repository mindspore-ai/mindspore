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
from mindspore.mint.nn.functional import relu_
import copy
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


def inplace_relu_forward_func(x):
    out = relu_(x)
    return out


def inplace_relu_forward_func_grad(x):
    x = x * 1
    out = relu_(x)
    return out


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
def test_inplace_relu_std():
    """
    Feature: standard forward, backward features.
    Description: test function zero.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 2, 3, 4), np.float32)
    y = copy.deepcopy(x)

    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    output_x = inplace_relu_forward_func(ms.Tensor(x))
    except_out = np.maximum(y, 0)
    np.allclose(output_x.asnumpy(), except_out, rtol=1e-5, equal_nan=True)
    np.allclose(x, except_out, rtol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
def test_inplace_relu_dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test zero forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_x1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_x2 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))

    TEST_OP(inplace_relu_forward_func_grad, [[tensor_x1], [tensor_x2]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'])
