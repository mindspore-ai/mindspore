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
    def construct(self, x, index, source, accumulate):
        x.put_(index, source, accumulate)
        return x


def generate_random_input(shape, dtype):
    return np.random.randint(1, 9, shape).astype(dtype)


def generate_expect_forward_output(x, index, source, accumulate):
    x_reshape = ms.Tensor(x).reshape(36).asnumpy()
    if accumulate:
        i = 0
        for item in index:
            x_reshape[item] += source[i]
            i += 1
    else:
        i = 0
        for item in index:
            x_reshape[item] = source[i]
            i += 1
    return ms.Tensor(x_reshape).reshape(2, 3, 6)


def put_forward_func(x, index, source, accumulate):
    return Net()(x, index, source, accumulate)


def put_forward_func_with_x_mul_1(x, index, source, accumulate):
    return Net()(x * 1, index, source, accumulate)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_put_std():
    """
    Feature: standard forward, backward features.
    Description: test function zero.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 6), np.float32)
    index = [1, 2, 9]
    source = generate_random_input((3), np.float32)
    accumulate1 = False
    accumulate2 = True
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    output1 = put_forward_func(ms.Tensor(x), ms.Tensor(index), ms.Tensor(source), accumulate1)
    output2 = put_forward_func(ms.Tensor(x), ms.Tensor(index), ms.Tensor(source), accumulate2)
    expect_out1 = generate_expect_forward_output(x, index, source, accumulate1)
    expect_out2 = generate_expect_forward_output(x, index, source, accumulate2)
    assert np.allclose(output1.asnumpy(), expect_out1.asnumpy(), rtol=1e-5, equal_nan=True)
    assert np.allclose(output2.asnumpy(), expect_out2.asnumpy(), rtol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.0)
def test_put_dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test zero forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_x1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_x2 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))
    index1 = ms.Tensor(generate_random_input((2), np.int64))
    source1 = ms.Tensor(generate_random_input((2), np.float32))
    index2 = ms.Tensor(generate_random_input((3, 2), np.int64))
    source2 = ms.Tensor(generate_random_input((3, 2), np.float32))
    accumulate1 = False
    accumulate2 = True
    TEST_OP(put_forward_func_with_x_mul_1,
            [[tensor_x1, index1, source1, accumulate1],
             [tensor_x2, index2, source2, accumulate2]],
            disable_mode=['GRAPH_MODE_GE'],
            inplace_update=True)
