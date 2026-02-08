# Copyright 2025 Huawei Technologies Co., Ltd
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
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.ops.test_tools.ops_binary_cases import OpsBinaryCase, ops_binary_cases
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def index_copy__forward_func(x, dim, index, tensor):
    return x.index_copy_(dim, index, tensor)


@test_utils.run_with_cell
def index_copy__grad(x, dim, index, tensor):
    x = x * 1
    return x.index_copy_(dim, index, tensor)


@test_utils.run_with_cell
def index_copy__backward_func(x, dim, index, tensor):
    return ms.grad(index_copy__grad, (0, 3))(x, dim, index, tensor)


def index_copy__binary_compare(input_binary_data, output_binary_data, dim):
    def compare(expect, actual):
        if expect.dtype == np.float16:
            loss = 1e-3
        elif expect.dtype == np.float32:
            loss = 1e-4
        else:
            loss = 0
        np.testing.assert_allclose(expect, actual, rtol=loss)

    x = ms.Tensor(input_binary_data[0])
    index = ms.Tensor(input_binary_data[1])
    tensor = ms.Tensor(input_binary_data[2])
    output = index_copy__forward_func(x, dim, index, tensor)
    compare(output_binary_data[0], output.asnumpy())
    compare(output_binary_data[0], x.asnumpy())
    grads = index_copy__backward_func(x, dim, index, tensor)
    compare(output_binary_data[1], grads[0].asnumpy())
    compare(output_binary_data[2], grads[1].asnumpy())


@ops_binary_cases(OpsBinaryCase(input_info=[((10, 10, 10), np.float32), ((5,), np.int64), ((10, 5, 10), np.float32)],
                                output_info=[((10, 10, 10), np.float32), ((10, 10, 10), np.float32),
                                             ((10, 5, 10), np.float32)]))
def index_copy__binary_case1(input_binary_data=None, output_binary_data=None):
    index_copy__binary_compare(input_binary_data, output_binary_data, -2)


@ops_binary_cases(OpsBinaryCase(input_info=[((10, 10), np.float16), ((5,), np.int64), ((10, 5), np.float16)],
                                output_info=[((10, 10), np.float16), ((10, 10), np.float16), ((10, 5), np.float16)]))
def index_copy__binary_case2(input_binary_data=None, output_binary_data=None):
    index_copy__binary_compare(input_binary_data, output_binary_data, 1)


@ops_binary_cases(OpsBinaryCase(input_info=[((), np.float16), ((1,), np.int64), ((), np.float16)],
                                output_info=[((), np.float16), ((), np.float16), ((), np.float16)]))
def index_copy__binary_case3(input_binary_data=None, output_binary_data=None):
    index_copy__binary_compare(input_binary_data, output_binary_data, 0)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_tensor_index_copy__normal(mode):
    """
    Feature: Tensor.index_copy_
    Description: Verify the result of Tensor.index_copy_
    Expectation: success
    """
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    index_copy__binary_case1()
    index_copy__binary_case2()
    index_copy__binary_case3()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_tensor_index_copy__test_op():
    """
    Feature: Tensor.index_copy_
    Description: Verify the result of Tensor.index_copy_
    Expectation: success
    """
    x1 = ms.mint.randn((5, 5, 3), dtype=ms.float32)
    dim1 = 0
    index1 = ms.Tensor([2,], dtype=ms.int64)
    tensor1 = ms.mint.randn((1, 5, 3), dtype=ms.float32)

    x2 = ms.mint.randn((5, 3), dtype=ms.float32)
    dim2 = -1
    index2 = ms.Tensor([0, 2, 1], dtype=ms.int64)
    tensor2 = ms.mint.randn((5, 3), dtype=ms.float32)

    TEST_OP(index_copy__grad,
            [[x1, dim1, index1, tensor1], [x2, dim2, index2, tensor2]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={
                'disable_input_check': True,
                'all_dim_zero': True,
            })
