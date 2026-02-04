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
import pytest
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
import numpy as np
import mindspore as ms
from mindspore import Tensor, mint

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

@test_utils.run_with_cell
def diff_forward_func(input_x, n=1, dim=-1, prepend=None, append=None):
    return mint.diff(input_x, n, dim, prepend, append)

@test_utils.run_with_cell
def diff_backward_func(input_x, n=1, dim=-1, prepend=None, append=None):
    return ms.grad(diff_forward_func, (0))(input_x, n, dim, prepend, append)

@test_utils.run_with_cell
def diff_forward_dyn_func(input_x, n=1, dim=-1, prepend=None, append=None):
    return mint.diff(input_x, n, dim, prepend, append)

@ops_binary_cases(OpsBinaryCase(input_info=[((4, 6, 7, 8), np.float32)],
                                output_info=[((4, 6, 7, 7), np.float32), ((4, 6, 7, 8), np.float32)],
                                extra_info='SD5B'))
def mint_diff_binary_case1(input_binary_data=None, output_binary_data=None):
    output = diff_forward_func(Tensor(input_binary_data[0]))
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    output = diff_backward_func(Tensor(input_binary_data[0]))
    assert np.allclose(output.asnumpy(), output_binary_data[1], 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((5, 3, 6, 8, 9), np.float32), ((5, 3, 2, 8, 9), np.float32)],
                                output_info=[((5, 3, 5, 8, 9), np.float32), ((5, 3, 6, 8, 9), np.float32)],
                                extra_info='SD5B'))
def mint_diff_binary_case2(input_binary_data=None, output_binary_data=None):
    output = diff_forward_func(Tensor(input_binary_data[0]), n=3, dim=2, prepend=Tensor(input_binary_data[1]))
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    output = diff_backward_func(Tensor(input_binary_data[0]), n=3, dim=2, prepend=Tensor(input_binary_data[1]))
    assert np.allclose(output.asnumpy(), output_binary_data[1], 1e-04, 1e-04)


@pytest.mark.parametrize('context_mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ops_diff_binary_cases(context_mode):
    """
    Feature: Ops
    Description: test op rotated_iou pynative
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    mint_diff_binary_case1()
    mint_diff_binary_case2()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=4.0)
def test_ops_normalize_dynamic_shape():
    """
    Feature: Ops
    Description: test op diff dynamic shape
    Expectation: expect correct result.
    """
    input_x = generate_random_input((4, 3, 5), np.float32)
    input_x1 = generate_random_input((4, 3, 6), np.float32)
    input_x2 = generate_random_input((4, 3, 7), np.float32)

    input_y = generate_random_input((2, 3, 4, 5), np.float32)
    input_y1 = generate_random_input((2, 3, 2, 5), np.float32)
    input_y2 = generate_random_input((2, 3, 6, 5), np.float32)
    TEST_OP(diff_forward_dyn_func, [[Tensor(input_x), 2, -1, Tensor(input_x1), Tensor(input_x2)],
                                    [Tensor(input_y), 3, 2, Tensor(input_y1), Tensor(input_y2)]],
            disable_mode=['GRAPH_MODE_GE'], disable_case=['EmptyTensor', 'ScalarTensor'])
