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
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
import numpy as np
import mindspore as ms
from mindspore import mint, Tensor

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

@test_utils.run_with_cell
def sub_forward_func(input1, other1):
    return mint.sub(input1, other1)

@test_utils.run_with_cell
def sub_forward_dyn_func(input2, other2, alpha):
    return mint.sub(input2, other2, alpha=alpha)

@test_utils.run_with_cell
def sub_backward_func(input3, other3):
    return ms.grad(sub_forward_func, (0))(input3, other3)

@ops_binary_cases(OpsBinaryCase(input_info=[((1, 1, 1, 288, 64), np.float32)],
                                output_info=[((1, 1, 1, 288, 64), np.float32), ((1, 1, 1, 288, 64), np.float32)],
                                extra_info='SD5B'))
def ops_subs_binary_case1(input_binary_data=None, output_binary_data=None):
    output = sub_forward_func(Tensor(input_binary_data[0]), 8.6)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    grad_output = sub_backward_func(Tensor(input_binary_data[0]), 8.6)
    assert np.allclose(grad_output.asnumpy(), output_binary_data[1], 1e-04, 1e-4)

@ops_binary_cases(OpsBinaryCase(input_info=[((1, 576, 128, 16, 2), np.float32)],
                                output_info=[((1, 576, 128, 16, 2), np.float32), ((1, 576, 128, 16, 2), np.float32)],
                                extra_info='SD5B'))
def ops_subs_binary_case2(input_binary_data=None, output_binary_data=None):
    output = sub_forward_func(Tensor(input_binary_data[0]), 4)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    grad_output = sub_backward_func(Tensor(input_binary_data[0]), 4)
    assert np.allclose(grad_output.asnumpy(), output_binary_data[1], 1e-04, 1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=40.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_sub_binary_cases(context_mode):
    """
    Feature: Ops
    Description: test op rotated_iou pynative
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    ops_subs_binary_case1()
    ops_subs_binary_case2()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=4.0)
def test_ops_sub_dynamic_shape():
    """
    Feature: Ops
    Description: test op rotated_iou dynamic shape
    Expectation: expect correct result.
    """
    input1 = generate_random_input((6, 4, 8, 9, 4), np.float32)
    other1 = 1.0
    alpha1 = 3

    input2 = generate_random_input((3, 7, 8, 5), np.float32)
    other2 = 2.4
    alpha2 = 9
    TEST_OP(sub_forward_dyn_func, [[Tensor(input1), other1, alpha1], [Tensor(input2), other2, alpha2]],
            disable_mode=['GRAPH_MODE_GE'])
