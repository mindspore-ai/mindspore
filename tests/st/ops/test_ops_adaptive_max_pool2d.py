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
"""Tests for ops.adaptive_max_pool2d."""
import pytest
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor, context

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

@test_utils.run_with_cell
def adaptive_max_pool2d_forward_func(input1, output_size, return_indices):
    return ops.adaptive_max_pool2d(input1, output_size, return_indices)

@test_utils.run_with_cell
def adaptive_max_pool2d_forward_dyn_func(input2):
    return ops.adaptive_max_pool2d(input2, (8, 8), False)

@test_utils.run_with_cell
def adaptive_max_pool2d_backward_func(input3, output_size, return_indices):
    return ms.grad(adaptive_max_pool2d_forward_func, (0))(input3, output_size, return_indices)

@ops_binary_cases(OpsBinaryCase(input_info=[((16, 4, 16, 16), np.float64)],
                                output_info=[((16, 4, 8, 8), np.float64), ((16, 4, 8, 8), np.int64),
                                             ((16, 4, 16, 16), np.float64)],
                                extra_info='SD5B'))
def ops_adaptive_max_pool2d_case1(input_binary_data=None, output_binary_data=None):
    output = adaptive_max_pool2d_forward_func(Tensor(input_binary_data[0]), (8, 8), True)
    assert np.allclose(output[0].asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    assert np.allclose(output[1].asnumpy(), output_binary_data[1], 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((7, 4, 6, 3), np.float32)],
                                output_info=[((7, 4, 4, 7), np.float32), ((7, 4, 4, 7), np.int64),
                                             ((7, 4, 6, 3), np.float32)],
                                extra_info='SD5B'))
def ops_adaptive_max_pool2d_case2(input_binary_data=None, output_binary_data=None):
    output = adaptive_max_pool2d_forward_func(Tensor(input_binary_data[0]), (4, 7), True)
    assert np.allclose(output[0].asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    assert np.allclose(output[1].asnumpy(), output_binary_data[1], 1e-04, 1e-04)
    grad_output = adaptive_max_pool2d_backward_func(Tensor(input_binary_data[0]), (4, 7), True)
    assert np.allclose(grad_output.asnumpy(), output_binary_data[2], 1e-04, 1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=20.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_adaptive_max_pool2d_binary_cases(context_mode):
    """
    Feature: Ops
    Description: test op rotated_iou pynative
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    ops_adaptive_max_pool2d_case1()
    ops_adaptive_max_pool2d_case2()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=20.0)
def test_ops_adaptive_max_pool2d_dynamic_shape():
    """
    Feature: Ops
    Description: test op rotated_iou dynamic shape
    Expectation: expect correct result.
    """
    input1 = generate_random_input((6, 4, 8, 9), np.float32)
    input2 = generate_random_input((3, 7, 8, 5), np.float32)
    TEST_OP(adaptive_max_pool2d_forward_dyn_func, [[Tensor(input1)], [Tensor(input2)]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True})

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=4.0)
def test_ops_adaptive_max_pool2d_binary_cases_GE():
    """
    Feature: Ops
    Description: test op rotated_iou GE
    Expectation: expect correct result.
    Note: For GE, the pooling do not support float64, so only test float32
    """
    context.set_context(mode=ms.GRAPH_MODE, jit_level='O2', device_target="Ascend")

    ops_adaptive_max_pool2d_case2()
