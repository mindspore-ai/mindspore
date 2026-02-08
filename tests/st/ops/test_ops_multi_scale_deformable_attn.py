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
from mindspore import Tensor, ops

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

@test_utils.run_with_cell
def msda_forward_func(value, shape, offset, locations, weight):
    return ops.multi_scale_deformable_attn_function(value, shape, offset, locations, weight)

@test_utils.run_with_cell
def msda_backward_func(value, shape, offset, locations, weight):
    return ms.grad(msda_forward_func, (0, 3, 4))(value, shape, offset, locations, weight)

@test_utils.run_with_cell
def msda_forward_dyn_func(value, shape, offset, locations, weight):
    return ops.multi_scale_deformable_attn_function(value, shape, offset, locations, weight)

@ops_binary_cases(OpsBinaryCase(input_info=[((1, 10000, 4, 32), np.float32), ((1, 2), np.int64), ((1,), np.int64),
                                            ((1, 32, 4, 1, 8, 2), np.float32), ((1, 32, 4, 1, 8), np.float32)],
                                output_info=[((1, 32, 128), np.float32), ((1, 10000, 4, 32), np.float32),
                                             ((1, 32, 4, 1, 8, 2), np.float32), ((1, 32, 4, 1, 8), np.float32)],
                                extra_info='SD5B'))
def ops_multi_scale_deformable_attn_case1(input_binary_data=None, output_binary_data=None):
    output = msda_forward_func(Tensor(input_binary_data[0]), Tensor(input_binary_data[1]), Tensor(input_binary_data[2]),
                               Tensor(input_binary_data[3]), Tensor(input_binary_data[4]))
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-06, 1e-06)
    output = msda_backward_func(Tensor(input_binary_data[0]), Tensor(input_binary_data[1]),
                                Tensor(input_binary_data[2]), Tensor(input_binary_data[3]),
                                Tensor(input_binary_data[4]))
    assert np.allclose(output[0].asnumpy(), output_binary_data[1], 1e-06, 1e-06)
    assert np.allclose(output[1].asnumpy(), output_binary_data[2], 1e-06, 1e-06)
    assert np.allclose(output[2].asnumpy(), output_binary_data[3], 1e-06, 1e-06)

@ops_binary_cases(OpsBinaryCase(input_info=[((2, 30000, 4, 16), np.float32), ((3, 2), np.int64), ((3,), np.int64),
                                            ((2, 32, 4, 3, 4, 2), np.float32), ((2, 32, 4, 3, 4), np.float32)],
                                output_info=[((2, 32, 64), np.float32), ((2, 30000, 4, 16), np.float32),
                                             ((2, 32, 4, 3, 4, 2), np.float32), ((2, 32, 4, 3, 4), np.float32)],
                                extra_info='SD5B'))
def ops_multi_scale_deformable_attn_case2(input_binary_data=None, output_binary_data=None):
    output = msda_forward_func(Tensor(input_binary_data[0]), Tensor(input_binary_data[1]), Tensor(input_binary_data[2]),
                               Tensor(input_binary_data[3]), Tensor(input_binary_data[4]))
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-06, 1e-06)
    output = msda_backward_func(Tensor(input_binary_data[0]), Tensor(input_binary_data[1]),
                                Tensor(input_binary_data[2]), Tensor(input_binary_data[3]),
                                Tensor(input_binary_data[4]))
    assert np.allclose(output[0].asnumpy(), output_binary_data[1], 1e-06, 1e-06)
    assert np.allclose(output[1].asnumpy(), output_binary_data[2], 1e-06, 1e-06)
    assert np.allclose(output[2].asnumpy(), output_binary_data[3], 1e-06, 1e-06)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=78.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_msda_binary_cases(context_mode):
    """
    Feature: Ops
    Description: test op rotated_iou pynative
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    ms.context.set_context(deterministic="ON")

    ops_multi_scale_deformable_attn_case1()
    ops_multi_scale_deformable_attn_case2()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=72.0)
def test_ops_msda_dynamic_shape():
    """
    Feature: Ops
    Description: test op diff dynamic shape
    Expectation: expect correct result.
    """
    value1 = generate_random_input((1, 10000, 4, 32), np.float32)
    shape1 = np.array([[1, 2]], dtype=np.int64)
    offset1 = np.array([1], dtype=np.int64)
    locations1 = generate_random_input((1, 32, 4, 1, 8, 2), np.float32)
    weight1 = generate_random_input((1, 32, 4, 1, 8), np.float32)

    value2 = generate_random_input((2, 10000, 4, 16), np.float32)
    shape2 = np.array([[3, 6]], dtype=np.int64)
    offset2 = np.array([4], dtype=np.int64)
    locations2 = generate_random_input((2, 32, 4, 1, 4, 2), np.float32)
    weight2 = generate_random_input((2, 32, 4, 1, 4), np.float32)

    TEST_OP(msda_forward_dyn_func,
            [[Tensor(value1), Tensor(shape1), Tensor(offset1), Tensor(locations1), Tensor(weight1)],
             [Tensor(value2), Tensor(shape2), Tensor(offset2), Tensor(locations2), Tensor(weight2)]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True})
