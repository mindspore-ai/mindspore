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
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor, context

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def rotated_iou_forward_func(boxes, query_boxes, trans, mode, is_cross, v_threshold, e_threshold):
    return ops.rotated_iou(boxes, query_boxes, trans, mode, is_cross, v_threshold, e_threshold)

@ops_binary_cases(OpsBinaryCase(input_info=[((4, 7, 5), np.float16), ((4, 6, 5), np.float32)],
                                output_info=[((4, 7, 6), np.float16)],
                                extra_info='SD5B'))
def ops_rotated_iou_binary_case1(input_binary_data=None, output_binary_data=None):
    output = rotated_iou_forward_func(Tensor(input_binary_data[0]), Tensor(input_binary_data[1]),
                                      trans=False, mode=0, is_cross=True, v_threshold=0.0, e_threshold=0.0)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((8, 10, 5), np.float32), ((8, 12, 5), np.float32)],
                                output_info=[((8, 10, 12), np.float32)],
                                extra_info='SD5B'))
def ops_rotated_iou_binary_case2(input_binary_data=None, output_binary_data=None):
    output = rotated_iou_forward_func(Tensor(input_binary_data[0]), Tensor(input_binary_data[1]),
                                      trans=False, mode=0, is_cross=True, v_threshold=0.0, e_threshold=0.0)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((7, 11, 5), np.float32), ((7, 6, 5), np.float32)],
                                output_info=[((7, 11, 6), np.float32)],
                                extra_info='SD5B'))
def ops_rotated_iou_binary_case3(input_binary_data=None, output_binary_data=None):
    output = rotated_iou_forward_func(Tensor(input_binary_data[0]), Tensor(input_binary_data[1]),
                                      trans=True, mode=0, is_cross=True, v_threshold=0.0, e_threshold=0.0)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((9, 8, 5), np.float32), ((9, 4, 5), np.float32)],
                                output_info=[((9, 8, 4), np.float32)],
                                extra_info='SD5B'))
def ops_rotated_iou_binary_case4(input_binary_data=None, output_binary_data=None):
    output = rotated_iou_forward_func(Tensor(input_binary_data[0]), Tensor(input_binary_data[1]),
                                      trans=False, mode=1, is_cross=True, v_threshold=0.0, e_threshold=0.0)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((6, 14, 5), np.float32), ((6, 14, 5), np.float32)],
                                output_info=[((6, 14, 14), np.float32)],
                                extra_info='SD5B'))
def ops_rotated_iou_binary_case5(input_binary_data=None, output_binary_data=None):
    output = rotated_iou_forward_func(Tensor(input_binary_data[0]), Tensor(input_binary_data[1]),
                                      trans=False, mode=0, is_cross=False, v_threshold=0.0, e_threshold=0.0)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((3, 8, 5), np.float32), ((3, 9, 5), np.float32)],
                                output_info=[((3, 8, 9), np.float32)],
                                extra_info='SD5B'))
def ops_rotated_iou_binary_case6(input_binary_data=None, output_binary_data=None):
    output = rotated_iou_forward_func(Tensor(input_binary_data[0], ms.bfloat16), Tensor(input_binary_data[1]),
                                      trans=False, mode=0, is_cross=True, v_threshold=0.0, e_threshold=0.0)
    output = output.astype(ms.float32)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)


@pytest.mark.parametrize('context_mode', [ms.PYNATIVE_MODE])
def test_ops_rotated_iou_binary_cases(context_mode):
    """
    Feature: Ops
    Description: test op rotated_iou pynative
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    ops_rotated_iou_binary_case1()
    ops_rotated_iou_binary_case2()
    ops_rotated_iou_binary_case3()
    ops_rotated_iou_binary_case4()
    ops_rotated_iou_binary_case5()
    ops_rotated_iou_binary_case6()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=4.0)
def test_ops_rotated_iou_binary_cases_GE():
    """
    Feature: Ops
    Description: test op rotated_iou GE
    Expectation: expect correct result.
    """
    context.set_context(mode=ms.GRAPH_MODE, jit_level='O2', device_target="Ascend")

    ops_rotated_iou_binary_case1()
    ops_rotated_iou_binary_case2()
    ops_rotated_iou_binary_case3()
    ops_rotated_iou_binary_case4()
    ops_rotated_iou_binary_case5()
    ops_rotated_iou_binary_case6()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=4.0)
def test_ops_round_iou_dynamic_shape():
    """
    Feature: Ops
    Description: test op rotated_iou dynamic shape
    Expectation: expect correct result.
    """
    boxes1 = generate_random_input((4, 3, 5), np.float32)
    query_boxes1 = generate_random_input((4, 6, 5), np.float32)
    boxes2 = generate_random_input((3, 3, 5), np.float32)
    query_boxes2 = generate_random_input((3, 4, 5), np.float32)
    TEST_OP(rotated_iou_forward_func, [[Tensor(boxes1), Tensor(query_boxes1), False, 0, True, 0.0, 0.0],
                                       [Tensor(boxes2), Tensor(query_boxes2), False, 0, True, 0.0, 0.0]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_grad': True})

    boxes1 = generate_random_input((8, 11, 5), np.float32)
    query_boxes1 = generate_random_input((8, 12, 5), np.float32)
    boxes2 = generate_random_input((11, 15, 5), np.float32)
    query_boxes2 = generate_random_input((11, 18, 5), np.float32)
    TEST_OP(rotated_iou_forward_func, [[Tensor(boxes1), Tensor(query_boxes1), False, 0, True, 0.0, 0.0],
                                       [Tensor(boxes2), Tensor(query_boxes2), False, 0, True, 0.0, 0.0]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_grad': True})
