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
from mindspore import Tensor, mint

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def normalize_forward_func(input_x, p=2.0, dim=1, eps=1e-12):
    return mint.nn.functional.normalize(input_x, p, dim, eps)

def normalize_backward_func(input_x, p=2.0, dim=1, eps=1e-12):
    return ms.grad(normalize_forward_func, (0))(input_x, p, dim, eps)

def normalize_forward_dyn_func(input_x, eps=1e-12):
    return mint.nn.functional.normalize(input_x, p=2.0, dim=1, eps=eps)

@ops_binary_cases(OpsBinaryCase(input_info=[((4, 7), np.float32)],
                                output_info=[((4, 7), np.float32), ((4, 7), np.float32)],
                                extra_info='SD5B'))
def mint_nn_binary_case1(input_binary_data=None, output_binary_data=None):
    output = normalize_forward_func(Tensor(input_binary_data[0]))
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    output = normalize_backward_func(Tensor(input_binary_data[0]))
    assert np.allclose(output.asnumpy(), output_binary_data[1], 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((4, 7, 5), np.float32)],
                                output_info=[((4, 7, 5), np.float32), ((4, 7, 5), np.float32)],
                                extra_info='SD5B'))
def mint_nn_binary_case2(input_binary_data=None, output_binary_data=None):
    output = normalize_forward_func(Tensor(input_binary_data[0]))
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    output = normalize_backward_func(Tensor(input_binary_data[0]))
    assert np.allclose(output.asnumpy(), output_binary_data[1], 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((4, 7, 5, 6), np.float32)],
                                output_info=[((4, 7, 5, 6), np.float32), ((4, 7, 5, 6), np.float32)],
                                extra_info='SD5B'))
def mint_nn_binary_case3(input_binary_data=None, output_binary_data=None):
    output = normalize_forward_func(Tensor(input_binary_data[0]))
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    output = normalize_backward_func(Tensor(input_binary_data[0]))
    assert np.allclose(output.asnumpy(), output_binary_data[1], 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((4, 7, 5, 6, 8), np.float32)],
                                output_info=[((4, 7, 5, 6, 8), np.float32), ((4, 7, 5, 6, 8), np.float32)],
                                extra_info='SD5B'))
def mint_nn_binary_case4(input_binary_data=None, output_binary_data=None):
    output = normalize_forward_func(Tensor(input_binary_data[0]))
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    output = normalize_backward_func(Tensor(input_binary_data[0]))
    assert np.allclose(output.asnumpy(), output_binary_data[1], 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((4, 7, 5), np.float32)],
                                output_info=[((4, 7, 5), np.float32), ((4, 7, 5), np.float32)],
                                extra_info='SD5B'))
def mint_nn_binary_case5(input_binary_data=None, output_binary_data=None):
    output = normalize_forward_func(Tensor(input_binary_data[0]), dim=0)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    output = normalize_backward_func(Tensor(input_binary_data[0]), dim=0)
    assert np.allclose(output.asnumpy(), output_binary_data[1], 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((4, 7, 5), np.float32)],
                                output_info=[((4, 7, 5), np.float32), ((4, 7, 5), np.float32)],
                                extra_info='SD5B'))
def mint_nn_binary_case6(input_binary_data=None, output_binary_data=None):
    output = normalize_forward_func(Tensor(input_binary_data[0]), dim=0, eps=1)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    output = normalize_backward_func(Tensor(input_binary_data[0]), dim=0, eps=1)
    assert np.allclose(output.asnumpy(), output_binary_data[1], 1e-04, 1e-04)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ops_rotated_iou_binary_cases(context_mode):
    """
    Feature: Ops
    Description: test op rotated_iou pynative
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    mint_nn_binary_case1()
    mint_nn_binary_case2()
    mint_nn_binary_case3()
    mint_nn_binary_case4()
    mint_nn_binary_case5()
    mint_nn_binary_case6()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=4.0)
def test_ops_normalize_dynamic_shape():
    """
    Feature: Ops
    Description: test op rotated_iou dynamic shape
    Expectation: expect correct result.
    """
    input_x = generate_random_input((4, 3, 5), np.float32)
    input_x1 = generate_random_input((2, 6, 8, 9), np.float32)
    TEST_OP(normalize_forward_dyn_func,
            [[Tensor(input_x), 1e-8], [Tensor(input_x1), 1e-6]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ScalarTensor'])
