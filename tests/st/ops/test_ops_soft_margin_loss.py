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
from mindspore import ops, Tensor, context

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

@test_utils.run_with_cell
def soft_margin_loss_forward_func(input1, target, reduction):
    return ops.soft_margin_loss(input1, target, reduction)

@test_utils.run_with_cell
def soft_margin_loss_forward_dyn_func(input2, target, reduction):
    return ops.soft_margin_loss(input2, target, reduction)

@test_utils.run_with_cell
def soft_margin_loss_backward_func(input3, target, reduction):
    return ms.grad(soft_margin_loss_forward_func, (0))(input3, target, reduction)

@ops_binary_cases(OpsBinaryCase(input_info=[((3, 4, 2, 5), np.float32), ((3, 4, 2, 5), np.float32)],
                                output_info=[((3, 4, 2, 5), np.float32), ((3, 4, 2, 5), np.float32)],
                                extra_info='SD5B'))
def ops_soft_margin_loss_case1(input_binary_data=None, output_binary_data=None):
    output = soft_margin_loss_forward_func(Tensor(input_binary_data[0]), Tensor(input_binary_data[1]), "none")
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    grad_output = soft_margin_loss_backward_func(Tensor(input_binary_data[0]), Tensor(input_binary_data[1]), "none")
    assert np.allclose(grad_output.asnumpy(), output_binary_data[1], 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((6, 3, 8, 7), np.float32), ((6, 3, 8, 7), np.float32)],
                                output_info=[((), np.float32), ((6, 3, 8, 7), np.float32)],
                                extra_info='SD5B'))
def ops_soft_margin_loss_case2(input_binary_data=None, output_binary_data=None):
    output = soft_margin_loss_forward_func(Tensor(input_binary_data[0]), Tensor(input_binary_data[1]), "sum")
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    grad_output = soft_margin_loss_backward_func(Tensor(input_binary_data[0]), Tensor(input_binary_data[1]), "sum")
    assert np.allclose(grad_output.asnumpy(), output_binary_data[1], 1e-04, 1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_soft_margin_loss_binary_cases(context_mode):
    """
    Feature: Ops
    Description: test op rotated_iou pynative
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    ops_soft_margin_loss_case1()
    ops_soft_margin_loss_case2()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_soft_margin_loss_dynamic_shape(reduction):
    """
    Feature: Ops
    Description: test op rotated_iou dynamic shape
    Expectation: expect correct result.
    """
    input1 = ms.Tensor(np.random.rand(7, 8, 9).astype(np.float32))
    target1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))

    input2 = ms.Tensor(np.random.rand(9, 8).astype(np.float32))
    target2 = ms.Tensor(generate_random_input((9, 8), np.float32))

    TEST_OP(soft_margin_loss_forward_dyn_func, [[input1, target1, reduction], [input2, target2, reduction]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True})

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=4.0)
def test_ops_soft_margin_loss_binary_cases_GE():
    """
    Feature: Ops
    Description: test op rotated_iou GE
    Expectation: expect correct result.
    """
    context.set_context(mode=ms.GRAPH_MODE, jit_level='O2', device_target="Ascend")

    ops_soft_margin_loss_case1()
    ops_soft_margin_loss_case2()
