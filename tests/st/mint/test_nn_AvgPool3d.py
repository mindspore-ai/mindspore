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
from mindspore import Tensor
from mindspore import context, mint
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase
from tests.mark_utils import arg_mark


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert True


@test_utils.run_with_cell
def avg_pool3d_forward_func(input_x, kernel_size, stride=None, padding=0,
                            ceil_mode=False, count_include_pad=True, divisor_override=None):
    return mint.nn.functional.avg_pool3d(input_x, kernel_size, stride, padding,
                                         ceil_mode, count_include_pad, divisor_override)


@test_utils.run_with_cell
def avg_pool3d_backward_func(input_x, kernel_size, stride=None, padding=0, ceil_mode=False,
                             count_include_pad=True, divisor_override=None):
    return ms.grad(avg_pool3d_forward_func, (0,))(input_x, kernel_size, stride, padding,
                                                  ceil_mode, count_include_pad, divisor_override)


def mint_avg_pool3d_binary_compare(input_binary_data, output_binary_data, loss, mode):
    if mode == "pynative":
        context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_x = Tensor(input_binary_data[0])

    output = avg_pool3d_forward_func(input_x, 3, 3, 1)
    allclose_nparray(output.asnumpy(), output_binary_data[0], loss, loss)

    grad = avg_pool3d_backward_func(input_x, 3, 3, 1)
    allclose_nparray(grad.asnumpy(), output_binary_data[1], loss, loss)


@ops_binary_cases(
    OpsBinaryCase(
        input_info=[
            ((2, 5, 10, 10, 10), np.float32),
        ],
        output_info=[
            ((2, 5, 4, 4, 4), np.float32),
            ((2, 5, 10, 10, 10), np.float32),
        ],
        extra_info="auto_drive",
    )
)
def mint_nn_avg_pool3d_binary_case1(input_binary_data=None, output_binary_data=None, loss=1e-04, mode="pynative"):
    mint_avg_pool3d_binary_compare(input_binary_data, output_binary_data, loss, mode)


@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_mint_avg_pool3d_binary_cases(mode):
    """
    Feature: Ops
    Description: test op AvgPool3DExt and AvgPool3DGradExt.
    Expectation: expect correct result.
    """
    mint_nn_avg_pool3d_binary_case1(loss=1e-04, mode=mode)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=20.01
)
def test_mint_avg_pool3d_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op AvgPool3DExt and AvgPool3DGradExt.
    Expectation: expect AvgPool3DExt and AvgPool3DGradExt. result.
    """

    input_case1 = Tensor(np.random.randn(5, 10, 10, 10), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(2, 5, 5, 5, 5), dtype=ms.float32)
    TEST_OP(
        avg_pool3d_forward_func,
        [
            [input_case1, 4, (2, 2, 2), (1,), False, True, 1],
            [input_case2, 2, (1, 1, 1), (1,), True, False, 2],
        ],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['EmptyTensor',
                      'ScalarTensor'],
        case_config={'disable_input_check': True}
    )
