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
"""
Test cases for conv2d functional operator.
"""
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn
from mindspore import ops
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase
from tests.device_utils import set_device, get_device


class Net2d(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mint_conv2d = ms.mint.nn.functional.conv2d

    def construct(self, input_x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return self.mint_conv2d(input_x, weight, bias, stride, padding, dilation, groups)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', mem_peak=4.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_conv2d_with_mixed_precision(mode):
    """
    Feature: conv2d with mixed precision (bfloat16 input, float32 weight/bias).
    Description: Test conv2d when input is bfloat16 and weight/bias are float32.
    Expectation: Success.
    """
    ms.context.set_context(mode=mode)
    ms.device_context.ascend.op_precision.conv_allow_hf32(False)
    x = ms.Tensor(np.ones([2, 2, 4, 4]), ms.bfloat16)
    weight = ms.Tensor(np.ones([2, 2, 1, 1]), ms.float32)
    bias = ms.Tensor(np.ones([2]), ms.float32)
    net = Net2d()
    output = net(x, weight, bias=bias)
    expected = np.array([[[3., 3., 3., 3.],
                          [3., 3., 3., 3.],
                          [3., 3., 3., 3.],
                          [3., 3., 3., 3.]],
                         [[3., 3., 3., 3.],
                          [3., 3., 3., 3.],
                          [3., 3., 3., 3.],
                          [3., 3., 3., 3.]]])
    cpu_cast = ops.Cast().set_device("CPU")
    output = cpu_cast(output, ms.float32)
    assert np.allclose(output.asnumpy(), expected)


def ops_conv2d_binary_compare(input_binary_data, output_binary_data, stride, padding, dilation, groups):
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
    def conv2d_binary_backward_func(inputx, weight, bias, stride, padding, dilation, groups):
        if bias is not None:
            grad_op = ms.grad(Net2d(), (0, 1, 2))
        else:
            grad_op = ms.grad(Net2d(), (0, 1))
        return grad_op(inputx, weight, bias, stride, padding, dilation, groups)

    inputx = ms.Tensor(input_binary_data[0])
    weight = ms.Tensor(input_binary_data[1])
    bias = None
    if len(input_binary_data) == 3:
        bias = ms.Tensor(input_binary_data[2])

    output = Net2d()(inputx, weight, bias, stride, padding, dilation, groups)
    allclose_nparray(output.asnumpy(), output_binary_data[0], 6e-03, 6e-03)
    output = conv2d_binary_backward_func(inputx, weight, bias, stride, padding, dilation, groups)
    allclose_nparray(output[0].asnumpy(), output_binary_data[1], 6e-03, 6e-03)
    allclose_nparray(output[1].asnumpy(), output_binary_data[2], 6e-03, 6e-03)
    if len(output_binary_data) == 4:
        allclose_nparray(output[2].asnumpy(), output_binary_data[3], 6e-03, 6e-03)


@ops_binary_cases(OpsBinaryCase(input_info=[((6, 256, 44, 80), np.float32), ((512, 256, 2, 2), np.float32),
                                            ((512,), np.float32)],
                                output_info=[((6, 512, 22, 40), np.float32), ((6, 256, 44, 80), np.float32),
                                             ((512, 256, 2, 2), np.float32), ((512,), np.float32)],
                                extra_info='auto_drive'))
def ops_conv2d_binary_case1(input_binary_data=None, output_binary_data=None):
    ops_conv2d_binary_compare(input_binary_data, output_binary_data, (2, 2), (0, 0), (1, 1), 1)


@ops_binary_cases(OpsBinaryCase(input_info=[((12, 128, 80, 64), np.float32), ((128, 128, 1, 1), np.float32)],
                                output_info=[((12, 128, 40, 32), np.float32), ((12, 128, 80, 64), np.float32),
                                             ((128, 128, 1, 1), np.float32)],
                                extra_info='auto_drive'))
def ops_conv2d_binary_case2(input_binary_data=None, output_binary_data=None):
    ops_conv2d_binary_compare(input_binary_data, output_binary_data, (2, 2), (0, 0), (1, 1), 1)


@ops_binary_cases(OpsBinaryCase(input_info=[((4, 64, 288, 64), np.float32), ((128, 64, 4, 4), np.float32),
                                            ((128,), np.float32)],
                                output_info=[((4, 128, 72, 16), np.float32), ((4, 64, 288, 64), np.float32),
                                             ((128, 64, 4, 4), np.float32), ((128,), np.float32)],
                                extra_info='auto_drive'))
def ops_conv2d_binary_case3(input_binary_data=None, output_binary_data=None):
    ops_conv2d_binary_compare(input_binary_data, output_binary_data, (4, 4), (0, 0), (1, 1), 1)


@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_conv2d_binary_cases(context_mode):
    """
    Feature: mint.nn.functional.conv2d.
    Description: test conv2d op with binary data.
    Expectation: expect correct result.
    """
    ms.set_context(mode=context_mode, jit_level='O0')
    set_device()
    if get_device() == "Ascend":
        ms.device_context.ascend.op_precision.conv_allow_hf32(False)

    ops_conv2d_binary_case1()
    ops_conv2d_binary_case2()
    ops_conv2d_binary_case3()
