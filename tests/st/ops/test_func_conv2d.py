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
from mindspore import Tensor
from mindspore import ops
from mindspore.mint.nn.functional import conv2d
from tests.st.utils import test_utils
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.device_utils import set_device, get_device


class Net2d(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mint_conv2d = conv2d

    def construct(self, input_x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return self.mint_conv2d(input_x, weight, bias, stride, padding, dilation, groups)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_conv2d_default(mode):
    """
    Feature: mint.nn.functional.conv2d
    Description: Verify the result of conv2d
    Expectation: success
    Note: There is a precision problem on Ascend, #I6PT9L
    """
    ms.set_context(jit_level='O0')
    ms.set_context(mode=mode)
    set_device()
    if get_device() == "Ascend":
        ms.device_context.ascend.op_precision.conv_allow_hf32(False)
    x = Tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                 [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]]],
                [[[18.0, 19.0, 20.0], [21.0, 22.0, 23.0], [24.0, 25.0, 26.0]],
                 [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0], [33.0, 34.0, 35.0]]],
                [[[36.0, 37.0, 38.0], [39.0, 40.0, 41.0], [42.0, 43.0, 44.0]],
                 [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0], [51.0, 52.0, 53.0]]]], ms.float32)
    bias = Tensor([0.7297250824055579, 0.6472988621466479], ms.float32)
    weight = Tensor([[[[-1.090221803810641]], [[-0.044567894776783905]]],
                     [[[0.04005113957734308]], [[0.22892450020231897]]]], ms.float32)
    net = Net2d()
    output = net(x, weight, bias=bias)
    expected = np.array([[[[0.3286, -0.8062, -1.9410],
                           [-3.0758, -4.2105, -5.3453],
                           [-6.4801, -7.6149, -8.7497]],
                          [[2.7076, 2.9766, 3.2456],
                           [3.5145, 3.7835, 4.0525],
                           [4.3215, 4.5904, 4.8594]]],
                         [[[-20.0976, -21.2324, -22.3672],
                           [-23.5020, -24.6368, -25.7715],
                           [-26.9063, -28.0411, -29.1759]],
                          [[7.5492, 7.8182, 8.0871],
                           [8.3561, 8.6251, 8.8941],
                           [9.1630, 9.4320, 9.7010]]],
                         [[[-40.5238, -41.6586, -42.7934],
                           [-43.9282, -45.0630, -46.1978],
                           [-47.3326, -48.4673, -49.6021]],
                          [[12.3907, 12.6597, 12.9287],
                           [13.1977, 13.4666, 13.7356],
                           [14.0046, 14.2736, 14.5425]]]])
    assert np.allclose(output.asnumpy(), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_conv2d_padding_same(mode):
    """
    Feature: mint.nn.functional.conv2d
    Description: Verify the result of conv2d
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.set_context(mode=mode)
    set_device()
    if get_device() == "Ascend":
        ms.device_context.ascend.op_precision.conv_allow_hf32(False)
    x = Tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                 [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]]],
                [[[18.0, 19.0, 20.0], [21.0, 22.0, 23.0], [24.0, 25.0, 26.0]],
                 [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0], [33.0, 34.0, 35.0]]],
                [[[36.0, 37.0, 38.0], [39.0, 40.0, 41.0], [42.0, 43.0, 44.0]],
                 [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0], [51.0, 52.0, 53.0]]]], ms.float32)
    bias = Tensor([0.7297250824055579, 0.6472988621466479], ms.float32)
    weight = Tensor([[[[-1.090221803810641]], [[-0.044567894776783905]]],
                     [[[0.04005113957734308]], [[0.22892450020231897]]]], ms.float32)
    net = Net2d()
    output = net(x, weight, bias=bias, padding="same")
    expected = np.array([[[[0.3286, -0.8062, -1.9410],
                           [-3.0758, -4.2105, -5.3453],
                           [-6.4801, -7.6149, -8.7497]],
                          [[2.7076, 2.9766, 3.2456],
                           [3.5145, 3.7835, 4.0525],
                           [4.3215, 4.5904, 4.8594]]],
                         [[[-20.0976, -21.2324, -22.3672],
                           [-23.5020, -24.6368, -25.7715],
                           [-26.9063, -28.0411, -29.1759]],
                          [[7.5492, 7.8182, 8.0871],
                           [8.3561, 8.6251, 8.8941],
                           [9.1630, 9.4320, 9.7010]]],
                         [[[-40.5238, -41.6586, -42.7934],
                           [-43.9282, -45.0630, -46.1978],
                           [-47.3326, -48.4673, -49.6021]],
                          [[12.3907, 12.6597, 12.9287],
                           [13.1977, 13.4666, 13.7356],
                           [14.0046, 14.2736, 14.5425]]]])
    assert np.allclose(output.asnumpy(), expected, atol=1e-4, rtol=1e-4)


def test_conv2d_with_bf16():
    """
    Feature: The weight init of conv2d with type of bfloat16.
    Description: The weight init of conv 2d is implemented by numpy, test type of bfloat16.
    Expectation: Success.
    """
    set_device()
    if get_device() == "Ascend":
        ms.device_context.ascend.op_precision.conv_allow_hf32(False)
    x = ms.Tensor(np.ones([2, 2, 4, 4]), ms.bfloat16)
    weight = ms.Tensor(np.ones([2, 2, 1, 1]), ms.bfloat16)
    bias = ms.Tensor(np.ones([2]), ms.bfloat16)
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


def test_conv2d_dynamic():
    """
    Feature: mint.nn.functional.conv2d
    Description: dynamic shape and rank
    Expectation: success
    """
    set_device()
    if get_device() == "Ascend":
        ms.device_context.ascend.op_precision.conv_allow_hf32(False)
    x1 = ms.Tensor(np.ones([2, 2, 4, 4]), ms.float16)
    weight1 = ms.Tensor(np.ones([2, 2, 1, 1]), ms.float16)
    x2 = ms.Tensor(np.ones([1, 2, 6, 8]), ms.float16)
    weight2 = ms.Tensor(np.ones([2, 2, 2, 3]), ms.float16)
    bias = ms.Tensor(np.ones([2]), ms.float16)
    stride = 1
    padding = 1
    dilation = 1
    groups = 1
    TEST_OP(conv2d, [[x1, weight1, bias, stride, padding, dilation, groups],
                     [x2, weight2, bias, stride, padding, dilation, groups]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True})


@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_conv2d_backward(context_mode):
    """
    Feature: mint.nn.functional.conv2d.
    Description: test conv2d op backward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    set_device()
    if get_device() == "Ascend":
        ms.device_context.ascend.op_precision.conv_allow_hf32(False)

    net = Net2d()
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    x = Tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                 [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]]],
                [[[18.0, 19.0, 20.0], [21.0, 22.0, 23.0], [24.0, 25.0, 26.0]],
                 [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0], [33.0, 34.0, 35.0]]],
                [[[36.0, 37.0, 38.0], [39.0, 40.0, 41.0], [42.0, 43.0, 44.0]],
                 [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0], [51.0, 52.0, 53.0]]]], ms.float32)
    bias = Tensor([0.7297250824055579, 0.6472988621466479], ms.float32)
    weight = Tensor([[[[-1.090221803810641]], [[-0.044567894776783905]]],
                     [[[0.04005113957734308]], [[0.22892450020231897]]]], ms.float32)

    grad_output = ms.grad(net, (0, 1, 2))(x, weight, bias, stride, padding, dilation, groups)
    expected_x_grad = np.array([[[[-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502]],
                                 [[0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844]]],
                                [[[-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502]],
                                 [[0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844]]],
                                [[[-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502]],
                                 [[0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844]]]])
    expected_weight_grad = np.array([[[[594.]], [[837.]]], [[[594.]], [[837.]]]])
    expected_bias_grad = np.array([27., 27.])
    assert np.allclose(grad_output[0].asnumpy(), expected_x_grad, atol=1e-4, rtol=1e-4)
    assert np.allclose(grad_output[1].asnumpy(), expected_weight_grad, atol=1e-4, rtol=1e-4)
    assert np.allclose(grad_output[2].asnumpy(), expected_bias_grad, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_conv2d_backward_fp16(context_mode):
    """
    Feature: mint.nn.functional.conv2d.
    Description: test conv2d op backward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    set_device()
    if get_device() == "Ascend":
        ms.device_context.ascend.op_precision.conv_allow_hf32(False)

    net = Net2d()
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    x = Tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                 [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]]],
                [[[18.0, 19.0, 20.0], [21.0, 22.0, 23.0], [24.0, 25.0, 26.0]],
                 [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0], [33.0, 34.0, 35.0]]],
                [[[36.0, 37.0, 38.0], [39.0, 40.0, 41.0], [42.0, 43.0, 44.0]],
                 [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0], [51.0, 52.0, 53.0]]]], ms.float16)
    bias = Tensor([0.7297250824055579, 0.6472988621466479], ms.float16)
    weight = Tensor([[[[-1.090221803810641]], [[-0.044567894776783905]]],
                     [[[0.04005113957734308]], [[0.22892450020231897]]]], ms.float16)

    grad_output = ms.grad(net, (0, 1, 2))(x, weight, bias, stride, padding, dilation, groups)
    expected_x_grad = np.array([[[[-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502]],
                                 [[0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844]]],
                                [[[-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502]],
                                 [[0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844]]],
                                [[[-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502],
                                  [-1.0502, -1.0502, -1.0502]],
                                 [[0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844],
                                  [0.1844, 0.1844, 0.1844]]]])
    expected_weight_grad = np.array([[[[594.]], [[837.]]], [[[594.]], [[837.]]]])
    expected_bias_grad = np.array([27., 27.])
    assert np.allclose(grad_output[0].asnumpy(), expected_x_grad, atol=1e-3, rtol=1e-3)
    assert np.allclose(grad_output[1].asnumpy(), expected_weight_grad, atol=1e-3, rtol=1e-3)
    assert np.allclose(grad_output[2].asnumpy(), expected_bias_grad, atol=1e-3, rtol=1e-3)



@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_conv2d_vmap(context_mode):
    """
    Feature: mint.nn.functional.conv2d.
    Description: test conv2d op vmap.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.set_context(mode=context_mode)
    set_device()
    if get_device() == "Ascend":
        ms.device_context.ascend.op_precision.conv_allow_hf32(False)
    x = Tensor(np.ones((3, 2, 2, 4, 4)), ms.float32)
    bias = Tensor(np.ones((2,)), ms.float32)
    weight = Tensor(np.ones((2, 2, 2, 2)), ms.float32)
    stride = 1
    padding = "same"
    dilation = 1
    groups = 1
    net = Net2d()
    in_axes = (0, None, None, None, None, None, None)
    net_vmap = ops.vmap(net, in_axes=in_axes, out_axes=0)
    out = net_vmap(x, weight, bias, stride, padding, dilation, groups)
    assert out.asnumpy().shape == (3, 2, 2, 4, 4)


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
