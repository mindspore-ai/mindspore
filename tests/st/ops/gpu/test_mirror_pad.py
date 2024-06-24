# Copyright 2020 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import pytest
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.context as context

from mindspore import Tensor
from mindspore.ops.composite import GradOperation


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_mirror_pad():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    test1_arr_in = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    test_1_paddings = ((0, 0), (0, 0), (1, 1), (2, 2))
    test1_arr_exp = [[[[6, 5, 4, 5, 6, 5, 4], [3, 2, 1, 2, 3, 2, 1], [6, 5, 4, 5, 6, 5, 4],
                       [9, 8, 7, 8, 9, 8, 7], [6, 5, 4, 5, 6, 5, 4]]]]

    test2_arr_in = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    test_2_paddings = ((0, 0), (0, 0), (1, 1), (2, 2))
    test2_arr_exp = [[[[2, 1, 1, 2, 3, 3, 2], [2, 1, 1, 2, 3, 3, 2], [5, 4, 4, 5, 6, 6, 5],
                       [8, 7, 7, 8, 9, 9, 8], [8, 7, 7, 8, 9, 9, 8]]]]

    reflect_op = nn.Pad(mode='REFLECT', paddings=test_1_paddings)
    symm_op = nn.Pad(mode='SYMMETRIC', paddings=test_2_paddings)

    x_test_1 = Tensor(np.array(test1_arr_in), dtype=mindspore.float32)
    x_test_2 = Tensor(np.array(test2_arr_in), dtype=mindspore.float32)

    y_test_1 = reflect_op(x_test_1).asnumpy()
    y_test_2 = symm_op(x_test_2).asnumpy()

    print(np.array(test1_arr_in))
    print(y_test_1)

    np.testing.assert_equal(np.array(test1_arr_exp), y_test_1)
    np.testing.assert_equal(np.array(test2_arr_exp), y_test_2)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True)
        self.network = network

    def construct(self, input_):
        return self.grad(self.network)(input_)


class Net(nn.Cell):
    def __init__(self, pads, mode_):
        super(Net, self).__init__()
        self.pad = nn.Pad(mode=mode_, paddings=pads)

    def construct(self, x):
        return self.pad(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_mirror_pad_backprop():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_arr_in = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]  # size -> 3*3
    test_arr_in = Tensor(test_arr_in, dtype=mindspore.float32)
    expected_dx = np.array([[[[0.2, 0.2, 0.1],
                              [0.4, 0.4, 0.2],
                              [0.2, 0.2, 0.1]]]])
    net = Grad(Net(((0, 0), (0, 0), (1, 0), (0, 2)), "REFLECT"))
    dx = net(test_arr_in)
    dx = dx[0].asnumpy()
    dx = dx * 0.1
    np.testing.assert_array_almost_equal(dx, expected_dx)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_mirror_pad_fwd_back_4d_int32_reflect():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # set constants
    shape = (2, 3, 3, 5)
    pads = ((1, 0), (2, 0), (1, 2), (3, 4))
    total_val = np.prod(shape)
    test_arr_np = np.arange(total_val).reshape(shape) + 1
    test_arr_ms = Tensor(test_arr_np, dtype=mindspore.int32)
    # fwd_pass_check
    op = nn.Pad(mode="REFLECT", paddings=pads)
    expected_np_result = np.pad(test_arr_np, pads, 'reflect')
    obtained_ms_res = op(test_arr_ms).asnumpy()
    np.testing.assert_array_equal(expected_np_result, obtained_ms_res)
    # backwards pass check
    grad_net = Grad(Net(pads, "REFLECT"))
    dx_value_obtained = grad_net(test_arr_ms)[0].asnumpy()
    dx_value_expected = np.array([[[[4, 6, 6, 6, 2],
                                    [6, 9, 9, 9, 3],
                                    [2, 3, 3, 3, 1]],
                                   [[8, 12, 12, 12, 4],
                                    [12, 18, 18, 18, 6],
                                    [4, 6, 6, 6, 2]],
                                   [[8, 12, 12, 12, 4],
                                    [12, 18, 18, 18, 6],
                                    [4, 6, 6, 6, 2]]],
                                  [[[8, 12, 12, 12, 4],
                                    [12, 18, 18, 18, 6],
                                    [4, 6, 6, 6, 2]],
                                   [[16, 24, 24, 24, 8],
                                    [24, 36, 36, 36, 12],
                                    [8, 12, 12, 12, 4]],
                                   [[16, 24, 24, 24, 8],
                                    [24, 36, 36, 36, 12],
                                    [8, 12, 12, 12, 4]]]], dtype=np.int32)
    np.testing.assert_array_equal(dx_value_expected, dx_value_obtained)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_mirror_pad_fwd_back_4d_int32_symm():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # set constants
    shape = (2, 3, 3, 5)
    pads = ((1, 0), (2, 0), (1, 2), (3, 4))
    total_val = np.prod(shape)
    test_arr_np = np.arange(total_val).reshape(shape) + 1
    test_arr_ms = Tensor(test_arr_np, dtype=mindspore.int32)
    # fwd_pass_check
    op = nn.Pad(mode="SYMMETRIC", paddings=pads)
    expected_np_result = np.pad(test_arr_np, pads, 'symmetric')
    obtained_ms_res = op(test_arr_ms).asnumpy()
    np.testing.assert_array_equal(expected_np_result, obtained_ms_res)
    # backwards pass check
    grad_net = Grad(Net(pads, "SYMMETRIC"))
    dx_value_obtained = grad_net(test_arr_ms)[0].asnumpy()
    dx_value_expected = np.array([[[[16, 24, 24, 16, 16],
                                    [16, 24, 24, 16, 16],
                                    [16, 24, 24, 16, 16]],
                                   [[16, 24, 24, 16, 16],
                                    [16, 24, 24, 16, 16],
                                    [16, 24, 24, 16, 16]],
                                   [[8, 12, 12, 8, 8],
                                    [8, 12, 12, 8, 8],
                                    [8, 12, 12, 8, 8]]],
                                  [[[8, 12, 12, 8, 8],
                                    [8, 12, 12, 8, 8],
                                    [8, 12, 12, 8, 8]],
                                   [[8, 12, 12, 8, 8],
                                    [8, 12, 12, 8, 8],
                                    [8, 12, 12, 8, 8]],
                                   [[4, 6, 6, 4, 4],
                                    [4, 6, 6, 4, 4],
                                    [4, 6, 6, 4, 4]]]], dtype=np.int32)
    np.testing.assert_array_equal(dx_value_expected, dx_value_obtained)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_mirror_pad_dynamic():
    """
    Feature: test dynamic shape of mirror_pad
    Description: test dynamic shape of mirror_pad
    Expectation: same to none dynamic
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    test1_arr_in = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    test_1_paddings = ((0, 0), (0, 0), (1, 1), (2, 2))
    test1_arr_exp = [[[[6, 5, 4, 5, 6, 5, 4], [3, 2, 1, 2, 3, 2, 1], [6, 5, 4, 5, 6, 5, 4],
                       [9, 8, 7, 8, 9, 8, 7], [6, 5, 4, 5, 6, 5, 4]]]]

    test2_arr_in = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    test_2_paddings = ((0, 0), (0, 0), (1, 1), (2, 2))
    test2_arr_exp = [[[[2, 1, 1, 2, 3, 3, 2], [2, 1, 1, 2, 3, 3, 2], [5, 4, 4, 5, 6, 6, 5],
                       [8, 7, 7, 8, 9, 9, 8], [8, 7, 7, 8, 9, 9, 8]]]]

    reflect_op = nn.Pad(mode='REFLECT', paddings=test_1_paddings)
    symm_op = nn.Pad(mode='SYMMETRIC', paddings=test_2_paddings)

    x_test_1 = Tensor(np.array(test1_arr_in), dtype=mindspore.float32)
    x_test_2 = Tensor(np.array(test2_arr_in), dtype=mindspore.float32)
    in_dyn = Tensor(shape=[1, 1, 3, None], dtype=mindspore.float32)
    reflect_op.set_inputs(in_dyn)
    symm_op.set_inputs(in_dyn)
    y_test_1 = reflect_op(x_test_1).asnumpy()
    y_test_2 = symm_op(x_test_2).asnumpy()

    print(np.array(test1_arr_in))
    print(y_test_1)

    np.testing.assert_equal(np.array(test1_arr_exp), y_test_1)
    np.testing.assert_equal(np.array(test2_arr_exp), y_test_2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_mirror_pad_grad_dynamic_shape():
    """
    Feature: dynamic shape support of MirrorPadGrad.
    Description: input Tensor with dynamic shape.
    Expectation: output shape coincide with expect_shape.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_arr_in_dyn = Tensor(shape=[1, 1, None, 3], dtype=mindspore.float32)
    test_arr_in = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    test_arr_in = Tensor(test_arr_in, dtype=mindspore.float32)
    expected_shape = (1, 1, 3, 3)
    net = Grad(Net(((0, 0), (0, 0), (1, 0), (0, 2)), "REFLECT"))
    net.set_inputs(test_arr_in_dyn)
    dx = net(test_arr_in)
    assert dx[0].asnumpy().shape == expected_shape
