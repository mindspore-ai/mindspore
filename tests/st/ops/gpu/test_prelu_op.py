# Copyright 2021 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.common import dtype as mstype


class PReLUOpNet(nn.Cell):
    def __init__(self):
        super(PReLUOpNet, self).__init__()
        self.prelu = P.PReLU()

    def construct(self, x, weight):
        return self.prelu(x, weight)


class PReLUOpGradNet(nn.Cell):
    def __init__(self, net):
        super(PReLUOpGradNet, self).__init__()
        self.forward = net
        self.grad = C.GradOperation(get_all=True, sens_param=False)

    def construct(self, x, weight):
        return self.grad(self.forward)(x, weight)


def judge_result_correct(result, expect):
    result = result.asnumpy()
    expect = expect.asnumpy()
    assert result.dtype == expect.dtype
    assert result.shape == expect.shape
    assert np.allclose(result, expect, rtol=1.e-2)


def prelu_test(x, weight, expect_forward, expect_dx, expect_dw):
    prelu_forward = PReLUOpNet()
    prelu_backward = PReLUOpGradNet(prelu_forward)
    forward_output = prelu_forward(x, weight)
    judge_result_correct(forward_output, expect_forward)

    backward_output = prelu_backward(x, weight)
    assert len(backward_output) == 2
    judge_result_correct(backward_output[0], expect_dx)
    judge_result_correct(backward_output[1], expect_dw)


context.set_context(device_target="GPU", mode=context.GRAPH_MODE)
dtypes = [mstype.float16, mstype.float32]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_prelu_single_weight():
    x = np.arange(-10, 26).reshape((2, 3, 2, 3)) * 0.7
    weight = np.array([0.6])
    expect_forward = np.where(x >= 0, x, weight * x)
    expect_dx = np.where(x > 0, 1, weight)
    expect_dw = np.sum(np.where(x >= 0, 0, x)).reshape((1,))

    for dtype in dtypes:
        x = Tensor(x, dtype)
        weight = Tensor(weight, dtype)
        expect_forward = Tensor(expect_forward, dtype)
        expect_dx = Tensor(expect_dx, dtype)
        expect_dw = Tensor(expect_dw, dtype)
        prelu_test(x, weight, expect_forward, expect_dx, expect_dw)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_prelu_multiple_weight():
    x = np.arange(-10, 26).reshape((2, 3, 2, 3)) * 0.6
    weight = np.array([0.2, 0.3, 0.4])
    expect_forward = np.array([[[[-1.20, -1.08, -0.96],
                                 [-0.84, -0.72, -0.60]],
                                [[-0.72, -0.54, -0.36],
                                 [-0.18, 0.00, 0.60]],
                                [[1.20, 1.80, 2.40],
                                 [3.00, 3.60, 4.20]]],
                               [[[4.80, 5.40, 6.00],
                                 [6.60, 7.20, 7.80]],
                                [[8.40, 9.00, 9.60],
                                 [10.20, 10.80, 11.40]],
                                [[12.00, 12.60, 13.20],
                                 [13.80, 14.40, 15.00]]]])
    expect_dx = np.array([[[[0.2, 0.2, 0.2],
                            [0.2, 0.2, 0.2]],
                           [[0.3, 0.3, 0.3],
                            [0.3, 0.3, 1.0]],
                           [[1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0]]],
                          [[[1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0]],
                           [[1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0]],
                           [[1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0]]]])
    expect_dw = np.array([-27.0, -6.0, 0.0])

    for dtype in dtypes:
        x = Tensor(x, dtype)
        weight = Tensor(weight, dtype)
        expect_forward = Tensor(expect_forward, dtype)
        expect_dx = Tensor(expect_dx, dtype)
        expect_dw = Tensor(expect_dw, dtype)
        prelu_test(x, weight, expect_forward, expect_dx, expect_dw)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_prelu_single_weight_0_D():
    x = np.array(-0.8)
    weight = np.array([0.6])
    expect_forward = np.array(-0.48)
    expect_dx = np.array(0.6)
    expect_dw = np.array([-0.8])

    for dtype in dtypes:
        x = Tensor(x, dtype)
        weight = Tensor(weight, dtype)
        expect_forward = Tensor(expect_forward, dtype)
        expect_dx = Tensor(expect_dx, dtype)
        expect_dw = Tensor(expect_dw, dtype)
        prelu_test(x, weight, expect_forward, expect_dx, expect_dw)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_prelu_single_weight_1_D():
    x = np.arange(-10, 26).reshape((36,)) * 0.7
    weight = np.array([0.6])
    expect_forward = np.where(x >= 0, x, weight * x)
    expect_dx = np.where(x > 0, 1, weight)
    expect_dw = np.sum(np.where(x >= 0, 0, x)).reshape((1,))

    for dtype in dtypes:
        x = Tensor(x, dtype)
        weight = Tensor(weight, dtype)
        expect_forward = Tensor(expect_forward, dtype)
        expect_dx = Tensor(expect_dx, dtype)
        expect_dw = Tensor(expect_dw, dtype)
        prelu_test(x, weight, expect_forward, expect_dx, expect_dw)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_prelu_single_weight_2_D():
    x = np.arange(-10, 26).reshape((4, 9)) * 0.7
    weight = np.array([0.6])
    expect_forward = np.where(x >= 0, x, weight * x)
    expect_dx = np.where(x > 0, 1, weight)
    expect_dw = np.sum(np.where(x >= 0, 0, x)).reshape((1,))

    for dtype in dtypes:
        x = Tensor(x, dtype)
        weight = Tensor(weight, dtype)
        expect_forward = Tensor(expect_forward, dtype)
        expect_dx = Tensor(expect_dx, dtype)
        expect_dw = Tensor(expect_dw, dtype)
        prelu_test(x, weight, expect_forward, expect_dx, expect_dw)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_prelu_multiple_weight_2_D():
    x = np.arange(-6, 6).reshape((3, 4)) * 0.6
    weight = np.array([0.2, 0.4, 0.7, 0.9])
    expect_forward = np.array([[-0.72, -1.20, -1.68, -1.62],
                               [-0.24, -0.24, 0.00, 0.60],
                               [1.20, 1.80, 2.40, 3.00]])
    expect_dx = np.array([[0.2, 0.4, 0.7, 0.9],
                          [0.2, 0.4, 0.7, 1.0],
                          [1.0, 1.0, 1.0, 1.0]])
    expect_dw = np.array([-4.8, -3.6, -2.4, -1.8])

    for dtype in dtypes:
        x = Tensor(x, dtype)
        weight = Tensor(weight, dtype)
        expect_forward = Tensor(expect_forward, dtype)
        expect_dx = Tensor(expect_dx, dtype)
        expect_dw = Tensor(expect_dw, dtype)
        prelu_test(x, weight, expect_forward, expect_dx, expect_dw)
