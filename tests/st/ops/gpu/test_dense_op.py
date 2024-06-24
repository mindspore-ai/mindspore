# Copyright 2019-2023 Huawei Technologies Co., Ltd
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
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.operations import nn_ops as ops


class BiasAdd(nn.Cell):
    def __init__(self):
        super(BiasAdd, self).__init__()
        self.ba = P.BiasAdd()

    def construct(self, x, b):
        return self.ba(x, b)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_biasadd():
    x = Tensor(np.array([[0.1, 0.2, 0.3, 0.4],
                         [0.5, 0.6, 0.7, 0.8],
                         [0.9, 1.0, 1.1, 1.2]]).astype(np.float32))
    b = Tensor(np.array([0.1, 0.2, 0.3, 0.4]).astype(np.float32))
    expect = np.array([[0.2, 0.4, 0.6, 0.8],
                       [0.6, 0.8, 1.0, 1.2],
                       [1.0, 1.2, 1.4, 1.6]])
    error = np.ones(shape=[3, 4]) * 1.0e-6

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    ba = BiasAdd()
    result = ba(x, b)
    diff = result.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ba = BiasAdd()
    result = ba(x, b)
    diff = result.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


class GradData(nn.Cell):
    def __init__(self, network):
        super(GradData, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, inputs, output_grad):
        return self.grad(self.network)(inputs, output_grad)


class GradWeight(nn.Cell):
    def __init__(self, network):
        super(GradWeight, self).__init__()
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)

    def construct(self, x, output_grad):
        weights = self.weights
        grads = self.grad(self.network, weights)(x, output_grad)
        return grads


class DenseNet(nn.Cell):
    def __init__(self):
        super(DenseNet, self).__init__()
        w = np.array([[0.1, 0.8, 0.1, 0.1],
                      [1, 1, 1, 1]]).astype(np.float32)
        b = np.array([0.3, 0.6]).astype(np.float32)
        self.dense = nn.Dense(4, 2, weight_init=Tensor(w), bias_init=Tensor(b))

    def construct(self, x):
        return self.dense(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dx():
    x = np.array([[0.1, 0.2, 0.3, 0.4],
                  [0.1, 0.2, 0.3, 0.4],
                  [0.1, 0.2, 0.3, 0.4]]).astype(np.float32)
    dy = np.array([[1, 1],
                   [1, 1],
                   [1, 1]]).astype(np.float32)
    dx_expect = np.array([[1.1, 1.8, 1.1, 1.1],
                          [1.1, 1.8, 1.1, 1.1],
                          [1.1, 1.8, 1.1, 1.1]]).astype(np.float32)
    error = np.ones(shape=[3, 4]) * 1.0e-6

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = GradData(DenseNet())
    dx = net(Tensor(x), Tensor(dy))
    diff = dx[0].asnumpy() - dx_expect
    assert np.all(diff < error)
    assert np.all(-diff < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = GradData(DenseNet())
    dx = net(Tensor(x), Tensor(dy))
    diff = dx[0].asnumpy() - dx_expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dx_nd():
    """
    Feature: Dense gpu kernel
    Description: test the rightness of Dense gpu kernel
    Expectation: the output is same as expected result
    """
    x = np.array([[[0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4]],
                  [[0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4]]
                  ]).astype(np.float32)
    dy = np.array([[[1, 1],
                    [1, 1],
                    [1, 1]],
                   [[1, 1],
                    [1, 1],
                    [1, 1]]]).astype(np.float32)
    dx_expect = np.array([[[1.1, 1.8, 1.1, 1.1],
                           [1.1, 1.8, 1.1, 1.1],
                           [1.1, 1.8, 1.1, 1.1]],
                          [[1.1, 1.8, 1.1, 1.1],
                           [1.1, 1.8, 1.1, 1.1],
                           [1.1, 1.8, 1.1, 1.1]]
                          ]).astype(np.float32)
    error = np.ones(shape=[2, 3, 4]) * 1.0e-6

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = GradData(DenseNet())
    dx = net(Tensor(x), Tensor(dy))
    diff = dx[0].asnumpy() - dx_expect
    assert np.all(diff < error)
    assert np.all(-diff < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = GradData(DenseNet())
    dx = net(Tensor(x), Tensor(dy))
    diff = dx[0].asnumpy() - dx_expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dw():
    x = np.array([[0.1, 0.2, 0.3, 0.4],
                  [0.1, 0.2, 0.3, 0.4],
                  [0.1, 0.2, 0.3, 0.4]]).astype(np.float32)
    dy = np.array([[1, 1],
                   [1, 1],
                   [1, 1]]).astype(np.float32)
    dw_expect = np.array([[0.3, 0.6, 0.9, 1.2],
                          [0.3, 0.6, 0.9, 1.2]]).astype(np.float32)
    dw_error = np.ones(shape=[2, 4]) * 1.0e-6
    db_expect = np.array([3, 3]).astype(np.float32)
    db_error = np.ones(shape=[2]) * 1.0e-6

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = GradWeight(DenseNet())
    dw, db = net(Tensor(x), Tensor(dy))
    diff = dw.asnumpy() - dw_expect
    assert np.all(diff < dw_error)
    assert np.all(-diff < dw_error)
    diff = db.asnumpy() - db_expect
    assert np.all(diff < db_error)
    assert np.all(-diff < db_error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = GradWeight(DenseNet())
    dw, db = net(Tensor(x), Tensor(dy))
    diff = dw.asnumpy() - dw_expect
    assert np.all(diff < dw_error)
    assert np.all(-diff < dw_error)
    diff = db.asnumpy() - db_expect
    assert np.all(diff < db_error)
    assert np.all(-diff < db_error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dw_nd():
    """
    Feature: Dense gpu kernel
    Description: test the rightness of Dense gpu kernel
    Expectation: the output is same as expected result
    """
    x = np.array([[[0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4]],
                  [[0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4]]]).astype(np.float32)
    dy = np.array([[[1, 1],
                    [1, 1],
                    [1, 1]],
                   [[1, 1],
                    [1, 1],
                    [1, 1]]]).astype(np.float32)
    dw_expect = 2 * np.array([[0.3, 0.6, 0.9, 1.2],
                              [0.3, 0.6, 0.9, 1.2]]).astype(np.float32)
    dw_error = np.ones(shape=[2, 4]) * 1.0e-6
    db_expect = 2 * np.array([3, 3]).astype(np.float32)
    db_error = np.ones(shape=[2]) * 1.0e-6

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = GradWeight(DenseNet())
    dw, db = net(Tensor(x), Tensor(dy))
    diff = dw.asnumpy() - dw_expect
    assert np.all(diff < dw_error)
    assert np.all(-diff < dw_error)
    diff = db.asnumpy() - db_expect
    assert np.all(diff < db_error)
    assert np.all(-diff < db_error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = GradWeight(DenseNet())
    dw, db = net(Tensor(x), Tensor(dy))
    diff = dw.asnumpy() - dw_expect
    assert np.all(diff < dw_error)
    assert np.all(-diff < dw_error)
    diff = db.asnumpy() - db_expect
    assert np.all(diff < db_error)
    assert np.all(-diff < db_error)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_, bias, dy):
        return self.grad(self.network)(input_, bias, dy)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_biasadd_3d():
    x = Tensor(np.array([[[1, 2, 3, 4, 5, 6, 7, 8],
                          [9, 10, 11, 12, 13, 14, 15, 16],
                          [17, 18, 19, 20, 21, 22, 23, 24],
                          [25, 26, 27, 28, 29, 30, 31, 32]],

                         [[33, 34, 35, 36, 37, 38, 39, 40],
                          [41, 42, 43, 44, 45, 46, 47, 48],
                          [49, 50, 51, 52, 53, 54, 55, 56],
                          [57, 58, 59, 60, 61, 62, 63, 64]],

                         [[65, 66, 67, 68, 69, 70, 71, 72],
                          [73, 74, 75, 76, 77, 78, 79, 80],
                          [81, 82, 83, 84, 85, 86, 87, 88],
                          [89, 90, 91, 92, 93, 94, 95, 96]]]).astype(np.float32))
    b = Tensor(np.array([1, 2, 3, 4]).astype(np.float32))
    dy = Tensor(np.array([[[1, 2, 3, 4, 5, 6, 7, 8],
                           [9, 10, 11, 12, 13, 14, 15, 16],
                           [17, 18, 19, 20, 21, 22, 23, 24],
                           [25, 26, 27, 28, 29, 30, 31, 32]],

                          [[33, 34, 35, 36, 37, 38, 39, 40],
                           [41, 42, 43, 44, 45, 46, 47, 48],
                           [49, 50, 51, 52, 53, 54, 55, 56],
                           [57, 58, 59, 60, 61, 62, 63, 64]],

                          [[65, 66, 67, 68, 69, 70, 71, 72],
                           [73, 74, 75, 76, 77, 78, 79, 80],
                           [81, 82, 83, 84, 85, 86, 87, 88],
                           [89, 90, 91, 92, 93, 94, 95, 96]]]).astype(np.float32))

    expect = np.array([[[2, 3, 4, 5, 6, 7, 8, 9],
                        [11, 12, 13, 14, 15, 16, 17, 18],
                        [20, 21, 22, 23, 24, 25, 26, 27],
                        [29, 30, 31, 32, 33, 34, 35, 36]],

                       [[34, 35, 36, 37, 38, 39, 40, 41],
                        [43, 44, 45, 46, 47, 48, 49, 50],
                        [52, 53, 54, 55, 56, 57, 58, 59],
                        [61, 62, 63, 64, 65, 66, 67, 68]],

                       [[66, 67, 68, 69, 70, 71, 72, 73],
                        [75, 76, 77, 78, 79, 80, 81, 82],
                        [84, 85, 86, 87, 88, 89, 90, 91],
                        [93, 94, 95, 96, 97, 98, 99, 100]]])

    error = np.ones(shape=[3, 4, 8]) * 1.0e-6
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = BiasAdd()
    net.set_grad()
    result = net(x, b)
    diff = result.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)

    net = Grad(net)
    _, result = net(x, b, dy)
    expect = np.array([876., 1068., 1260., 1452.])
    diff = result.asnumpy() - expect
    error = np.ones(shape=[4]) * 1.0e-6
    assert np.all(diff < error)
    assert np.all(-diff < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_biasadd_4d():
    x = Tensor(np.array([[[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]],

                          [[17, 18, 19, 20],
                           [21, 22, 23, 24],
                           [25, 26, 27, 28],
                           [29, 30, 31, 32]],

                          [[33, 34, 35, 36],
                           [37, 38, 39, 40],
                           [41, 42, 43, 44],
                           [45, 46, 47, 48]]],

                         [[[49, 50, 51, 52],
                           [53, 54, 55, 56],
                           [57, 58, 59, 60],
                           [61, 62, 63, 64]],

                          [[65, 66, 67, 68],
                           [69, 70, 71, 72],
                           [73, 74, 75, 76],
                           [77, 78, 79, 80]],

                          [[81, 82, 83, 84],
                           [85, 86, 87, 88],
                           [89, 90, 91, 92],
                           [93, 94, 95, 96]]]]).astype(np.float32))
    b = Tensor(np.array([1, 2, 3]).astype(np.float32))
    dy = Tensor(np.array([[[[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16]],

                           [[17, 18, 19, 20],
                            [21, 22, 23, 24],
                            [25, 26, 27, 28],
                            [29, 30, 31, 32]],

                           [[33, 34, 35, 36],
                            [37, 38, 39, 40],
                            [41, 42, 43, 44],
                            [45, 46, 47, 48]]],

                          [[[49, 50, 51, 52],
                            [53, 54, 55, 56],
                            [57, 58, 59, 60],
                            [61, 62, 63, 64]],

                           [[65, 66, 67, 68],
                            [69, 70, 71, 72],
                            [73, 74, 75, 76],
                            [77, 78, 79, 80]],

                           [[81, 82, 83, 84],
                            [85, 86, 87, 88],
                            [89, 90, 91, 92],
                            [93, 94, 95, 96]]]]).astype(np.float32))

    expect = np.array([[[[2, 3, 4, 5],
                         [6, 7, 8, 9],
                         [10, 11, 12, 13],
                         [14, 15, 16, 17]],

                        [[19, 20, 21, 22],
                         [23, 24, 25, 26],
                         [27, 28, 29, 30],
                         [31, 32, 33, 34]],

                        [[36, 37, 38, 39],
                         [40, 41, 42, 43],
                         [44, 45, 46, 47],
                         [48, 49, 50, 51]]],

                       [[[50, 51, 52, 53],
                         [54, 55, 56, 57],
                         [58, 59, 60, 61],
                         [62, 63, 64, 65]],

                        [[67, 68, 69, 70],
                         [71, 72, 73, 74],
                         [75, 76, 77, 78],
                         [79, 80, 81, 82]],

                        [[84, 85, 86, 87],
                         [88, 89, 90, 91],
                         [92, 93, 94, 95],
                         [96, 97, 98, 99]]]])
    error = np.ones(shape=[2, 3, 4, 4]) * 1.0e-6

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    ba = BiasAdd()
    result = ba(x, b)
    diff = result.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = BiasAdd()
    result = net(x, b)
    diff = result.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)

    net = Grad(net)
    _, result = net(x, b, dy)
    expect = np.array([1040., 1552., 2064.])
    diff = result.asnumpy() - expect
    error = np.ones(shape=[3]) * 1.0e-6
    assert np.all(diff < error)
    assert np.all(-diff < error)


class BiasAddDynamic(nn.Cell):
    def __init__(self):
        super(BiasAddDynamic, self).__init__()
        self.ba = P.BiasAdd()
        self.test_dynamic = inner.GpuConvertToDynamicShape()

    def construct(self, x, b):
        x = self.test_dynamic(x)
        output = self.ba(x, b)
        return output


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bias_add_dynamic_two_inputs():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = BiasAddDynamic()

    x_1 = Tensor(np.array([[0.1, 0.2, 0.3, 0.4],
                           [0.5, 0.6, 0.7, 0.8],
                           [0.9, 1.0, 1.1, 1.2]]).astype(np.float32))
    b_1 = Tensor(np.array([0.1, 0.2, 0.3, 0.4]).astype(np.float32))
    expect_1 = np.array([[0.2, 0.4, 0.6, 0.8],
                         [0.6, 0.8, 1.0, 1.2],
                         [1.0, 1.2, 1.4, 1.6]])
    error_1 = np.ones(shape=[3, 4]) * 1.0e-6
    result_1 = net(x_1, b_1)
    diff_1 = result_1.asnumpy() - expect_1
    assert np.all(diff_1 < error_1)
    assert np.all(-diff_1 < error_1)

    x_2 = Tensor(np.array([[[1, 2, 3, 4, 5, 6, 7, 8],
                            [9, 10, 11, 12, 13, 14, 15, 16],
                            [17, 18, 19, 20, 21, 22, 23, 24],
                            [25, 26, 27, 28, 29, 30, 31, 32]],
                           [[33, 34, 35, 36, 37, 38, 39, 40],
                            [41, 42, 43, 44, 45, 46, 47, 48],
                            [49, 50, 51, 52, 53, 54, 55, 56],
                            [57, 58, 59, 60, 61, 62, 63, 64]],
                           [[65, 66, 67, 68, 69, 70, 71, 72],
                            [73, 74, 75, 76, 77, 78, 79, 80],
                            [81, 82, 83, 84, 85, 86, 87, 88],
                            [89, 90, 91, 92, 93, 94, 95, 96]]]).astype(np.float32))
    b_2 = Tensor(np.array([1, 2, 3, 4]).astype(np.float32))
    expect_2 = np.array([[[2, 3, 4, 5, 6, 7, 8, 9],
                          [11, 12, 13, 14, 15, 16, 17, 18],
                          [20, 21, 22, 23, 24, 25, 26, 27],
                          [29, 30, 31, 32, 33, 34, 35, 36]],
                         [[34, 35, 36, 37, 38, 39, 40, 41],
                          [43, 44, 45, 46, 47, 48, 49, 50],
                          [52, 53, 54, 55, 56, 57, 58, 59],
                          [61, 62, 63, 64, 65, 66, 67, 68]],
                         [[66, 67, 68, 69, 70, 71, 72, 73],
                          [75, 76, 77, 78, 79, 80, 81, 82],
                          [84, 85, 86, 87, 88, 89, 90, 91],
                          [93, 94, 95, 96, 97, 98, 99, 100]]])
    error_2 = np.ones(shape=[3, 4, 8]) * 1.0e-6
    result_2 = net(x_2, b_2)
    diff_2 = result_2.asnumpy() - expect_2
    assert np.all(diff_2 < error_2)
    assert np.all(-diff_2 < error_2)


class DenseOp(nn.Dense):
    def __init__(self):
        w = np.array([[0.1, 0.8, 0.1, 0.1],
                      [1, 1, 1, 1]]).astype(np.float32)
        b = np.array([0.3, 0.6]).astype(np.float32)
        super(DenseOp, self).__init__(4, 2, weight_init=Tensor(w), bias_init=Tensor(b))
        self.dense = ops.Dense()

    def construct(self, x):
        x = self.dense(x, self.weight, self.bias)
        if self.activation_flag:
            x = self.activation(x)
        return x


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dx_op():
    """
    Feature: Dense gpu kernel
    Description: test the rightness of Dense gpu kernel
    Expectation: the output is same as expected result
    """
    x = np.array([[0.1, 0.2, 0.3, 0.4],
                  [0.1, 0.2, 0.3, 0.4],
                  [0.1, 0.2, 0.3, 0.4]]).astype(np.float32)
    dy = np.array([[1, 1],
                   [1, 1],
                   [1, 1]]).astype(np.float32)
    dx_expect = np.array([[1.1, 1.8, 1.1, 1.1],
                          [1.1, 1.8, 1.1, 1.1],
                          [1.1, 1.8, 1.1, 1.1]]).astype(np.float32)
    error = np.ones(shape=[3, 4]) * 1.0e-6

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = GradData(DenseOp())
    dx = net(Tensor(x), Tensor(dy))
    diff = dx[0].asnumpy() - dx_expect
    assert np.all(diff < error)
    assert np.all(-diff < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = GradData(DenseOp())
    dx = net(Tensor(x), Tensor(dy))
    diff = dx[0].asnumpy() - dx_expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dx_nd_op():
    """
    Feature: Dense gpu kernel
    Description: test the rightness of Dense gpu kernel
    Expectation: the output is same as expected result
    """
    x = np.array([[[0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4]],
                  [[0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4]]
                  ]).astype(np.float32)
    dy = np.array([[[1, 1],
                    [1, 1],
                    [1, 1]],
                   [[1, 1],
                    [1, 1],
                    [1, 1]]]).astype(np.float32)
    dx_expect = np.array([[[1.1, 1.8, 1.1, 1.1],
                           [1.1, 1.8, 1.1, 1.1],
                           [1.1, 1.8, 1.1, 1.1]],
                          [[1.1, 1.8, 1.1, 1.1],
                           [1.1, 1.8, 1.1, 1.1],
                           [1.1, 1.8, 1.1, 1.1]]
                          ]).astype(np.float32)
    error = np.ones(shape=[2, 3, 4]) * 1.0e-6

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = GradData(DenseOp())
    dx = net(Tensor(x), Tensor(dy))
    diff = dx[0].asnumpy() - dx_expect
    assert np.all(diff < error)
    assert np.all(-diff < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = GradData(DenseOp())
    dx = net(Tensor(x), Tensor(dy))
    diff = dx[0].asnumpy() - dx_expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dw_op():
    """
    Feature: Dense gpu kernel
    Description: test the rightness of Dense gpu kernel
    Expectation: the output is same as expected result
    """
    x = np.array([[0.1, 0.2, 0.3, 0.4],
                  [0.1, 0.2, 0.3, 0.4],
                  [0.1, 0.2, 0.3, 0.4]]).astype(np.float32)
    dy = np.array([[1, 1],
                   [1, 1],
                   [1, 1]]).astype(np.float32)
    dw_expect = np.array([[0.3, 0.6, 0.9, 1.2],
                          [0.3, 0.6, 0.9, 1.2]]).astype(np.float32)
    dw_error = np.ones(shape=[2, 4]) * 1.0e-6
    db_expect = np.array([3, 3]).astype(np.float32)
    db_error = np.ones(shape=[2]) * 1.0e-6

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = GradWeight(DenseOp())
    dw, db = net(Tensor(x), Tensor(dy))
    diff = dw.asnumpy() - dw_expect
    assert np.all(diff < dw_error)
    assert np.all(-diff < dw_error)
    diff = db.asnumpy() - db_expect
    assert np.all(diff < db_error)
    assert np.all(-diff < db_error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = GradWeight(DenseOp())
    dw, db = net(Tensor(x), Tensor(dy))
    diff = dw.asnumpy() - dw_expect
    assert np.all(diff < dw_error)
    assert np.all(-diff < dw_error)
    diff = db.asnumpy() - db_expect
    assert np.all(diff < db_error)
    assert np.all(-diff < db_error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dw_nd_op():
    """
    Feature: Dense gpu kernel
    Description: test the rightness of Dense gpu kernel
    Expectation: the output is same as expected result
    """
    x = np.array([[[0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4]],
                  [[0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4]]]).astype(np.float32)
    dy = np.array([[[1, 1],
                    [1, 1],
                    [1, 1]],
                   [[1, 1],
                    [1, 1],
                    [1, 1]]]).astype(np.float32)
    dw_expect = 2 * np.array([[0.3, 0.6, 0.9, 1.2],
                              [0.3, 0.6, 0.9, 1.2]]).astype(np.float32)
    dw_error = np.ones(shape=[2, 4]) * 1.0e-6
    db_expect = 2 * np.array([3, 3]).astype(np.float32)
    db_error = np.ones(shape=[2]) * 1.0e-6

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = GradWeight(DenseOp())
    dw, db = net(Tensor(x), Tensor(dy))
    diff = dw.asnumpy() - dw_expect
    assert np.all(diff < dw_error)
    assert np.all(-diff < dw_error)
    diff = db.asnumpy() - db_expect
    assert np.all(diff < db_error)
    assert np.all(-diff < db_error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = GradWeight(DenseOp())
    dw, db = net(Tensor(x), Tensor(dy))
    diff = dw.asnumpy() - dw_expect
    assert np.all(diff < dw_error)
    assert np.all(-diff < dw_error)
    diff = db.asnumpy() - db_expect
    assert np.all(diff < db_error)
    assert np.all(-diff < db_error)


class Dense(nn.Cell):
    def __init__(self):
        super(Dense, self).__init__()
        self.dense = ops.Dense()

    def construct(self, x, w, b):
        return self.dense(x, w, b)


class DenseGrad(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.grad = GradOperation(get_all=True, get_by_list=True, sens_param=True)
        self.params = ParameterTuple(self.network.trainable_params())

    def construct(self, *inputs):
        return self.grad(self.network, self.params)(*inputs)


def test_1d_forward():
    """
    Feature: Test dense 1d.
    Description: Test dense 1d forward for Graph mode.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dtype = np.float32
    error = 1e-3
    x_np = np.array([1, 2, 3]).astype(dtype)
    w_np = np.array([1, 2, 3]).astype(dtype)
    b_np = np.array(7).astype(dtype)
    out_np = np.array(21).astype(dtype)
    x_ms = Tensor(x_np)
    w_ms = Tensor(w_np)
    b_ms = Tensor(b_np)
    net = Dense()
    net.set_train()

    # dynamic shape
    net.set_inputs(
        Tensor(shape=[None for _ in x_ms.shape], dtype=x_ms.dtype),
        Tensor(shape=[None for _ in w_ms.shape], dtype=w_ms.dtype),
        b_ms,
    )
    out_ms = net(x_ms, w_ms, b_ms).asnumpy()
    assert np.abs(out_ms - out_np).mean() < error

    # dynamic rank
    net.set_inputs(
        Tensor(dtype=x_ms.dtype),
        Tensor(dtype=w_ms.dtype),
        b_ms,
    )
    out_ms = net(x_ms, w_ms, b_ms).asnumpy()
    assert np.abs(out_ms - out_np).mean() < error


def test_3d_forward():
    """
    Feature: Test dense 3d forward.
    Description: Test dense 3d forward for Graph mode.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dtype = np.float32
    error = 1e-3
    x_np = np.array([[[1, 2, 3], [4, 5, 6]]]).astype(dtype)
    w_np = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]]).astype(dtype)
    b_np = np.array([19, 20, 21, 22]).astype(dtype)
    out_np = np.array([[69, 88, 107, 126], [141, 187, 233, 279]]).astype(dtype)
    x_ms = Tensor(x_np)
    w_ms = Tensor(w_np)
    b_ms = Tensor(b_np)
    net = Dense()
    net.set_train()

    # dynamic shape
    net.set_inputs(
        Tensor(shape=[None for _ in x_ms.shape], dtype=x_ms.dtype),
        Tensor(shape=[None for _ in w_ms.shape], dtype=w_ms.dtype),
        b_ms,
    )
    out_ms = net(x_ms, w_ms, b_ms).asnumpy()
    assert np.abs(out_ms - out_np).mean() < error

    # dynamic rank
    net.set_inputs(
        Tensor(dtype=x_ms.dtype),
        Tensor(dtype=w_ms.dtype),
        b_ms,
    )
    out_ms = net(x_ms, w_ms, b_ms).asnumpy()
    assert np.abs(out_ms - out_np).mean() < error


def test_1d_backward():
    """
    Feature: Test dense 1d.
    Description: Test dense 1d backward for Graph mode.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dtype = np.float32
    error = 1e-3
    x_np = np.array([1, 2, 3]).astype(dtype)
    w_np = np.array([4, 5, 6]).astype(dtype)
    b_np = np.array(7).astype(dtype)
    dout_np = np.array(8).astype(dtype)
    dx_np = np.array([32, 40, 48]).astype(dtype)
    dw_np = np.array([8, 16, 24]).astype(dtype)
    db_np = np.array(8).astype(dtype)
    x_ms = Tensor(x_np)
    w_ms = Tensor(w_np)
    b_ms = Tensor(b_np)
    dout_ms = Tensor(dout_np)
    net = Dense()
    grad_net = DenseGrad(net)
    grad_net.set_train()

    # dynamic shape
    grad_net.set_inputs(
        Tensor(dtype=x_ms.dtype),
        Tensor(dtype=w_ms.dtype),
        b_ms,
        dout_ms,
    )
    input_grad = grad_net(x_ms, w_ms, b_ms, dout_ms)
    dx_ms = input_grad[0][0].asnumpy()
    dw_ms = input_grad[0][1].asnumpy()
    db_ms = input_grad[0][2].asnumpy()

    assert np.abs(dx_ms - dx_np).mean() < error
    assert np.abs(dw_ms - dw_np).mean() < error
    assert np.abs(db_ms - db_np).mean() < error

    # dynamic rank
    grad_net.set_inputs(
        Tensor(shape=[None for _ in x_ms.shape], dtype=x_ms.dtype),
        Tensor(shape=[None for _ in w_ms.shape], dtype=w_ms.dtype),
        b_ms,
        dout_ms,
    )
    input_grad = grad_net(x_ms, w_ms, b_ms, dout_ms)
    dx_ms = input_grad[0][0].asnumpy()
    dw_ms = input_grad[0][1].asnumpy()
    db_ms = input_grad[0][2].asnumpy()

    assert np.abs(dx_ms - dx_np).mean() < error
    assert np.abs(dw_ms - dw_np).mean() < error
    assert np.abs(db_ms - db_np).mean() < error


def test_3d_backward():
    """
    Feature: Test dense 3d.
    Description: Test dense 3d backward for Graph mode.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dtype = np.float32
    error = 1e-3
    x_np = np.array([[[1, 2, 3], [4, 5, 6]]]).astype(dtype)
    w_np = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]]).astype(dtype)
    b_np = np.array([19, 20, 21, 22]).astype(dtype)
    dout_np = np.array([[[23, 24, 25, 26], [27, 28, 29, 30]]]).astype(np.float32)
    dx_np = np.array([[[1142, 1240, 1338], [1326, 1440, 1554]]]).astype(dtype)
    dw_np = np.array([[131, 181, 231], [136, 188, 240], [141, 195, 249.], [146, 202, 258]]).astype(dtype)
    db_np = np.array([50, 52, 54, 56]).astype(dtype)
    x_ms = Tensor(x_np)
    w_ms = Tensor(w_np)
    b_ms = Tensor(b_np)
    dout_ms = Tensor(dout_np)
    net = Dense()
    grad_net = DenseGrad(net)
    grad_net.set_train()

    # dynamic shape
    grad_net.set_inputs(
        Tensor(dtype=x_ms.dtype),
        Tensor(dtype=w_ms.dtype),
        b_ms,
        dout_ms,
    )
    input_grad = grad_net(x_ms, w_ms, b_ms, dout_ms)
    dx_ms = input_grad[0][0].asnumpy()
    dw_ms = input_grad[0][1].asnumpy()
    db_ms = input_grad[0][2].asnumpy()

    assert np.abs(dx_ms - dx_np).mean() < error
    assert np.abs(dw_ms - dw_np).mean() < error
    assert np.abs(db_ms - db_np).mean() < error

    # dynamic rank
    grad_net.set_inputs(
        Tensor(shape=[None for _ in x_ms.shape], dtype=x_ms.dtype),
        Tensor(shape=[None for _ in w_ms.shape], dtype=w_ms.dtype),
        b_ms,
        dout_ms,
    )
    input_grad = grad_net(x_ms, w_ms, b_ms, dout_ms)
    dx_ms = input_grad[0][0].asnumpy()
    dw_ms = input_grad[0][1].asnumpy()
    db_ms = input_grad[0][2].asnumpy()

    assert np.abs(dx_ms - dx_np).mean() < error
    assert np.abs(dw_ms - dw_np).mean() < error
    assert np.abs(db_ms - db_np).mean() < error


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_1d_complex64_backward():
    """
    Feature: Test dense 1d complex64 backward.
    Description: Test dense 1d complex64 backward for Graph mode.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dtype = np.complex64
    error = 1e-3
    x_np = np.array([0.6 + 0.7j, 0.7 + 0.9j, 0.0 + 0.7j]).astype(dtype)
    w_np = np.array([0.2 + 0.1j, 0.8 + 0.4j, 0.9 + 0.7j]).astype(dtype)
    b_np = np.array(-0.5 + 0.6j).astype(dtype)
    dout_np = np.array(-1.3 - 0.1j).astype(dtype)
    dx_np = np.array([-0.27 + 0.11j, -1.08 + 0.44j, -1.24 + 0.82j]).astype(dtype)
    dw_np = np.array([-0.85 + 0.85j, -1.0 + 1.1j, -0.07 + 0.91j]).astype(dtype)
    db_np = np.array(-1.3 - 0.1j).astype(dtype)
    x_ms = Tensor(x_np)
    w_ms = Tensor(w_np)
    b_ms = Tensor(b_np)
    dout_ms = Tensor(dout_np)
    net = Dense()
    grad_net = DenseGrad(net)
    grad_net.set_train()

    input_grad = grad_net(x_ms, w_ms, b_ms, dout_ms)
    dx_ms = input_grad[0][0].asnumpy()
    dw_ms = input_grad[0][1].asnumpy()
    db_ms = input_grad[0][2].asnumpy()

    assert np.abs(dx_ms - dx_np).mean() < error
    assert np.abs(dw_ms - dw_np).mean() < error
    assert np.abs(db_ms - db_np).mean() < error


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_2d_complex128_backward():
    """
    Feature: Test dense 2d complex128 backward.
    Description: Test dense 2d complex128 backward for Graph mode.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dtype = np.complex128
    error = 1e-3
    x_np = np.array([[0.6 + 0.7j, 0.7 + 0.9j, 0.0 + 0.7j]]).astype(dtype)
    w_np = np.array([[0.2 + 0.1j, 0.8 + 0.4j, 0.9 + 0.7j]]).astype(dtype)
    b_np = np.array([-0.5 + 0.6j]).astype(dtype)
    dout_np = np.array([[-1.3 - 0.1j]]).astype(dtype)
    dx_np = np.array([[-0.27 + 0.11j, -1.08 + 0.44j, -1.24 + 0.82j]]).astype(dtype)
    dw_np = np.array([[-0.85 + 0.85j, -1.0 + 1.1j, -0.07 + 0.91j]]).astype(dtype)
    db_np = np.array([-1.3 - 0.1j]).astype(dtype)
    x_ms = Tensor(x_np)
    w_ms = Tensor(w_np)
    b_ms = Tensor(b_np)
    dout_ms = Tensor(dout_np)
    net = Dense()
    grad_net = DenseGrad(net)
    grad_net.set_train()

    input_grad = grad_net(x_ms, w_ms, b_ms, dout_ms)
    dx_ms = input_grad[0][0].asnumpy()
    dw_ms = input_grad[0][1].asnumpy()
    db_ms = input_grad[0][2].asnumpy()

    assert np.abs(dx_ms - dx_np).mean() < error
    assert np.abs(dw_ms - dw_np).mean() < error
    assert np.abs(db_ms - db_np).mean() < error


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_2d_dtypes_forward():
    """
    Feature: Test dense 2d dtypes forward.
    Description: Test dense 2d dtypes forward for Graph mode.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dtypes = (np.float16, np.float32, np.float64, np.complex64, np.complex128)
    error = 1e-3
    net = Dense()
    for dtype in dtypes:
        x_np = np.array([_ for _ in range(6)]).reshape(2, 3).astype(dtype)
        w_np = np.array([_ for _ in range(12)]).reshape(4, 3).astype(dtype)
        b_np = np.array([_ for _ in range(4)]).astype(dtype)
        x_ms = Tensor(x_np)
        w_ms = Tensor(w_np)
        b_ms = Tensor(b_np)
        out_ms = net(x_ms, w_ms, b_ms).asnumpy()
        out_np = np.array([5, 15, 25, 35, 14, 51, 88, 125]).reshape(2, 4).astype(dtype)
        assert np.abs(out_ms - out_np).mean() < error


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_0d_bias():
    """
    Feature: Test dense 0d bias.
    Description: Test dense 0d bias for Graph mode.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dtype = np.float32
    error = 1e-3
    net = Dense()
    x_np = np.array([2, 0, 1, 3]).reshape(2, 2).astype(dtype)
    w_np = np.array([1, 2, 2, 7]).reshape(2, 2).astype(dtype)
    b_np = np.array(118).astype(dtype)
    x_ms = Tensor(x_np)
    w_ms = Tensor(w_np)
    b_ms = Tensor(b_np)
    out_ms = net(x_ms, w_ms, b_ms).asnumpy()
    out_np = np.array([120, 122, 125, 141]).reshape(2, 2).astype(dtype)
    assert np.abs(out_ms - out_np).mean() < error
    dtype = np.float64
    x_np = np.array([2, 0, 1, 3]).astype(dtype)
    w_np = np.array([1, 2, 2, 7]).astype(dtype)
    b_np = np.array(118).astype(dtype)
    x_ms = Tensor(x_np)
    w_ms = Tensor(w_np)
    b_ms = Tensor(b_np)
    out_ms = net(x_ms, w_ms, b_ms).asnumpy()
    out_np = np.array(143).astype(dtype)
    assert np.abs(out_ms - out_np).mean() < error
