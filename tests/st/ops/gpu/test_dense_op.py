# Copyright 2019 Huawei Technologies Co., Ltd
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

class BiasAdd(nn.Cell):
    def __init__(self):
        super(BiasAdd, self).__init__()
        self.ba = P.BiasAdd()

    def construct(self, x, b):
        return self.ba(x, b)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dx_ND():
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dw_ND():
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
