# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.context as context
from mindspore.ops import composite as C


class NetTensorDot(nn.Cell):
    def __init__(self, axes):
        super(NetTensorDot, self).__init__()
        self.axes = axes

    def construct(self, x, y):
        return C.tensor_dot(x, y, self.axes)


class GradNetwork(nn.Cell):
    def __init__(self, network):
        super(GradNetwork, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data_a, input_data_b, sens):
        gout = self.grad(self.network)(input_data_a, input_data_b, sens)
        return gout


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_dot_fp32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    np.random.seed(12876)
    shape_x1 = (1, 3, 9, 7)
    shape_x2 = (9, 7, 3, 1)
    axes = ((1, 3), (2, 1))
    x1 = np.random.random(shape_x1).astype(np.float32)
    x2 = np.random.random(shape_x2).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=mindspore.float32)
    x2_tensor = Tensor(x2, dtype=mindspore.float32)

    network = NetTensorDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    np_result = np.tensordot(x1, x2, axes)
    np.testing.assert_array_almost_equal(ms_result_np, np_result)

    # 1D
    shape_x1 = (200)
    shape_x2 = (200)
    axes = 1
    x1 = np.random.random(shape_x1).astype(np.float32)
    x2 = np.random.random(shape_x2).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=mindspore.float32)
    x2_tensor = Tensor(x2, dtype=mindspore.float32)

    network = NetTensorDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    np_result = np.tensordot(x1, x2, axes)
    np.allclose(ms_result_np, np_result)

    # 2D
    shape_x1 = (100, 300)
    shape_x2 = (300, 700)
    axes = ([1], [0])
    x1 = np.random.random(shape_x1).astype(np.float32)
    x2 = np.random.random(shape_x2).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=mindspore.float32)
    x2_tensor = Tensor(x2, dtype=mindspore.float32)

    network = NetTensorDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    np_result = np.tensordot(x1, x2, axes)
    np.allclose(ms_result_np, np_result)

    # 3D
    shape_x1 = (110, 30, 900)
    shape_x2 = (900, 70, 30)
    axes = ((1, 2), (2, 0))
    x1 = np.random.random(shape_x1).astype(np.float32)
    x2 = np.random.random(shape_x2).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=mindspore.float32)
    x2_tensor = Tensor(x2, dtype=mindspore.float32)

    network = NetTensorDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    np_result = np.tensordot(x1, x2, axes)
    np.allclose(ms_result_np, np_result)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_dot_fp16():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    np.random.seed(41329)
    shape_x1 = (1, 3, 4, 1)
    shape_x2 = (4, 1, 7, 5)
    axes = 2  # select first N from
    x1 = np.random.random(shape_x1).astype(np.float16)
    x2 = np.random.random(shape_x2).astype(np.float16)
    x1_tensor = Tensor(x1, dtype=mindspore.float16)
    x2_tensor = Tensor(x2, dtype=mindspore.float16)

    network = NetTensorDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    np_result = np.tensordot(x1, x2, axes)
    np.testing.assert_array_almost_equal(ms_result_np, np_result)

    # 1D
    shape_x1 = (300)
    shape_x2 = (300)
    axes = 1
    x1 = np.random.random(shape_x1).astype(np.float16)
    x2 = np.random.random(shape_x2).astype(np.float16)
    x1_tensor = Tensor(x1, dtype=mindspore.float16)
    x2_tensor = Tensor(x2, dtype=mindspore.float16)

    network = NetTensorDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    np_result = np.tensordot(x1, x2, axes)
    np.testing.assert_array_almost_equal(ms_result_np, np_result)

    # 2D
    shape_x1 = (100, 300)
    shape_x2 = (300, 100)
    axes = ([1], [0])
    x1 = np.random.random(shape_x1).astype(np.float16)
    x2 = np.random.random(shape_x2).astype(np.float16)
    x1_tensor = Tensor(x1, dtype=mindspore.float16)
    x2_tensor = Tensor(x2, dtype=mindspore.float16)

    network = NetTensorDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    np_result = np.tensordot(x1, x2, axes)
    assert np.allclose(ms_result_np, np_result, rtol=1e-3, atol=1e-3)

    # 3D
    shape_x1 = (60, 30, 450)
    shape_x2 = (450, 90, 30)
    axes = ((1, 2), (2, 0))
    x1 = np.random.random(shape_x1).astype(np.float16)
    x2 = np.random.random(shape_x2).astype(np.float16)
    x1_tensor = Tensor(x1, dtype=mindspore.float16)
    x2_tensor = Tensor(x2, dtype=mindspore.float16)

    network = NetTensorDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    np_result = np.tensordot(x1, x2, axes)
    assert np.allclose(ms_result_np, np_result, rtol=1e-3, atol=6e0)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_dot_outer():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(2746)
    shape_x1 = (1, 2, 3)  # incompatible dims for x1 and x2
    shape_x2 = (4, 5, 6)
    axes = 0  # outer product does not require multiplicable dims
    x1 = np.random.random(shape_x1).astype(np.float32)
    x2 = np.random.random(shape_x2).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=mindspore.float32)
    x2_tensor = Tensor(x2, dtype=mindspore.float32)

    network = NetTensorDot(axes)

    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    np_result = np.tensordot(x1, x2, axes)
    np.testing.assert_array_almost_equal(ms_result_np, np_result)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_dot_reverse_axes():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(2746)
    shape_x1 = (1, 2, 3)
    shape_x2 = (1, 2, 3)
    axes = ((1, 0), (0, 1))
    x1 = np.random.random(shape_x1).astype(np.float32)
    x2 = np.random.random(shape_x2).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=mindspore.float32)
    x2_tensor = Tensor(x2, dtype=mindspore.float32)

    network = NetTensorDot(axes)

    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    expected_result = np.array([[0.58561826, 0.21897982, 0.906598],
                                [0.19630799, 0.10731681, 0.2680981],
                                [0.8003185, 0.457294, 1.0721111]]).astype(np.float32)
    np.testing.assert_array_almost_equal(ms_result_np, expected_result)


    np.random.seed(1526)
    shape_x1 = (1, 2, 3, 4, 5)
    shape_x2 = (1, 2, 3)
    axes = ((0, 2), (2, 0))
    x1 = np.random.random(shape_x1).astype(np.float32)
    x2 = np.random.random(shape_x2).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=mindspore.float32)
    x2_tensor = Tensor(x2, dtype=mindspore.float32)

    network = NetTensorDot(axes)
    ms_result_np = network(x1_tensor, x2_tensor).asnumpy()
    expected_result = np.array([[[[1.106365, 0.9736746],
                                  [0.91042125, 0.7814131],
                                  [0.5576207, 0.327488],
                                  [0.93404585, 0.7108171],
                                  [1.078351, 0.87405884]],
                                 [[1.1720579, 0.9948833],
                                  [1.1594493, 1.0185612],
                                  [0.7251004, 0.60322404],
                                  [0.4724398, 0.2930961],
                                  [0.9711088, 0.8482977]],
                                 [[1.4110168, 1.1171235],
                                  [0.81948525, 0.778057],
                                  [0.7914786, 0.78767675],
                                  [0.77509344, 0.6020987],
                                  [0.8986199, 0.7100061]],
                                 [[0.7270926, 0.35752398],
                                  [0.5529937, 0.31682697],
                                  [0.73876995, 0.48478222],
                                  [0.96520174, 0.73099715],
                                  [0.96569407, 0.8556314]]],
                                [[[1.2093457, 0.90222925],
                                  [1.3758272, 0.8189213],
                                  [1.2997738, 1.045748],
                                  [1.1460838, 0.67475325],
                                  [0.95835257, 0.67791444]],
                                 [[0.84732395, 0.8058369],
                                  [1.1979935, 0.57202166],
                                  [0.2577264, 0.22021212],
                                  [0.8855853, 0.5440637],
                                  [0.8993537, 0.4622679]],
                                 [[0.6797033, 0.58302796],
                                  [0.7820443, 0.49587217],
                                  [0.64423263, 0.5469],
                                  [1.0270302, 0.5271675],
                                  [1.0278721, 0.9446807]],
                                 [[1.2069539, 1.0113767],
                                  [0.86160654, 0.7664283],
                                  [0.9797001, 0.7087945],
                                  [0.47638205, 0.4660839],
                                  [0.6920749, 0.36285543]]]]).astype(np.float32)
    np.testing.assert_array_almost_equal(ms_result_np, expected_result)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_dot_backprop():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # TEST 1
    shape_x1 = (2, 4, 2)
    shape_x2 = (3, 2, 3)
    axes = ((0,), (1,))  # select first N from
    network = NetTensorDot(axes)

    np.random.seed(115)
    x1 = np.random.random(shape_x1).astype(np.float16)
    np.random.seed(1467)
    x2 = np.random.random(shape_x2).astype(np.float16)
    x1_tensor = Tensor(x1, dtype=mindspore.float16)
    x2_tensor = Tensor(x2, dtype=mindspore.float16)

    np.random.seed(157)
    grad = np.random.random((4, 2, 3, 3))
    grad_tensor = Tensor(grad, dtype=mindspore.float16)
    grad_network = GradNetwork(network)
    dx1, dx2 = grad_network(x1_tensor, x2_tensor, grad_tensor)
    dx1, dx2 = dx1.asnumpy(), dx2.asnumpy()

    # precomputed
    expect_dx1 = np.array([[[2.0293, 2.4473],
                            [2.9727, 1.4873],
                            [1.7910, 3.4727],
                            [2.4160, 1.7227]],
                           [[2.5547, 2.5039],
                            [3.4062, 2.3320],
                            [2.6270, 3.1543],
                            [2.1406, 1.7666]]])
    expect_dx2 = np.array([[[2.1523, 2.9199, 0.8350],
                            [2.0254, 2.7734, 1.3213]],
                           [[2.6836, 2.4707, 1.0156],
                            [2.9746, 3.0254, 1.9199]],
                           [[1.8545, 1.7803, 1.3457],
                            [2.2676, 2.1797, 1.2764]]])
    np.allclose(dx1, expect_dx1)
    np.allclose(dx2, expect_dx2)

    # TEST 2
    shape_x1 = (10, 35)
    shape_x2 = (20, 10)
    axes = ((0,), (1,))  # select first N from
    network = NetTensorDot(axes)

    np.random.seed(215)
    x1 = np.random.random(shape_x1).astype(np.float16)
    np.random.seed(2467)
    x2 = np.random.random(shape_x2).astype(np.float16)
    x1_tensor = Tensor(x1, dtype=mindspore.float16)
    x2_tensor = Tensor(x2, dtype=mindspore.float16)

    np.random.seed(257)
    grad = np.random.random((35, 20))
    grad_tensor = Tensor(grad, dtype=mindspore.float16)
    grad_network = GradNetwork(network)
    dx1, dx2 = grad_network(x1_tensor, x2_tensor, grad_tensor)
    dx1, dx2 = dx1.asnumpy(), dx2.asnumpy()

    # precomputed
    expect_dx1 = np.array([[5.9727, 4.6484, 5.1836, 4.3906, 5.1641, 5.1406, 5.1211, 6.5352, 4.9922,
                            4.4297, 4.4648, 6.5469, 6.2305, 4.8789, 6.8320, 5.3906, 4.7383, 6.0352,
                            4.7383, 4.4844, 5.3711, 6.2617, 4.6484, 5.8672, 4.7500, 6.0234, 3.6387,
                            5.3789, 5.9727, 5.7227, 6.0234, 4.9609, 5.0117, 5.4141, 5.1406],
                           [5.2305, 4.0078, 4.6328, 3.9238, 4.2773, 4.2539, 4.6797, 5.1289, 3.7910,
                            3.8887, 3.2930, 5.5898, 5.4219, 3.6211, 5.5234, 3.5391, 4.8516, 4.7539,
                            4.2500, 2.9785, 4.8867, 5.4648, 5.0195, 6.0195, 4.7109, 3.9727, 3.4922,
                            4.1484, 4.7969, 5.3555, 4.9414, 5.2969, 3.1992, 5.2031, 4.4648],
                           [5.2266, 5.2617, 5.3750, 4.7930, 4.9062, 5.4102, 4.9336, 6.9414, 4.4961,
                            4.4023, 4.7344, 5.8125, 4.9180, 4.7891, 5.9805, 5.2383, 4.6445, 6.1172,
                            4.8477, 3.7578, 4.3047, 5.7969, 4.5859, 6.0273, 4.3438, 4.7305, 4.0938,
                            4.8398, 5.8320, 5.3438, 5.3281, 4.8320, 4.0938, 4.9375, 5.3281],
                           [7.4297, 5.1484, 6.3477, 5.4844, 5.7852, 6.3906, 5.5234, 7.2383, 5.2969,
                            4.9844, 4.5625, 7.3047, 7.3789, 6.4453, 8.2266, 6.6172, 5.5547, 7.0234,
                            4.8594, 4.9531, 6.0469, 6.9258, 6.1055, 6.7539, 6.6953, 6.0430, 4.5117,
                            5.7344, 7.4297, 6.4219, 6.8125, 6.4141, 5.2773, 6.8828, 6.0430],
                           [5.7969, 4.7109, 5.8281, 4.5703, 5.5078, 6.4219, 4.8359, 7.1484, 4.2617,
                            4.8477, 4.2539, 5.6016, 6.4414, 5.7305, 6.4766, 5.4648, 4.5859, 6.5547,
                            5.5156, 3.3848, 5.1523, 5.5352, 4.9531, 6.5938, 5.2969, 4.6055, 5.2109,
                            4.4961, 5.8984, 5.4531, 5.8086, 5.7930, 5.0742, 5.4102, 4.9453],
                           [7.2188, 5.8789, 6.9453, 6.0039, 6.7188, 7.3359, 6.7695, 8.6172, 5.6680,
                            6.4219, 6.1836, 7.7695, 7.5391, 6.5312, 8.2812, 7.5352, 5.8867, 7.7070,
                            6.0039, 5.1172, 6.4844, 7.4297, 5.9219, 7.5078, 6.3125, 6.9805, 5.3750,
                            5.9805, 7.2148, 7.6484, 7.8828, 6.7695, 5.7109, 6.8828, 6.9023],
                           [5.7656, 4.3633, 4.5039, 4.4375, 4.3867, 5.4336, 4.3672, 5.5469, 3.5742,
                            4.0508, 3.7402, 5.9141, 5.7734, 4.5781, 5.6719, 4.5625, 4.5391, 5.1719,
                            4.3945, 3.4844, 4.9297, 5.7227, 4.8203, 5.8125, 4.8633, 4.3125, 3.6641,
                            4.3789, 5.6133, 5.1758, 4.9141, 5.8008, 4.0391, 5.8984, 4.3594],
                           [4.7734, 3.4238, 4.3477, 3.6270, 4.4883, 5.2031, 3.9023, 5.0078, 2.9355,
                            3.8477, 3.4648, 5.1445, 4.8398, 4.4297, 5.1641, 4.2422, 4.2695, 4.6992,
                            4.5039, 2.5176, 4.2500, 5.6680, 4.1875, 5.4141, 3.6094, 3.1758, 3.8398,
                            3.9180, 5.3320, 4.6523, 3.9531, 4.8281, 3.9863, 4.8867, 4.3711],
                           [6.7578, 5.3164, 6.0000, 4.4531, 5.8789, 6.3750, 5.1094, 7.0391, 4.5781,
                            4.8633, 4.5156, 6.6641, 6.3594, 5.5664, 6.9453, 5.5820, 5.1992, 6.9570,
                            5.3242, 3.8574, 5.1445, 6.0547, 5.0273, 6.9180, 5.1914, 4.6914, 4.6445,
                            5.1289, 5.8711, 6.2070, 6.1953, 5.7695, 4.7617, 5.5898, 4.9492],
                           [4.9180, 4.0117, 4.1211, 3.4629, 3.6445, 4.6602, 3.7031, 4.9062, 4.1133,
                            3.0020, 3.2246, 4.6562, 4.4727, 3.3828, 5.2695, 4.0078, 3.2559, 4.9688,
                            3.5742, 3.1133, 3.8223, 4.7578, 3.7949, 4.8438, 4.0664, 4.4336, 3.0957,
                            4.4375, 4.2969, 4.1758, 4.5234, 4.2930, 3.9434, 4.8281, 3.0703]])
    expect_dx2 = np.array([[6.7930, 7.0000, 8.8203, 9.7031, 8.1250,
                            6.7422, 8.4844, 8.7031, 7.2891, 10.1484],
                           [8.5781, 8.1641, 9.9609, 9.2344, 9.3281,
                            8.1484, 9.8984, 9.0391, 7.9805, 11.0469],
                           [8.1016, 7.0781, 8.9688, 10.0938, 9.6641,
                            7.1523, 8.2969, 8.8594, 8.3047, 10.2578],
                           [7.0938, 7.3477, 9.3594, 8.2422, 7.9141,
                            6.5156, 8.2812, 8.2266, 6.9766, 8.5703],
                           [9.2891, 9.2500, 11.6875, 9.5234, 10.1172,
                            8.8125, 9.5781, 9.5547, 8.9688, 11.2266],
                           [9.3594, 7.7539, 9.2500, 9.2500, 8.1094,
                            8.0859, 8.7344, 8.2031, 8.5859, 10.3203],
                           [8.7344, 7.7227, 10.2578, 10.1641, 9.3984,
                            8.1719, 8.0156, 8.6953, 8.6797, 10.6875],
                           [8.8750, 7.9922, 10.2422, 10.3984, 9.5234,
                            8.5156, 8.7266, 8.8125, 8.2578, 10.2578],
                           [9.5703, 8.9844, 10.0547, 10.3047, 10.4062,
                            8.2422, 10.7031, 9.7891, 9.2969, 11.0078],
                           [9.2891, 9.5391, 10.5938, 10.5078, 9.8203,
                            8.5156, 9.0859, 9.0703, 8.7812, 10.8750],
                           [8.6094, 8.2734, 10.2734, 9.7891, 9.4531,
                            7.5820, 8.4609, 8.6094, 7.7578, 10.3438],
                           [8.2891, 8.7578, 9.3906, 9.6016, 9.4375,
                            7.1016, 8.6875, 8.1875, 8.2188, 9.3672],
                           [7.2969, 6.6953, 9.3984, 8.2422, 8.3438,
                            7.5547, 7.6445, 7.5820, 7.5156, 9.0781],
                           [8.3906, 7.3516, 8.5938, 9.2422, 8.7734,
                            8.0781, 9.1250, 7.8359, 7.7891, 10.9375],
                           [9.9219, 8.8281, 9.4141, 10.2500, 9.8047,
                            8.5234, 8.5391, 8.4609, 8.5859, 11.2422],
                           [6.8984, 6.4570, 8.0000, 6.4688, 7.4609,
                            6.6016, 7.0352, 6.6797, 6.5586, 7.7070],
                           [8.0625, 7.4805, 8.7578, 8.3281, 8.2188,
                            7.4023, 8.5312, 7.5312, 7.1445, 10.3750],
                           [7.7773, 6.6484, 9.1094, 8.0078, 7.8281,
                            7.1016, 8.2422, 8.1562, 6.8828, 10.3281],
                           [8.3281, 8.3672, 9.7656, 10.4922, 8.2500,
                            7.5625, 8.4922, 8.9844, 8.0703, 10.3438],
                           [7.5195, 7.0430, 7.9453, 8.4375, 7.6641,
                            6.9688, 7.7734, 8.7734, 6.3672, 9.4766]])
    np.allclose(dx1, expect_dx1)
    np.allclose(dx2, expect_dx2)
