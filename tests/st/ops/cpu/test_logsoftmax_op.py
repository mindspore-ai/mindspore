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

import numpy as np
import scipy.special
import pytest

from mindspore import Tensor, ops, nn, context
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import dtype as mstype


class LogSoftmax(nn.Cell):
    def __init__(self, axis=-1):
        super(LogSoftmax, self).__init__()
        self.log_softmax = P.LogSoftmax(axis)

    def construct(self, x):
        return self.log_softmax(x)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, sens):
        gout = self.grad(self.network)(input_data, sens)
        return gout


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('axis', [-1, 0, 2])
@pytest.mark.parametrize('dtype, error', [(np.float32, 1.0e-5), (np.float16, 1.0e-3)])
def test_logsoftmax(axis, dtype, error):
    """
    Feature: ALL To ALL
    Description: test cases for LogSoftmax
    Expectation: the result match to scipy
    """
    np.random.seed(0)
    x = np.random.random((3, 5, 7, 4)).astype(dtype)
    expect = scipy.special.log_softmax(x, axis).astype(dtype)
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    output = ops.log_softmax(Tensor(x), axis)
    assert np.allclose(output.asnumpy(), expect, atol=error, rtol=error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('axis', [-1, 0, 2])
@pytest.mark.parametrize('dtype, error', [(np.float32, 1.0e-5)])
def test_logsoftmax_vmap(axis, dtype, error):
    """
    Feature: ALL To ALL
    Description: test cases for LogSoftmax vmap
    Expectation: the result match to scipy
    """
    np.random.seed(0)
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    vmap_dim = 2
    x = np.random.random((3, 5, 7, 4)).astype(dtype)

    output = ops.vmap(ops.log_softmax, (vmap_dim, None), vmap_dim)(Tensor(x), axis)

    expect = np.zeros_like(x).astype(dtype)
    axis = axis + 3 if axis < 0 else axis
    axis = axis if axis <= vmap_dim else axis + 1
    for i in range(x.shape[vmap_dim]):
        expect[:, :, i, :] = scipy.special.log_softmax(x[:, :, i, :], axis)

    assert np.allclose(output.asnumpy(), expect, atol=error, rtol=error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logsoftmax_dynamic_shape():
    """
    Feature: ALL To ALL
    Description: test cases for LogSoftmax
    Expectation: the result match to scipy
    """
    axis = -1
    dtype, error = np.float32, 1.0e-5

    np.random.seed(0)
    x = np.random.random((3, 5, 7, 4)).astype(dtype)
    expect = scipy.special.log_softmax(x, axis).astype(dtype)
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    dynamic_net = LogSoftmax(axis)
    place_holder = Tensor(shape=[3, 5, None, 4], dtype=mstype.float32)
    dynamic_net.set_inputs(place_holder)

    output = dynamic_net(Tensor(x))
    assert np.allclose(output.asnumpy(), expect, atol=error, rtol=error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logsoftmax_positive_result():
    """
    Feature: ALL To ALL
    Description: solve issue #I5HU62
    Expectation: the result has no positive number
    """
    context.set_context(device_target='CPU')
    x = [-8.696573257446289, -10.154170989990234, -6.408495903015137, -8.64416790008545, -7.133539199829102,
         -5.636314392089844, -6.203953742980957, 1.137353539466858, -1.3180590867996216, -2.010432481765747,
         1.2304869890213013, -1.0495449304580688, 0.08905300498008728, 0.07220810651779175, 0.6883723139762878,
         1.1946378946304321, -0.25151532888412476, 0.09144830703735352, -2.5387685298919678, 0.4569661617279053,
         1.6665842533111572, -1.1546639204025269, 1.789994239807129, -1.8672490119934082, 0.09714305400848389,
         23.740997314453125, -0.7563713788986206, 0.007991373538970947, 3.242292642593384, 0.7086143493652344,
         -0.05716127157211304, -1.8195757865905762, 1.2769628763198853, 1.0851389169692993, 0.43163713812828064,
         -0.17872026562690735, 0.7721619009971619, 0.38897693157196045, 1.7030436992645264, -0.8432123064994812,
         0.2874690294265747, -0.37078866362571716, -0.471019983291626, 0.4507289528846741, -1.433785080909729,
         -0.3246658444404602, -3.194831132888794, -5.469168663024902, -5.003396034240723, -7.386798858642578,
         -6.215921401977539]
    x = np.array(x).astype('float32').reshape((1, -1))
    log_softmax = nn.LogSoftmax(axis=1)
    y = log_softmax(Tensor(x, mstype.float32)).asnumpy()
    assert y.max() <= 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logsoftmaxgrad():
    """
    Feature: ALL To ALL
    Description: test cases for LogSoftmax Grad
    Expectation: the result match result
    """
    x = np.array([[-0.47705367, 0.48267725, -1.0453935, 1.574488, 0.20362134, 0.4435456, -0.23984082, -0.43684655,
                   -0.7725506, 1.4481013],
                  [1.1012247, 1.7069651, 0.55062026, 0.3361901, -1.1082426, -0.5001939, -0.3255393, -0.7972024,
                   -0.27965206, -0.702805],
                  [0.19450496, 0.87596166, 0.6467245, -1.044987, 0.5248943, -2.6166635, 1.6719198, 0.06600758,
                   -0.4099178, 1.1861311],
                  [1.1305193, -1.97308, 2.1047623, -1.5105937, 0.93052036, 1.2467804, 0.5310002, 0.7084912, -1.3681422,
                   -0.9686862],
                  [1.871408, 0.14219497, -0.41050452, -0.749807, 1.4900619, -1.8172716, -0.73839617, 0.17565694,
                   -0.4553867, -1.5423119]]).astype(np.float32)
    dy = np.array([[1.516363, -0.15196544, 0.598733, 0.64357865, 0.16265012, -1.3521105, 0.22621834, 0.7168259,
                    -0.6709239, 0.79757756],
                   [-0.32457778, 1.2831115, 1.1211495, -0.02665559, 1.9170904, -1.3397789, 1.4124829, -1.4298155,
                    0.758519, -0.25322974],
                   [-0.24226122, -1.2555921, 0.6492511, -0.34847677, 0.19916506, 0.628554, -0.19658111, 0.44939864,
                    -0.11677749, -1.2131723],
                   [0.24267715, 0.28106326, 1.1075432, -0.29006946, 0.31335673, 0.8833154, 0.13152207, 1.5482179,
                    0.29770762, -0.16246222],
                   [0.02145994, 0.80424, -0.95061, 1.5875458, -0.00308682, 0.17964548, 0.49912593, 0.46977136,
                    0.2151897, 0.30908248]]).astype(np.float32)
    expect = np.array([[1.4219905, -0.39837134, 0.5452743, -0.09062839, -0.02375537, -1.5890603, 0.10658137, 0.6185817,
                        -0.7411523, 0.15054005],
                       [-0.94926417, 0.13830578, 0.7609547, -0.31733334, 1.8485254, -1.4657221, 1.2625053, -1.523396,
                        0.601499, -0.35607445],
                       [-0.14447737, -1.0622973, 0.80294746, -0.32016528, 0.33523226, 0.63443416, 0.23186903,
                        0.53539133, -0.0633494, -0.9495847],
                       [-0.36894822, 0.253609, -0.5127511, -0.33366728, -0.18740037, 0.19628316, -0.20430653, 1.1471655,
                        0.24743511, -0.23741922],
                       [-1.2582518, 0.57718843, -1.0812542, 1.4944922, -0.8770549, 0.1476463, 0.40500447, 0.23499368,
                        0.09027944, 0.26695627]]).astype(np.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = LogSoftmax()
    dx = Grad(net)(Tensor(x), Tensor(dy))
    diff = dx[0].asnumpy() - expect
    err = np.ones(shape=expect.shape) * 1.0e-5
    assert np.all(diff < err)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logsoftmaxgrad1():
    """
    Feature: ALL To ALL
    Description: test cases for LogSoftmax Grad
    Expectation: the result match result
    """
    x = np.array([[-0.47705367, 0.48267725, -1.0453935, 1.574488, 0.20362134, 0.4435456, -0.23984082, -0.43684655,
                   -0.7725506, 1.4481013],
                  [1.1012247, 1.7069651, 0.55062026, 0.3361901, -1.1082426, -0.5001939, -0.3255393, -0.7972024,
                   -0.27965206, -0.702805],
                  [0.19450496, 0.87596166, 0.6467245, -1.044987, 0.5248943, -2.6166635, 1.6719198, 0.06600758,
                   -0.4099178, 1.1861311],
                  [1.1305193, -1.97308, 2.1047623, -1.5105937, 0.93052036, 1.2467804, 0.5310002, 0.7084912, -1.3681422,
                   -0.9686862],
                  [1.871408, 0.14219497, -0.41050452, -0.749807, 1.4900619, -1.8172716, -0.73839617, 0.17565694,
                   -0.4553867, -1.5423119]]).astype(np.float32)
    dy = np.array([[1.516363, -0.15196544, 0.598733, 0.64357865, 0.16265012, -1.3521105, 0.22621834, 0.7168259,
                    -0.6709239, 0.79757756],
                   [-0.32457778, 1.2831115, 1.1211495, -0.02665559, 1.9170904, -1.3397789, 1.4124829, -1.4298155,
                    0.758519, -0.25322974],
                   [-0.24226122, -1.2555921, 0.6492511, -0.34847677, 0.19916506, 0.628554, -0.19658111, 0.44939864,
                    -0.11677749, -1.2131723],
                   [0.24267715, 0.28106326, 1.1075432, -0.29006946, 0.31335673, 0.8833154, 0.13152207, 1.5482179,
                    0.29770762, -0.16246222],
                   [0.02145994, 0.80424, -0.95061, 1.5875458, -0.00308682, 0.17964548, 0.49912593, 0.46977136,
                    0.2151897, 0.30908248]]).astype(np.float32)
    expect = np.array([[1.464194, -0.29578894, 0.5296974, -0.39600563, -0.1479242, -1.0869746, 0.04521982, 0.5064515,
                        -0.7515615, 1.0554069],
                       [-0.5774203, 0.793861, 0.7805745, -0.32800734, 1.8334473, -1.236596, 1.2463496, -1.5765365,
                        0.6265108, -0.22322391],
                       [-0.34437084, -1.4687154, 0.27432096, -0.42420125, -0.22908019, 0.640983, -1.4210342, 0.10155854,
                        -0.23266247, -1.0147638],
                       [-0.01768187, 0.26872346, -0.5037259, -0.3376058, -0.3291146, 1.4752979, -0.25972134, 0.8869053,
                        0.25325722, -0.13946185],
                       [-0.5247209, 0.70192003, -1.0808672, 1.4858199, -1.1273282, 0.20728993, 0.38918605, 0.08162117,
                        0.10445589, 0.3220427]]).astype(np.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = LogSoftmax(0)
    dx = Grad(net)(Tensor(x), Tensor(dy))
    diff = dx[0].asnumpy() - expect
    err = np.ones(shape=expect.shape) * 1.0e-5
    assert np.all(diff < err)


class LogSoftmaxForForward(nn.Cell):
    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis
        self.logsoftmax = P.LogSoftmax(axis=axis)
        self.stack = P.Stack(axis=axis)

    def construct(self, x):
        out = []
        for i in range(x.shape[self.axis]):
            out.append(self.logsoftmax(x[i]))
        out = self.stack(out)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logsoftmaxgrad_vmap():
    """
    Feature: ALL To ALL
    Description: test cases for LogSoftmax Grad vmap
    Expectation: the result match result
    """
    seed = np.random.RandomState()
    x = Tensor(seed.random((3, 5, 1)).astype(np.float32))
    sens = Tensor(seed.random((3, 5, 1)).astype(np.float32))

    forward = LogSoftmax(axis=0)
    for_forward = LogSoftmaxForForward(axis=0)
    backward = Grad(forward)
    for_backward = Grad(for_forward)

    forward_result = forward(x)
    backward_vmap = ops.vmap(backward, in_axes=0, out_axes=0)(forward_result, sens)
    backward_for = for_backward(forward_result, sens)

    np.testing.assert_allclose(backward_for[0].asnumpy(), backward_vmap[0].asnumpy(), rtol=1e-5)
