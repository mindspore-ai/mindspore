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
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logsoftmax():
    x = np.array([[-0.08082921, -0.13706027, -0.4711177, -0.05606057],
                  [-0.46082982, 1.1761844, -1.016654, -1.743829],
                  [-1.5062045, 0.6910976, 0.4839723, 1.1502692]]).astype(np.float32)
    expect = np.array([[-1.2939762, -1.3502073, -1.6842647, -1.2692076],
                       [-1.9445671, -0.3075528, -2.5003912, -3.2275662],
                       [-3.452001, -1.2546989, -1.4618242, -0.79552734]]).astype(np.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    logSoftmax = P.LogSoftmax()
    output = logSoftmax(Tensor(x))
    diff = output.asnumpy() - expect
    err = np.ones(shape=expect.shape) * 1.0e-5
    assert np.all(diff < err)


class LogSoftmax(nn.Cell):
    def __init__(self, axis=-1):
        super(LogSoftmax, self).__init__()
        self.logsoftmax = P.LogSoftmax(axis)

    def construct(self, x):
        return self.logsoftmax(x)


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
def test_logsoftmaxgrad():
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
                        0.10445589, 0.3220427]],).astype(np.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = LogSoftmax(0)
    dx = Grad(net)(Tensor(x), Tensor(dy))
    diff = dx[0].asnumpy() - expect
    err = np.ones(shape=expect.shape) * 1.0e-5
    assert np.all(diff < err)
