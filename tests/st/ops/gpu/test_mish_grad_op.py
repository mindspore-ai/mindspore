# Copyright 2022 Huawei Technologies Co., Ltd
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


class MishNet(nn.Cell):
    def __init__(self):
        super(MishNet, self).__init__()
        self.mish = P.Mish()

    def construct(self, x):
        return self.mish(x)


class MishGradNet(nn.Cell):
    def __init__(self, network):
        super(MishGradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, x, dy):
        gout = self.grad(self.network)(x, dy)
        return gout


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype, tol', [(np.float16, 1e-3), (np.float32, 1e-4)])
def test_mish_grad(mode, dtype, tol):
    """
    Feature: ALL To ALL
    Description: test cases for MishGrad
    Expectation: the result match to the expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = Tensor(np.array([[[[1.7641, 0.4002, 0.9787],
                           [2.2409, 1.8676, -0.9773]],
                          [[0.9501, -0.1514, -0.1032],
                           [0.4106, 0.1440, 1.4543]]],
                         [[[0.7610, 0.1217, 0.4439],
                           [0.3337, 1.4941, -0.2052]],
                          [[0.3131, -0.8541, -2.5530],
                           [0.6536, 0.8644, -0.7422]]]]).astype(dtype))
    dy = Tensor(np.array([[[[2.2698, -1.4544, 0.0458],
                            [-0.1872, 1.5328, 1.4694]],
                           [[0.1549, 0.3782, -0.8878],
                            [-1.9808, -0.3479, 0.1563]]],
                          [[[1.2303, 1.2024, -0.3873],
                            [-0.3023, -1.0486, -1.4200]],
                           [[-1.7063, 1.9508, -0.5097],
                            [-0.4381, -1.2528, 0.7775]]]]).astype(dtype))
    expect = np.array([[[[2.4551, -1.2174, 0.0478],
                         [-0.1975, 1.6503, 0.0989]],
                        [[0.1610, 0.1901, -0.4737],
                         [-1.6688, -0.2403, 0.1702]]],
                       [[[1.2171, 0.8138, -0.3328],
                         [-0.2423, -1.1413, -0.6649]],
                        [[-1.3482, 0.2244, 0.0553],
                         [-0.4169, -1.2767, 0.1278]]]]).astype(dtype)
    net = MishNet()
    grad = MishGradNet(net)
    output = grad(x, dy)
    assert np.allclose(output[0].asnumpy(), expect, atol=tol, rtol=tol, equal_nan=True)
