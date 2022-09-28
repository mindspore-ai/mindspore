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
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner


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


class MishGradDynamicShapeNet(nn.Cell):
    def __init__(self, network):
        super(MishGradDynamicShapeNet, self).__init__()
        self.test_dynamic = inner.GpuConvertToDynamicShape()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, x, dy):
        x = self.test_dynamic(x)
        dy = self.test_dynamic(dy)
        return self.grad(self.network)(x, dy)


@pytest.mark.level1
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_mish_grad_vmap(mode):
    """
    Feature: test mish_grad vmap feature.
    Description: test mish_grad vmap feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="GPU")
    x = Tensor(np.array([[[[1.7641, 0.4002, 0.9787],
                           [2.2409, 1.8676, -0.9773]],
                          [[0.9501, -0.1514, -0.1032],
                           [0.4106, 0.1440, 1.4543]]],
                         [[[0.7610, 0.1217, 0.4439],
                           [0.3337, 1.4941, -0.2052]],
                          [[0.3131, -0.8541, -2.5530],
                           [0.6536, 0.8644, -0.7422]]]]).astype(np.float32))
    dout = Tensor(np.array([[[[2.2698, -1.4544, 0.0458],
                              [-0.1872, 1.5328, 1.4694]],
                             [[0.1549, 0.3782, -0.8878],
                              [-1.9808, -0.3479, 0.1563]]],
                            [[[1.2303, 1.2024, -0.3873],
                              [-0.3023, -1.0486, -1.4200]],
                             [[-1.7063, 1.9508, -0.5097],
                              [-0.4381, -1.2528, 0.7775]]]]).astype(np.float32))
    # Case 1
    output = F.vmap(MishGradNet(MishNet()), (0, 0), 0)(x, dout)
    expect_output = np.array([[[[2.4551494, -1.2175093, 0.04786031],
                                [-0.1975334, 1.6502876, 0.098847]],
                               [[0.16096734, 0.19009684, -0.4737671],
                                [-1.6688104, -0.24026635, 0.17010784]]],
                              [[[1.2171272, 0.8138411, -0.33282048],
                                [-0.24231756, -1.1413976, -0.6648672]],
                               [[-1.3482721, 0.22441003, 0.05531899],
                                [-0.41695648, -1.2767013, 0.12779452]]]]).astype(np.float32)
    assert np.allclose(output[0].asnumpy(), expect_output, atol=1e-4, rtol=1e-4, equal_nan=True)

    # # Case 2
    output = F.vmap(MishGradNet(MishNet()), (0, 1), 0)(x, dout)
    expect_output = np.array([[[[2.4551494, -1.2175093, 0.04786031],
                                [-0.1975334, 1.6502876, 0.098847]],
                               [[1.2784901, 0.6043692, -0.20667945],
                                [-0.25468567, -0.7241831, -1.5454454]]],
                              [[[0.1532415, 0.25598362, -0.76291764],
                                [-1.5877693, -0.378688, 0.07318222]],
                               [[-1.3482721, 0.22441003, 0.05531899],
                                [-0.41695648, -1.2767013, 0.12779452]]]]).astype(np.float32)
    assert np.allclose(output[0].asnumpy(), expect_output, atol=1e-4, rtol=1e-4, equal_nan=True)

    # # Case 3
    output = F.vmap(MishGradNet(MishNet()), (0, 0), 1)(x, dout)
    expect_output = np.array([[[[2.4551494, -1.2175093, 0.04786031],
                                [-0.1975334, 1.6502876, 0.098847]],
                               [[1.2171272, 0.8138411, -0.33282048],
                                [-0.24231756, -1.1413976, -0.6648672]]],
                              [[[0.16096734, 0.19009684, -0.4737671],
                                [-1.6688104, -0.24026635, 0.17010784]],
                               [[-1.3482721, 0.22441003, 0.05531899],
                                [-0.41695648, -1.2767013, 0.12779452]]]]).astype(np.float32)
    assert np.allclose(output[0].asnumpy(), expect_output, atol=1e-4, rtol=1e-4, equal_nan=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_mish_grad_dynamic_shape(mode):
    """
    Feature: test mish_grad dynamic_shape feature.
    Description: test mish_grad dynamic_shape feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="GPU")
    x = Tensor(np.array([[[[1.7641, 0.4002, 0.9787],
                           [2.2409, 1.8676, -0.9773]],
                          [[0.9501, -0.1514, -0.1032],
                           [0.4106, 0.1440, 1.4543]]],
                         [[[0.7610, 0.1217, 0.4439],
                           [0.3337, 1.4941, -0.2052]],
                          [[0.3131, -0.8541, -2.5530],
                           [0.6536, 0.8644, -0.7422]]]]).astype(np.float32))
    dout = Tensor(np.array([[[[2.2698, -1.4544, 0.0458],
                              [-0.1872, 1.5328, 1.4694]],
                             [[0.1549, 0.3782, -0.8878],
                              [-1.9808, -0.3479, 0.1563]]],
                            [[[1.2303, 1.2024, -0.3873],
                              [-0.3023, -1.0486, -1.4200]],
                             [[-1.7063, 1.9508, -0.5097],
                              [-0.4381, -1.2528, 0.7775]]]]).astype(np.float32))
    output = MishGradDynamicShapeNet(MishNet())(x, dout)
    expect_output = np.array([[[[2.4551494, -1.2175093, 0.04786031],
                                [-0.1975334, 1.6502876, 0.098847]],
                               [[0.16096734, 0.19009684, -0.4737671],
                                [-1.6688104, -0.24026635, 0.17010784]]],
                              [[[1.2171272, 0.8138411, -0.33282048],
                                [-0.24231756, -1.1413976, -0.6648672]],
                               [[-1.3482721, 0.22441003, 0.05531899],
                                [-0.41695648, -1.2767013, 0.12779452]]]]).astype(np.float32)
    assert np.allclose(output[0].asnumpy(), expect_output, atol=1e-4, rtol=1e-4, equal_nan=True)
