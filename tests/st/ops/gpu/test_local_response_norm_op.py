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
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore import dtype as mstype


class MSLRNOpNet(nn.Cell):
    def __init__(self):
        super(MSLRNOpNet, self).__init__()
        self.lrn1 = P.LRN(depth_radius=2, bias=1.0, alpha=0.0001, beta=0.75)

    def construct(self, x):
        x = self.lrn1(x)
        return x


class MSGradNet(nn.Cell):
    def __init__(self, network):
        super(MSGradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True, get_by_list=True)
        self.network = network
        self.params = ParameterTuple(network.trainable_params())

    def construct(self, x, dy):
        grad_op = self.grad(self.network, self.params)
        output = grad_op(x, dy)
        return output


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lrn_ms():
    x = Tensor(np.array([[[[1.6243454, -0.6117564],
                           [-0.5281718, -1.0729686]],
                          [[0.86540765, -2.3015387],
                           [1.7448118, -0.7612069]],
                          [[0.3190391, -0.24937038],
                           [1.4621079, -2.0601406]]]]).astype(np.float32))
    y_exp = np.array([[[[1.6239204, -0.61149347],
                        [-0.5279556, -1.0724881]],
                       [[0.86518127, -2.3005495],
                        [1.7440975, -0.760866]],
                       [[0.31895563, -0.2492632],
                        [1.4615093, -2.059218]]]]).astype(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = MSLRNOpNet()
    output = net(x)
    assert np.allclose(output.asnumpy(), y_exp)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = MSLRNOpNet()
    output = net(x)
    assert np.allclose(output.asnumpy(), y_exp)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lrn_grad():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = Tensor(np.array([[[[1.6243454, -0.6117564],
                           [-0.5281718, -1.0729686]],
                          [[0.86540765, -2.3015387],
                           [1.7448118, -0.7612069]],
                          [[0.3190391, -0.24937038],
                           [1.4621079, -2.0601406]]]]).astype(np.float32))
    dy = Tensor(np.array([[[[-0.3224172, -0.38405436],
                            [1.1337694, -1.0998913]],
                           [[-0.1724282, -0.8778584],
                            [0.04221375, 0.58281523]],
                           [[-1.1006192, 1.1447237],
                            [0.9015907, 0.50249434]]]]).astype(np.float32))
    dx_exp = np.array([[[[-0.3220835, -0.3837087],
                         [1.133368, -1.0994467]],
                        [[-0.17225023, -0.8768017],
                         [0.04198911, 0.5825201]],
                        [[-1.1002823, 1.1443052],
                         [0.9010479, 0.50217706]]]]).astype(np.float32)
    net = MSLRNOpNet()
    grad_net = MSGradNet(net)
    grad_net.set_train(True)
    output = grad_net(x, dy)
    dx = output[0][0].asnumpy()
    assert np.allclose(dx, dx_exp, atol=1.0e-4, rtol=1.0e-4, equal_nan=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lrn_grad_dynamic_shape():
    """
    Feature: test lrn_grad op in gpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_dyn = Tensor(shape=[1, 32, 64, None], dtype=mstype.float32)
    dy_dyn = Tensor(shape=[1, 32, 64, None], dtype=mstype.float32)
    net = MSLRNOpNet()
    grad_net = MSGradNet(net)
    grad_net.set_train(True)
    grad_net.set_inputs(x_dyn, dy_dyn)
    x = np.random.randn(1, 32, 64, 64)
    dy = np.random.randn(1, 32, 64, 64)
    output = grad_net(Tensor(x, mstype.float32), Tensor(dy, mstype.float32))
    expect_shape = (1, 32, 64, 64)
    assert output[0][0].asnumpy().shape == expect_shape
