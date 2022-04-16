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
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import operations as P
from mindspore.ops import composite as C


class LrnNet(nn.Cell):
    def __init__(self):
        super(LrnNet, self).__init__()
        self.lrn = P.LRN(depth_radius=2, bias=1.0, alpha=0.0001, beta=0.75)

    def construct(self, x):
        x = self.lrn(x)
        return x


class LrnGradNet(nn.Cell):
    def __init__(self, lrn_network):
        super(LrnGradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True, get_by_list=True)
        self.lrn_network = lrn_network
        self.params = ParameterTuple(lrn_network.trainable_params())

    def construct(self, x, dout):
        grad_op = self.grad(self.lrn_network, self.params)
        output = grad_op(x, dout)
        return output


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_lrn_grad(mode, data_type):
    """
    Feature: Test LrnGrad.
    Description: The input shape need to match to output shape.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[[1.6243454, -0.6117564],
                           [-0.5281718, -1.0729686]],
                          [[0.86540765, -2.3015387],
                           [1.7448118, -0.7612069]],
                          [[0.3190391, -0.24937038],
                           [1.4621079, -2.0601406]]]]).astype(data_type))
    dy = Tensor(np.array([[[[-0.3224172, -0.38405436],
                            [1.1337694, -1.0998913]],
                           [[-0.1724282, -0.8778584],
                            [0.04221375, 0.58281523]],
                           [[-1.1006192, 1.1447237],
                            [0.9015907, 0.50249434]]]]).astype(data_type))
    dx_exp = np.array([[[[-0.3220835, -0.3837087],
                         [1.133368, -1.0994467]],
                        [[-0.17225023, -0.8768017],
                         [0.04198911, 0.5825201]],
                        [[-1.1002823, 1.1443052],
                         [0.9010479, 0.50217706]]]]).astype(data_type)
    error = 1e-4
    if data_type == np.float16:
        error = 1e-3
    lrn_net = LrnNet()
    grad_net = LrnGradNet(lrn_net)
    grad_net.set_train(True)
    output = grad_net(x, dy)
    dx = output[0][0].asnumpy()
    assert np.allclose(dx, dx_exp, atol=error, rtol=error, equal_nan=True)
