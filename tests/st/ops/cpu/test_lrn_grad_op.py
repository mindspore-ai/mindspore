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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.functional import vmap


class LrnNet(nn.Cell):
    def __init__(self):
        super(LrnNet, self).__init__()
        self.lrn = P.LRN(depth_radius=2, bias=1.0, alpha=0.0001, beta=0.75)

    def construct(self, x):
        out = self.lrn(x)
        return out


class LrnGradNet(nn.Cell):
    def __init__(self):
        super(LrnGradNet, self).__init__()
        self.lrn_grad = G.LRNGrad(depth_radius=2, bias=1.0, alpha=0.0001, beta=0.75)

    def construct(self, dy, x, y):
        out = self.lrn_grad(dy, x, y)
        return out


class LrnGradVMapNet(nn.Cell):
    def __init__(self, forward_net, in_axes, out_axes):
        super(LrnGradVMapNet, self).__init__()
        self.net = forward_net
        self.in_axes = in_axes
        self.out_axes = out_axes

    def construct(self, dy, x, y):
        return vmap(self.net, self.in_axes, self.out_axes)(dy, x, y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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
    y = Tensor(np.array([[[[1.6239204, -0.61149347],
                           [-0.5279556, -1.0724881]],
                          [[0.86518127, -2.3005495],
                           [1.7440975, -0.760866]],
                          [[0.31895563, -0.2492632],
                           [1.4615093, -2.059218]]]]).astype(data_type))
    dx_exp = np.array([[[[-0.3220835, -0.3837087],
                         [1.133368, -1.0994467]],
                        [[-0.17225023, -0.8768017],
                         [0.04198911, 0.5825201]],
                        [[-1.1002823, 1.1443052],
                         [0.9010479, 0.50217706]]]]).astype(data_type)
    loss = 1e-6
    if data_type == np.float16:
        loss = 1e-3
    lrn_grad_net = LrnGradNet()
    dx = lrn_grad_net(dy, x, y)
    assert np.allclose(dx.asnumpy(), dx_exp, atol=loss, rtol=loss, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lrn_grad_vmap():
    """
    Feature: Test LRN Grad Vmap on CPU.
    Description: The output shape match to input shape.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    data_type = np.float32
    loss = 1e-6
    x = Tensor(np.array([[[[[1.6243454, -0.6117564],
                            [-0.5281718, -1.0729686]],
                           [[0.86540765, -2.3015387],
                            [1.7448118, -0.7612069]],
                           [[0.3190391, -0.24937038],
                            [1.4621079, -2.0601406]]]],
                         [[[[1.6243454, -0.6117564],
                            [-0.5281718, -1.0729686]],
                           [[0.86540765, -2.3015387],
                            [1.7448118, -0.7612069]],
                           [[0.3190391, -0.24937038],
                            [1.4621079, -2.0601406]]]]]).astype(data_type))
    y = Tensor(np.array([[[[[1.6239204, -0.61149347],
                            [-0.5279556, -1.0724881]],
                           [[0.86518127, -2.3005495],
                            [1.7440975, -0.760866]],
                           [[0.31895563, -0.2492632],
                            [1.4615093, -2.059218]]]],
                         [[[[1.6239204, -0.61149347],
                            [-0.5279556, -1.0724881]],
                           [[0.86518127, -2.3005495],
                            [1.7440975, -0.760866]],
                           [[0.31895563, -0.2492632],
                            [1.4615093, -2.059218]]]]]).astype(data_type))
    dy = Tensor(np.array([[[[[-0.3224172, -0.38405436],
                             [1.1337694, -1.0998913]],
                            [[-0.1724282, -0.8778584],
                             [0.04221375, 0.58281523]],
                            [[-1.1006192, 1.1447237],
                             [0.9015907, 0.50249434]]]],
                          [[[[-0.3224172, -0.38405436],
                             [1.1337694, -1.0998913]],
                            [[-0.1724282, -0.8778584],
                             [0.04221375, 0.58281523]],
                            [[-1.1006192, 1.1447237],
                             [0.9015907, 0.50249434]]]]]).astype(data_type))
    dx_exp = np.array([[[[[-0.3220835, -0.3837087],
                          [1.133368, -1.0994467]],
                         [[-0.17225023, -0.8768017],
                          [0.04198911, 0.5825201]],
                         [[-1.1002823, 1.1443052],
                          [0.9010479, 0.50217706]]]],
                       [[[[-0.3220835, -0.3837087],
                          [1.133368, -1.0994467]],
                         [[-0.17225023, -0.8768017],
                          [0.04198911, 0.5825201]],
                         [[-1.1002823, 1.1443052],
                          [0.9010479, 0.50217706]]]]]).astype(data_type)
    lrn_grad_net = LrnGradNet()
    in_axes = 0
    out_axes = 0
    output = LrnGradVMapNet(lrn_grad_net, in_axes, out_axes)(dy, x, y)
    dx = output.asnumpy()
    np.testing.assert_allclose(dx, dx_exp, rtol=loss, atol=loss)
