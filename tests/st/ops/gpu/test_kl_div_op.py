# Copyright 2020 Huawei Technologies Co., Ltd
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

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class Net(nn.Cell):
    def __init__(self, reduction="none"):
        super(Net, self).__init__()
        self.KLDivLoss = P.KLDivLoss("none")

    def construct(self, x, y):
        return self.KLDivLoss(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_binary_cross_entropy_loss():
    np.random.seed(42)
    prediction = np.random.rand(20).astype(np.float32)
    target = np.random.rand(20).astype(np.float32)
    net = Net()
    loss = net(Tensor(prediction), Tensor(target))
    expect = [-0.5297444, -0.40738472, -0.5733339, -0.58720195, -0.42922008, -0.31237593,
              -0.3332863, -0.78742254, -0.6662671, -0.17546377, -0.31526336, -0.46702948,
              -0.23191005, -0.2512708, -0.20934652, -0.32021108, -0.45477402, -0.278453,
              -0.5551879, -0.48938933]
    assert np.allclose(loss.asnumpy(), expect)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, x1, x2, sens):
        gout = self.grad(self.network)(x1, x2, sens)
        return gout


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_binary_cross_entropy_loss_grad():
    np.random.seed(42)
    prediction = np.random.rand(20).astype(np.float32)
    target = np.random.rand(20).astype(np.float32)
    sens = np.random.rand(20).astype(np.float32)
    grad = Grad(Net())
    dx = grad(Tensor(prediction), Tensor(target), Tensor(sens))

    dx1_expect = [-0.07466945, -0.06907414, -0.01004642, -0.3331403, -0.11802178, -0.52019656,
                  -0.06224053, -0.2674369, -0.32387912, -0.00858657, -0.58906615, -0.13217884,
                  -0.06111591, -0.8490888, -0.57735133, -0.7452407, -0.02695603, -0.01914206,
                  -0.03094601, -0.14319494]

    dx2_expect = [0.0163771, -0.950962, -0.03309895, -0.5481312, 0.01523498, 0.39894313,
                  -0.20858267, -0.27628726, -0.06815486, -0.5134226, 0.46645382, -1.3477919,
                  -2.409831, 0.65787154, 0.4682768, 0.55671424, -0.04362264, -0.36274382,
                  0.00852979, -0.03639247]

    assert np.allclose(dx[0].asnumpy(), dx1_expect)
    assert np.allclose(dx[1].asnumpy(), dx2_expect)
