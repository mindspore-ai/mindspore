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

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=True)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_smoothl1loss():
    np.random.seed(42)
    prediction = np.random.randn(20).astype(np.float32)
    target = np.random.randn(20).astype(np.float32)
    sigma = 1.0

    net = nn.SmoothL1Loss(sigma)
    loss = net(Tensor(prediction), Tensor(target))
    expect = [0.46941718, 0.00382918, 0.16829303, 2.447778, 0.04812113, 0.05953304,
              2.2302065, 0.07672881, 0.00860204, 0.34798968, 0.00956192, 1.818008,
              0.03262977, 0.36599946, 2.047463, 0.2168481, 0.7216947, 1.7739174,
              0.08826803, 1.109165]
    assert np.allclose(loss.asnumpy(), expect)



class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(name="get_all", get_all=True, sens_param=True)
        self.network = network

    def construct(self, x1, x2, sens):
        gout = self.grad(self.network)(x1, x2, sens)
        return gout


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_smoothl1loss_grad():
    np.random.seed(42)
    prediction = np.random.randn(20).astype(np.float32)
    target = np.random.randn(20).astype(np.float32)
    sens = np.random.randn(20).astype(np.float32)
    sigma = 1.0

    net = nn.SmoothL1Loss(sigma)
    grad = Grad(net)
    dx = grad(Tensor(prediction), Tensor(target), Tensor(sens))

    dx1_expect = [-0.71552587, 0.01499678, -0.06709455, -0.30110368, -0.45868093,
                  0.24838912, -0.46063876, 0.41411355, 0.04507046, -1.4708229,
                  0.04481723, 0.38508227, -0.17292616, -0.52333146, -1.0309995,
                  0.61330026, 0.83921754, -0.3092124, 0.1391843, -0.9755451]

    dx2_expect = [0.71552587, -0.01499678, 0.06709455, 0.30110368, 0.45868093,
                  -0.24838912, 0.46063876, -0.41411355, -0.04507046, 1.4708229,
                  -0.04481723, -0.38508227, 0.17292616, 0.52333146, 1.0309995,
                  -0.61330026, -0.83921754, 0.3092124, -0.1391843, 0.9755451]

    assert np.allclose(dx[0].asnumpy(), dx1_expect)
    assert np.allclose(dx[1].asnumpy(), dx2_expect)
