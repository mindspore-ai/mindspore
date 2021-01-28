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

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def __init__(self, reduction="none"):
        super(Net, self).__init__()
        self.BinaryCrossEntropy = P.BinaryCrossEntropy(reduction)

    def construct(self, x, y, weight):
        return self.BinaryCrossEntropy(x, y, weight)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_binary_cross_entropy_loss():
    np.random.seed(42)
    prediction = np.random.rand(20).astype(np.float32)
    target = np.random.rand(20).astype(np.float32)
    weight = np.random.rand(20).astype(np.float32)
    reduction = "none"
    net = Net(reduction)
    loss = net(Tensor(prediction), Tensor(target), Tensor(weight))
    expect = [0.09555826, 1.2861121, 0.03518666, 0.6969416, 0.24313456, 0.99062896,
              0.19205657, 0.5465214, 0.36964455, 0.21999404, 2.2953863, 2.2566645,
              1.5803775, 1.3266402, 0.9883408, 1.2997618, 0.05439841, 0.14389999,
              0.03405444, 0.23934692]
    assert np.allclose(loss.asnumpy(), expect)

def test_binary_cross_entropy_loss_mean():
    np.random.seed(42)
    prediction = np.random.rand(20).astype(np.float32)
    target = np.random.rand(20).astype(np.float32)
    weight = np.random.rand(20).astype(np.float32)
    reduction = "mean"
    net = Net(reduction)
    loss = net(Tensor(prediction), Tensor(target), Tensor(weight))
    expect = [0.7447324991226196]
    assert loss.asnumpy() == expect

def test_binary_cross_entropy_loss_sum():
    np.random.seed(42)
    prediction = np.random.rand(20).astype(np.float32)
    target = np.random.rand(20).astype(np.float32)
    weight = np.random.rand(20).astype(np.float32)
    reduction = "sum"
    net = Net(reduction)
    loss = net(Tensor(prediction), Tensor(target), Tensor(weight))
    expect = [14.894649505615234]
    assert loss.asnumpy() == expect

def test_binary_cross_entropy_loss_16():
    np.random.seed(42)
    prediction = np.random.rand(20).astype(np.float16)
    target = np.random.rand(20).astype(np.float16)
    weight = np.random.rand(20).astype(np.float16)
    reduction = "none"
    net = Net(reduction)
    loss = net(Tensor(prediction), Tensor(target), Tensor(weight))
    expect = [0.09552, 1.28613, 0.0351868, 0.696777, 0.243164, 0.990234,
              0.192139, 0.546875, 0.370117, 0.219971, 2.29492, 2.25391,
              1.58105, 1.32812, 0.987305, 1.30078, 0.0544434, 0.143921,
              0.0340576, 0.239258]
    assert np.allclose(loss.asnumpy(), expect)

def test_binary_cross_entropy_loss_mean_16():
    np.random.seed(42)
    prediction = np.random.rand(20).astype(np.float16)
    target = np.random.rand(20).astype(np.float16)
    weight = np.random.rand(20).astype(np.float16)
    reduction = "mean"
    net = Net(reduction)
    loss = net(Tensor(prediction), Tensor(target), Tensor(weight))
    expect = [0.74462890625]
    assert loss.asnumpy() == expect

def test_binary_cross_entropy_loss_sum_16():
    np.random.seed(42)
    prediction = np.random.rand(20).astype(np.float16)
    target = np.random.rand(20).astype(np.float16)
    weight = np.random.rand(20).astype(np.float16)
    reduction = "sum"
    net = Net(reduction)
    loss = net(Tensor(prediction), Tensor(target), Tensor(weight))
    expect = [14.890625]
    assert loss.asnumpy() == expect

class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, x1, x2, sens, weight):
        gout = self.grad(self.network)(x1, x2, sens, weight)
        return gout


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_binary_cross_entropy_loss_grad():
    np.random.seed(42)
    prediction = np.random.rand(20).astype(np.float32)
    target = np.random.rand(20).astype(np.float32)
    sens = np.random.rand(20).astype(np.float32)
    weight = np.random.rand(20).astype(np.float32)
    reduction = "none"
    grad = Grad(Net(reduction))
    dx = grad(Tensor(prediction), Tensor(target), Tensor(sens), Tensor(weight))

    dx1_expect = [-4.80516590e-02, 2.32625079e+00, 6.38972521e-02, 3.13642323e-01,
                  -1.65661633e-01, -1.71821892e+00, -1.13685496e-01, 1.26669514e+00,
                  1.47891801e-03, 5.83921909e-01, -2.17992840e+01, 4.21899414e+00,
                  2.85430793e-02, -3.21346498e+00, -2.22674108e+00, -2.80453944e+00,
                  -1.19787852e-04, 2.48514321e-02, -1.66696273e-02, -2.71965731e-02]

    assert np.allclose(dx[0].asnumpy(), dx1_expect)
