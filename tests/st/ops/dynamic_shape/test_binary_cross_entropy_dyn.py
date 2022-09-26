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

"""test BinaryCrossEntropy forward and backward dynamic shape"""

import numpy as np
import pytest

import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self, reduction):
        super().__init__()
        self.bce = P.BinaryCrossEntropy(reduction)

    def construct(self, x, y, weight=None):
        return self.bce(x, y, weight)


class Grad(nn.Cell):
    def __init__(self, network, sens):
        super().__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network
        self.sens = sens

    def construct(self, x, y, weight=None):
        gout = self.grad(self.network)(x, y, weight, self.sens)
        return gout


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_binary_cross_entropy_loss():
    """
    Feature: test binary_cross_entropy op with reduction none.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    prediction = np.random.rand(20).astype(np.float32)
    target = np.random.rand(20).astype(np.float32)
    weight = np.random.rand(20).astype(np.float32)
    prediction_dyn = Tensor(shape=(None,), dtype=mstype.float32)
    target_dyn = Tensor(shape=(None,), dtype=mstype.float32)
    weight_dyn = Tensor(shape=(None,), dtype=mstype.float32)
    reduction = "none"

    net = Net(reduction)
    net.set_inputs(prediction_dyn, target_dyn, weight_dyn)
    loss = net(Tensor(prediction), Tensor(target), Tensor(weight))
    assert loss.asnumpy().shape == prediction.shape

    grad_net = Grad(net, loss)
    grad_net.set_inputs(prediction_dyn, target_dyn, weight_dyn)
    grad = grad_net(Tensor(prediction), Tensor(target), Tensor(weight))
    assert grad[0].asnumpy().shape == prediction.shape
    assert grad[1].asnumpy().shape == target.shape
    assert grad[2].asnumpy().shape == weight.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_binary_cross_entropy_loss_mean_reduction():
    """
    Feature: test binary_cross_entropy op with reduction mean.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    prediction = np.random.rand(20).astype(np.float32)
    target = np.random.rand(20).astype(np.float32)
    weight = np.random.rand(20).astype(np.float32)
    prediction_dyn = Tensor(shape=(None,), dtype=mstype.float32)
    target_dyn = Tensor(shape=(None,), dtype=mstype.float32)
    weight_dyn = Tensor(shape=(None,), dtype=mstype.float32)
    reduction = "mean"

    net = Net(reduction)
    net.set_inputs(prediction_dyn, target_dyn, weight_dyn)
    loss = net(Tensor(prediction), Tensor(target), Tensor(weight))
    assert loss.asnumpy().shape == tuple()

    grad_net = Grad(net, loss)
    grad_net.set_inputs(prediction_dyn, target_dyn, weight_dyn)
    grad = grad_net(Tensor(prediction), Tensor(target), Tensor(weight))
    assert grad[0].asnumpy().shape == prediction.shape
    assert grad[1].asnumpy().shape == target.shape
    assert grad[2].asnumpy().shape == weight.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_binary_cross_entropy_loss_sum_reduction():
    """
    Feature: test binary_cross_entropy op with reduction sum.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    prediction = np.random.rand(20).astype(np.float32)
    target = np.random.rand(20).astype(np.float32)
    weight = np.random.rand(20).astype(np.float32)
    prediction_dyn = Tensor(shape=(None,), dtype=mstype.float32)
    target_dyn = Tensor(shape=(None,), dtype=mstype.float32)
    weight_dyn = Tensor(shape=(None,), dtype=mstype.float32)
    reduction = "mean"

    net = Net(reduction)
    net.set_inputs(prediction_dyn, target_dyn, weight_dyn)
    loss = net(Tensor(prediction), Tensor(target), Tensor(weight))
    assert loss.asnumpy().shape == tuple()

    grad_net = Grad(net, loss)
    grad_net.set_inputs(prediction_dyn, target_dyn, weight_dyn)
    grad = grad_net(Tensor(prediction), Tensor(target), Tensor(weight))
    assert grad[0].asnumpy().shape == prediction.shape
    assert grad[1].asnumpy().shape == target.shape
    assert grad[2].asnumpy().shape == weight.shape
