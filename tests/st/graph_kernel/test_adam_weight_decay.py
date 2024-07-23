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
from tests.mark_utils import arg_mark

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn import Dense
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.optim.optimizer import Optimizer


def _adam_opt(opt, beta1, beta2, eps, lr, weight_decay, param, m, v, gradient):
    """
    Update parameters by AdamWeightDecay op.
    """
    success = True
    next_param = opt(param, m, v, lr, beta1, beta2, eps, weight_decay, gradient)
    return F.depend(success, next_param)


class AdamWeightDecayOp(Optimizer):
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(AdamWeightDecayOp, self).__init__(learning_rate, params, weight_decay)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self.parameters.clone(prefix="adam_m", init='zeros')
        self.moments2 = self.parameters.clone(prefix="adam_v", init='zeros')
        self.opt = P.AdamWeightDecay()

    def construct(self, gradients):
        """AdamWeightDecayOp"""
        lr = self.get_lr()
        optim_result = self.map_reverse(F.partial(_adam_opt, self.opt, self.beta1, self.beta2, self.eps, lr,
                                                  self.weight_decay), self.parameters, self.moments1, self.moments2,
                                        gradients)
        return optim_result


class NetAdamWeightDecay(nn.Cell):
    def __init__(self):
        super(NetAdamWeightDecay, self).__init__()
        self.batch_size = 1
        self.reshape = P.Reshape()
        weight = Tensor(np.ones([10, 16]).astype(np.float32) * 0.01)
        bias = Tensor(np.zeros(10).astype(np.float32))
        self.fc1 = Dense(16, 10, weight_init=weight, bias_init=bias)

    def construct(self, input_x):
        output = self.reshape(input_x, (self.batch_size, -1))
        output = self.fc1(output)
        return output


def run_adam_weight_decay(enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    epoch = 3
    net = NetAdamWeightDecay()
    optimizer = AdamWeightDecayOp(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate=0.01)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()

    losses = []
    for _ in range(epoch):
        data = Tensor(np.arange(0, 16).reshape(1, 1, 4, 4).astype(np.float32) * 0.01)
        label = Tensor(np.array([0]).astype(np.int32))
        loss = train_network(data, label)
        losses.append(loss.asnumpy())
    return losses


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_adam_weight_decay():
    """
    Feature: graph kernel testcase for AdamWeightDecay
    Description: fixed input when using graph_kernel in graph mode
    Expectation: get the same result when using and not using graph kernel
    """
    context.set_context(mode=context.GRAPH_MODE)
    expect = run_adam_weight_decay(False)
    result = run_adam_weight_decay(True)
    res1 = np.allclose(expect[0], result[0], rtol=1.e-4, atol=1.e-4, equal_nan=True)
    assert res1
    res2 = np.allclose(expect[1], result[1], rtol=1.e-4, atol=1.e-4, equal_nan=True)
    assert res2
    res3 = np.allclose(expect[2], result[2], rtol=1.e-4, atol=1.e-4, equal_nan=True)
    assert res3
