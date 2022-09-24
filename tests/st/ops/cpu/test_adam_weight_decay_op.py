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
from mindspore.nn import Dense
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore._checkparam import Rel
from mindspore._checkparam import Validator as validator
from mindspore.nn.optim.optimizer import Optimizer

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


def _adam_opt(opt, beta1, beta2, eps, lr, weight_decay, param, m, v, gradient):
    """
    Update parameters by AdamWeightDecay op.
    """
    opt(param, m, v, lr, beta1, beta2, eps, weight_decay, gradient)
    return True


def _check_param_value(beta1, beta2, eps, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_float_range(beta1, 0.0, 1.0, Rel.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, Rel.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)


class AdamWeightDecayOp(Optimizer):
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(AdamWeightDecayOp, self).__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self.parameters.clone(prefix="adam_m", init='zeros')
        self.moments2 = self.parameters.clone(prefix="adam_v", init='zeros')
        self.opt = P.AdamWeightDecay()
        self.opt.set_device("CPU")

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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adam_weight_decay():
    epoch = 3
    net = NetAdamWeightDecay()
    optimizer = AdamWeightDecayOp(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate=0.01)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()

    losses1 = []
    for _ in range(epoch):
        data = Tensor(np.arange(0, 16).reshape(1, 1, 4, 4).astype(np.float32) * 0.01)
        label = Tensor(np.array([0]).astype(np.int32))
        loss = train_network(data, label)
        losses1.append(loss.asnumpy())
    assert losses1[0] > losses1[1]
    assert losses1[1] > losses1[2]
