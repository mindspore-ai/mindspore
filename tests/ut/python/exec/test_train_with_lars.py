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
""" test model train """
import mindspore.nn as nn
from mindspore import Tensor, Model
from mindspore.common import dtype as mstype
from mindspore.common.parameter import ParameterTuple, Parameter
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim import Momentum
from mindspore.ops import composite as C
from mindspore.ops import operations as P


def get_reordered_parameters(parameters):
    """get_reordered_parameters"""
    # put the bias parameter to the end
    non_bias_param = []
    bias_param = []
    for item in parameters:
        if item.name.find("bias") >= 0:
            bias_param.append(item)
        else:
            non_bias_param.append(item)
    reordered_params = tuple(non_bias_param + bias_param)
    return len(non_bias_param), len(reordered_params), reordered_params


def get_net_trainable_reordered_params(net):
    params = net.trainable_params()
    return get_reordered_parameters(params)


class TrainOneStepWithLarsCell(nn.Cell):
    """TrainOneStepWithLarsCell definition"""

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepWithLarsCell, self).__init__(auto_prefix=False)
        self.network = network
        self.slice_index, self.params_len, weights = get_net_trainable_reordered_params(self.network)
        self.weights = ParameterTuple(weights)
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.sens = Parameter(Tensor([sens], mstype.float32), name='sens', requires_grad=False)
        self.weight_decay = 1.0
        self.lars = P.Lars(epsilon=1.0, hyperpara=1.0)

    def construct(self, data, label):
        weights = self.weights
        loss = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label, self.sens)
        non_bias_weights = weights[0: self.slice_index]
        non_bias_grads = grads[0: self.slice_index]
        bias_grads = grads[self.slice_index: self.params_len]
        lars_grads = self.lars(non_bias_weights, non_bias_grads, self.weight_decay)
        new_grads = lars_grads + bias_grads
        self.optimizer(new_grads)
        return loss


# fn is a function use i as input
def lr_gen(fn, epoch_size):
    for i in range(epoch_size):
        yield fn(i)


def me_train_tensor(net, input_np, label_np, epoch_size=2):
    """me_train_tensor"""
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # reorder the net parameters , leave the parameters that need to be passed into lars to the end part

    opt = Momentum(get_net_trainable_reordered_params(net)[2], lr_gen(lambda i: 0.1, epoch_size), 0.9, 0.01, 1024)
    Model(net, loss, opt)
    _network = nn.WithLossCell(net, loss)
    TrainOneStepWithLarsCell(_network, opt)
    data = Tensor(input_np)
    label = Tensor(label_np)
    net(data, label)
