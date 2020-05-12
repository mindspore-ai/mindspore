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
from mindspore import nn
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore.nn.layer.activation import get_activation
from src.metrics import Loss, Accuracy


def glorot(shape):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = np.random.uniform(-init_range, init_range, shape).astype(np.float32)
    return Tensor(initial)


class GraphConvolution(nn.Cell):
    def __init__(self,
                 feature_in_dim,
                 feature_out_dim,
                 dropout_ratio=None,
                 activation=None):
        super(GraphConvolution, self).__init__()
        self.in_dim = feature_in_dim
        self.out_dim = feature_out_dim
        self.weight_init = glorot([self.out_dim, self.in_dim])
        self.fc = nn.Dense(self.in_dim,
                           self.out_dim,
                           weight_init=self.weight_init,
                           has_bias=False)
        self.dropout_ratio = dropout_ratio
        if self.dropout_ratio is not None:
            self.dropout = nn.Dropout(keep_prob=1-self.dropout_ratio)
        self.dropout_flag = self.dropout_ratio is not None
        self.activation = get_activation(activation)
        self.activation_flag = self.activation is not None
        self.matmul = P.MatMul()

    def construct(self, adj, input_feature):
        dropout = input_feature
        if self.dropout_flag:
            dropout = self.dropout(dropout)

        fc = self.fc(dropout)
        output_feature = self.matmul(adj, fc)

        if self.activation_flag:
            output_feature = self.activation(output_feature)
        return output_feature


class GCN(nn.Cell):
    def __init__(self, config, adj, feature, output_dim):
        super(GCN, self).__init__()
        self.adj = Tensor(adj)
        self.feature = Tensor(feature)
        input_dim = feature.shape[1]
        self.layer0 = GraphConvolution(input_dim, config.hidden1, activation="relu", dropout_ratio=config.dropout)
        self.layer1 = GraphConvolution(config.hidden1, output_dim, dropout_ratio=None)

    def construct(self):
        output0 = self.layer0(self.adj, self.feature)
        output1 = self.layer1(self.adj, output0)
        return output1


class LossAccuracyWrapper(nn.Cell):
    def __init__(self, network, label, mask, weight_decay):
        super(LossAccuracyWrapper, self).__init__()
        self.network = network
        self.loss = Loss(label, mask, weight_decay, network.trainable_params()[0])
        self.accuracy = Accuracy(label, mask)

    def construct(self):
        preds = self.network()
        loss = self.loss(preds)
        accuracy = self.accuracy(preds)
        return loss, accuracy


class LossWrapper(nn.Cell):
    def __init__(self, network, label, mask, weight_decay):
        super(LossWrapper, self).__init__()
        self.network = network
        self.loss = Loss(label, mask, weight_decay, network.trainable_params()[0])

    def construct(self):
        preds = self.network()
        loss = self.loss(preds)
        return loss


class TrainOneStepCell(nn.Cell):
    r"""
    Network training package class.

    Wraps the network with an optimizer. The resulting Cell be trained without inputs.
    Backward graph will be created in the construct function to do parameter updating. Different
    parallel modes are available to run the training.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The scaling number to be filled as the input of backpropagation. Default value is 1.0.

    Outputs:
        Tensor, a scalar Tensor with shape :math:`()`.

    Examples:
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> loss_net = nn.WithLossCell(net, loss_fn)
        >>> train_net = nn.TrainOneStepCell(loss_net, optim)
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation('grad', get_by_list=True, sens_param=True)
        self.sens = sens

    def construct(self):
        weights = self.weights
        loss = self.network()
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(sens)
        return F.depend(loss, self.optimizer(grads))


class TrainNetWrapper(nn.Cell):
    def __init__(self, network, label, mask, config):
        super(TrainNetWrapper, self).__init__(auto_prefix=True)
        self.network = network
        loss_net = LossWrapper(network, label, mask, config.weight_decay)
        optimizer = nn.Adam(loss_net.trainable_params(),
                            learning_rate=config.learning_rate)
        self.loss_train_net = TrainOneStepCell(loss_net, optimizer)
        self.accuracy = Accuracy(label, mask)

    def construct(self):
        loss = self.loss_train_net()
        accuracy = self.accuracy(self.network())
        return loss, accuracy
