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

"""Utils for model verification."""

# pylint: disable=arguments-differ

import numpy as np

import mindspore.nn as nn
from mindspore import Parameter, ParameterTuple, Tensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P


class SquaredLoss(nn.Cell):
    """Squared loss function."""

    def __init__(self):
        super(SquaredLoss, self).__init__()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.two = Tensor(np.array([2.0]).astype(np.float32))
        self.reduce_sum = P.ReduceSum()

    def construct(self, y_hat, y):
        ret = y_hat - self.reshape(y, self.shape(y_hat))
        return self.reduce_sum((ret * ret) / self.two, (0,))


opt_step = C.MultitypeFuncGraph("opt_step")


@opt_step.register("Tensor", "Tensor",
                   "Tensor", "Tensor")
def update_opt_step(learning_rate, batch_size, parameter, gradient):
    """
    Update opt step.

    Args:
        learning_rate (Tensor): Learning rate.
        batch_size (Tensor): Batch Size.
        parameter (Tensor): Parameter.
        gradient (Tensor): Gradients.

    Returns:
    """
    next_param = parameter - learning_rate * gradient / batch_size
    F.assign(parameter, next_param)
    return next_param


class SGD(nn.Cell):
    """SGD optimizer."""

    def __init__(self, parameters, learning_rate=0.001, batch_size=1):
        super(SGD, self).__init__()
        self.parameters = ParameterTuple(parameters)
        self.learning_rate = Tensor(np.array([learning_rate]).astype(np.float32))
        self.batch_size = Tensor(np.array([batch_size]).astype(np.float32))
        self.hyper_map = C.HyperMap()

    def set_params(self, parameters):
        self.parameters = parameters

    def construct(self, gradients):
        success = self.hyper_map(F.partial(opt_step, self.learning_rate, self.batch_size),
                                 self.parameters, gradients)
        return success


class Linreg(nn.Cell):
    """Linear regression model."""

    def __init__(self, num_features):
        super(Linreg, self).__init__()
        self.matmul = P.MatMul()
        self.w = Parameter(Tensor(np.random.normal(scale=0.01, size=(num_features, 1)).astype(np.float32)), name='w')
        self.b = Parameter(Tensor(np.zeros(shape=(1,)).astype(np.float32)), name='b')

    def construct(self, x):
        return self.matmul(x, self.w) + self.b


class Model:
    """Simplified model."""

    def __init__(self, network, loss_fn, optimizer):
        self.optimizer = optimizer
        self.step = nn.TrainOneStepCell(nn.WithLossCell(network, loss_fn), self.optimizer)

    def optimize(self, data, label):
        return self.step(data, label)

    def train(self, num_epochs, train_dataset):
        train_dataset = list(train_dataset)
        for epoch in range(num_epochs):
            for x, y in train_dataset:
                loss = self.optimize(x, y)
            print('epoch %d, loss %f' % (epoch + 1, loss.asnumpy().mean()))
        return loss
