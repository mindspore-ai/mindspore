# Copyright 2019 Huawei Technologies Co., Ltd
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

import numpy as np

import mindspore as ms
import mindspore.common.dtype as DT
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.nn import WithLossCell
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.train.model import Model
from mindspore.train.parallel_utils import ParallelMode
from tests.dataset_mock import MindData


class Dataset(MindData):
    def __init__(self, predict, label, length=3):
        super(Dataset, self).__init__(size=length)
        self.predict = predict
        self.label = label
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.predict, self.label

    def reset(self):
        self.index = 0


class FusedBatchNorm(nn.Cell):
    """Batch Normalization base class."""

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 gamma_init='ones',
                 beta_init='zeros',
                 moving_mean_init='zeros',
                 moving_var_init='ones'):
        super(FusedBatchNorm, self).__init__()
        if num_features < 1:
            raise ValueError("num_features must be at least 1")

        if momentum < 0 or momentum > 1:
            raise ValueError("momentum should be a number in range [0, 1], but got {}".format(momentum))

        self.num_features = num_features
        self.eps = eps
        self.momentum = Tensor(1.0 - momentum, DT.float32)
        self.gamma = Parameter(initializer(
            gamma_init, num_features), name="gamma", requires_grad=affine)
        self.beta = Parameter(initializer(
            beta_init, num_features), name="beta", requires_grad=affine)
        self.moving_mean = Parameter(initializer(
            moving_mean_init, num_features), name="mean", requires_grad=False)
        self.moving_variance = Parameter(initializer(
            moving_var_init, num_features), name="variance", requires_grad=False)

        self.bn_train = P.BatchNorm(is_training=True,
                                    epsilon=self.eps)
        self.bn_infer = P.BatchNorm(is_training=False,
                                    epsilon=self.eps)
        self.sub_mean = P.Sub().set_strategy(((1), (1)))
        self.sub_var = P.Sub().set_strategy(((1), (1)))
        self.mul_mean = P.Mul().set_strategy(((1,), ()))
        self.mul_var = P.Mul().set_strategy(((1,), ()))
        self.assign_sub_mean = P.AssignSub().set_strategy(((1,), (1,)))
        self.assign_sub_var = P.AssignSub().set_strategy(((1), (1)))
        self.sub_mean2 = P.Sub().set_strategy(((1), (1)))
        self.sub_var2 = P.Sub().set_strategy(((1), (1)))

    def set_strategy(self, strategy):
        self.bn_train.set_strategy(strategy)
        self.bn_infer.set_strategy(strategy)

    def _check_data_dim(self, x):
        raise NotImplementedError

    def construct(self, x):
        if self.training:
            y, batch_mean, batch_var, _, _ = \
                self.bn_train(x,
                              self.gamma,
                              self.beta,
                              None,
                              None)

            mean_sub = self.sub_mean(self.moving_mean, batch_mean)
            temp_mean = self.mul_mean(mean_sub, self.momentum)
            mean_sub2 = self.sub_var(self.moving_variance, batch_var)
            temp_variance = self.mul_var(mean_sub2, self.momentum)
            y = F.depend(y, self.assign_sub_mean(self.moving_mean, temp_mean))
            y = F.depend(y, self.assign_sub_var(self.moving_variance, temp_variance))

        else:
            y = self.bn_infer(x,
                              self.gamma,
                              self.beta,
                              self.moving_mean,
                              self.moving_variance)[0]
        return y

    def extend_repr(self):
        return 'num_features={}, eps={}, momentum={}, ' \
               'beta={}, gamma={}, ' \
               'moving_mean={}, moving_variance={} ' \
            .format(self.num_features,
                    self.eps,
                    self.momentum,
                    self.beta,
                    self.gamma,
                    self.moving_mean,
                    self.moving_variance)


class PReLU(nn.Cell):
    """
    PReLU activation function.

    Computes prelu value of a 4-dim tensor(NCHW).
    PReLU: out = max(0, A) + min(0, wA)

    Args:
        channel: Integer. The dimensionality of w. Default: 1.
        w: Float. The initial value of w. Default: 0.25.

    Returns:
        Tensor, has the same type as features.

    Examples:
        prelu = nn.PReLU(1, [np.float32(0.25)]) # or prelu = nn.PReLU(33, Tensor(np.random.rand(33), ms.float32)])
        input_data = Tensor(np.random.rand(1, 33, 4, 4), ms.float32)
        output = prelu.construct(input_data)
    """

    def __init__(self, channel=1, w=0.25):
        super(PReLU, self).__init__()
        if isinstance(w, (np.float32, float)):
            tmp = np.empty((channel,), dtype=np.float32)
            tmp.fill(w)
            w = tmp
        elif isinstance(w, (int, bool, complex, str)):
            raise TypeError("w only support input type float32 and float")

        if not isinstance(w, Tensor):
            w = Tensor(w)
        self.w = Parameter(initializer(w, [channel, ]), name='a')
        self.prelu = P.PReLU()
        self.relu = P.ReLU().set_strategy(((1)))

    def construct(self, x):
        self.w = self.relu(self.w)
        return self.prelu(x, self.w)


class BNNet(nn.Cell):
    def __init__(self, strategy0, strategy1, strategy2):
        super(BNNet, self).__init__()
        self.bn = FusedBatchNorm(512)
        self.prelu = PReLU(512)

    def construct(self, x):
        x = self.bn(x)
        x = self.prelu(x)
        return x


def bn_net(strategy0, strategy1, strategy2):
    return BNNet(strategy0=strategy0, strategy1=strategy1, strategy2=strategy2)


def bn_common(parallel_mode, train_flag, strategy0=None, strategy1=None, strategy2=None, strategy_loss=None):
    context.set_context(mode=context.GRAPH_MODE)
    batch_size = 32
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2
    rank_size = 8

    predict = Tensor(np.ones([32, 512]), dtype=ms.float32)
    label = Tensor(np.ones([32]), dtype=ms.int32)
    dataset = Dataset(predict, label, 2)
    net = bn_net(strategy0, strategy1, strategy2)

    loss = SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
    loss.softmax_cross_entropy.set_strategy(strategy_loss)
    opt = Momentum(net.trainable_params(), learning_rate, momentum, 0.0001, 1024 * rank_size)

    if not train_flag:
        net = WithLossCell(net, loss)
        net.set_train()

    if parallel_mode == ParallelMode.DATA_PARALLEL:
        context.set_auto_parallel_context(parameter_broadcast=True)
    context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=8)
    model = Model(net, loss, opt)
    if train_flag:
        model.train(epoch_size, dataset, dataset_sink_mode=False)
    else:
        model._predict(predict, label)


def test_data_parallel():
    parallel_mode = ParallelMode.DATA_PARALLEL
    train_flag = True
    bn_common(parallel_mode, train_flag)


def auto_parallel():
    train_flag = True
    parallel_mode = ParallelMode.AUTO_PARALLEL
    bn_common(parallel_mode, train_flag)


def Xtest_data_parallel_predict():
    parallel_mode = ParallelMode.DATA_PARALLEL
    train_flag = False
    bn_common(parallel_mode, train_flag)


def Xtest_semi_auto_parallel_predict():
    train_flag = False
    parallel_mode = ParallelMode.SEMI_AUTO_PARALLEL
    bn_common(parallel_mode, train_flag)


def Xtest_auto_parallel_predict():
    train_flag = False
    parallel_mode = ParallelMode.AUTO_PARALLEL
    bn_common(parallel_mode, train_flag)


if __name__ == '__main__':
    auto_parallel()
