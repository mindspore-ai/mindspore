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
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.common.api import _executor
from mindspore.nn.cell import Cell
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.train import Model
from mindspore.context import ParallelMode
from tests.dataset_mock import MindData
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


device_num = 16
device_id = 2


class StrategyModel():
    onehot_strategy = ((1, device_num), (), ())
    twod_strategy = ((1, device_num),)
    twod_strategy_m = ((device_num, 1),)
    scalar_twod_strategy = ((), (1, device_num))
    twod_scalar_strategy = ((1, device_num), ())
    scalar_strategy = ((),)
    oned_strategy = ((1,),)
    scalar_scalar_strategy = ((), ())
    twod_twod_strategy = ((1, device_num), (1, device_num))
    twod_twodbc_strategy = ((1, device_num), (1, 1))
    twodbc_twod_strategy = ((1, 1), (device_num, 1))


class StrategyBatch():
    onehot_strategy = ((device_num, 1), (), ())
    twod_strategy = ((1, device_num),)
    twod_strategy_m = ((device_num, 1),)
    scalar_twod_strategy = ((), (1, device_num))
    twod_scalar_strategy = ((1, device_num), ())
    scalar_strategy = ((),)
    oned_strategy = ((1,),)
    scalar_scalar_strategy = ((), ())
    twod_twod_strategy = ((1, device_num), (1, device_num))
    twod_twodbc_strategy = ((1, device_num), (1, 1))
    twodbc_twod_strategy = ((1, 1), (device_num, 1))


class Args():
    a = 1
    b = 2
    c = 3
    d = 4
    e = 5
    num_classes = 512
    emb_size = 512


class SemiAutoOneHotNet(Cell):
    def __init__(self, args, strategy):
        super(SemiAutoOneHotNet, self).__init__()
        self.a = args.a
        self.b = args.b
        self.c = args.c
        self.d = args.d
        self.e = args.e
        self.cast = P.Cast()
        self.cast.shard(strategy=strategy.twod_strategy)
        self.cast1 = P.Cast()
        self.cast1.shard(strategy=strategy.twod_strategy)
        self.cast2 = P.Cast()
        self.cast2.shard(strategy=strategy.twod_strategy)
        self.cast3 = P.Cast()
        self.cast3.shard(strategy=strategy.scalar_strategy)
        self.cast4 = P.Cast()
        self.cast4.shard(strategy=strategy.scalar_strategy)
        self.a_const = Tensor(self.a, dtype=mstype.float32)
        self.b_const = Tensor(self.b, dtype=mstype.float32)
        self.c_const = Tensor(self.c, dtype=mstype.float32)
        self.d_const = Tensor(self.d, dtype=mstype.float32)
        self.e_const = Tensor(self.e, dtype=mstype.float32)
        self.m_const_zero = Tensor(0, dtype=mstype.float32)
        self.a_const_one = Tensor(1, dtype=mstype.float32)
        self.onehot = P.OneHot()
        self.onehot.shard(strategy=strategy.onehot_strategy)
        self.exp = P.Exp()
        self.exp.shard(strategy=strategy.twod_strategy)
        self.exp2 = P.Exp()
        self.exp2.shard(strategy=strategy.twod_strategy)
        self.exp3 = P.Exp()
        self.exp3.shard(strategy=strategy.twod_strategy)
        self.mul_const = P.Mul()
        self.mul_const.shard(strategy=strategy.scalar_twod_strategy)
        self.mul_const2 = P.Add()
        self.mul_const2.shard(strategy=strategy.scalar_twod_strategy)
        self.mul_const3 = P.Sub()
        self.mul_const3.shard(strategy=strategy.twod_scalar_strategy)
        self.mul_const4 = P.Sub()
        self.mul_const4.shard(strategy=strategy.scalar_twod_strategy)
        self.mul_const5 = P.Mul()
        self.mul_const5.shard(strategy=strategy.twod_scalar_strategy)
        self.mul = P.Mul()
        self.mul.shard(strategy=strategy.twod_twod_strategy)
        self.mul2 = P.Mul()
        self.mul2.shard(strategy=strategy.twod_twod_strategy)
        self.mul3 = P.Add()
        self.mul3.shard(strategy=strategy.twod_twod_strategy)
        self.mul4 = P.Sub()
        self.mul4.shard(strategy=strategy.twod_twodbc_strategy)
        self.mul5 = P.RealDiv()
        self.mul5.shard(strategy=strategy.twod_twodbc_strategy)
        self.mul6 = P.Mul()
        self.mul6.shard(strategy=strategy.twod_twod_strategy)
        self.mul7 = P.Mul()
        self.mul7.shard(strategy=strategy.twod_scalar_strategy)
        self.mul8 = P.RealDiv()
        self.mul8.shard(strategy=strategy.scalar_scalar_strategy)
        self.mul9 = P.Add()
        self.mul9.shard(strategy=strategy.twod_scalar_strategy)

        self.reduce_max = P.ReduceMax(keep_dims=True)
        self.reduce_max.shard(strategy=strategy.twod_strategy)

        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.reduce_sum.shard(strategy=strategy.twod_strategy)
        self.reduce_sum_2 = P.ReduceSum(keep_dims=False)
        self.reduce_sum_2.shard(strategy=strategy.twod_strategy)
        self.reduce_sum_3 = P.ReduceSum(keep_dims=False)
        self.reduce_sum_3.shard(strategy=strategy.oned_strategy)

        self.reshape = P.Reshape()
        self.log = P.Log()
        self.log.shard(strategy=strategy.twod_strategy)

        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.normalize = P.L2Normalize(axis=1)
        self.normalize.shard(strategy=strategy.twod_strategy_m)
        self.normalize2 = P.L2Normalize(axis=1)
        self.normalize2.shard(strategy=strategy.twod_strategy_m)
        self.fc = P.MatMul(transpose_b=True)
        self.fc.shard(strategy=strategy.twodbc_twod_strategy)
        weight_shape = [args.num_classes, args.emb_size]
        weight_np = np.zeros(weight_shape, np.float32)
        self.weight = Parameter(Tensor(weight_np), name='model_parallel_weight')

    def construct(self, input_, label):
        input_n = self.normalize(input_)
        w = self.normalize2(self.weight)
        fc_o = self.fc(input_n, w)
        fc_o_shape = F.shape(fc_o)
        one_hot_float = self.onehot(label, fc_o_shape[1], self.on_value, self.off_value)
        local_label = self.cast(one_hot_float, mstype.int32)

        exp_o = self.exp(fc_o)
        mul_const_o = self.mul_const(self.a_const, exp_o)
        mul_const2_o = self.mul_const2(self.b_const, mul_const_o)
        exp2_o = self.exp2(mul_const2_o)
        mul_const3_o = self.mul_const3(exp2_o, self.c_const)
        mul_const4_o = self.mul_const4(F.scalar_to_array(1), local_label)
        mul6_o = self.mul6(self.mul(mul_const3_o, one_hot_float),
                           self.mul2(fc_o, self.cast2(mul_const4_o, mstype.float32)))
        mul_const5_o = self.mul_const5(mul6_o, self.d_const)

        max_o = self.reduce_max(mul_const5_o, -1)
        mul4_o = self.mul4(mul_const5_o, max_o)
        exp3_o = self.exp3(mul4_o)
        sum_o = self.reduce_sum(exp3_o, -1)
        reshape_o = self.reshape(sum_o, (F.shape(sum_o)[0], 1))
        mul5_o = self.mul5(exp3_o, reshape_o)
        log_o = self.log(self.mul9(mul5_o, self.e_const))
        mul3_o = self.mul3(log_o, one_hot_float)
        mul7_o = self.mul7(mul3_o, self.cast3(F.scalar_to_array(-1), mstype.float32))
        sum2_o = self.reduce_sum_2(mul7_o, -1)
        loss = self.mul8(self.reduce_sum_3(sum2_o, -1),
                         self.cast4(F.scalar_to_array(F.shape(mul_const5_o)[0]), mstype.float32))
        return loss


class Dataset(MindData):
    def __init__(self, predict, label, length=3, input_num=2):
        super(Dataset, self).__init__(size=length)
        self.predict = predict
        self.label = label
        self.index = 0
        self.length = length
        self.input_num = input_num

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        if self.input_num == 2:
            return (self.predict, self.label)
        return (self.predict,)

    def reset(self):
        self.index = 0


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, b):
        predict = self.network(x, b)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, b):
        return grad_all(self.network)(x, b)


def bn_with_initialize(out_channels):
    bn = nn.BatchNorm2d(out_channels, momentum=0.3, eps=1e-5).add_flags_recursive(fp32=True)
    return bn


def fc_with_initialize(input_channels, out_channels):
    return nn.Dense(input_channels, out_channels)


class BNReshapeDenseBNNet(nn.Cell):
    def __init__(self):
        super(BNReshapeDenseBNNet, self).__init__()
        self.batch_norm = bn_with_initialize(2)
        self.reshape = P.Reshape()
        self.batch_norm2 = nn.BatchNorm1d(512, affine=False)
        self.fc = fc_with_initialize(2 * 32 * 32, 512)
        self.loss = SemiAutoOneHotNet(args=Args(), strategy=StrategyBatch())

    def construct(self, x, label):
        x = self.batch_norm(x)
        x = self.reshape(x, (16, 2 * 32 * 32))
        x = self.fc(x)
        x = self.batch_norm2(x)
        loss = self.loss(x, label)
        return loss


def test_bn_reshape_dense_bn_train_loss():
    batch_size = 16
    context.set_auto_parallel_context(device_num=device_num, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    input_ = Tensor(np.ones([batch_size, 2, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([batch_size]), dtype=ms.int32)

    net = GradWrap(NetWithLoss(BNReshapeDenseBNNet()))
    net.set_auto_parallel()

    net.set_train()
    _executor.compile(net, input_, label)


def test_semi_one_hot_net_batch():
    batch_size = 16
    context.set_auto_parallel_context(device_num=device_num, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    input_ = Tensor(np.ones([batch_size * 1, 512]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([batch_size]), dtype=ms.int32)

    net = SemiAutoOneHotNet(args=Args(), strategy=StrategyBatch())
    net = GradWrap(NetWithLoss(net))
    net.set_auto_parallel()

    net.set_train()
    _executor.compile(net, input_, label)


def test_semi_one_hot_net_model():
    batch_size = 16
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2

    predict = Tensor(np.ones([batch_size, 512]), dtype=ms.float32)
    label = Tensor(np.ones([batch_size]), dtype=ms.int32)
    dataset = Dataset(predict, label, 2, input_num=2)

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, device_num=16)
    context.set_context(mode=context.GRAPH_MODE)
    net = SemiAutoOneHotNet(args=Args(), strategy=StrategyModel())
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, optimizer=opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)
