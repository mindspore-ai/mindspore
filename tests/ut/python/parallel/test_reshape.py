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
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _executor
from mindspore.common.parameter import Parameter
from mindspore.common.parameter import ParameterTuple
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.parallel import set_algo_parameters
from mindspore.train import Model
from mindspore.context import ParallelMode
from tests.dataset_mock import MindData
from tests.ut.python.ops.test_math_ops import VirtualLoss

context.set_context(mode=context.GRAPH_MODE)
context.reset_auto_parallel_context()


grad_all = C.GradOperation(get_all=True)


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


class ReshapeNet(nn.Cell):
    def __init__(self, strategy0, strategy1, strategy2):
        super(ReshapeNet, self).__init__()
        self.relu = P.ReLU().shard(strategy0)
        self.reshape = P.Reshape().shard(strategy1)
        self.matmul = P.MatMul().shard(strategy2)
        self.matmul_weight = Parameter(Tensor(np.ones([25088, 256]), dtype=ms.float32), name="weight")

    def construct(self, x):
        x = self.relu(x)
        x = self.reshape(x, (256, 25088))
        x = self.matmul(x, self.matmul_weight)
        return x


def reshape_net(strategy0, strategy1, strategy2):
    return ReshapeNet(strategy0=strategy0, strategy1=strategy1, strategy2=strategy2)


def reshape_common(parallel_mode, strategy0, strategy1, strategy2, strategy_loss):
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=8)
    predict = Tensor(np.ones([32, 512, 7, 7]), dtype=ms.float32)
    label = Tensor(np.ones([32]), dtype=ms.int32)
    dataset = Dataset(predict, label, 2)
    net = reshape_net(strategy0, strategy1, strategy2)

    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss.softmax_cross_entropy.shard(strategy_loss)
    loss.one_hot.shard(((8, 1), (), ()))
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, loss, opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)


def test_reshape1():
    strategy0 = ((8, 1, 1, 1),)
    strategy1 = None
    strategy2 = ((8, 1), (1, 1))
    strategy_loss = ((8, 1), (8, 1))
    reshape_common(ParallelMode.SEMI_AUTO_PARALLEL, strategy0, strategy1, strategy2, strategy_loss)


def test_reshape1_strategy_1():
    strategy0 = ((8, 1, 1, 1),)
    strategy1 = ((8, 1, 1, 1),)
    strategy2 = ((8, 1), (1, 1))
    strategy_loss = ((8, 1), (8, 1))
    try:
        reshape_common(ParallelMode.SEMI_AUTO_PARALLEL, strategy0, strategy1, strategy2, strategy_loss)
    except ValueError:
        pass
    except TypeError:
        pass
    except RuntimeError:
        pass


def test_reshape1_strategy_2():
    strategy0 = ((8, 1, 1, 1),)
    strategy1 = ((8, 1, 1, 1),)
    strategy2 = ((8, 1), (1, 1))
    strategy_loss = ((8, 1), (8, 1))
    try:
        reshape_common(ParallelMode.AUTO_PARALLEL, strategy0, strategy1, strategy2, strategy_loss)
    except ValueError:
        pass
    except TypeError:
        pass
    except RuntimeError:
        pass


def test_reshape2():
    strategy0 = ((8, 1, 1, 1),)
    strategy1 = None
    strategy2 = ((8, 1), (1, 1))
    strategy_loss = ((8, 1), (8, 1))
    reshape_common(ParallelMode.SEMI_AUTO_PARALLEL, strategy0, strategy1, strategy2, strategy_loss)


def test_reshape3():
    strategy0 = ((2, 1, 1, 1),)
    strategy1 = None
    strategy2 = ((8, 1), (1, 1))
    strategy_loss = ((8, 1), (8, 1))
    reshape_common(ParallelMode.SEMI_AUTO_PARALLEL, strategy0, strategy1, strategy2, strategy_loss)


def test_reshape4():
    strategy0 = ((1, 1, 1, 1),)
    strategy1 = None
    strategy2 = ((8, 1), (1, 1))
    strategy_loss = ((8, 1), (8, 1))
    reshape_common(ParallelMode.SEMI_AUTO_PARALLEL, strategy0, strategy1, strategy2, strategy_loss)


def test_reshape5():
    strategy0 = ((2, 1, 1, 1),)
    strategy1 = None
    strategy2 = ((1, 8), (8, 1))
    strategy_loss = ((8, 1), (8, 1))
    reshape_common(ParallelMode.SEMI_AUTO_PARALLEL, strategy0, strategy1, strategy2, strategy_loss)


def test_reshape_auto():
    strategy0 = None
    strategy1 = None
    strategy2 = None
    strategy_loss = None
    reshape_common(ParallelMode.AUTO_PARALLEL, strategy0, strategy1, strategy2, strategy_loss)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x):
        return grad_all(self.network)(x)


class ReshapeNet1(nn.Cell):
    def __init__(self, strategy0):
        super(ReshapeNet1, self).__init__()
        self.reshape = P.Reshape()
        self.matmul = P.MatMul().shard(strategy0)
        self.matmul_weight = Parameter(Tensor(np.ones([25088, 256]), dtype=ms.float32), name="weight")
        self.reshape2 = P.Reshape()

    def construct(self, x):
        x = self.reshape(x, (256, 25088))
        x = self.matmul(x, self.matmul_weight)
        x = self.reshape2(x, (256 * 256,))
        return x


class ReshapeNet2(nn.Cell):
    def __init__(self, strategy0):
        super(ReshapeNet2, self).__init__()
        self.reshape = P.Reshape()
        self.matmul = P.MatMul().shard(strategy0)
        self.matmul_weight = Parameter(Tensor(np.ones([25088, 256]), dtype=ms.float32), name="weight")
        self.reshape2 = P.Reshape()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.reshape3 = P.Reshape()

    def construct(self, x):
        x = self.reshape(x, (256, 25088))
        x = self.matmul(x, self.matmul_weight)
        x = self.reshape2(x, (256 * 256,))
        x = self.reduce_sum(x, -1)
        x = self.reshape3(x, ())
        return x


class ReshapeNet3(nn.Cell):
    def __init__(self, strategy0):
        super(ReshapeNet3, self).__init__()
        self.reshape = P.Reshape()
        self.matmul = P.MatMul().shard(strategy0)
        self.matmul_weight = Parameter(Tensor(np.ones([25088, 256]), dtype=ms.float32), name="weight")
        self.reshape2 = P.Reshape()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.reshape3 = P.Reshape()

    def construct(self, x):
        x = self.reshape(x, (256, 25088))
        x = self.matmul(x, self.matmul_weight)
        x = self.reshape2(x, (256 * 256,))
        x = self.reduce_sum(x, -1)
        x = self.reshape3(x, (1, 1))
        return x


class ReshapeNet4(nn.Cell):
    def __init__(self, strategy0):
        super(ReshapeNet4, self).__init__()
        self.reshape = P.Reshape()
        self.reshape2 = P.Reshape()
        self.matmul = P.MatMul().shard(strategy0)
        self.matmul_weight = Parameter(Tensor(np.ones([25088, 256]), dtype=ms.float32), name="weight")

    def construct(self, x):
        x = self.reshape(x, (256, 25088))
        w = self.reshape2(self.matmul_weight, (25088, 256))
        x = self.matmul(x, w)
        return x


class ReshapeNet5(nn.Cell):
    def __init__(self, strategy0):
        super(ReshapeNet5, self).__init__()
        self.reshape = P.Reshape()
        self.matmul1 = P.MatMul().shard(strategy0)
        self.matmul1_weight = Parameter(Tensor(np.ones([25088, 256]), dtype=ms.float32), name="weight")
        self.matmul2 = P.MatMul().shard(strategy0)

    def construct(self, x):
        x = self.reshape(x, (256, 25088))
        matmul1_o = self.matmul1(x, self.matmul1_weight)
        matmul2_o = self.matmul2(matmul1_o, x)
        return matmul2_o


class ReshapeNet6(nn.Cell):
    def __init__(self, strategy0):
        super(ReshapeNet6, self).__init__()
        self.reshape = P.Reshape()
        self.matmul1_1 = P.MatMul().shard(strategy0)
        self.matmul1_2 = P.MatMul().shard(strategy0)
        self.matmul1_weight = Parameter(Tensor(np.ones([25088, 256]), dtype=ms.float32), name="weight")
        self.matmul2 = P.MatMul().shard(strategy0)
        self.add = P.Add()

    def construct(self, x):
        x = self.reshape(x, (256, 25088))
        matmul1_1_o = self.matmul1_1(x, self.matmul1_weight)
        matmul1_2_o = self.matmul1_2(x, self.matmul1_weight)
        matmul1_o = self.add(matmul1_1_o, matmul1_2_o)
        matmul2_o = self.matmul2(matmul1_o, x)
        return matmul2_o


def compile_net(net, input_):
    net.set_auto_parallel()
    net.set_train()
    _executor.compile(net, input_)


def reshape_net2(backbone):
    batch_size = 16
    device_num = 16
    context.set_auto_parallel_context(device_num=device_num, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    input_ = Tensor(np.ones([batch_size * device_num, 512, 7, 7]).astype(np.float32) * 0.01)

    net = GradWrap(NetWithLoss(backbone))

    compile_net(net, input_)


def test_reshape_net1_1():
    reshape_net2(_VirtualDatasetCell(ReshapeNet1(((1, 8), (8, 1)))))


def test_reshape_net1_2():
    reshape_net2(_VirtualDatasetCell(ReshapeNet1(((1, 8), (8, 2)))))


def test_reshape_net2_1():
    reshape_net2(_VirtualDatasetCell(ReshapeNet2(((1, 8), (8, 1)))))


def test_reshape_net2_2():
    reshape_net2(_VirtualDatasetCell(ReshapeNet2(((1, 8), (8, 2)))))


def test_reshape_net3_1():
    reshape_net2(_VirtualDatasetCell(ReshapeNet3(((1, 8), (8, 1)))))


def test_reshape_net3_2():
    reshape_net2(_VirtualDatasetCell(ReshapeNet3(((1, 8), (8, 2)))))


def test_reshape_net4_1():
    try:
        reshape_net2(_VirtualDatasetCell(ReshapeNet4(((1, 8), (8, 1)))))
    except ValueError:
        pass
    except TypeError:
        pass
    except RuntimeError:
        pass


def test_reshape_net4_2():
    try:
        reshape_net2(_VirtualDatasetCell(ReshapeNet4(((1, 8), (8, 2)))))
    except ValueError:
        pass
    except TypeError:
        pass
    except RuntimeError:
        pass


def test_reshape_net5_1():
    reshape_net2(_VirtualDatasetCell(ReshapeNet5(((1, 8), (8, 1)))))


def test_reshape_net5_2():
    reshape_net2(_VirtualDatasetCell(ReshapeNet5(((1, 8), (8, 2)))))


def test_reshape_net6_1():
    reshape_net2(_VirtualDatasetCell(ReshapeNet6(((1, 8), (8, 1)))))


def test_reshape_net6_2():
    reshape_net2(_VirtualDatasetCell(ReshapeNet6(((1, 8), (8, 2)))))


class TrainOneStepCell(nn.Cell):
    """
    Network training package class.

    Append an optimizer to the training network after that the construct function
    can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.

    Examples:
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> loss_net = WithLossCell(net, loss_fn)
        >>> train_net = TrainOneStepCell(loss_net, optim)
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.sens = sens

    def construct(self, data):
        weights = self.weights
        loss = self.network(data)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(data, sens)

        return F.depend(loss, self.optimizer(grads))


def reshape_common2(parallel_mode, net):
    batch_size = 16
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2

    predict = Tensor(np.ones([batch_size, 512, 7, 7]), dtype=ms.float32)
    label = Tensor(np.ones([batch_size]), dtype=ms.int32)
    dataset = Dataset(predict, label, 2, input_num=1)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=16)

    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    train_net = TrainOneStepCell(net, opt).set_train()
    model = Model(train_net)
    model.train(epoch_size, dataset, dataset_sink_mode=False)


def test_reshape_common2_0():
    reshape_common2(ParallelMode.SEMI_AUTO_PARALLEL, _VirtualDatasetCell(ReshapeNet1(((1, 8), (8, 1)))))


def test_reshape_common2_1():
    reshape_common2(ParallelMode.SEMI_AUTO_PARALLEL, _VirtualDatasetCell(ReshapeNet1(((1, 8), (8, 2)))))


def test_reshape_common2_2():
    reshape_common2(ParallelMode.SEMI_AUTO_PARALLEL, _VirtualDatasetCell(ReshapeNet2(((1, 8), (8, 1)))))


def test_reshape_common2_3():
    reshape_common2(ParallelMode.SEMI_AUTO_PARALLEL, _VirtualDatasetCell(ReshapeNet2(((1, 8), (8, 2)))))


def test_reshape_common2_4():
    reshape_common2(ParallelMode.SEMI_AUTO_PARALLEL, _VirtualDatasetCell(ReshapeNet3(((1, 8), (8, 1)))))


def test_reshape_common2_5():
    reshape_common2(ParallelMode.SEMI_AUTO_PARALLEL, _VirtualDatasetCell(ReshapeNet3(((1, 8), (8, 2)))))


class BatchNormReshapeNet(nn.Cell):
    def __init__(self):
        super(BatchNormReshapeNet, self).__init__()
        self.batch_norm = nn.BatchNorm1d(512, affine=False)
        self.reshape = P.Reshape()
        self.prelu = nn.PReLU(channel=256)

    def construct(self, x):
        x = self.batch_norm(x)
        x = self.reshape(x, (512, 256))
        x = self.prelu(x)
        return x


def test_batchnorm_reshape_train():
    batch_size = 16
    device_num = 16
    context.set_auto_parallel_context(device_num=device_num, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    input_ = Tensor(np.ones([batch_size * device_num, 512]).astype(np.float32) * 0.01)

    net = GradWrap(NetWithLoss(_VirtualDatasetCell(BatchNormReshapeNet())))

    compile_net(net, input_)


def bn_with_initialize(out_channels):
    bn = nn.BatchNorm2d(out_channels, momentum=0.3, eps=1e-5).add_flags_recursive(fp32=True)
    return bn


def fc_with_initialize(input_channels, out_channels):
    return nn.Dense(input_channels, out_channels).add_flags_recursive(fp16=True)


class BNReshapeDenseBNNet(nn.Cell):
    def __init__(self):
        super(BNReshapeDenseBNNet, self).__init__()
        self.batch_norm = bn_with_initialize(2)
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.batch_norm2 = nn.BatchNorm1d(512, affine=False)
        self.fc = fc_with_initialize(2 * 32 * 32, 512)

    def construct(self, x):
        x = self.batch_norm(x)
        x = self.reshape(x, (16, 2 * 32 * 32))
        x = self.fc(x)
        x = self.batch_norm2(x)
        return x


def test_bn_reshape_dense_bn_train():
    batch_size = 16
    device_num = 16
    context.set_auto_parallel_context(device_num=device_num, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    input_ = Tensor(np.ones([batch_size, 2, 32, 32]).astype(np.float32) * 0.01)

    net = GradWrap(NetWithLoss(BNReshapeDenseBNNet()))

    compile_net(net, input_)


class ParallelReduceMeanNet(nn.Cell):
    def __init__(self, conv_in_channel, conv_out_channel,
                 reducemean_keep_dims=False, reducemean_axis=-1, strategy=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=conv_in_channel, out_channels=conv_out_channel,
                              kernel_size=1, stride=1, pad_mode='valid', has_bias=True,
                              weight_init='ones', bias_init='ones')
        self.reduce_mean = P.ReduceMean(keep_dims=reducemean_keep_dims)
        self.flat = nn.Flatten()
        self.reducemean_axis = reducemean_axis
        if strategy is not None:
            self.reduce_mean.shard(strategy)

    def construct(self, inputs):
        x = self.conv(inputs)
        x = self.reduce_mean(x, self.reducemean_axis)
        x = self.flat(x)
        return x


class CrossEntropyLoss(nn.Cell):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()

        self.reduce_mean = P.ReduceMean()
        self.cross_entropy = SoftmaxCrossEntropyWithLogits()
        self.reduction = reduction

    def construct(self, logits, label):
        loss = self.cross_entropy(logits, label)
        if self.reduction == 'mean':
            loss = self.reduce_mean(loss, (-1,))
        return loss


def test_flatten_reshape(parallel_mode="auto_parallel"):
    batch_size = 16
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=8)
    net = ParallelReduceMeanNet(conv_in_channel=3, conv_out_channel=64, reducemean_axis=(2, 3),
                                strategy=((4, 2, 1, 1),))
    loss = CrossEntropyLoss()
    predict = Tensor(np.ones([batch_size, 3, 32, 32]), dtype=ms.float32)
    label = Tensor(np.ones([batch_size, 64]), dtype=ms.float32)
    dataset = Dataset(predict, label, 2, input_num=2)

    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, loss_fn=loss, optimizer=opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)


def test_flatten_reshape2(parallel_mode="auto_parallel"):
    batch_size = 16
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=8)
    set_algo_parameters(fully_use_devices=False)
    net = ParallelReduceMeanNet(conv_in_channel=3, conv_out_channel=64, reducemean_axis=(2, 3),
                                strategy=((4, 1, 1, 1),))
    loss = CrossEntropyLoss()
    predict = Tensor(np.ones([batch_size, 3, 32, 32]), dtype=ms.float32)
    label = Tensor(np.ones([batch_size, 64]), dtype=ms.float32)
    dataset = Dataset(predict, label, 2, input_num=2)

    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, loss_fn=loss, optimizer=opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)


class ParallelReshapeNet(nn.Cell):
    def __init__(self, dense_in_channel, dense_out_channel, shape, strategy=None):
        super().__init__()
        self.flat = nn.Flatten()
        self.dense = nn.Dense(in_channels=dense_in_channel,
                              out_channels=dense_out_channel,
                              weight_init='ones',
                              bias_init='ones',
                              has_bias=True)
        self.reshape = P.Reshape()
        self.shape = shape
        self.reshape.shard(strategy)

    def construct(self, inputs):
        x = self.flat(inputs)
        x = self.dense(x)
        x = self.reshape(x, self.shape)
        return x


# the shape of input and output of reshape is the same
# reshape is optimized before step_parallel
def test_flatten_reshape3(parallel_mode="auto_parallel"):
    batch_size = 16
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=8)
    set_algo_parameters(fully_use_devices=False)
    net = ParallelReshapeNet(dense_in_channel=2048, dense_out_channel=1000, shape=(128, 1000), strategy=((16, 1),))
    loss = CrossEntropyLoss()
    predict = Tensor(np.ones([batch_size, 1, 2, 1024]), dtype=ms.float32)
    label = Tensor(np.ones([batch_size, 1000]), dtype=ms.float32)
    dataset = Dataset(predict, label, 2, input_num=2)

    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, loss_fn=loss, optimizer=opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)


class CrossEntropyLoss2(nn.Cell):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss2, self).__init__()
        self.cross_entropy = SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logits, label):
        loss = self.cross_entropy(logits, label)
        return loss


def test_flatten_reshape4(parallel_mode="semi_auto_parallel"):
    batch_size = 16
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=8)
    set_algo_parameters(fully_use_devices=False)
    net = ParallelReduceMeanNet(conv_in_channel=3, conv_out_channel=64, reducemean_keep_dims=True,
                                strategy=((4, 1, 1, 1),))
    loss = CrossEntropyLoss2()
    predict = Tensor(np.ones([batch_size, 3, 32, 32]), dtype=ms.float32)
    label = Tensor(np.ones([batch_size, 2048]), dtype=ms.float32)
    dataset = Dataset(predict, label, 2, input_num=2)

    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, loss_fn=loss, optimizer=opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)
