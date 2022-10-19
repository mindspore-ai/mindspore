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
# ============================================================================

import os
import numpy as np
from numpy import allclose

import mindspore as ms
import mindspore.communication.management as distributedTool
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.train import Model
from mindspore.train.callback import Callback

np.set_printoptions(threshold=np.inf)
device_num = 2
device_id = int(os.getenv('DEVICE_ID'))
rank_id = 0
embed = 128
classes = 32
batch_size = 32 * 2
MatmulParamShape = (classes, embed)


def setup_module():
    global device_num
    global rank_id
    np.random.seed(0)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(device_id=device_id)
    distributedTool.init()
    rank_id = distributedTool.get_rank()
    device_num = distributedTool.get_group_size()
    context.set_auto_parallel_context(device_num=device_num,
                                      global_rank=device_id)


def teardown_module():
    distributedTool.release()


class DataGenerator():
    def get_parallel_blocks(self, input_, strategy):
        blocks = [input_]
        i = 0
        for stra in strategy:
            temp = []
            while blocks:
                block = blocks.pop(0)
                temp.extend(np.split(block, stra, axis=i))
            blocks.extend(temp)
            i += 1
        return blocks

    def generate_data(self, shape):
        size = np.cumprod(shape)[-1]
        num_range = min(size, 1000)
        data = (np.arange(0, size) % num_range) / num_range
        data = np.reshape(data, shape)
        return data

    def input_data(self, shape):
        data = (self.generate_data(shape) * 0.1).astype(np.float32)
        stra = [1] * len(shape)
        stra[0] = device_num
        data_list = self.get_parallel_blocks(data, stra)
        return Tensor(data), Tensor(data_list[rank_id])

    def label_data(self, shape, embed_):
        data = (self.generate_data(shape) * (embed_ - 1)).astype(np.int32)
        stra = [1] * len(shape)
        stra[0] = device_num
        data_list = self.get_parallel_blocks(data, stra)
        return Tensor(data), Tensor(data_list[rank_id])


class Dataset():
    def __init__(self, predict, label, length=1, input_num=2):
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

    def get_dataset_size(self):
        return self.length

    def get_repeat_count(self):
        return self.length


class ModelCallback(Callback):
    def __init__(self):
        super(ModelCallback, self).__init__()
        self.loss_list = []

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        result = cb_params.net_outputs
        self.loss_list.append(result.asnumpy().mean())


class SoftmaxCrossEntropyExpand(Cell):
    def __init__(self, sparse=False, stra_list=None):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        if stra_list is None:
            stra_list = []
        if len(stra_list) < 11:
            stra_list = [None] * 11
        self.exp = P.Exp()
        self.reduce_sum = P.ReduceSum(keep_dims=True).shard(strategy=stra_list[1])
        self.onehot = P.OneHot().shard(strategy=stra_list[2])
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.div = P.Div().shard(strategy=stra_list[3])
        self.log = P.Log().shard(strategy=stra_list[4])
        self.sum_cross_entropy = P.ReduceSum(keep_dims=False).shard(strategy=stra_list[5])
        self.mul = P.Mul().shard(strategy=stra_list[6])
        self.mul2 = P.Mul().shard(strategy=stra_list[7])
        self.cast = P.Cast()
        self.reduce_mean = P.ReduceMean(keep_dims=False).shard(strategy=stra_list[8])
        self.sparse = sparse
        self.reduce_max = P.ReduceMax(keep_dims=True).shard(strategy=stra_list[9])
        self.sub = P.Sub().shard(strategy=stra_list[10])

    def construct(self, logit, label):
        logit_max = self.reduce_max(logit, -1)
        exp = self.exp(self.sub(logit, logit_max))
        exp_sum = self.reduce_sum(exp, -1)
        softmax_result = self.div(exp, exp_sum)
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        softmax_result_log = self.log(softmax_result)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        loss = self.mul2(F.scalar_to_tensor(-1.0), loss)
        loss = self.reduce_mean(loss, -1)
        return loss


class MatmulNet(Cell):
    def __init__(self, matmul_stra=None, loss_stra_list=None):
        super(MatmulNet, self).__init__()
        if loss_stra_list is None:
            loss_stra_list = []
        self.matmul = P.MatMul(transpose_b=True).shard(strategy=matmul_stra)
        self.loss = SoftmaxCrossEntropyExpand(sparse=True, stra_list=loss_stra_list)
        self.weight = Parameter(Tensor(np.ones(MatmulParamShape), dtype=ms.float32), name="weight")

    def construct(self, x, label):
        loss_input = self.matmul(x, self.weight)
        out = self.loss(loss_input, label)
        return out


class LossFactory():
    def __init__(self):
        data_gen = DataGenerator()
        self.input_full, self.input_part = data_gen.input_data((batch_size, embed))
        self.label_full, self.label_part = data_gen.label_data((batch_size,), embed)

    def single_matmul_trains(self):
        single_callback = ModelCallback()
        net = MatmulNet()
        optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        model = Model(net, optimizer=optimizer)
        epoch_size = 6
        dataset = Dataset(self.input_full, self.label_full)
        model.train(epoch_size, dataset, callbacks=single_callback, dataset_sink_mode=False)
        loss_value = np.array(single_callback.loss_list)
        return loss_value

    def data_parallel_matmul_trains(self):
        parallel_callback = ModelCallback()
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        net = MatmulNet()
        optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        model = Model(net, optimizer=optimizer)
        epoch_size = 6
        dataset = Dataset(self.input_part, self.label_part)
        model.train(epoch_size, dataset, callbacks=parallel_callback, dataset_sink_mode=False)
        loss_value = np.array(parallel_callback.loss_list)
        return loss_value

    def model_parallel_matmul_trains(self):
        parallel_callback = ModelCallback()
        matmul_stra = ((1, 1), (device_num, 1))
        reduce_max_stra = ((1, device_num),)
        sub_stra = ((1, device_num), (1, 1))
        exp_stra = ((1, device_num),)
        reduce_sum_stra = ((1, device_num),)
        div_stra = ((1, device_num), (1, 1))
        log_stra = ((1, device_num),)
        mul_stra = ((1, device_num), (1, device_num))
        sum_cross_entropy_stra = ((1, device_num),)
        mul2_stra = ((), (device_num,))
        reduce_mean_stra = ((device_num,),)
        onehot_stra = ((1, device_num), (), ())
        loss_stra_list = [exp_stra, reduce_sum_stra, onehot_stra, div_stra, log_stra,
                          sum_cross_entropy_stra, mul_stra, mul2_stra, reduce_mean_stra, reduce_max_stra, sub_stra]
        context.set_auto_parallel_context(parallel_mode="auto_parallel")
        net = MatmulNet(matmul_stra=matmul_stra, loss_stra_list=loss_stra_list)
        optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        model = Model(net, optimizer=optimizer)
        epoch_size = 6
        dataset = Dataset(self.input_part, self.label_part)
        model.train(epoch_size, dataset, callbacks=parallel_callback, dataset_sink_mode=False)
        loss_value = np.array(parallel_callback.loss_list)
        return loss_value

    def mix_parallel_matmul_trains(self):
        parallel_callback = ModelCallback()
        matmul_stra = ((device_num, 1), (1, 1))
        reduce_max_stra = ((1, device_num),)
        sub_stra = ((device_num, 1), (device_num, 1))
        exp_stra = ((1, device_num),)
        reduce_sum_stra = ((1, device_num),)
        div_stra = ((1, device_num), (1, 1))
        log_stra = ((1, device_num),)
        mul_stra = ((1, device_num), (1, device_num))
        sum_cross_entropy_stra = ((1, device_num),)
        mul2_stra = ((), (device_num,))
        reduce_mean_stra = ((device_num,),)
        onehot_stra = ((1, device_num), (), ())
        loss_stra_list = [exp_stra, reduce_sum_stra, onehot_stra, div_stra, log_stra,
                          sum_cross_entropy_stra, mul_stra, mul2_stra, reduce_mean_stra, reduce_max_stra, sub_stra]
        context.set_auto_parallel_context(parallel_mode="auto_parallel")
        net = MatmulNet(matmul_stra=matmul_stra, loss_stra_list=loss_stra_list)
        optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        model = Model(net, optimizer=optimizer)
        epoch_size = 6
        dataset = Dataset(self.input_part, self.label_part)
        model.train(epoch_size, dataset, callbacks=parallel_callback, dataset_sink_mode=False)
        loss_value = np.array(parallel_callback.loss_list)
        return loss_value


def test_all_trains():
    loss_factory = LossFactory()
    context.reset_auto_parallel_context()
    single_loss = loss_factory.single_matmul_trains()
    model_parallel_loss = loss_factory.model_parallel_matmul_trains()
    mix_parallel_loss = loss_factory.mix_parallel_matmul_trains()
    assert allclose(single_loss, model_parallel_loss)
    assert allclose(single_loss, mix_parallel_loss)
