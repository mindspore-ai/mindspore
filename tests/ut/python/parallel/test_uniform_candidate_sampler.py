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
import pytest
import mindspore as ms
import mindspore.context as context
from mindspore import Tensor, Parameter
import mindspore.nn as nn
from mindspore.common.api import _executor
from mindspore.nn import TrainOneStepCell, Momentum
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, embedding_weight, num_true, num_sampled, unique, range_max, seed, remove_accidential,
                 strategy1=None):
        super(Net, self).__init__()
        self.sampler = P.UniformCandidateSampler(num_true, num_sampled, unique, range_max, seed,
                                                 remove_accidential)
        if strategy1:
            self.sampler.shard(strategy1)
        self.embedding_table = Parameter(embedding_weight, "embedding_weight")
        self.gatherv2 = P.Gather()
        self.reduce_sum = P.ReduceSum()
        self.reduce_sum2 = P.ReduceSum()
        self.reduce_sum3 = P.ReduceSum()

    def construct(self, x):
        out1, out2, out3 = self.sampler(x)
        lookup = self.gatherv2(self.embedding_table, out1, 0)
        loss = out1 - out3
        loss = self.reduce_sum(loss, (0,))
        loss2 = self.reduce_sum2(lookup, (0, 1))
        loss3 = self.reduce_sum3(out2, (0, 1))
        loss4 = loss + loss2 + loss3
        return loss4


class Net2(nn.Cell):
    def __init__(self, mul_weight, num_true, num_sampled, unique, range_max, seed, remove_accidential,
                 strategy1=None):
        super(Net2, self).__init__()
        self.sampler = P.UniformCandidateSampler(num_true, num_sampled, unique, range_max, seed,
                                                 remove_accidential)
        self.cast = P.Cast()
        self.weight = Parameter(mul_weight, "w1")
        self.mul = P.Mul()
        if strategy1:
            self.sampler.shard(strategy1)

    def construct(self, x):
        x = self.mul(x, self.weight)
        x = self.cast(x, ms.int32)
        _, out2, _ = self.sampler(x)
        return out2


_w = Tensor(np.ones([48, 16]), dtype=ms.float32)
_w1 = Tensor(np.ones([96, 64]), dtype=ms.float32)
_x = Tensor(np.ones([48, 16]), dtype=ms.int32)


def compile_net(net):
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    train_net.set_train()
    _executor.compile(train_net, _x)
    context.reset_auto_parallel_context()


def test_uniform_candidate_sampler_no_full_0d_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((4, 1),)
    net = Net(_w1, num_true=16, num_sampled=16, unique=True, range_max=20, seed=1,
              remove_accidential=False, strategy1=strategy1)
    compile_net(net)


def test_uniform_candidate_sampler_no_full_1d_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4),)
    net = Net(_w1, num_true=16, num_sampled=16, unique=True, range_max=20, seed=1,
              remove_accidential=False, strategy1=strategy1)
    compile_net(net)


def test_uniform_candidate_sampler_full_0d_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1),)
    net = Net(_w1, num_true=16, num_sampled=16, unique=True, range_max=20, seed=1,
              remove_accidential=False, strategy1=strategy1)
    compile_net(net)


def test_uniform_candidate_sampler_full_1d_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8),)
    net = Net(_w1, num_true=16, num_sampled=16, unique=True, range_max=20, seed=1,
              remove_accidential=False, strategy1=strategy1)
    compile_net(net)


def test_uniform_candidate_sampler_full_1d_unqiue_false():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8),)
    net = Net(_w1, num_true=16, num_sampled=16, unique=False, range_max=20, seed=1,
              remove_accidential=False, strategy1=strategy1)
    compile_net(net)


def test_uniform_candidate_sampler_auto_parllel():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net(_w1, num_true=16, num_sampled=16, unique=False, range_max=20, seed=1,
              remove_accidential=False, strategy1=None)
    compile_net(net)


def test_uniform_candidate_sampler_auto_parllel_unqiue_true():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net(_w1, num_true=16, num_sampled=16, unique=True, range_max=20, seed=1,
              remove_accidential=False, strategy1=None)
    compile_net(net)


def test_uniform_candidate_sampler_auto_parllel_remove_true():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net(_w1, num_true=16, num_sampled=16, unique=True, range_max=20, seed=1,
              remove_accidential=True, strategy1=None)
    compile_net(net)


def test_uniform_candidate_sampler_full_1d_remove_true():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8),)
    net = Net(_w1, num_true=16, num_sampled=16, unique=False, range_max=20, seed=1,
              remove_accidential=True, strategy1=strategy1)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_uniform_candidate_sampler_as_final():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8),)
    net = Net2(_w, num_true=16, num_sampled=16, unique=False, range_max=20, seed=1, remove_accidential=False,
               strategy1=strategy1)
    with pytest.raises(RuntimeError):
        compile_net(net)
