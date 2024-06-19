# Copyright 2024 Huawei Technologies Co., Ltd
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
import subprocess
import shutil
import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor, Parameter, Symbol
from mindspore.nn import Cell, Momentum
from mindspore.ops import operations as P
from mindspore.train import Model
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


_x = Tensor(np.ones([1, 8, 64]), dtype=ms.float32)
_b = Tensor(np.ones([1, 8, 64]), dtype=ms.float32)


def compile_net(net):
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2
    dataset = Dataset(_x, _b)
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    s = Symbol(divisor=8)
    x = Tensor(shape=[1, s, 64], dtype=ms.float32)
    y = Tensor(shape=[None, None, None], dtype=ms.float32)
    net.set_inputs(x, y)
    model = Model(net, optimizer=opt, amp_level="O2")
    model.train(epoch_size, dataset, dataset_sink_mode=False)
    context.reset_auto_parallel_context()


def test_dynamic_shape_merge_allgather():
    """
    Feature: test dynamic shape merge allgather
    Description: merge 3 allgather to 1
    Expectation: compile success
    """

    class DynamicMulNet(Cell):
        def __init__(self, strategy1, strategy2, strategy3, strategy4):
            super().__init__()
            self.norm = P.RmsNorm().shard(strategy1)
            self.w = Parameter(Tensor(np.ones([64]), dtype=ms.float32), "w0")
            self.w1 = Parameter(Tensor(np.ones([64, 64]), dtype=ms.float32), "w1")
            self.w2 = Parameter(Tensor(np.ones([64, 64]), dtype=ms.float32), "w2")
            self.w3 = Parameter(Tensor(np.ones([64, 64]), dtype=ms.float32), "w3")
            self.m1 = P.MatMul().shard(strategy2)
            self.m2 = P.MatMul().shard(strategy2)
            self.m3 = P.MatMul().shard(strategy2)
            self.reshape = P.Reshape()
            self.add = P.Add().shard(strategy3)
            self.sum = P.ReduceSum().shard(strategy4)

        def construct(self, x, y):
            out, _ = self.norm(x, self.w)
            s = x.shape
            r1 = self.reshape(out, (s[0] * s[1], s[2]))
            r2 = self.reshape(out, (s[0] * s[1], s[2]))
            r3 = self.reshape(out, (s[0] * s[1], s[2]))
            out1 = self.m1(r1, self.w1)
            out2 = self.m2(r2, self.w2)
            out3 = self.m3(r3, self.w3)
            out = self.add(out1, out2)
            out = self.add(out, out3)
            out = self.sum(out)
            return out

    strategy1 = ((1, 8, 1), (1,))
    strategy2 = ((1, 1), (1, 8))
    strategy3 = ((1, 8), (1, 8))
    strategy4 = ((1, 8),)
    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True, full_batch=True)
    context.set_auto_parallel_context(dataset_strategy=((1, 8, 1), (1, 8, 1)))
    net = DynamicMulNet(strategy1, strategy2, strategy3, strategy4)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(save_graphs=True, save_graphs_path="./dynamic_shape_merge_allgather")
    if os.path.exists("./dynamic_shape_merge_allgather/rank_0"):
        shutil.rmtree("./dynamic_shape_merge_allgather/rank_0")

    compile_net(net)

    file = "./dynamic_shape_merge_allgather/rank_0/*step_parallel_end*.ir"
    para = "= AllGather("
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "3"

    file = "./dynamic_shape_merge_allgather/rank_0/*merge_comm*.ir"
    para = "= AllGather("
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "1"

    if os.path.exists("./dynamic_shape_merge_allgather/rank_0"):
        shutil.rmtree("./dynamic_shape_merge_allgather/rank_0")
