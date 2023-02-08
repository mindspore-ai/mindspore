# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import context, Parameter, ms_function
from mindspore import dataset as ds
from mindspore.ops import operations as P
from mindspore.common.jit_config import JitConfig
from mindspore.common.initializer import initializer
from mindspore.ops import functional as F


context.set_context(mode=ms.GRAPH_MODE)
context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch")


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Layer1(nn.Cell):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.weight = Parameter(initializer(0.03, [in_dim, hidden_dim]), "w")
        self.matmul = P.MatMul().shard(((2, 1), (1, 2)))

    def construct(self, x):
        out = self.matmul(x, self.weight)
        return out


class Layer2(nn.Cell):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.weight2 = Parameter(initializer(0.03, [hidden_dim, out_dim]), "w2")
        self.matmul2 = P.MatMul().shard(((2, 2), (2, 1)))
        self.relu = P.ReLU()

    def construct(self, x):
        out = self.relu(x)
        out = self.matmul2(out, self.weight2)
        return out


class Net(nn.Cell):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layer1 = Layer1(in_dim, hidden_dim)
        self.layer2 = Layer2(hidden_dim, out_dim)
        self.layer1.pipeline_stage = 0
        self.layer2.pipeline_stage = 1

    def construct(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


class NetWithLoss(nn.Cell):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = Net(in_dim, hidden_dim, out_dim)
        self.loss = nn.MSELoss()

    def construct(self, x, y):
        out = self.net(x)
        out = self.loss(out, y)
        return out


def pipeline_clear_grad(accu_grad, grad):
    accu_grad = F.depend(accu_grad, grad)
    zeros = F.tensor_mul(accu_grad, 0.0)
    return F.assign(accu_grad, zeros)


def funcs(in_dim, hidden_dim, out_dim):
    net = NetWithLoss(in_dim, hidden_dim, out_dim)
    net_pipeline = nn.PipelineCell(net, 2)
    net_pipeline.set_train()
    opt = nn.Momentum(net.trainable_params(), 0.01, 0.1)
    accu_grads = opt.parameters.clone(prefix="accu_grads", init="zeros")
    hyper_map = ms.ops.HyperMap()

    def net_forward(x, y):
        loss = net_pipeline(x, y)
        return loss

    grad_net = ms.value_and_grad(net_forward, grad_position=None, weights=net.trainable_params())
    enable_opt_shard = context.get_auto_parallel_context("enable_parallel_optimizer")
    @ms_function
    def train_one_step(x, y):
        loss, grads = grad_net(x, y)
        if enable_opt_shard:
            opt(grads)
        else:
            opt(accu_grads)
        status = hyper_map(pipeline_clear_grad, accu_grads, grads)
        return F.depend(loss, status)
    return train_one_step


def test_sink_with_grad_pipeline_without_opt_shard():
    """
    Feature: Function mode with pipeline parallel in auto parallel
    Description: without optimizer shard
    Expectation: compile ok
    """
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch",
                                      device_num=8, pipeline_stages=2, global_rank=4, enable_parallel_optimizer=False)

    batch_size = 128
    in_dim = 32
    hidden_dim = 8
    out_dim = 16
    data = {"input": np.ones([32, batch_size, in_dim]).astype(np.float32),
            "label": np.zeros([32, batch_size, out_dim]).astype(np.float32)}
    dataset = ds.NumpySlicesDataset(data=data)
    train_one_step = funcs(in_dim, hidden_dim, out_dim)
    jitconfig = JitConfig(jit_level="O1", exc_mode='no_sink')
    sink_process = ms.train.data_sink(train_one_step, dataset, sink_size=4, jit_config=jitconfig)
    _ = sink_process()


def test_sink_with_grad_pipeline_with_opt_shard():
    """
    Feature: Function mode with pipeline parallel in auto parallel
    Description: with optimizer shard
    Expectation: compile ok
    """
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch",
                                      device_num=8, pipeline_stages=2, global_rank=4, enable_parallel_optimizer=True)

    batch_size = 128
    in_dim = 32
    hidden_dim = 8
    out_dim = 16
    data = {"input": np.ones([32, batch_size, in_dim]).astype(np.float32),
            "label": np.zeros([32, batch_size, out_dim]).astype(np.float32)}
    dataset = ds.NumpySlicesDataset(data=data)
    train_one_step = funcs(in_dim, hidden_dim, out_dim)
    jitconfig = JitConfig(jit_level="O1", exc_mode='no_sink')
    sink_process = ms.train.data_sink(train_one_step, dataset, sink_size=4, jit_config=jitconfig)
    _ = sink_process()
