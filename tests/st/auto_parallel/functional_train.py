# Copyright 2022 Huawei Technologies Co., Ltd
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

from mindspore import Parameter, jit
from mindspore.nn import Cell, Momentum
from mindspore.nn import MSELoss
import mindspore.dataset as ds
import mindspore.ops as ops
import mindspore as ms
from mindspore.common.initializer import initializer
from mindspore.communication import init


def get_dataset(batch_size, step_per_epoch, in_dim, out_dim):
    input_data = np.ones((batch_size, in_dim), dtype=np.float32) * 0.1
    label_data = np.ones((batch_size, out_dim), dtype=np.float32) * 0.1
    def generate():
        for _ in range(step_per_epoch):
            yield (input_data, label_data)
    return generate


class Net(Cell):
    """define net"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.weight = Parameter(initializer(0.03, [self.in_dim, self.hidden_dim]), "w")
        self.weight2 = Parameter(initializer(0.04, [self.hidden_dim, self.out_dim]), "w2")
        self.matmul = ops.MatMul()

        self.relu = ops.ReLU()
        self.matmul2 = ops.MatMul()

    def construct(self, x):
        out = self.matmul(x, self.weight)
        out = self.relu(out)
        out = self.matmul2(out, self.weight2)
        return out


def test_pynative_func():
    '''
    Feature: Object Oriented and Functional Mixed Programming
    Description: pynative mode, run one step
    Expectation: Run success
    '''
    var_step_per_epoch = 2
    var_single_batch_size = 16
    var_in_dim = 32
    var_hidden_dim = 8
    var_out_dim = 16

    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, search_mode="sharding_propagation")

    init()

    # dataset
    fake_dataset = get_dataset(var_single_batch_size, var_step_per_epoch, var_in_dim, var_out_dim)
    dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"])

    # define net
    net = Net(var_in_dim, var_hidden_dim, var_out_dim)

    # define shard
    net.shard(in_strategy=((2, 4),), parameter_plan={"weight": (4, 1)})

    # define loss
    loss_fn = MSELoss()

    # define opt
    learning_rate = 0.3
    momentum = 0.1
    opt = Momentum(net.trainable_params(), learning_rate, momentum)

    # define forward function
    def net_forward(x, y):
        out = net(x)
        loss = loss_fn(out, y)
        return loss

    grad_net = ops.value_and_grad(net_forward, grad_position=None, weights=net.trainable_params())

    def train_one_step(x, y):
        loss, grads = grad_net(x, y)
        opt(grads)
        return loss

    loss = 0.0
    for _ in range(1):
        for input_x, label in dataset:
            loss = train_one_step(input_x, label)
    assert np.allclose(np.array([loss.asnumpy()]), np.array([0.0047495714]), 0.00001, 0.00001)
    ms.reset_auto_parallel_context()


def test_graph_func():
    '''
    Feature: Object Oriented and Functional Mixed Programming
    Description: graph mode, run two step, ops parallel + opt parallel
    Expectation: Run success
    '''
    var_step_per_epoch = 2
    var_single_batch_size = 16
    var_in_dim = 32
    var_hidden_dim = 8
    var_out_dim = 16

    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, search_mode="sharding_propagation",
                                 enable_parallel_optimizer=True, dataset_strategy="full_batch",
                                 parallel_optimizer_config={"parallel_optimizer_threshold": 0})

    init()

    # dataset
    fake_dataset = get_dataset(var_single_batch_size, var_step_per_epoch, var_in_dim, var_out_dim)
    dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"])

    # define net
    net = Net(var_in_dim, var_hidden_dim, var_out_dim)

    # define shard
    net.matmul.shard(((2, 4), (4, 1)))

    # define loss
    loss_fn = MSELoss()

    # define opt
    learning_rate = 0.3
    momentum = 0.1
    opt = Momentum(net.trainable_params(), learning_rate, momentum)

    # define forward function
    def net_forward(x, y):
        out = net(x)
        loss = loss_fn(out, y)
        return loss

    grad_net = ops.value_and_grad(net_forward, grad_position=None, weights=net.trainable_params())

    @jit
    def train_one_step(x, y):
        loss, grads = grad_net(x, y)
        opt(grads)
        return loss

    loss = 0.0
    for _ in range(1):
        for input_x, label in dataset:
            loss = train_one_step(input_x, label)
    assert np.allclose(np.array([loss.asnumpy()]), np.array([0.0047495714]), 0.00001, 0.00001)
    ms.reset_auto_parallel_context()


def test_graph_func_sink():
    '''
    Feature: Object Oriented and Functional Mixed Programming
    Description: graph mode, run two step, ops parallel + opt parallel + data sink
    Expectation: Run success
    '''
    var_step_per_epoch = 2
    var_single_batch_size = 16
    var_in_dim = 32
    var_hidden_dim = 8
    var_out_dim = 16

    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, search_mode="sharding_propagation",
                                 enable_parallel_optimizer=True, dataset_strategy="full_batch",
                                 parallel_optimizer_config={"parallel_optimizer_threshold": 0})

    init()

    # dataset
    fake_dataset = get_dataset(var_single_batch_size, var_step_per_epoch, var_in_dim, var_out_dim)
    dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"])

    # define net
    net = Net(var_in_dim, var_hidden_dim, var_out_dim)

    # define shard
    net.matmul.shard(((2, 4), (4, 1)))

    # define loss
    loss_fn = MSELoss()

    # define opt
    learning_rate = 0.3
    momentum = 0.1
    opt = Momentum(net.trainable_params(), learning_rate, momentum)

    # define forward function
    def net_forward(x, y):
        out = net(x)
        loss = loss_fn(out, y)
        return loss

    grad_net = ops.value_and_grad(net_forward, grad_position=None, weights=net.trainable_params())

    @jit
    def train_one_step(x, y):
        loss, grads = grad_net(x, y)
        opt(grads)
        return loss

    sink_size = 1
    epoch = 1
    sink_process = ms.train.data_sink(train_one_step, dataset, sink_size=sink_size, jit_config=ms.JitConfig())
    loop_num = int(epoch * var_step_per_epoch / sink_size)

    loss = 0.0
    for _ in range(loop_num):
        loss = sink_process()
    assert np.allclose(np.array([loss.asnumpy()]), np.array([0.0047495714]), 0.00001, 0.00001)
    ms.reset_auto_parallel_context()


def test_pynative_func_sink():
    '''
    Feature: Object Oriented and Functional Mixed Programming
    Description: pynative mode, run two step, auto_parallel + data sink with jit
    Expectation: Run success
    '''
    var_step_per_epoch = 2
    var_single_batch_size = 16
    var_in_dim = 32
    var_hidden_dim = 8
    var_out_dim = 16

    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, search_mode="sharding_propagation",
                                 enable_parallel_optimizer=True, dataset_strategy="full_batch")

    init()

    # dataset
    fake_dataset = get_dataset(var_single_batch_size, var_step_per_epoch, var_in_dim, var_out_dim)
    dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"])

    # define net
    net = Net(var_in_dim, var_hidden_dim, var_out_dim)

    # define loss
    loss_fn = MSELoss()

    # define opt
    learning_rate = 0.3
    momentum = 0.1
    opt = Momentum(net.trainable_params(), learning_rate, momentum)

    # define forward function
    def net_forward(x, y):
        out = net(x)
        loss = loss_fn(out, y)
        return loss

    grad_net = ops.value_and_grad(net_forward, grad_position=None, weights=net.trainable_params())

    def train_one_step(x, y):
        loss, grads = grad_net(x, y)
        opt(grads)
        return loss

    sink_size = 1
    epoch = 1
    sink_process = ms.train.data_sink(train_one_step, dataset, sink_size=sink_size, jit_config=ms.JitConfig())
    loop_num = int(epoch * var_step_per_epoch / sink_size)

    loss = 0.0
    for _ in range(loop_num):
        loss = sink_process()
    assert np.allclose(np.array([loss.asnumpy()]), np.array([0.0047495714]), 0.00001, 0.00001)
    ms.reset_auto_parallel_context()
