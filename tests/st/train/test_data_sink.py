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

""" test data sink"""
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops as P
import mindspore.dataset as ds
from mindspore import Tensor, context
from mindspore.train.data_sink import data_sink


def fixed_dataset_generator():
    for _ in range(1, 10):
        yield (
            np.ones((3, 2048, 7, 7), dtype=np.float32),
            np.ones((3, 1000), dtype=np.float32))


def dynamic_dataset_generator_cell():
    for i in range(1, 10):
        yield (
            np.ones((i, 2048, 7, 7), dtype=np.float32),
            np.ones((i, 1000), dtype=np.float32))


def dynamic_dataset_generator_func():
    for i in range(1, 10):
        yield (
            np.ones((i), dtype=np.float32),
            np.ones((i), dtype=np.float32))


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.dense = nn.Dense()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.dense(x)
        x = self.relu(x)
        return x


class ReluReduceMeanDenseRelu(nn.Cell):
    def __init__(self, kernel, bias, in_channel, num_class):
        super().__init__()
        self.relu = P.ReLU()
        self.mean = P.ReduceMean(keep_dims=False)
        self.dense = nn.Dense(in_channel, num_class, kernel, bias)

    def construct(self, x_):
        x_ = self.relu(x_)
        x_ = self.mean(x_, (2, 3))
        x_ = self.dense(x_)
        x_ = self.relu(x_)
        return x_


def _train_func_sink(model, dataset, loss_fn, opt, input_signature=None):
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = P.value_and_grad(forward_fn, None, opt.parameters, has_aux=True)
    model.set_train()

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = P.depend(loss, opt(grads))
        return loss

    data_size = dataset.get_dataset_size()
    epochs = 5
    steps = data_size * epochs
    sink_size = data_size
    jit = ms.JitConfig()

    sink_process = data_sink(train_step, dataset, sink_size=sink_size, jit_config=jit, input_signature=input_signature)
    for _ in range(steps):
        loss = sink_process()
        print("loss: ", loss)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_data_sink_fixed_shape(mode):
    """
    Feature: mindspore.train.data_sink
    Description: test data_sink with fixed-shape dataset.
    Expectation: Success.
    """
    context.set_context(mode=mode)
    weight = Tensor(np.ones((1000, 2048)).astype(np.float32))
    bias = Tensor(np.ones((1000,)).astype(np.float32))
    network = ReluReduceMeanDenseRelu(weight, bias, 2048, 1000)

    dataset = ds.GeneratorDataset(
        fixed_dataset_generator, ["data", "label"])
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)

    _train_func_sink(network, dataset, loss_fn, opt)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.skip(reason='Have ops issue, not support yet')
def test_data_sink_dynamic_shape(mode):
    """
    Feature: mindspore.train.data_sink
    Description: test data_sink with dynamic shape dataset.
    Expectation: Success.
    """
    context.set_context(mode=mode)
    weight = Tensor(np.ones((1000, 2048)).astype(np.float32))
    bias = Tensor(np.ones((1000,)).astype(np.float32))

    network = ReluReduceMeanDenseRelu(weight, bias, 2048, 1000)

    dataset = ds.GeneratorDataset(dynamic_dataset_generator_cell, ["data", "label"])
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)

    input_signature = (Tensor(shape=[None, 2048, 7, 7], dtype=ms.float32),
                       Tensor(shape=[None, 1000], dtype=ms.float32))
    _train_func_sink(network, dataset, loss_fn, opt, input_signature)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_function_data_sink_dynamic_shape(mode):
    """
    Feature: mindspore.train.data_sink
    Description: test data_sink with dynamic shape dataset.
    Expectation: Success.
    """
    context.set_context(mode=mode)
    dataset = ds.GeneratorDataset(dynamic_dataset_generator_func, ["data", "label"])

    def func_net(x, y):
        out = x + y
        return out

    data_size = dataset.get_dataset_size()
    epochs = 5
    steps = data_size * epochs
    sink_size = data_size
    jit = ms.JitConfig()

    input_signature = (Tensor(shape=[None,], dtype=ms.float32), Tensor(shape=[None,], dtype=ms.float32))

    sink_process = data_sink(func_net, dataset, sink_size=sink_size, jit_config=jit, input_signature=input_signature)
    for _ in range(steps):
        out = sink_process()
        print("out: ", out)
