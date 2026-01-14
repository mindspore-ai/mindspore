# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
"""test case of tensor index getitem"""

import numpy as np
import pytest
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.log as logger
import mindspore.dataset as ds
from mindspore import Tensor
from mindspore import jit
from mindspore import nn
from mindspore import ops
from mindspore.nn import Cell
from mindspore.train import Model
from mindspore.common import dtype
from mindspore.common.initializer import Zero


class LinearNet(Cell):
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(2, 2, Zero(), Zero())

    def construct(self, x):
        return self.fc(x)


def generator(data_np, label_np, loop_num=2):
    for _ in range(loop_num):
        yield (data_np, label_np)


def use_model_train_cell(network, epoch, dataset, input_x, sink_size=-1, dataset_sink_mode=True):
    optimizer = nn.Momentum(network.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss = nn.MSELoss(reduction='mean')
    model = Model(network, loss, optimizer)
    for _ in range(epoch):
        model.train(1, dataset, sink_size=sink_size, dataset_sink_mode=dataset_sink_mode)
    out = network(input_x)
    return out


def use_train_one_step_cell(network, epoch, dataset, input_x):
    """test mindspore train step"""
    optimizer = nn.Momentum(network.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss = nn.MSELoss(reduction='mean')
    net_with_loss = nn.WithLossCell(network, loss)
    train_network = nn.TrainOneStepCell(net_with_loss, optimizer)
    train_network.set_train()
    for _ in range(epoch):
        for data, label in dataset:
            train_network(data, label)

    out = network(input_x)
    return out


def use_ms_function_train(network, dataset, input_x, epoch=2, sink_size=1,
                          jit_config=None, input_signature=None, enough=True):
    """test mindspore train"""
    optimizer = nn.Momentum(network.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.MSELoss(reduction='mean')

    def net_forward(input_x, label):
        out = network(input_x)
        loss = loss_fn(out, label)
        return loss, out

    grad_fn = ops.value_and_grad(net_forward, grad_position=None,
                                 weights=network.trainable_params(), has_aux=True)

    @jit
    def train_one_step(input_x, label):
        (loss, _), grads = grad_fn(input_x, label)
        optimizer(grads)
        return loss

    data_size = dataset.get_dataset_size()
    steps = data_size * epoch

    sink_process = ms.train.data_sink(train_one_step, dataset,
                                      sink_size=sink_size,
                                      jit_config=jit_config,
                                      input_signature=input_signature)
    if enough:
        loop_num = int(steps/sink_size)
    else:
        loop_num = int(steps/sink_size) + 1
    for _ in range(loop_num):
        loss = sink_process()
        logger.info(f"========The loss with jit:{loss}========")
    out = network(input_x)
    return out


def use_train_without_ms_function(network, dataset, input_x, epoch=2, sink_size=1,
                                  jit_config=None, input_signature=None):
    """test mindspore net"""
    optimizer = nn.Momentum(network.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.MSELoss(reduction='mean')

    def net_forward(input_x, label):
        out = network(input_x)
        loss = loss_fn(out, label)
        return loss, out

    grad_fn = ops.value_and_grad(net_forward, grad_position=None,
                                 weights=network.trainable_params(), has_aux=True)

    def train_one_step(input_x, label):
        (loss, _), grads = grad_fn(input_x, label)
        optimizer(grads)
        return loss

    data_size = dataset.get_dataset_size()
    steps = data_size * epoch

    sink_process = ms.train.data_sink(train_one_step, dataset,
                                      sink_size=sink_size,
                                      jit_config=jit_config,
                                      input_signature=input_signature)

    for _ in range(int(steps/sink_size)):
        loss = sink_process()
        logger.info(f"========The loss without jit:{loss}========")
    out = network(input_x)
    return out


def use_feed_train(network, epoch, dataset, input_x):
    """test mindspore feed net"""
    optimizer = nn.Momentum(network.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.MSELoss(reduction='mean')

    def net_forward(input_x, label):
        out = network(input_x)
        loss = loss_fn(out, label)
        return loss, out

    grad_fn = ops.value_and_grad(net_forward, grad_position=None,
                                 weights=network.trainable_params(), has_aux=True)

    @jit
    def train_one_step(input_x, label):
        (loss, _), grads = grad_fn(input_x, label)
        optimizer(grads)
        return loss

    for _ in range(epoch):
        for inputs, targets in dataset:
            loss = train_one_step(inputs, targets)
            logger.info(f"========The loss with feed:{loss}========")
    out = network(input_x)
    return out


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_data_sink_fn_ms_function_jit_config_none(mode):
    """
    Feature: PyNative data sink fn ms function jit config none
    Description: Test PyNative data sink fn ms function jit config none.
    Expectation: run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    input_x = Tensor(np.ones([2, 2]), dtype.float32)
    input_np = np.ones([2, 2]).astype(np.float32)
    label_np = np.ones([2, 2]).astype(np.float32)

    net0 = LinearNet()
    dataset0 = ds.GeneratorDataset(lambda: generator(input_np, label_np), ["data", "label"])
    out_0 = use_model_train_cell(network=net0, epoch=2, dataset=dataset0, input_x=input_x)
    logger.info(f"====out_0.asnumpy():{out_0.asnumpy()}====")

    net_1 = LinearNet()
    dataset1 = ds.GeneratorDataset(lambda: generator(input_np, label_np), ["data", "label"])
    out_1 = use_train_one_step_cell(network=net_1, epoch=2, dataset=dataset1, input_x=input_x)
    logger.info(f"====out_1.asnumpy():{out_1.asnumpy()}====")
    assert np.allclose(out_1.asnumpy(), out_0.asnumpy(), 0.001, 0.001)

    net_2 = LinearNet()
    dataset2 = ds.GeneratorDataset(lambda: generator(input_np, label_np), ["data", "label"])
    out_2 = use_ms_function_train(network=net_2, dataset=dataset2, input_x=input_x)
    logger.info(f"====out_2.asnumpy():{out_2.asnumpy()}====")
    assert np.allclose(out_1.asnumpy(), out_0.asnumpy(), 0.001, 0.001)

    net_3 = LinearNet()
    dataset3 = ds.GeneratorDataset(lambda: generator(input_np, label_np), ["data", "label"])
    out_3 = use_train_without_ms_function(network=net_3, dataset=dataset3, input_x=input_x)
    logger.info(f"====out_3.asnumpy():{out_3.asnumpy()}====")
    assert np.allclose(out_1.asnumpy(), out_0.asnumpy(), 0.001, 0.001)

    net_4 = LinearNet()
    dataset4 = ds.GeneratorDataset(lambda: generator(input_np, label_np), ["data", "label"])
    out_4 = use_feed_train(network=net_4, epoch=2, dataset=dataset4, input_x=input_x)
    logger.info(f"====out_4.asnumpy():{out_4.asnumpy()}====")
    assert np.allclose(out_1.asnumpy(), out_0.asnumpy(), 0.001, 0.001)
