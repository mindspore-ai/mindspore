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

import os
import numpy as np
import pytest
from numpy import allclose as allclose_nparray

import mindspore.communication.management as distributedTool
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops.composite import grad_all_with_sens

device_num = 4
device_id = int(os.environ["RANK_ID"])
path = "./output/"


def setup_module():
    print("~~~~~~~~~~~set up~~~~~~~~~~~~~")
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=device_num, global_rank=device_id)
    distributedTool.init()
    distributedTool.create_group("0-3", [0, 1, 2, 3])
    print("~~~~~~~~~~~set up finished~~~~~~~~~~~~~")


def teardown_module():
    print("~~~~~~~~~~~~tear down~~~~~~~~~~")


class Grad(Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.network = network

    def construct(self, x, y, output_grad):
        return grad_all_with_sens(self.network)(x, y, output_grad)


class Reshape(Cell):
    def __init__(self, target_shape, strategy0=None, strategy1=None):
        super(Reshape, self).__init__()
        self.add = P.TensorAdd(strategy=strategy0)
        self.reshape = P.Reshape(strategy=strategy1)
        self.shape = tuple(target_shape)

    def construct(self, input1, input2):
        x = self.add(input1, input2)
        return self.reshape(x, self.shape)


class ReshapeFactory:
    def __init__(self, input_shape, target_shape, strategy0, strategy1):
        prefix = ""
        size = 1
        for s in input_shape:
            prefix = prefix + str(s)
            size = size * s
        self.prefix = prefix
        number_range = min(1000, size)
        self.input_np1 = np.reshape(np.arange(0, size) % number_range - number_range / 2, input_shape).astype(
            np.float32)
        self.input_np2 = np.reshape(np.arange(0, size) % number_range - number_range / 4, input_shape).astype(
            np.float32)
        target_size = 1
        for s in target_shape:
            target_size = target_size * s
        number_range = min(1000, target_size)
        self.output_grad_np = np.reshape(np.arange(0, target_size) % number_range - number_range / 2,
                                         target_shape).astype(np.float32)
        self.target_shape = target_shape
        self.strategy0 = strategy0
        self.strategy1 = strategy1
        out_strategy = [1] * len(target_shape)
        out_strategy[0] = strategy1[1][0]
        self.out_strategy = out_strategy

        need_dev_num0 = 1
        need_dev_num1 = 1
        for s in strategy0[1]:
            need_dev_num0 = need_dev_num0 * s
        for s in out_strategy:
            need_dev_num1 = need_dev_num1 * s
        self.x_id = device_id % need_dev_num0
        self.y_id = device_id % need_dev_num0
        self.out_id = device_id % need_dev_num1

    def get_parallel_blocks(self, input_, strategy):
        blocks = [input_]
        i = 0
        for stra in strategy:
            temp = []
            while len(blocks) > 0:
                block = blocks.pop(0)
                temp.extend(np.split(block, stra, axis=i))
            blocks.extend(temp)
            i += 1
        return blocks

    def forward_reshape_mindspore_impl(self):
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2)
        net = Reshape(self.target_shape)
        out = net(x, y)
        return out.asnumpy()

    def forward_reshape_mindspore_parallel_impl(self):
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2)
        inputs_x = self.get_parallel_blocks(self.input_np1, self.strategy0[1])
        inputs_y = self.get_parallel_blocks(self.input_np2, self.strategy0[1])
        x1 = Tensor(inputs_x[self.x_id])
        y1 = Tensor(inputs_y[self.y_id])
        net = Reshape(self.target_shape, strategy0=self.strategy0, strategy1=self.strategy1)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        net.set_auto_parallel()
        out = net(x, y, parallel_inputs_compile=[x, y], parallel_inputs_run=[x1, y1])
        return out.asnumpy()

    def grad_reshape_mindspore_impl(self):
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2)
        output_grad = Tensor(self.output_grad_np)
        net = Reshape(self.target_shape)
        grad_net = Grad(net)
        grad_net.set_train()
        input_grad = grad_net(x, y, output_grad)
        return input_grad

    def grad_reshape_mindspore_parallel_impl(self):
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2)
        output_grad = Tensor(self.output_grad_np)
        inputs_x = self.get_parallel_blocks(self.input_np1, self.strategy0[1])
        inputs_y = self.get_parallel_blocks(self.input_np2, self.strategy0[1])
        outgrads = self.get_parallel_blocks(self.output_grad_np, self.out_strategy)
        x1 = Tensor(inputs_x[self.x_id])
        y1 = Tensor(inputs_y[self.y_id])
        output_grad1 = Tensor(outgrads[self.out_id])
        net = Reshape(self.target_shape, strategy0=self.strategy0, strategy1=self.strategy1)
        grad_net = Grad(net)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        grad_net.set_auto_parallel()
        grad_net.set_train()
        input_grad = grad_net(x, y, output_grad, parallel_inputs_compile=[x, y, output_grad1],
                              parallel_inputs_run=[x1, y1, output_grad1])
        return input_grad

    def forward_reshape_cmp(self):
        out_mindspore = self.forward_reshape_mindspore_impl()
        out_mindspore_parallel = self.forward_reshape_mindspore_parallel_impl()
        out_blocks = self.get_parallel_blocks(out_mindspore, self.out_strategy)
        assert np.allclose(out_blocks[self.out_id], out_mindspore_parallel, 0.0001, 0.001)

    def grad_reshape_cmp(self):
        input_grad_mindspore = self.grad_reshape_mindspore_impl()
        input_grad_mindspore_parallel = self.grad_reshape_mindspore_parallel_impl()
        input_grad_mindspore0 = input_grad_mindspore[0].asnumpy()
        input_grad_mindspore1 = input_grad_mindspore[1].asnumpy()
        input_grad_mindspore_parallel0 = input_grad_mindspore_parallel[0].asnumpy()
        input_grad_mindspore_parallel1 = input_grad_mindspore_parallel[1].asnumpy()
        input_grad_blocks_0 = self.get_parallel_blocks(input_grad_mindspore0, self.strategy0[1])
        input_grad_blocks_1 = self.get_parallel_blocks(input_grad_mindspore1, self.strategy0[2])
        assert allclose_nparray(input_grad_blocks_0[self.x_id], input_grad_mindspore_parallel0, 0.0001, 0.0001)
        assert allclose_nparray(input_grad_blocks_1[self.y_id], input_grad_mindspore_parallel1, 0.0001, 0.0001)


@pytest.mark.reid_forward
def test_reid_reshape_input_128x512x7x7_target_128x25088():
    fact = ReshapeFactory(input_shape=(128, 512, 7, 7), target_shape=(128, 25088),
                          strategy0=(0, (4, 1, 1, 1), (4, 1, 1, 1)), strategy1=(0, (4, 1, 1, 1)))
    fact.forward_reshape_cmp()


def test_reid_reshape_grad_input_128x512x7x7_target_128x25088():
    fact = ReshapeFactory(input_shape=(128, 512, 7, 7), target_shape=(128, 25088),
                          strategy0=(0, (4, 1, 1, 1), (4, 1, 1, 1)), strategy1=(0, (4, 1, 1, 1)))
    fact.grad_reshape_cmp()


@pytest.mark.reid_forward
def test_reid_reshape_input_128x64_target_128x64x1x1():
    fact = ReshapeFactory(input_shape=(128, 64), target_shape=(128, 64, 1, 1), strategy0=(0, (2, 1), (2, 1)),
                          strategy1=(0, (2, 1)))
    fact.forward_reshape_cmp()


@pytest.mark.reid_grad
def test_reid_reshape_grad_input_128x64_target_128x64x1x1():
    fact = ReshapeFactory(input_shape=(128, 64), target_shape=(128, 64, 1, 1), strategy0=(0, (2, 1), (2, 1)),
                          strategy1=(0, (2, 1)))
    fact.grad_reshape_cmp()
