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
import pytest
import numpy as np
from mindspore.nn import Cell
from mindspore import context
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
import mindspore.communication.management as distributedTool
from mindspore.ops.composite import grad_all

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


class AddRelu(Cell):
    def __init__(self, strategy0=None, strategy1=None):
        super(AddRelu, self).__init__()
        self.add = P.TensorAdd(strategy=strategy0)
        self.relu = P.ReLU(strategy=strategy1)

    def construct(self, x, y):
        out = self.add(x, y)
        out = self.relu(out)
        return out


class NetWithLoss(Cell):
    def __init__(self, network, strategy2=None):
        super(NetWithLoss, self).__init__()
        self.loss = P.SoftmaxCrossEntropyWithLogits(strategy=strategy2)
        self.network = network

    def construct(self, x, y, b):
        predict = self.network(x, y)
        return self.loss(predict, b)[0]


class Grad(Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return grad_all(self.network)(x, y, b)


class AddReluFactory:
    def __init__(self, input_shape, strategy0, strategy1, strategy2):
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
        target_shape = input_shape
        self.target_shape = target_shape
        target_size = 1
        for s in target_shape:
            target_size = target_size * s
        number_range = min(10, target_size)
        self.output_grad_np = np.reshape((np.arange(0, target_size) % number_range) * 0.1, target_shape).astype(
            np.float32)
        self.strategy0 = strategy0
        self.strategy1 = strategy1
        self.strategy2 = strategy2
        out_strategy = strategy1[1]
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

    def grad_mindspore_impl(self):
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2)
        output_grad = Tensor(self.output_grad_np)
        net = AddRelu()
        net_with_loss = NetWithLoss(net)
        grad_net = Grad(net_with_loss)
        grad_net.set_train()
        input_grads = []
        for i in range(0, 3):
            input_grad = grad_net(x, y, output_grad)
            input_grads.append(input_grad)
        return input_grads

    def grad_mindspore_parallel_impl(self):
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2)
        output_grad = Tensor(self.output_grad_np)
        inputs_x = self.get_parallel_blocks(self.input_np1, self.strategy0[1])
        inputs_y = self.get_parallel_blocks(self.input_np2, self.strategy0[1])
        outgrads = self.get_parallel_blocks(self.output_grad_np, self.out_strategy)
        x1 = Tensor(inputs_x[self.x_id])
        y1 = Tensor(inputs_y[self.y_id])
        output_grad1 = Tensor(outgrads[self.out_id])
        net = AddRelu(strategy0=self.strategy0, strategy1=self.strategy1)
        net_with_loss = NetWithLoss(net, strategy2=self.strategy2)
        grad_net = Grad(net_with_loss)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        grad_net.set_auto_parallel()
        grad_net.set_train()
        input_grads = []
        for i in range(0, 3):
            input_grad = grad_net(x, y, output_grad, parallel_inputs_compile=[x, y, output_grad],
                                  parallel_inputs_run=[x1, y1, output_grad1])
            input_grads.append(input_grad)
        return input_grads

    def grad_cmp(self):
        input_grad_mindspores = self.grad_mindspore_impl()
        input_grad_mindspore_parallels = self.grad_mindspore_parallel_impl()
        for i in range(0, len(input_grad_mindspores)):
            input_grad_mindspore = input_grad_mindspores[i]
            input_grad_mindspore_parallel = input_grad_mindspore_parallels[i]
            input_grad_mindspore0 = input_grad_mindspore[0].asnumpy()
            input_grad_mindspore1 = input_grad_mindspore[1].asnumpy()
            input_grad_mindspore_parallel0 = input_grad_mindspore_parallel[0].asnumpy()
            input_grad_mindspore_parallel1 = input_grad_mindspore_parallel[1].asnumpy()
            input_grad_blocks_0 = self.get_parallel_blocks(input_grad_mindspore0, self.strategy0[1])
            input_grad_blocks_1 = self.get_parallel_blocks(input_grad_mindspore1, self.strategy0[2])
            np.save(path + str(i) + "_" + str(device_id) + "_" + self.prefix + "_grad_single0.npy",
                    input_grad_blocks_0[self.x_id])
            np.save(path + str(i) + "_" + str(device_id) + "_" + self.prefix + "_grad_single1.npy",
                    input_grad_blocks_1[self.y_id])
            np.save(path + str(i) + "_" + str(device_id) + "_" + self.prefix + "_grad_parallel0.npy",
                    input_grad_mindspore_parallel0)
            np.save(path + str(i) + "_" + str(device_id) + "_" + self.prefix + "_grad_parallel1.npy",
                    input_grad_mindspore_parallel1)
            assert np.allclose(input_grad_blocks_0[self.x_id], input_grad_mindspore_parallel0, 0.0001, 0.0001)
            assert np.allclose(input_grad_blocks_1[self.y_id], input_grad_mindspore_parallel1, 0.0001, 0.0001)


def test_reid_l2normalize_grad_input_128_512():
    input_shape = (128, 512)
    fact = AddReluFactory(input_shape, strategy0=(0, (4, 1), (4, 1)), strategy1=(0, (4, 1)),
                          strategy2=(0, (4, 1), (4, 1)))
    fact.grad_cmp()


def test_reid_l2normalize_grad_input_128_512_stridesplit():
    input_shape = (128, 512)
    fact = AddReluFactory(input_shape, strategy0=(0, (1, 1), (1, 1)), strategy1=(0, (4, 1)),
                          strategy2=(0, (4, 1), (4, 1)))
    fact.grad_cmp()
