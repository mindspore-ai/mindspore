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
import os
import pytest
from numpy import allclose as allclose_nparray

import mindspore as ms
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


class GradScalar(Cell):
    def __init__(self, network):
        super(GradScalar, self).__init__()
        self.network = network
        self.sens = Tensor([1.0], dtype=ms.float32)

    def construct(self, x, y):
        return grad_all_with_sens(self.network)(x, y, self.sens)


class ReduceMean(Cell):
    def __init__(self, keep_dims, axis, strategy0=None, strategy1=None):
        super(ReduceMean, self).__init__()
        self.add = P.TensorAdd(strategy=strategy0)
        self.reduce_mean = P.ReduceMean(keep_dims=keep_dims).set_strategy(strategy=strategy1)
        self.axis = axis

    def construct(self, x, y):
        out = self.add(x, y)
        return self.reduce_mean(out, self.axis)


class ReduceMeanFactory:
    def __init__(self, input_shape, keep_dims, axis, strategy0=None, strategy1=None):
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
        self.keep_dims = keep_dims
        self.axis = axis
        target_shape = self.input_np1.mean(axis=axis, keepdims=keep_dims).shape
        target_size = 1
        for s in target_shape:
            target_size = target_size * s
        number_range = min(1000, target_size)
        self.output_grad_np = np.array([1.0], dtype=np.float32)
        if len(target_shape) > 0:
            self.output_grad_np = np.reshape(np.arange(0, target_size) % number_range, target_shape).astype(
                np.float32) + 1.0
        self.shape = target_shape
        self.strategy0 = strategy0
        self.strategy1 = strategy1
        out_strategy = []
        axis_ = list(axis)
        if axis_[0] == -1:
            axis_[0] = len(input_shape) - 1
        for i in range(0, len(input_shape)):
            if i in axis_:
                if keep_dims:
                    out_strategy.append(1)
            else:
                out_strategy.append(strategy1[1][i])
        self.out_strategy = out_strategy
        need_dev_num0 = 1
        need_dev_num1 = 1
        for s in strategy0[1]:
            need_dev_num0 = need_dev_num0 * s
        for s in out_strategy:
            need_dev_num1 = need_dev_num1 * s
        self.x_id = device_id % need_dev_num0
        self.y_id = device_id % need_dev_num0
        block_id = device_id % need_dev_num0
        device_index = self.id_to_list(block_id, self.strategy1[1])
        print(device_index)
        for i in axis:
            device_index[i] = 0
        print(device_index)
        self.out_id = self.list_to_id(device_index, self.out_strategy)
        print(self.out_id)

    def id_to_list(self, id, shape):
        result = []
        r = id
        for i in range(0, len(shape)):
            v = 1
            for j in range(i + 1, len(shape)):
                v = v * shape[j]
            result.append(r // v)
            r = r % v
        return result

    def list_to_id(self, id_list, shape):
        result = 0
        for i in range(0, len(id_list)):
            v = 1
            for j in range(i + 1, len(id_list)):
                v = v * shape[j]
            result = result + id_list[i] * v
        return result

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

    def forward_mindspore_impl(self):
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2)
        net = ReduceMean(keep_dims=self.keep_dims, axis=self.axis)
        out = net(x, y)
        return out.asnumpy()

    def forward_mindspore_parallel_impl(self):
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2)
        inputs_x = self.get_parallel_blocks(self.input_np1, self.strategy0[1])
        inputs_y = self.get_parallel_blocks(self.input_np2, self.strategy0[1])
        x1 = Tensor(inputs_x[self.x_id])
        y1 = Tensor(inputs_y[self.y_id])
        net = ReduceMean(keep_dims=self.keep_dims, axis=self.axis, strategy0=self.strategy0, strategy1=self.strategy1)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        net.set_auto_parallel()
        out = net(x, y, parallel_inputs_compile=[x, y], parallel_inputs_run=[x1, y1])
        return out.asnumpy()

    def grad_mindspore_impl(self):
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2)
        out_grad = Tensor(self.output_grad_np)
        net = ReduceMean(keep_dims=self.keep_dims, axis=self.axis)
        grad_net = Grad(net)
        grad_net.set_train()
        input_grad = grad_net(x, y, out_grad)
        return input_grad

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
        net = ReduceMean(keep_dims=self.keep_dims, axis=self.axis, strategy0=self.strategy0, strategy1=self.strategy1)
        grad_net = Grad(net)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        grad_net.set_auto_parallel()
        grad_net.set_train()
        input_grad = grad_net(x, y, output_grad, parallel_inputs_compile=[x, y, output_grad1],
                              parallel_inputs_run=[x1, y1, output_grad1])
        return input_grad

    def forward_cmp(self):
        out_mindspore = self.forward_mindspore_impl()
        out_mindspore_parallel = self.forward_mindspore_parallel_impl()
        out_blocks = self.get_parallel_blocks(out_mindspore, self.out_strategy)
        assert np.allclose(out_blocks[self.out_id], out_mindspore_parallel, 0.0001, 0.001)

    def grad_cmp(self):
        input_grad_mindspore = self.grad_mindspore_impl()
        input_grad_mindspore_parallel = self.grad_mindspore_parallel_impl()
        input_grad_mindspore0 = input_grad_mindspore[0].asnumpy()
        input_grad_mindspore1 = input_grad_mindspore[1].asnumpy()
        input_grad_mindspore_parallel0 = input_grad_mindspore_parallel[0].asnumpy()
        input_grad_mindspore_parallel1 = input_grad_mindspore_parallel[1].asnumpy()
        input_grad_blocks_0 = self.get_parallel_blocks(input_grad_mindspore0, self.strategy0[1])
        input_grad_blocks_1 = self.get_parallel_blocks(input_grad_mindspore1, self.strategy0[2])
        assert allclose_nparray(input_grad_blocks_0[self.x_id], input_grad_mindspore_parallel0, 0.0001, 0.0001)
        assert allclose_nparray(input_grad_blocks_1[self.y_id], input_grad_mindspore_parallel1, 0.0001, 0.0001)


def test_reid_reducemean_input_64x16():
    fact = ReduceMeanFactory(input_shape=(64 * 16,), keep_dims=False, axis=(-1,), strategy0=(0, (4,), (4,)),
                             strategy1=(0, (4,)))
    fact.forward_cmp()


def test_grad_reid_reducemean_input_64x16():
    fact = ReduceMeanFactory(input_shape=(64 * 16,), keep_dims=False, axis=(-1,), strategy0=(0, (4,), (4,)),
                             strategy1=(0, (4,)))
    fact.grad_cmp()


def test_reid_reducemean_input_64x128x28x28():
    fact = ReduceMeanFactory(input_shape=(64, 128, 32, 32), keep_dims=True, axis=(2, 3),
                             strategy0=(0, (2, 1, 2, 1), (2, 1, 2, 1)), strategy1=(0, (2, 1, 2, 1)))
    fact.forward_cmp()


def test_grad_reid_reducemean_input_64x128x28x28():
    fact = ReduceMeanFactory(input_shape=(64, 128, 32, 32), keep_dims=True, axis=(2, 3),
                             strategy0=(0, (2, 1, 2, 1), (2, 1, 2, 1)), strategy1=(0, (2, 1, 2, 1)))
    fact.grad_cmp()
