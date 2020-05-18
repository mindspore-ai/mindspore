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

    def construct(self, input1, input2, output_grad):
        return grad_all_with_sens(self.network)(input1, input2, output_grad)


class Max(Cell):
    def __init__(self, axis, keep_dims, strategy0=None, strategy1=None):
        super(Max, self).__init__()
        self.add = P.TensorAdd(strategy=strategy0)
        self.reduce_max = P.ReduceMax(keep_dims=keep_dims).set_strategy(strategy=strategy1)
        self.axis = axis

    def construct(self, input1, input2):
        out = self.add(input1, input2)
        return self.reduce_max(out, self.axis)


class MaxFactory:
    def __init__(self, input_shape, axis, keep_dims, strategy0, strategy1):
        self.strategy0 = strategy0
        self.strategy1 = strategy1
        self.axis = axis
        self.keep_dims = keep_dims
        input_size = 1
        prefix = ""
        for s in input_shape:
            prefix = prefix + str(s) + "_"
            input_size = input_size * s
        number_range = min(1000, input_size)
        self.input_np1 = np.reshape(np.arange(0, input_size) % number_range - number_range / 2, input_shape).astype(
            np.float32)
        self.input_np2 = self.input_np1.copy()
        self.out_grad_np = None
        out_shape = list(input_shape)
        out_shape.pop(axis)
        out_size = input_size / input_shape[axis]
        number_range_ = min(1000, out_size)
        self.out_grad_np = np.reshape(np.arange(0, out_size) % number_range_ - number_range_ / 2, out_shape).astype(
            np.float32)
        out_strategy = list(strategy1[1])
        out_strategy.pop(axis)
        self.out_strategy = out_strategy
        need_dev_num = 1
        need_dev_num_ = 1
        for s in strategy0[1]:
            need_dev_num = need_dev_num * s
        for s in out_strategy:
            need_dev_num_ = need_dev_num_ * s
        self.x_id = device_id % need_dev_num
        self.y_id = device_id % need_dev_num
        self.out_id = device_id % need_dev_num_

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
        input1 = Tensor(self.input_np1)
        input2 = Tensor(self.input_np2)
        net = Max(axis=self.axis, keep_dims=self.keep_dims)
        out = net(input1, input2)
        return out.asnumpy()

    def forward_mindspore_parallel_impl(self):
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2)
        xs = self.get_parallel_blocks(self.input_np1, self.strategy0[1])
        ys = self.get_parallel_blocks(self.input_np2, self.strategy0[1])
        x1 = Tensor(xs[self.x_id])
        y1 = Tensor(ys[self.y_id])
        net = Max(axis=self.axis, keep_dims=self.keep_dims, strategy0=self.strategy0, strategy1=self.strategy1)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        net.set_auto_parallel()
        out = net(x, y, parallel_inputs_compile=[x, y], parallel_inputs_run=[x1, y1])
        return out.asnumpy()

    def grad_mindspore_impl(self):
        input1 = Tensor(self.input_np1)
        input2 = Tensor(self.input_np2)
        out_grad = Tensor(self.out_grad_np)
        net = Max(axis=self.axis, keep_dims=self.keep_dims)
        grad_net = Grad(net)
        grad_net.set_train()
        input_grad = grad_net(input1, input2, out_grad)
        return input_grad

    def grad_mindspore_parallel_impl(self):
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2)
        output_grads = self.get_parallel_blocks(self.out_grad_np, self.out_strategy)
        out_grad = Tensor(output_grads[self.out_id])
        xs = self.get_parallel_blocks(self.input_np1, self.strategy0[1])
        ys = self.get_parallel_blocks(self.input_np2, self.strategy0[1])
        x1 = Tensor(xs[self.x_id])
        y1 = Tensor(ys[self.y_id])
        net = Max(axis=self.axis, keep_dims=self.keep_dims, strategy0=self.strategy0, strategy1=self.strategy1)
        grad_net = Grad(net)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        grad_net.set_auto_parallel()
        grad_net.set_train()
        input_grad = grad_net(x, y, out_grad, parallel_inputs_compile=[x, y, out_grad],
                              parallel_inputs_run=[x1, y1, out_grad])
        return input_grad

    def forward_cmp(self):
        out_mindspore = self.forward_mindspore_impl()
        out_mindspore_parallel = self.forward_mindspore_parallel_impl()
        print(out_mindspore)
        print(out_mindspore_parallel)
        out_blocks = self.get_parallel_blocks(out_mindspore, self.out_strategy)
        assert np.allclose(out_blocks[self.out_id], out_mindspore_parallel, 0.001, 0.001)

    def grad_cmp(self):
        input_grad_mindspore = self.grad_mindspore_impl()
        input_grad_mindspore_parallel = self.grad_mindspore_parallel_impl()
        input_grad_mindspore0 = input_grad_mindspore[0].asnumpy()
        input_grad_mindspore1 = input_grad_mindspore[1].asnumpy()
        input_grad_mindspore_parallel0 = input_grad_mindspore_parallel[0].asnumpy()
        input_grad_mindspore_parallel1 = input_grad_mindspore_parallel[1].asnumpy()
        input_grad_blocks_0 = self.get_parallel_blocks(input_grad_mindspore0, self.strategy0[1])
        input_grad_blocks_1 = self.get_parallel_blocks(input_grad_mindspore1, self.strategy0[2])
        assert np.allclose(input_grad_blocks_0[self.x_id], input_grad_mindspore_parallel0, 0.0001, 0.0001)
        assert np.allclose(input_grad_blocks_1[self.y_id], input_grad_mindspore_parallel1, 0.0001, 0.0001)


def test_reid_max_forward_input_256_64():
    fact = MaxFactory(input_shape=(256, 64), axis=1, keep_dims=False, strategy0=(0, (4, 1), (4, 1)),
                      strategy1=(0, (4, 1)))
    fact.forward_cmp()


def test_reid_max_grad_input_256_64():
    fact = MaxFactory(input_shape=(256, 64), axis=1, keep_dims=False, strategy0=(0, (4, 1), (4, 1)),
                      strategy1=(0, (4, 1)))
    fact.grad_cmp()


def test_reid_max_forward_input_128_64_32_32():
    fact = MaxFactory(input_shape=(128, 64, 32, 32), axis=3, keep_dims=False, strategy0=(0, (2, 1, 2, 1), (2, 1, 2, 1)),
                      strategy1=(0, (2, 1, 2, 1)))
    fact.forward_cmp()


def test_reid_max_grad_input_128_64_32_32():
    fact = MaxFactory(input_shape=(128, 64, 32, 32), axis=3, keep_dims=False, strategy0=(0, (2, 1, 2, 1), (2, 1, 2, 1)),
                      strategy1=(0, (2, 1, 2, 1)))
    fact.grad_cmp()


def test_reid_max_forward_input_256_64_repeat():
    fact = MaxFactory(input_shape=(256, 64), axis=1, keep_dims=False, strategy0=(0, (2, 1), (2, 1)),
                      strategy1=(0, (2, 1)))
    fact.forward_cmp()


def test_reid_max_grad_input_256_64_repeat():
    fact = MaxFactory(input_shape=(256, 64), axis=1, keep_dims=False, strategy0=(0, (2, 1), (2, 1)),
                      strategy1=(0, (2, 1)))
    fact.grad_cmp()
