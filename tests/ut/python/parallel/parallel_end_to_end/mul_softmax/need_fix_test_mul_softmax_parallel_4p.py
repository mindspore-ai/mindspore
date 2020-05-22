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


class MulSoftmax(Cell):
    def __init__(self, strategy0=None, strategy1=None, axis=0):
        super(MulSoftmax, self).__init__()
        self.mul = P.Mul(strategy=strategy0)
        self.softmax = P.Softmax(axis=axis, strategy=strategy1)

    def construct(self, x, z):
        out = self.mul(x, z)
        return self.softmax(out)


class Grad(Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.network = network

    def construct(self, x, y, output_grad):
        return grad_all_with_sens(self.network)(x, y, output_grad)


class MulSoftmaxFactory:
    def __init__(self, input_shape, strategy0, strategy1):
        prefix = ""
        size = 1
        for s in input_shape:
            prefix = prefix + str(s)
            size = size * s
        self.prefix = prefix
        number_range = min(1000, size)
        self.input_np1 = np.reshape(np.arange(0, size) % number_range - number_range / 2, input_shape).astype(
            np.float32)
        self.input_np2 = 1.0
        self.output_grad_np = np.reshape((np.arange(0, size) % (number_range - 10) - number_range / 2) * 0.1,
                                         input_shape).astype(np.float32)
        self.strategy0 = strategy0
        self.strategy1 = strategy1
        need_dev_num = 1
        need_dev_num_ = 1
        for s in strategy0[1]:
            need_dev_num = need_dev_num * s
        for s in strategy1[1]:
            need_dev_num_ = need_dev_num_ * s
        self.x_id = device_id % need_dev_num
        self.y_id = device_id % need_dev_num
        self.out_id = device_id % need_dev_num_

    def forward_mindspore_impl(self):
        net = MulSoftmax()
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2, ms.float32)
        out = net(x, y)
        return out.asnumpy()

    def forward_mindspore_parallel_impl(self):
        net = MulSoftmax(strategy0=self.strategy0, strategy1=self.strategy1)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        net.set_auto_parallel()
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2, ms.float32)
        inputs_x = self.get_parallel_blocks(self.input_np1, self.strategy0[1])
        x1 = Tensor(inputs_x[self.x_id])
        y1 = Tensor(self.input_np2, ms.float32)
        out = net(x, y, parallel_inputs_compile=[x, y], parallel_inputs_run=[x1, y1])
        return out.asnumpy()

    def grad_mindspore_impl(self):
        output_grad = Tensor(self.output_grad_np)
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2, ms.float32)
        net = MulSoftmax()
        grad_net = Grad(net)
        grad_net.set_train()
        input_grad = grad_net(x, y, output_grad)
        return input_grad

    def grad_mindspore_parallel_impl(self):
        output_grads = self.get_parallel_blocks(self.output_grad_np, self.strategy1[1])
        output_grad = Tensor(output_grads[self.out_id])
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2, ms.float32)
        net = MulSoftmax(strategy0=self.strategy0, strategy1=self.strategy1)
        grad_net = Grad(net)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        grad_net.set_train()
        grad_net.set_auto_parallel()
        inputs_x = self.get_parallel_blocks(self.input_np1, self.strategy0[1])
        x1 = Tensor(inputs_x[self.x_id])
        y1 = Tensor(self.input_np2, ms.float32)
        input_grad = grad_net(x, y, output_grad, parallel_inputs_compile=[x, y, output_grad],
                              parallel_inputs_run=[x1, y1, output_grad])
        return input_grad

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

    def forward_cmp(self):
        out_mindspore = self.forward_mindspore_impl()
        out_mindspore_parallel = self.forward_mindspore_parallel_impl()
        np.save(path + str(device_id) + "_" + self.prefix + "_forward_parallel.npy", out_mindspore_parallel)
        out_blocks = self.get_parallel_blocks(out_mindspore, self.strategy1[1])
        assert np.allclose(out_blocks[self.out_id], out_mindspore_parallel, 0.0001, 0.001)

    def grad_cmp(self):
        input_grad_mindspore = self.grad_mindspore_impl()
        input_grad_mindspore_parallel = self.grad_mindspore_parallel_impl()
        input_grad_mindspore0 = input_grad_mindspore[0].asnumpy()
        input_grad_mindspore1 = input_grad_mindspore[1].asnumpy()
        input_grad_mindspore_parallel0 = input_grad_mindspore_parallel[0].asnumpy()
        input_grad_mindspore_parallel1 = input_grad_mindspore_parallel[1].asnumpy()
        np.save(path + str(device_id) + "_" + self.prefix + "_grad_parallel0.npy", input_grad_mindspore_parallel0)
        np.save(path + str(device_id) + "_" + self.prefix + "_grad_parallel1.npy", input_grad_mindspore_parallel1)
        input_grad_blocks_0 = self.get_parallel_blocks(input_grad_mindspore0,
                                                       self.strategy0[1])  # 这里由于TensorMul两个输入X1没做广播，X2做了广播
        assert np.allclose(input_grad_blocks_0[self.x_id], input_grad_mindspore_parallel0, 0.0001, 0.0001)
        assert np.allclose(input_grad_mindspore1, input_grad_mindspore_parallel1, 0.0001, 0.0001)


@pytest.mark.reid_forward
def test_reid_mul_softmax_input_128x64():
    stra0 = (0, (1, 4), ())
    stra1 = (0, (1, 4))
    fact = MulSoftmaxFactory(input_shape=(128, 64), strategy0=stra0, strategy1=stra1)
    fact.forward_cmp()


@pytest.mark.reid_grad
def test_reid_grad_mul_softmax_input_128x64():
    stra0 = (0, (1, 4), ())
    stra1 = (0, (1, 4))
    fact = MulSoftmaxFactory(input_shape=(128, 64), strategy0=stra0, strategy1=stra1)
    fact.grad_cmp()


@pytest.mark.reid_forward
def test_reid_mul_softmax_input_128x64_all_to_all():
    stra0 = (0, (4, 1), ())
    stra1 = (0, (1, 4))
    fact = MulSoftmaxFactory(input_shape=(128, 64), strategy0=stra0, strategy1=stra1)
    fact.forward_cmp()


@pytest.mark.reid_grad
def test_reid_grad_mul_softmax_input_128x64_all_to_all():
    stra0 = (0, (4, 1), ())
    stra1 = (0, (1, 4))
    fact = MulSoftmaxFactory(input_shape=(128, 64), strategy0=stra0, strategy1=stra1)
    fact.grad_cmp()
