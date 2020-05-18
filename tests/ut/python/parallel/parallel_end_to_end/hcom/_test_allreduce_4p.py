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
import mindspore as ms
from mindspore.nn import Cell
from mindspore import context
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
import mindspore.communication.management as distributedTool
from mindspore.ops.composite import grad_all_with_sens

device_num = 4
device_id = int(os.environ["RANK_ID"])
path = "./output/"


def setup_module():
    print("~~~~~~~~~~~set up~~~~~~~~~~~~~")
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=device_num, global_rank=device_id)
    distributedTool.init()
    print("~~~~~~~~~~~set up finished~~~~~~~~~~~~~")


def teardown_module():
    print("~~~~~~~~~~~~tear down~~~~~~~~~~")


class MatmulSingle(Cell):
    def __init__(self, transpose_a=False, transpose_b=False):
        super(MatmulSingle, self).__init__()
        self.matmul1 = P.MatMul(transpose_a, transpose_b)
        self.matmul2 = P.MatMul(transpose_a, transpose_b)
        self.pow = P.Pow()
        self.reduce_sum = P.ReduceSum()

    def construct(self, x, y, z):
        out = self.matmul1(x, y)
        out = self.matmul2(out, z)
        out = self.pow(out, 2.0)
        out = self.reduce_sum(out, None)
        return out


class MatmulReduce(Cell):
    def __init__(self, group, transpose_a=False, transpose_b=False):
        super(MatmulReduce, self).__init__()
        self.matmul1 = P.MatMul(transpose_a, transpose_b)
        self.allreduce1 = P.AllReduce(group=group)
        self.matmul2 = P.MatMul(transpose_a, transpose_b)
        self.pow = P.Pow()
        self.reduce_sum = P.ReduceSum()
        self.allreduce2 = P.AllReduce(group=group)

    def construct(self, x, y, z):
        out = self.matmul1(x, y)
        out = self.allreduce1(out)
        out = self.matmul2(out, z)
        out = self.pow(out, 2.0)
        out = self.reduce_sum(out, None)
        out = self.allreduce2(out)
        return out


class Grad(Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.network = network

    def construct(self, x, y, z, sens):
        return grad_all_with_sens(self.network)(x, y, z, sens)


class MatmulReduceFactory:
    def __init__(self, inputx_shape, inputy_shape, inputz_shape, x_stra, y_stra, z_stra):
        self.inputx = self.GenValue(inputx_shape, 10)
        self.inputy = self.GenValue(inputy_shape, 20)
        self.inputz = self.GenValue(inputz_shape, 30)
        self.x_stra = x_stra
        self.y_stra = y_stra
        self.z_stra = z_stra
        stra_size = 1
        for s in x_stra:
            stra_size = stra_size * s
        self.stra_size = stra_size

    def GenValue(self, input_shape, delta):
        size = 1
        for s in input_shape:
            size = size * s
        number_range = min(100, size)
        input_np = np.reshape(np.arange(0, size) % number_range - delta, input_shape).astype(np.float32)
        return input_np

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

    def grad_mindspore_impl_single(self):
        x = Tensor(self.inputx)
        y = Tensor(self.inputy)
        z = Tensor(self.inputz)
        sens = Tensor(1.0, dtype=ms.float32)
        net = MatmulSingle()
        grad_net = Grad(net)
        grad_net.set_train()
        input_grad = grad_net(x, y, z, sens)
        return input_grad

    def grad_mindspore_impl_reduce(self):
        inputxs = self.get_parallel_blocks(self.inputx, self.x_stra)
        inputys = self.get_parallel_blocks(self.inputy, self.y_stra)
        inputzs = self.get_parallel_blocks(self.inputz, self.z_stra)
        x = Tensor(inputxs[device_id % self.stra_size])
        y = Tensor(inputys[device_id % self.stra_size])
        z = Tensor(inputzs[device_id % self.stra_size])
        repeat_num = device_num / self.stra_size
        v = self.stra_size * repeat_num * repeat_num * repeat_num
        sens = Tensor(1.0 / v, dtype=ms.float32)
        net = MatmulReduce("hccl_world_group")
        grad_net = Grad(net)
        grad_net.set_train()
        input_grad = grad_net(x, y, z, sens)
        return input_grad

    def grad_cmp(self):
        single_results = self.grad_mindspore_impl_single()
        reduce_results = self.grad_mindspore_impl_reduce()
        single_result0 = self.get_parallel_blocks(single_results[0].asnumpy(), self.x_stra)[device_id % self.stra_size]
        reduce_result0 = reduce_results[0].asnumpy()
        single_result1 = self.get_parallel_blocks(single_results[1].asnumpy(), self.y_stra)[device_id % self.stra_size]
        reduce_result1 = reduce_results[1].asnumpy()
        single_result2 = self.get_parallel_blocks(single_results[2].asnumpy(), self.z_stra)[device_id % self.stra_size]
        reduce_result2 = reduce_results[2].asnumpy()
        assert np.allclose(single_result0, reduce_result0, 0.0001, 0.0001)
        assert np.allclose(single_result1, reduce_result1, 0.0001, 0.0001)
        assert np.allclose(single_result2, reduce_result2, 0.0001, 0.0001)


def test_reduce_grad():
    inputx_shape = (32, 64)
    inputy_shape = (64, 64)
    inputz_shape = (64, 32)
    fact = MatmulReduceFactory(inputx_shape, inputy_shape, inputz_shape, (1, 4), (4, 1), (1, 4))
    fact.grad_cmp()


def test_reduce_grad_repeat():
    inputx_shape = (32, 64)
    inputy_shape = (64, 64)
    inputz_shape = (64, 32)
    fact = MatmulReduceFactory(inputx_shape, inputy_shape, inputz_shape, (1, 2), (2, 1), (1, 2))
    fact.grad_cmp()
