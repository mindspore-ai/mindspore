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
import mindspore as ms
import mindspore.communication.management as distributedTool
from numpy import allclose
from mindspore import context
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.ops.composite import grad_all_with_sens

device_num=4
device_id = int(os.environ["RANK_ID"])
path = "./output/"

def setup_module():
    print("~~~~~~~~~~~set up~~~~~~~~~~~~~")
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=device_num, global_rank=device_id)
    distributedTool.init()
    distributedTool.create_group("0-3", [0,1,2,3])
    print("~~~~~~~~~~~set up finished~~~~~~~~~~~~~")

def teardown_module():
    print("~~~~~~~~~~~~tear down~~~~~~~~~~")
    
class PReLU(Cell):
    def __init__(self, channel=1, w=0.25, strategy_=None, strategy1_=None):
        super(PReLU, self).__init__()
        self.add = P.TensorAdd(strategy=strategy1_)
        self.prelu = P.PReLU(strategy=strategy_)

    def construct(self, x, z, w):
        out = self.add(x, z)
        return self.prelu(out, w)


class Grad(Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.network = network

    def construct(self, input,z, w, output_grad):
        return grad_all_with_sens(self.network)(input,z,w, output_grad)


class PReLUFactory:
    def __init__(self, input_shape, strategy):
        n, c = input_shape[:2]
        prefix = ""
        size = 1
        for s in input_shape:
            prefix = prefix + str(s)
            size = size*s
        self.prefix = prefix
        number_range = min(1000, size)
        self.input_np = np.reshape(np.arange(0, size)%number_range - number_range/2, input_shape).astype(np.float32)
        self.output_grad_np = np.reshape((np.arange(0, size)%(number_range-10) - number_range/2)*0.1, input_shape).astype(np.float32)
        self.channel = c
        self.weight = np.array([np.float32(0.25)] * c)
        self.strategy = strategy

    def forward_mindspore_impl(self):
        net = PReLU(channel=self.channel, w=self.weight)
        x = Tensor(self.input_np)
        z = Tensor(np.zeros(self.input_np.shape), ms.float32)
        w = Tensor(self.weight)
        out = net(x, z, w)
        return out.asnumpy()

    def forward_mindspore_parallel_impl(self):
        net = PReLU(channel=self.channel, w=self.weight, strategy_=self.strategy, strategy1_=(self.strategy[0], self.strategy[1], self.strategy[1]))
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        x = Tensor(self.input_np)
        z = Tensor(np.zeros(self.input_np.shape), ms.float32)
        w = Tensor(self.weight)
        
        inputs = self.get_parallel_blocks(self.input_np, self.strategy[1])
        block_id = device_id%len(inputs)
        x1 = Tensor(inputs[block_id])
        z1 = Tensor(np.zeros(inputs[block_id].shape), ms.float32)
        w1 = Tensor(self.weight)
        
        out = net(x, z, w, parallel_inputs_compile=[x, z, w], parallel_inputs_run=[x1, z1 ,w1])
        return out.asnumpy()
        
    def grad_mindspore_impl(self):
        output_grad = Tensor(self.output_grad_np)
        x = Tensor(self.input_np)
        z = Tensor(np.zeros(self.input_np.shape), ms.float32)
        w = Tensor(self.weight)
        
        net = PReLU(channel=self.channel, w=self.weight)
        grad_net = Grad(net)
        grad_net.set_train()
        input_grad = grad_net(x, z, w, output_grad)
        return input_grad

    def grad_mindspore_parallel_impl(self):
        output_grads = self.get_parallel_blocks(self.output_grad_np, self.strategy[1])
        block_id = device_id%len(output_grads)
        output_grad = Tensor(output_grads[block_id])   
        x = Tensor(self.input_np)
        z = Tensor(np.zeros(self.input_np.shape), ms.float32)
        w = Tensor(self.weight)
        
        net = PReLU(channel=self.channel, w=self.weight, strategy_=self.strategy, strategy1_=(self.strategy[0], self.strategy[1], self.strategy[1]))
        grad_net = Grad(net)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        
        grad_net.set_train()
        inputs = self.get_parallel_blocks(self.input_np, self.strategy[1])
        x1 = Tensor(inputs[block_id])
        z1 = Tensor(np.zeros(inputs[block_id].shape), ms.float32)
        w1 = Tensor(self.weight)
        
        input_grad = grad_net(x, z, w, output_grad, parallel_inputs_compile=[x, z, w, output_grad], parallel_inputs_run=[x1, z1, w1, output_grad])
        return input_grad
    
    def get_parallel_blocks(self, input_, strategy):
        blocks = [input_]
        i = 0
        for stra in strategy:
            temp = []
            while len(blocks)>0:
                block = blocks.pop(0)
                temp.extend(np.split(block, stra, axis=i))
            blocks.extend(temp)
            i+=1
        return blocks
        
    def forward_cmp(self):
        out_mindspore = self.forward_mindspore_impl()
        out_mindspore_parallel = self.forward_mindspore_parallel_impl()
        out_blocks = self.get_parallel_blocks(out_mindspore, self.strategy[1])
        block_id = device_id%len(out_blocks)
        assert np.allclose(out_blocks[block_id], out_mindspore_parallel, 0.0001, 0.001)

    def grad_cmp(self):
        input_grad_mindspore = self.grad_mindspore_impl()
        input_grad_mindspore_parallel = self.grad_mindspore_parallel_impl()
        input_grad_mindspore0 = input_grad_mindspore[0].asnumpy()
        input_grad_mindspore1 = input_grad_mindspore[1].asnumpy()
        input_grad_mindspore2 = input_grad_mindspore[2].asnumpy()
        input_grad_mindspore_parallel0 = input_grad_mindspore_parallel[0].asnumpy()
        input_grad_mindspore_parallel1 = input_grad_mindspore_parallel[1].asnumpy()
        input_grad_mindspore_parallel2 = input_grad_mindspore_parallel[2].asnumpy()
        input_grad_blocks = self.get_parallel_blocks(input_grad_mindspore0, self.strategy[1])
        input1_grad_blocks = self.get_parallel_blocks(input_grad_mindspore1, self.strategy[1])
        block_id = device_id%len(input_grad_blocks)
        assert np.allclose(input_grad_blocks[block_id], input_grad_mindspore_parallel0, 0.0001, 0.0001)
        assert np.allclose(input_grad_mindspore2, input_grad_mindspore_parallel2, 0.0001, 0.0001)
        assert np.allclose(input1_grad_blocks[block_id], input_grad_mindspore_parallel1, 0.0001, 0.0001)

        
        
@pytest.mark.reid_grad
def test_reid_prelu_input_128x64x112x112_repeat():
    stra = (0,(1,1,2,1),(1))
    fact = PReLUFactory(input_shape=(128, 64, 112, 112), strategy=stra)
    fact.forward_cmp()       
 
@pytest.mark.reid_grad
def test_reid_grad_prelu_input_128x64x112x112_repeat():
    stra = (0,(1,1,2,1),(1))
    fact = PReLUFactory(input_shape=(128, 64, 112, 112), strategy=stra)
    fact.grad_cmp()
        
@pytest.mark.reid_grad
def test_reid_prelu_input_128x64x112x112_mix():
    stra = (0,(2,1,1,2),(1))
    fact = PReLUFactory(input_shape=(128, 64, 112, 112), strategy=stra)
    fact.forward_cmp()
        
@pytest.mark.reid_grad
def test_reid_grad_prelu_input_128x64x112x112_mix():
    stra = (0,(2,1,1,2),(1))
    fact = PReLUFactory(input_shape=(128, 64, 112, 112), strategy=stra)
    fact.grad_cmp()

