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
from numpy import allclose
from mindspore.nn import Cell
from mindspore import context
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
import mindspore.communication.management as distributedTool

device_num=4
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


class Onehot(Cell):
    def __init__(self, axis=-1, depth=1, on_value=1.0, off_value=0.0, strategy=None):
        super(Onehot, self).__init__()
        self.onehot = P.OneHot(axis, strategy=strategy)
        self.depth = depth
        self.on_value = Tensor(on_value, ms.float32)
        self.off_value = Tensor(off_value, ms.float32)

    def construct(self, indices):
        return self.onehot(indices, self.depth, self.on_value, self.off_value)


class OneHotFactory:
    def __init__(self, input_shape, depth, on_value=1.0, off_value=0.0, axis=None, dtype=None, strategy0=None):
        size = 1
        prefix = ""
        for s in input_shape:
            prefix = prefix + str(s)
            size = size*s
        self.prefix = prefix
        number_range = min(10, size)
        self.input_np = np.reshape(np.arange(0, size)%number_range, input_shape).astype(np.int32)
        self.depth = depth
        self.on_value = on_value
        self.off_value = off_value
        self.axis = axis
        self.dtype = dtype
        self.strategy0 = strategy0
        need_dev_num = 1
        for s in strategy0[1]:
            need_dev_num = need_dev_num*s
        self.x_id = device_id%need_dev_num
        self.out_id = device_id%need_dev_num
    
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
        
    def grad_mindspore_impl(self):
        output_grad = Tensor(self.output_grad_np)
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2, ms.float32)
        net = AddRelu()
        grad_net = Grad(net)
        grad_net.set_train()
        input_grad = grad_net(x, y, output_grad)
        return input_grad
        
    def forward_mindspore_impl(self):
        indices = Tensor(self.input_np)
        net = Onehot(axis=self.axis, 
                     depth=self.depth, 
                     on_value=self.on_value, 
                     off_value=self.off_value)
        out = net(indices)
        return out.asnumpy()
        
    def forward_mindspore_parallel_impl(self):
        x = Tensor(self.input_np)
        inputs_x = self.get_parallel_blocks(self.input_np, self.strategy0[1])
        x1 = Tensor(inputs_x[self.x_id])
        net = Onehot(axis=self.axis, 
                     depth=self.depth, 
                     on_value=self.on_value, 
                     off_value=self.off_value, strategy=self.strategy0)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        net.set_auto_parallel()
        out = net(x, parallel_inputs_compile=[x], parallel_inputs_run=[x1])
        return out.asnumpy()

    def forward_cmp(self):
        out_mindspore = self.forward_mindspore_impl()
        out_mindspore_parallel = self.forward_mindspore_parallel_impl()
        out_blocks = self.get_parallel_blocks(out_mindspore, self.strategy0[1])
        assert np.allclose(out_blocks[self.out_id], out_mindspore_parallel, 0.0001, 0.0001)


def test_reid_onehot_forward_int32_128_depth13000():
    fact = OneHotFactory(input_shape=(128,),
                         depth=131072,
                         on_value=1.000000,
                         off_value=0.000000,
                         axis=-1,
                         dtype="float32",
                         strategy0=(0,(2,)))
    fact.forward_cmp()


def test_reid_onehot_forward_int32_131072_depth127():
    fact = OneHotFactory(input_shape=(131072,),
                         depth=127,
                         on_value=1.000000,
                         off_value=0.000000,
                         axis=-1,
                         dtype="float32",
                         strategy0=(0,(4,)))
    fact.forward_cmp()

