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
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
import mindspore.context as context
import mindspore.communication.management as distributedTool

device_num = 2
device_id = int(os.getenv('DEVICE_ID'))
rank_id = 0

def setup_module():
    global device_num
    global rank_id
    np.random.seed(0)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(enable_hccl=True)
    context.set_context(enable_task_sink=True,
                        device_id=device_id)
    context.set_context(enable_ir_fusion=True)
    context.set_context(enable_loop_sink=False)
    distributedTool.init()
    device_num = distributedTool.get_group_size()
    rank_id = distributedTool.get_rank()
    context.set_auto_parallel_context(device_num=device_num,
                                      global_rank=rank_id)

def teardown_module():
    distributedTool.release()

class Onehot(Cell):
    def __init__(self, axis=-1, depth=1, on_value=1.0, off_value=0.0, strategy=None):
        super(Onehot, self).__init__()
        trans_stra = None
        if strategy:
            trans_stra = (strategy[0],)
        self.onehot = P.OneHot().set_strategy(strategy=strategy)
        self.depth = depth
        self.on_value = Tensor(on_value, ms.float32)
        self.off_value = Tensor(off_value, ms.float32)
        self.transpose = P.Transpose().set_strategy(strategy=trans_stra)
        self.sub = P.Sub().set_strategy(strategy=((1,1),(1,1)))

    def construct(self, input, indices):
        x = self.onehot(indices, self.depth, self.on_value, self.off_value)
        x = self.transpose(x, (1,0))
        x = self.sub(input, x)
        return x

class DataGenerator():
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

    def generate_data(self, shape):
        data = np.random.rand(*shape)
        return data

    def input_data(self, shape):
        data = (self.generate_data(shape)*2).astype(np.float32)
        stra = [1]*len(shape)
        stra[0] = device_num
        datas = self.get_parallel_blocks(data, stra)
        return Tensor(data), Tensor(datas[rank_id])

    def label_data(self, shape, classes):
        data = (self.generate_data(shape)*(classes-1)).astype(np.int32)
        stra = [1]*len(shape)
        stra[0] = device_num
        datas = self.get_parallel_blocks(data, stra)
        return Tensor(data),Tensor(datas[rank_id])

class OneHotFactory:
    def __init__(self, batch_size, classes, on_value=1.0, off_value=0.0, axis=None, strategy=None):
        dataGen = DataGenerator()
        self.input_full, self.input_part = dataGen.input_data((classes, batch_size))
        self.label_full, self.label_part = dataGen.label_data((batch_size,),classes)
        self.depth = classes
        self.on_value = on_value
        self.off_value = off_value
        self.axis = axis
        self.strategy = strategy
    
    def forward_mindspore_single_impl(self):
        net = Onehot(axis=self.axis, 
                     depth=self.depth, 
                     on_value=self.on_value, 
                     off_value=self.off_value)
        out = net(self.input_full, self.label_full)
        return out
        
    def forward_mindspore_parallel_impl(self):
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        net = Onehot(axis=self.axis, 
                     depth=self.depth, 
                     on_value=self.on_value, 
                     off_value=self.off_value, strategy=self.strategy)
        out = net.compile_and_run(self.input_full, self.label_full)
        return out

    def forward_cmp(self):
        out_mindspore_single = self.forward_mindspore_single_impl().asnumpy()
        context.reset_auto_parallel_context()
        out_mindspore_parallel = self.forward_mindspore_parallel_impl().asnumpy()
        context.reset_auto_parallel_context()
        assert np.allclose(out_mindspore_single, out_mindspore_parallel, 0.0001, 0.0001)


def test_reid_onehot_forward_int32_128_depth1024_model_parallel():
    fact = OneHotFactory(batch_size=128,
                         classes=1024,
                         on_value=1.000000,
                         off_value=0.000000,
                         axis=-1,
                         strategy=((1,device_num),(),()))
    fact.forward_cmp()


def test_reid_onehot_forward_int32_1024_depth128_model_parallel():
    fact = OneHotFactory(batch_size=1024,
                         classes=128,
                         on_value=1.000000,
                         off_value=0.000000,
                         axis=-1,
                         strategy=((1,device_num),(),()))
    fact.forward_cmp()
