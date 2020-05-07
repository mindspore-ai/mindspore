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
from numpy import allclose
from mindspore import context
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
import mindspore.communication.management as distributedTool
from mindspore.ops.composite import grad_all_with_sens

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

class Matmul(Cell):
    def __init__(self, transpose_a=False, transpose_b=False, strategy0=None, strategy1=None):
        super(Matmul, self).__init__()
        self.add = P.TensorAdd(strategy=strategy1)
        self.matmul = P.MatMul(transpose_a, transpose_b, strategy=strategy0)
    def construct(self, x, w, z):
        out = self.add(x, z)
        return self.matmul(out, w)

class BatchMatMul(Cell):
    def __init__(self, transpose_a=False, transpose_b=False, strategy0=None, strategy1=None):
        super(BatchMatMul, self).__init__()
        self.add = P.TensorAdd(strategy=strategy1)
        self.batchmatmul = P.BatchMatMul(transpose_a, transpose_b, strategy=strategy0)
    def construct(self, x, w, z):
        out = self.add(x, z)
        return self.batchmatmul(out, w)

class Grad(Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.network = network

    def construct(self, inputa, inputb, inputz, output_grad):
        gout = grad_all_with_sens(self.network)(inputa, inputb, inputz, output_grad)
        return gout

class BatchmatmulFactory:
    def __init__(self, inputa_shape, inputb_shape, transpose_a, transpose_b, strategy, strategy_):
        self.strategy = strategy
        self.strategy_ = strategy_
        inputa_size = 1
        inputb_size = 1
        prefix = ""
        for s in inputa_shape:
            prefix = prefix + str(s) + "_"
            inputa_size = inputa_size*s
        prefix = prefix + "and"
        for s in inputb_shape:
            prefix = prefix + str(s) + "_"
            inputb_size = inputb_size*s
        number_rangea = min(1000, inputa_size)
        number_rangeb = min(1000, inputb_size)
        self.inputa = np.reshape(np.arange(0, inputa_size)%number_rangea - number_rangea/2, inputa_shape).astype(np.float32)
        self.inputb = np.reshape(np.arange(0, inputb_size)%number_rangeb - number_rangeb/2, inputb_shape).astype(np.float32)
        self.inputz = np.zeros(self.inputa.shape).astype(np.float32)
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        
        out_shape = []
        device_matrix = []
        out_strategy = []
        if transpose_a:
            temp = inputa_shape[-1]
            inputa_shape[-1] = inputa_shape[-2]
            inputa_shape[-2] = temp
        if transpose_b:
            temp = inputb_shape[-1]
            inputb_shape[-1] = inputb_shape[-2]
            inputb_shape[-2] = temp
        
        if (len(inputa_shape) >= len(inputb_shape)):
            out_shape = list(inputa_shape)
            out_shape[-1] = inputb_shape[-1]
        else:
            out_shape = list(inputb_shape)
            out_shape[-2] = inputa_shape[-2]
        
       
        strategy1 = list(self.strategy[1])
        strategy2 = list(self.strategy[2])
        if transpose_a:
            temp = strategy1[-1]
            strategy1[-1] = strategy1[-2]
            strategy1[-2] = temp
        if transpose_b:
            temp = strategy2[-1]
            strategy2[-1] = strategy2[-2]
            strategy2[-2] = temp
            
        if (len(strategy1) >= len(strategy2)):
            out_strategy = strategy1.copy()
            out_strategy[-1] = strategy2[-1]
        else:
            out_strategy = strategy2.copy()
            out_strategy[-2] = strategy1[-2]
        device_matrix = out_strategy.copy()
        device_matrix.insert(-1, strategy1[-1])
        self.out_strategy = out_strategy
        
        need_dev_num = 1    
        for s in device_matrix:
            need_dev_num = need_dev_num*s
        self.need_dev_num = need_dev_num
        self.device_matrix = device_matrix
        
        out_size = 1
        for s in out_shape:
            out_size = out_size*s 
        number_range = min(1000, out_size)
        self.output_grad_np = np.reshape(np.arange(0, out_size)%number_range - number_range/2, out_shape).astype(np.float32)
        
        device_index = self.id_to_list(device_id%need_dev_num, self.device_matrix)
        x_index = device_index[:-1].copy()
        if transpose_a:
            temp = x_index[-1]
            x_index[-1] = x_index[-2]
            x_index[-2] = temp
        y_index = device_index[:-3].copy()
        y_index.append(device_index[-2])
        y_index.append(device_index[-1])
        if transpose_b:
            temp = y_index[-1]
            y_index[-1] = y_index[-2]
            y_index[-2] = temp
            
        out_index = device_index[:-2].copy()
        out_index.append(device_index[-1])
        
        print(device_matrix)
        print(device_index)
        
        need_dev_num_ = 1
        for s in strategy_[1]:
            need_dev_num_ = need_dev_num_*s
        self.x_id = device_id%need_dev_num_
        self.y_id = self.list_to_id(y_index, self.strategy[2])
        self.out_id = self.list_to_id(out_index, self.out_strategy)
        
        
    
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
    
    """
    shape：每一维的上限，如（2,4,8）
    """
    def id_to_list(self, id, shape):
        result = []
        r = id
        for i in range(0, len(shape)):
            v = 1
            for j in range(i+1, len(shape)):
                v = v*shape[j]
            result.append(r//v)
            r = r%v
        return result
    
    def list_to_id(self, id_list, shape):
        result = 0
        for i in range(0, len(id_list)):
            v = 1
            for j in range(i+1, len(id_list)):
                v = v*shape[j]
            result = result + id_list[i]*v
        return result
            
    
    def forward_mindspore_impl(self):
        if len(self.inputa.shape) > 2:
            matmul = BatchMatMul(self.transpose_a, self.transpose_b)
        else:
            matmul = Matmul(self.transpose_a, self.transpose_b)
        matmul.set_train()
        out_me = matmul(Tensor(self.inputa), Tensor(self.inputb), Tensor(self.inputz))
        return out_me.asnumpy()
        
    def forward_mindspore_parallel_impl(self):
        if len(self.inputa.shape) > 2:
            matmul = BatchMatMul(self.transpose_a, self.transpose_b, strategy0=self.strategy, strategy1=self.strategy_)
        else:
            matmul = Matmul(self.transpose_a, self.transpose_b, strategy0=self.strategy, strategy1=self.strategy_)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        x = Tensor(self.inputa)
        y = Tensor(self.inputb)
        z = Tensor(self.inputz)
        xs = self.get_parallel_blocks(self.inputa, self.strategy_[1])
        ys = self.get_parallel_blocks(self.inputb, self.strategy[2])
        zs = self.get_parallel_blocks(self.inputz, self.strategy_[1])
        x1 = Tensor(xs[self.x_id]) #
        y1 = Tensor(ys[self.y_id]) #需要从设备矩阵推导
        z1 = Tensor(zs[self.x_id])
        matmul.set_train()
        matmul.set_auto_parallel()
        out_me = matmul(x, y, z, parallel_inputs_compile=[x, y, z], parallel_inputs_run=[x1, y1, z1])
        return out_me.asnumpy()
        
    def grad_mindspore_impl(self):
        x = Tensor(self.inputa)
        y = Tensor(self.inputb)
        z = Tensor(self.inputz)
        if len(self.inputa.shape) > 2:
            matmul = BatchMatMul(self.transpose_a, self.transpose_b)
        else:
            matmul = Matmul(self.transpose_a, self.transpose_b)
        net_me = Grad(matmul)
        net_me.set_train()
        out_grad_me = Tensor(self.output_grad_np)
        out_grad = net_me(x, y, z, out_grad_me)
        return out_grad
        
    def grad_mindspore_parallel_impl(self):
        if len(self.inputa.shape) > 2:
            matmul = BatchMatMul(self.transpose_a, self.transpose_b, strategy0=self.strategy, strategy1=self.strategy_)
        else:
            matmul = Matmul(self.transpose_a, self.transpose_b, strategy0=self.strategy, strategy1=self.strategy_)
        x = Tensor(self.inputa)
        y = Tensor(self.inputb)
        z = Tensor(self.inputz)
        out_grad_me = Tensor(self.output_grad_np)
        
        xs = self.get_parallel_blocks(self.inputa, self.strategy_[1])
        ys = self.get_parallel_blocks(self.inputb, self.strategy[2])
        zs = self.get_parallel_blocks(self.inputz, self.strategy_[1])
        out_grads = self.get_parallel_blocks(self.output_grad_np, self.out_strategy)
        
        x1 = Tensor(xs[self.x_id]) #需要从设备矩阵推导
        y1 = Tensor(ys[self.y_id]) #
        z1 = Tensor(zs[self.x_id])
        out_grad1 = Tensor(out_grads[self.out_id])
        net_me = Grad(matmul)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        net_me.set_auto_parallel()
        net_me.set_train()
        
        out_grad = net_me(x, y, z, out_grad_me, parallel_inputs_compile = [x, y, z, out_grad1], parallel_inputs_run = [x1, y1, z1, out_grad1])
        return out_grad

    def forward_cmp(self):
        out_mindspore = self.forward_mindspore_impl()
        out_mindspores = self.get_parallel_blocks(out_mindspore, self.out_strategy)
        out_mindspore_parallel = self.forward_mindspore_parallel_impl()
        assert allclose(out_mindspores[self.out_id], out_mindspore_parallel, 0.0001, 0.0001)

    def grad_cmp(self):
        input_grad_mindspore = self.grad_mindspore_impl()
        input_grad_mindspore_parallel = self.grad_mindspore_parallel_impl()
        input_grad_mindspores0 = self.get_parallel_blocks(input_grad_mindspore[0].asnumpy(), self.strategy_[1])
        input_grad_mindspores1 = self.get_parallel_blocks(input_grad_mindspore[1].asnumpy(), self.strategy[2])
        input_grad_mindspores2 = self.get_parallel_blocks(input_grad_mindspore[2].asnumpy(), self.strategy_[1])
        assert allclose(input_grad_mindspores0[self.x_id], input_grad_mindspore_parallel[0].asnumpy(), 0.0001, 0.0001)
        assert allclose(input_grad_mindspores1[self.y_id], input_grad_mindspore_parallel[1].asnumpy(), 0.0001, 0.0001)
        assert allclose(input_grad_mindspores2[self.x_id], input_grad_mindspore_parallel[2].asnumpy(), 0.0001, 0.0001)
        
def test_reid_batchmatmul_inputa_128_512_inputb_2000_512():
    inputa = [128, 512]
    inputb = [2000, 512]
    fact = BatchmatmulFactory(inputa, inputb, False, True, (0,(2,2),(1,2)), (0,(2,2),(2,2)))
    fact.forward_cmp()

def test_reid_batchmatmul_grad_inputa_128_512_inputb_2000_512():
    inputa = [128, 512]
    inputb = [2000, 512]
    fact = BatchmatmulFactory(inputa, inputb, False, True, (0, (2,2),(1,2)), (0,(2,2),(2,2)))
    fact.grad_cmp()

def test_reid_batchmatmul_inputa_128_512_inputb_2000_512_redistribution():
    inputa = [128, 512]
    inputb = [2000, 512]
    fact = BatchmatmulFactory(inputa, inputb, False, True, (0,(1,2),(1,2)), (0,(2,2),(2,2)))
    fact.forward_cmp()

def test_reid_batchmatmul_grad_inputa_128_512_inputb_2000_512_redistribution():
    inputa = [128, 512]
    inputb = [2000, 512]
    fact = BatchmatmulFactory(inputa, inputb, False, True, (0, (1,2),(1,2)), (0,(2,2),(2,2)))
    fact.grad_cmp()
