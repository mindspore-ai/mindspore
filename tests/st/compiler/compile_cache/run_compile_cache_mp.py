# Copyright 2023 Huawei Technologies Co., Ltd
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
# ============================================================================
import sys
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.train import Model
from mindspore.nn.wrap.cell_wrapper import PipelineCell, Cell
from mindspore import lazy_inline
from mindspore.communication import init
from mindspore.nn.optim import Momentum
import mindspore.dataset as ds

ms.set_seed(1)

class MyIter:
    def __init__(self, data_input, length=1):
        self.data = data_input
        self.index = 1
        self.length = length

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.data

    def __iter__(self):
        self.index = 0
        return self

    def __len__(self):
        return self.length

    def reset(self):
        self.index = 0


class MatMulCell(Cell):
    def __init__(self):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        self.param1 = Parameter(initializer("zeros", [64, 64]), name="param1")
        self.matmul = P.MatMul()
        self.matmul1 = P.MatMul()

    def construct(self, x):
        out = self.matmul(x, self.param)
        out = self.matmul1(out, self.param1)
        return out


class Net(nn.Cell):
    @lazy_inline
    def __init__(self):
        super().__init__()
        self.block = nn.CellList()
        for i in range(8):
            cell = MatMulCell()
            cell.pipeline_stage = i
            self.block.append(cell)
        self.block[3].recompute()

    def construct(self, x):
        for i in range(8):
            x = self.block[i](x)
        return x


context.set_context(mode=context.GRAPH_MODE, enable_compile_cache=True, compile_cache_path=sys.argv[1])
context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", pipeline_stages=8)
init()
data1 = Tensor(np.ones([32, 64]), dtype=ms.float32)
net = PipelineCell(Net(), 8)
learning_rate = 0.01
momentum = 0.9
optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
model = Model(net, optimizer=optimizer)
dataset = ds.GeneratorDataset(source=MyIter(data1, 1), column_names=["data"])
model.build(dataset)
context.set_context(enable_compile_cache=False)
