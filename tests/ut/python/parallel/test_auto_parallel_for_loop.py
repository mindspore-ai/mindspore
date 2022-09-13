# Copyright 2020 Huawei Technologies Co., Ltd
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
import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
import mindspore.common.dtype as mstype
from mindspore.common.api import _cell_graph_executor
from mindspore.parallel._cost_model_context import _set_algo_single_loop
from tests.dataset_mock import MindData


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Dataset(MindData):
    def __init__(self, predict, label, length=3):
        super(Dataset, self).__init__(size=length)
        self.predict = predict
        self.label = label
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.predict, self.label

    def reset(self):
        self.index = 0


class LayerNorm(nn.Cell):
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = Parameter(initializer('ones', normalized_shape), name="gamma")
        self.beta = Parameter(initializer('zeros', normalized_shape), name="beta")
        self.mean = P.ReduceMean(keep_dims=True)
        self.eps = eps
        self.sub = P.Sub()
        self.add = P.Add()
        self.mul = P.Mul()
        self.div = P.RealDiv()
        self.square = P.Square()
        self.sqrt = P.Sqrt()

    def construct(self, x):
        mean = self.mean(x, -1)
        variance = self.mean(self.square(self.sub(x, mean)))
        add_variance = self.add(variance, self.eps)
        sqrt_variance = self.sqrt(add_variance)
        output = self.div(self.sub(x, mean), sqrt_variance)
        rescaled_output = self.add(self.mul(output, self.gamma), self.beta)
        return rescaled_output


class SubNet(Cell):
    def __init__(self, index):
        super().__init__()
        self.matmul = P.MatMul()
        self.relu = P.ReLU()
        self.weight = Parameter(Tensor(np.ones([128, 128]), dtype=ms.float32), "matmul_w"+str(index))
        self.layernorm1 = LayerNorm((128,)).to_float(mstype.float32)

    def construct(self, x):
        x = self.layernorm1(x)
        out = self.matmul(x, self.weight)
        out = self.relu(out)
        return out


class Net(Cell):
    def __init__(self, mul_weight, num_layers, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.neg = P.Neg().shard(strategy2)
        self.mul_weight = Parameter(mul_weight, "w1")
        self.num_layers = num_layers
        self.layers = nn.CellList()
        for i in range(num_layers):
            self.layers.append(SubNet(i))

    def construct(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        out = self.mul(x, self.mul_weight)
        out = self.neg(out)
        return out


class Full(Cell):
    def __init__(self, mul_weight, num_layers, strategy1=None, strategy2=None):
        super().__init__()
        self.network = Net(mul_weight, num_layers, strategy1, strategy2)
        self.relu = P.ReLU()

    def construct(self, x):
        out = self.network(x)
        out = self.relu(out)
        return out


_x = Tensor(np.ones([512, 128]), dtype=ms.float32)
_b = Tensor(np.ones([32]), dtype=ms.int32)
_w1 = Tensor(np.ones([512, 128]), dtype=ms.float32)


def test_auto_parallel():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=16, global_rank=0)
    _set_algo_single_loop(True)
    net = Full(_w1, 3)
    net.set_auto_parallel()
    net.set_train()
    _cell_graph_executor.compile(net, _x, phase='train')
    num_ops = _cell_graph_executor._get_num_parallel_ops(net)
    expected_num = 16
    assert num_ops == expected_num
