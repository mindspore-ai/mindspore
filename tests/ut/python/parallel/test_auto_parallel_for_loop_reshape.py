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
from mindspore.ops import operations as P, functional as F
from mindspore.common.initializer import initializer
import mindspore.common.dtype as mstype
from mindspore.common.api import _executor
from tests.dataset_mock import MindData


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

        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, x):
        x_origin_shape = self.shape(x)
        x_target_shape = x_origin_shape[:-1]
        x_shape = x_origin_shape + (1,)
        x = self.reshape(x, x_shape)
        x = self.reshape(x, x_target_shape)
        mean = self.mean(x, -1)
        variance = self.mean(F.square(self.sub(x, mean)))
        output = self.div(self.sub(x, mean), F.sqrt(self.add(variance, self.eps)))
        rescaled_output = self.add(self.mul(output, self.gamma), self.beta)
        output_shape = self.shape(rescaled_output) + (1,)
        rescaled_output = self.reshape(rescaled_output, output_shape)
        return rescaled_output


class SubNet(Cell):
    def __init__(self, index):
        super().__init__()
        self.relu = P.ReLU()
        self.layernorm1 = LayerNorm((128,)).to_float(mstype.float32)

    def construct(self, x):
        x = self.layernorm1(x)
        out = self.relu(x)
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


_x = Tensor(np.ones([512, 128, 1]), dtype=ms.float32)
_b = Tensor(np.ones([32]), dtype=ms.int32)
_w1 = Tensor(np.ones([512, 128, 1]), dtype=ms.float32)


def test_auto_parallel():
    context.set_context(save_graphs=False)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=16, global_rank=0)
    net = Full(_w1, 3)
    net.set_auto_parallel()
    net.set_train()
    _executor.compile(net, _x, phase='train')
    num_ops = _executor._get_num_parallel_ops(net)
    expected_num = 16
    assert num_ops == expected_num
