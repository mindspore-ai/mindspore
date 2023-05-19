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
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.train import Model
from tests.ut.python.parallel.test_pipeline_end_node import MatMulCell


class NetWithMultiOut(nn.Cell):
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.block = nn.CellList()
        for i in range(2):
            cell = MatMulCell(strategy1, strategy2)
            cell.pipeline_stage = i
            self.block.append(cell)
        self.max = P.ArgMaxWithValue()

    def construct(self, x):
        for i in range(2):
            x = self.block[i](x)
        x = self.max(x)
        return x


class NetWithTupleOut(nn.Cell):
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.block = nn.CellList()
        for i in range(2):
            cell = MatMulCell(strategy1, strategy2)
            cell.pipeline_stage = i
            self.block.append(cell)
        self.max = P.ArgMaxWithValue()

    def construct(self, x):
        for i in range(2):
            x = self.block[i](x)
        x = self.max(x)
        return (x, x[1])


class NetWithSingleOut(nn.Cell):
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.block = nn.CellList()
        for i in range(2):
            cell = MatMulCell(strategy1, strategy2)
            cell.pipeline_stage = i
            self.block.append(cell)
        self.max = P.ArgMaxWithValue()

    def construct(self, x):
        for i in range(2):
            x = self.block[i](x)
        x = self.max(x)
        return x[0]


def test_pipeline_with_multiout_stage0():
    """
    Feature:pipeline stage0
    Description:pipeline multiout end node
    Expectation:success
    """
    context.set_auto_parallel_context(device_num=16, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 2))
    strategy2 = ((2, 2), (2, 1))
    net = NetWithMultiOut(strategy1, strategy2)
    model = Model(net)
    model.predict(data)


def test_pipeline_with_tuple_out_stage0():
    """
    Feature:pipeline stage0
    Description:pipeline tuple end node
    Expectation:success
    """
    context.set_auto_parallel_context(device_num=16, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 2))
    strategy2 = ((2, 2), (2, 1))
    net = NetWithTupleOut(strategy1, strategy2)
    model = Model(net)
    model.predict(data)


def test_pipeline_with_single_out_stage0():
    """
    Feature:pipeline stage0
    Description:pipeline tuple end node
    Expectation:success
    """
    context.set_auto_parallel_context(device_num=16, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 2))
    strategy2 = ((2, 2), (2, 1))
    net = NetWithSingleOut(strategy1, strategy2)
    model = Model(net)
    model.predict(data)
