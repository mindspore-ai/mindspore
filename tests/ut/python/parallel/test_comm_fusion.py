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
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.train import Model
from mindspore.nn.wrap.cell_wrapper import PipelineCell
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from tests.ut.python.parallel.test_adafactor import compile_net
from tests.ut.python.parallel.test_adafactor import Net as Net2


class DatasetLenet():
    def __init__(self, data, label, length=3):
        self.data = data
        self.label = label
        self.index = 1
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.data, self.label

    def reset(self):
        self.index = 0

    @staticmethod
    def get_dataset_size():
        return 32

    @staticmethod
    def get_repeat_count():
        return 1

    @staticmethod
    def get_batch_size():
        return 32

    def create_tuple_iterator(self, num_epochs=1, do_copy=True):
        return self


class MatMulCell(nn.Cell):
    def __init__(self, strategy1, strategy2, param=None):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        if param is not None:
            self.param = param
        self.param1 = Parameter(initializer("zeros", [64, 64]), name="param1")
        self.matmul = P.MatMul().shard(strategy1)
        self.matmul1 = P.MatMul().shard(strategy2)

    def construct(self, x):
        out = self.matmul(x, self.param)
        out = self.matmul1(out, self.param1)
        return out


class Net(nn.Cell):
    def __init__(self, strategy1, strategy2, param=None):
        super().__init__()
        self.block = nn.CellList()
        for i in range(2):
            cell = MatMulCell(strategy1, strategy2, param)
            cell.pipeline_stage = i
            self.block.append(cell)

    def construct(self, x):
        for i in range(2):
            x = self.block[i](x)
        return x


class PipelineSplit(nn.Cell):
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.cell = Net(strategy1, strategy2)
        self.cell.block[0].matmul.add_prim_attr("parameter_start", 0)

    def construct(self, x, label):
        x = self.cell(x)
        return x

def test_fusion_size():
    """
    Feature: test_fusion_auto in size mode
    Description: allgather and reduce scatter fusion in size mode
    Expectation: success
    """
    allgather_threshold = 8
    reducescatter_threshold = 16
    comm_fusion_dict = {"allgather": {"mode": "size", "config": allgather_threshold},
                        "reducescatter": {"mode": "size", "config": reducescatter_threshold}}
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", comm_fusion=comm_fusion_dict,
                                      dataset_strategy="data_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0, pipeline_stages=2, enable_parallel_optimizer=True)
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
    assert auto_parallel_context().allgather_fusion_threshold_mb() == allgather_threshold
    assert auto_parallel_context().reducescatter_fusion_threshold_mb() == reducescatter_threshold

def test_fusion_auto():
    """
    Feature: test_fusion_auto in auto mode
    Description: allgather and reduce scatter fusion in auto mode
    Expectation: success
    """
    comm_fusion_dict = {"allgather": {"mode": "auto", "config": None},
                        "reducescatter": {"mode": "auto", "config": None}}
    context.set_auto_parallel_context(device_num=8, global_rank=0, pipeline_stages=2, enable_parallel_optimizer=True,
                                      parallel_mode="semi_auto_parallel", comm_fusion=comm_fusion_dict,
                                      dataset_strategy="data_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
    assert auto_parallel_context().allgather_fusion_threshold_mb() == 64
    assert auto_parallel_context().reducescatter_fusion_threshold_mb() == 64

def test_fusion_optimizer_parallel():
    """
    Feature: test_fusion_optimizer_parallel in size mode
    Description: allgather and reduce scatter size fusion in optimizer parallel
    Expectation: compile success
    """
    allgather_threshold = 16
    reducescatter_threshold = 8
    comm_fusion_dict = {"allgather": {"mode": "size", "config": allgather_threshold},
                        "reducescatter": {"mode": "size", "config": reducescatter_threshold}}
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0,
                                      enable_parallel_optimizer=True, comm_fusion=comm_fusion_dict,
                                      dataset_strategy="full_batch")
    _w0 = Tensor(np.ones([64, 16, 2]), dtype=ms.float32)
    _w1 = Tensor(np.ones([32, 32]), dtype=ms.float32)
    _w2 = Tensor(np.ones([32]), dtype=ms.float32)
    strategy1 = ((4, 2), (2, 2))
    strategy2 = ((4, 2), (2,))
    net = Net2(_w0, _w1, _w2, strategy1, strategy2)
    compile_net(net)

    comm_fusion_dict = {"allgather": {"mode": "auto", "config": None},
                        "reducescatter": {"mode": "auto", "config": None}}
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0,
                                      enable_parallel_optimizer=True, comm_fusion=comm_fusion_dict,
                                      dataset_strategy="full_batch")
    net1 = Net2(_w0, _w1, _w2, strategy1, strategy2)
    compile_net(net1)

def test_allgather_fusion_invalid_value_failed():
    """
    Feature: test_allgather_fusion with invalid value
    Description: test_allgather_fusion with invalid value
    Expectation: throw TypeError
    """
    with pytest.raises(TypeError):
        comm_fusion_dict = [1, 2]
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", comm_fusion=comm_fusion_dict)

    with pytest.raises(TypeError):
        comm_fusion_dict = {"allgather": [1, 2]}
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", comm_fusion=comm_fusion_dict)

    with pytest.raises(TypeError):
        comm_fusion_dict = {"allgather": {"mode": "size", "config": "30.12"}}
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", comm_fusion=comm_fusion_dict)

    with pytest.raises(KeyError):
        comm_fusion_dict = {"all": {"mode": "size", "config": 30}}
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", comm_fusion=comm_fusion_dict)

    with pytest.raises(KeyError):
        comm_fusion_dict = {"allgather": {"modes": "size", "config": 30}}
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", comm_fusion=comm_fusion_dict)

    with pytest.raises(KeyError):
        comm_fusion_dict = {"allgather": {"mode": "sizes", "config": 30}}
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", comm_fusion=comm_fusion_dict)

    with pytest.raises(KeyError):
        comm_fusion_dict = {"allgather": {"mode": "size"}}
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", comm_fusion=comm_fusion_dict)

def test_reducescatter_fusion_invalid_value_failed():
    """
    Feature: test_reducescatter_fusion with invalid value
    Description: test_reducescatter_fusion with invalid value
    Expectation: throw TypeError
    """

    with pytest.raises(TypeError):
        comm_fusion_dict = {"reducescatter": [1, 2]}
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", comm_fusion=comm_fusion_dict)

    with pytest.raises(TypeError):
        comm_fusion_dict = {"reducescatter": {"mode": "size", "config": "30.12"}}
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", comm_fusion=comm_fusion_dict)

    with pytest.raises(KeyError):
        comm_fusion_dict = {"reducescatter": {"modes": "size", "config": 30}}
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", comm_fusion=comm_fusion_dict)

    with pytest.raises(KeyError):
        comm_fusion_dict = {"reducescatter": {"mode": "sizes", "config": 30}}
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", comm_fusion=comm_fusion_dict)

    with pytest.raises(KeyError):
        comm_fusion_dict = {"reducescatter": {"mode": "size"}}
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", comm_fusion=comm_fusion_dict)


def test_openstate_comm_fusion():
    """
    Feature: test_openstate_comm_fusion
    Description: test openstate in comm_fusion
    Expectation: success
    """
    comm_fusion_dict = {"openstate": False}
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", comm_fusion=comm_fusion_dict)
    assert auto_parallel_context().get_enable_all_reduce_fusion() is False
    assert auto_parallel_context().get_enable_all_gather_fusion() is False
    assert auto_parallel_context().get_enable_reduce_scatter_fusion() is False
