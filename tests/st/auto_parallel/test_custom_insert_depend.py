# Copyright 2024 Huawei Technologies Co., Ltd
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
import shutil
import json
import subprocess

import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.train import Model
from mindspore.nn.wrap.cell_wrapper import PipelineCell
from mindspore.communication.management import init

context.set_context(mode=context.GRAPH_MODE)


class MatMulCell(nn.Cell):
    def __init__(self, strategy1, strategy2, param=None, dtype=ms.float32):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        if param is not None:
            self.param = param
        self.param1 = Parameter(initializer("zeros", [64, 64]), name="param1")
        self.matmul = P.MatMul().shard(strategy1)
        self.matmul1 = P.MatMul().shard(strategy2)
        self.cast = P.Cast()
        self.dtype = dtype

    def construct(self, x):
        out = self.matmul(self.cast(x, self.dtype),
                          self.cast(self.param, self.dtype))
        out = self.matmul1(out, self.cast(self.param1, self.dtype))
        return out


class Net(nn.Cell):
    def __init__(self, strategy1, strategy2, param=None, dtype=ms.float32):
        super().__init__()
        self.block = nn.CellList()
        for i in range(2):
            cell = MatMulCell(strategy1, strategy2, param, dtype)
            cell.pipeline_stage = i
            self.block.append(cell)

    def construct(self, x):
        for i in range(2):
            x = self.block[i](x)
        return x


class PipelineSplit(nn.Cell):
    def __init__(self, strategy1, strategy2, dtype=ms.float32):
        super().__init__()
        self.cell = Net(strategy1, strategy2, dtype=dtype)
        self.cell.block[0].matmul.add_prim_attr("parameter_start", 0)

    def construct(self, x, label):
        x = self.cell(x)
        return x


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

    def get_dataset_size(self):
        return 32

    def get_repeat_count(self):
        return 1

    def get_batch_size(self):
        return 32

    def create_tuple_iterator(self, num_epochs=1, do_copy=True):
        return self


def test_pipeline_split_stage0_custom_insert_depend_kbk():
    """
    Feature: test custom insert depend in pipeline stage0 with kbk
    Description: parallel subgraph inline in grad parallel
    Expectation: success
    """
    os.environ["MS_SIMULATION_LEVEL"] = "1"
    os.environ["RANK_SIZE"] = "16"
    os.environ["RANK_ID"] = "1"
    os.environ["GRAPH_OP_RUN"] = "1"
    context.set_context(save_graphs=True, save_graphs_path="./")
    if os.path.exists("./depend.json"):
        os.remove("./depend.json")
    a = {
        "get_full_op_name_list": True,
        "stage0": [
            {
                "graph_id": 0,
                "depend_src_list": ["Default/network-PipelineCell/micro_inputs-CellList/7-_MicroBatch/AllGather-op1"],
                "depend_dest_list": ["Default/network-PipelineCell/network-PipelineSplit/cell-Net/"
                                     "block-CellList/0-MatMulCell/MatMul-op32"],
            }
        ],
        "stage1": [
            {
                "graph_id": 0,
                "depend_src_list": ["Default/network-PipelineCell/network-PipelineSplit/cell-Net/block-CellList/"
                                    "1-MatMulCell/AllGather-op24"],
                "depend_dest_list": ["Default/network-PipelineCell/network-PipelineSplit/cell-Net/block-CellList/"
                                     "1-MatMulCell/Concat-op48"],
            }
        ]
    }
    f = open("depend.json", "w")
    f.write(json.dumps(a))
    f.close()
    os.environ["MS_CUSTOM_DEPEND_CONFIG_PATH"] = './depend.json'
    init()
    context.set_auto_parallel_context(
        device_num=16, global_rank=1, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([128, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((1, 1), (1, 8))
    strategy2 = ((1, 1), (1, 4))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 8)
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
    file = "rank_1/hwopt_d_after_stream_assign_0_*.ir"
    para = "Depend(%"
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "47"
    os.environ["MS_CUSTOM_DEPEND_CONFIG_PATH"] = ""
    os.environ["MS_SIMULATION_LEVEL"] = ""
    if os.path.exists("./rank_1"):
        shutil.rmtree("./rank_1")
    if os.path.exists("./depend.json"):
        os.remove("./depend.json")
    context.set_context(save_graphs=False)


def test_pipeline_split_stage1_custom_insert_depend_kbk():
    """
    Feature: test custom insert depend in pipeline stage1 with kbk
    Description: parallel subgraph inline in grad parallel
    Expectation: success
    """
    os.environ["MS_SIMULATION_LEVEL"] = "1"
    os.environ["RANK_SIZE"] = "16"
    os.environ["RANK_ID"] = "8"
    os.environ["GRAPH_OP_RUN"] = "1"
    context.set_context(save_graphs=True, save_graphs_path="./")
    if os.path.exists("./depend.json"):
        os.remove("./depend.json")
    a = {
        "get_full_op_name_list": True,
        "stage0": [
            {
                "graph_id": 0,
                "depend_src_list": ["Default/network-PipelineCell/micro_inputs-CellList/7-_MicroBatch/AllGather-op1"],
                "depend_dest_list": ["Default/network-PipelineCell/network-PipelineSplit/cell-Net/"
                                     "block-CellList/0-MatMulCell/MatMul-op32"],
            }
        ],
        "stage1": [
            {
                "graph_id": 0,
                "depend_src_list": ["Default/network-PipelineCell/network-PipelineSplit/cell-Net/block-CellList/"
                                    "1-MatMulCell/AllGather-op24"],
                "depend_dest_list": ["Default/network-PipelineCell/network-PipelineSplit/cell-Net/block-CellList/"
                                     "1-MatMulCell/Concat-op48"],
            }
        ]
    }
    f = open("depend.json", "w")
    f.write(json.dumps(a))
    f.close()
    os.environ["MS_CUSTOM_DEPEND_CONFIG_PATH"] = './depend.json'
    init()
    context.set_auto_parallel_context(
        device_num=16, global_rank=8, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([128, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((1, 1), (1, 8))
    strategy2 = ((1, 1), (1, 4))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 8)
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
    file = "rank_8/hwopt_d_after_stream_assign_0_*.ir"
    para = "Depend(%"
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "39"
    os.environ["MS_CUSTOM_DEPEND_CONFIG_PATH"] = ""
    os.environ["MS_SIMULATION_LEVEL"] = ""
    if os.path.exists("./rank_8"):
        shutil.rmtree("./rank_8")
    if os.path.exists("./depend.json"):
        os.remove("./depend.json")
    context.set_context(save_graphs=False)


def test_pipeline_split_stage1_custom_insert_depend():
    """
    Feature: test custom insert depend in pipeline stage1 with ge
    Description: parallel subgraph inline in grad parallel
    Expectation: success
    """
    os.environ["MS_SIMULATION_LEVEL"] = "1"
    os.environ["RANK_SIZE"] = "16"
    os.environ["RANK_ID"] = "8"
    context.set_context(save_graphs=True, save_graphs_path="./")
    if os.path.exists("./depend.json"):
        os.remove("./depend.json")
    a = {
        "get_full_op_name_list": True,
        "stage1": [
            {
                "graph_id": 0,
                "depend_src_list": ["Default/network-PipelineCell/network-PipelineSplit/"
                                    "cell-Net/block-CellList/1-MatMulCell/AllGather-op0"],
                "depend_dest_list": ["Default/network-PipelineCell/network-PipelineSplit/"
                                     "cell-Net/block-CellList/1-MatMulCell/Concat-op24"],
            }
        ]

    }
    f = open("depend.json", "w")
    f.write(json.dumps(a))
    f.close()
    os.environ["MS_CUSTOM_DEPEND_CONFIG_PATH"] = './depend.json'
    init()
    context.set_auto_parallel_context(
        device_num=16, global_rank=8, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([128, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((1, 1), (1, 8))
    strategy2 = ((1, 1), (1, 4))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 8)
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
    file = "rank_8/hwopt_d_end_opt_ge_graph_0*.ir"
    para = "Depend(%"
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "39"
    os.environ["MS_CUSTOM_DEPEND_CONFIG_PATH"] = ""
    os.environ["MS_SIMULATION_LEVEL"] = ""
    if os.path.exists("./rank_8"):
        shutil.rmtree("./rank_8")
    if os.path.exists("./depend.json"):
        os.remove("./depend.json")
    context.set_context(save_graphs=False)
