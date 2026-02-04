# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
import json
import os
import subprocess
import shutil
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.train import Model, FlopsUtilizationCollector
from mindspore.nn import PipelineCell, GradAccumulationCell
from mindspore.graph.lazy_inline import lazy_inline


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


class MatMulCell(nn.Cell):
    def __init__(self, strategy1, strategy2, dtype=None):
        super().__init__()
        if not dtype:
            self.param = Parameter(initializer("zeros", [64, 64]), name="param")
            self.param1 = Parameter(initializer("zeros", [64, 64]), name="param1")
        else:
            self.param = Parameter(initializer("zeros", [64, 64], dtype=dtype), name="param")
            self.param1 = Parameter(initializer("zeros", [64, 64], dtype=dtype), name="param1")
        self.matmul = P.MatMul().shard(strategy1)
        self.matmul1 = P.MatMul().shard(strategy2)

    def construct(self, x):
        out = self.matmul(x, P.Cast()(self.param, x.dtype))
        out = self.matmul1(out, P.Cast()(self.param1, x.dtype))
        return out, self.param


class MatMulCell2(nn.Cell):
    def __init__(self, strategy1, strategy2, dtype=None):
        super().__init__()
        if not dtype:
            self.param1 = Parameter(initializer("zeros", [64, 64]), name="param1")
        else:
            self.param1 = Parameter(initializer("zeros", [64, 64], dtype=dtype), name="param1")
        self.matmul = P.MatMul().shard(strategy1)
        self.matmul1 = P.MatMul().shard(strategy2)

    def construct(self, x, param):
        out = self.matmul(x, P.Cast()(param, x.dtype))
        out = self.matmul1(out, P.Cast()(self.param1, x.dtype))
        return out


class Net(nn.Cell):
    def __init__(self, strategy1, strategy2, param=None):
        super().__init__()
        self.cell1 = MatMulCell(strategy1, strategy2)
        self.cell1.pipeline_stage = 0
        self.cell2 = MatMulCell2(strategy1, strategy2)
        self.cell2.pipeline_stage = 1

    def construct(self, x, label):
        out, param = self.cell1(x)
        out = self.cell2(out, param)
        return out


class LargeMatMulCell(nn.Cell):
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 10240000]), name="param")
        self.param1 = Parameter(initializer("zeros", [10240000, 64]), name="param1")
        self.tile = P.Tile().shard(((strategy1[0][0], 1),))
        self.matmul = P.MatMul().shard(strategy1)
        self.matmul1 = P.MatMul().shard(strategy2)

    def construct(self, x):
        x = self.tile(x, (64, 1))
        for _ in range(32):
            x = self.matmul(x, self.param)
            x = self.matmul1(x, self.param1)
        return x, self.param


class LargeMatMulCell2(nn.Cell):
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.param1 = Parameter(initializer("zeros", [10240000, 64]), name="param1")
        self.matmul = P.MatMul().shard(strategy1)
        self.matmul1 = P.MatMul().shard(strategy2)

    def construct(self, x, param):
        out = self.matmul(x, param)
        out = self.matmul1(out, self.param1)
        return out


class LargeLazyInlineNet(nn.Cell):
    @lazy_inline
    def __init__(self, stra1, stra2, param=None):
        super().__init__()
        self.cell1 = LargeMatMulCell(stra1, stra2)
        self.cell1.pipeline_stage = 0
        self.cell2 = LargeMatMulCell2(stra1, stra2)
        self.cell2.pipeline_stage = 1

    def construct(self, x, label):
        out, param = self.cell1(x)
        out = self.cell2(out, param)
        return out


class LazyInlineNet(nn.Cell):
    @lazy_inline
    def __init__(self, stra1, stra2, dtype=None):
        super().__init__()
        self.cell1 = MatMulCell(stra1, stra2, dtype)
        self.cell1.pipeline_stage = 0
        self.cell2 = MatMulCell2(stra1, stra2, dtype)
        self.cell2.pipeline_stage = 1

    def construct(self, x, label):
        out, param = self.cell1(x)
        out = self.cell2(out, param)
        return out


class LazyInlineRecomputeNet(nn.Cell):
    @lazy_inline
    def __init__(self, stra1, stra2, param=None):
        super().__init__()
        self.cell1 = MatMulCell(stra1, stra2)
        self.softmax1 = P.Softmax(-1)
        self.abs1 = P.Abs()
        self.cell1.pipeline_stage = 0
        self.softmax1.pipeline_stage = 0
        self.abs1.pipeline_stage = 0
        self.cell1.recompute()
        self.softmax1.recompute()
        self.abs1.recompute()
        self.cell2 = MatMulCell2(stra1, stra2)
        self.softmax2 = P.Softmax(-1)
        self.abs2 = P.Abs()
        self.cell2.pipeline_stage = 1
        self.softmax2.pipeline_stage = 1
        self.abs2.pipeline_stage = 1
        self.cell2.recompute()
        self.softmax2.recompute()
        self.abs2.recompute()

    def construct(self, x, label):
        out, param = self.cell1(x)
        out = self.abs1(out)
        out = self.softmax1(out)
        out = self.cell2(out, param)
        out = self.softmax2(out)
        out = self.abs2(out)
        return out


class MatMulCell3(nn.Cell):
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        self.param1 = Parameter(initializer("zeros", [64, 64]), name="param1")
        self.matmul = P.MatMul().shard(strategy1)
        self.matmul1 = P.MatMul().shard(strategy2)

    def construct(self, x):
        out = self.matmul(x, self.param)
        out = self.matmul1(out, self.param1)
        return out


class LazyInlineVppNet(nn.Cell):
    @lazy_inline
    def __init__(self, stra1, stra2):
        super().__init__()
        self.cell1 = MatMulCell3(stra1, stra2)
        self.cell1.pipeline_stage = 0
        self.cell2 = MatMulCell3(stra1, stra2)
        self.cell2.pipeline_stage = 1
        self.cell3 = MatMulCell3(stra1, stra2)
        self.cell3.pipeline_stage = 0
        self.cell4 = MatMulCell2(stra1, stra2)
        self.cell4.pipeline_stage = 1

    def construct(self, x, label):
        out = self.cell1(x)
        out = self.cell2(out)
        out = self.cell3(out)
        out = self.cell4(out, self.cell1.param)
        return out


def test_vpp_with_shared_parameter_stage0():
    """
    Feature: parallel subgraph inline
    Description: parallel subgraph inline in pipeline parallel mode
    Expectation: success
    """
    context.set_auto_parallel_context(
        device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(pipeline_config={'pipeline_scheduler': '1f1b', 'pipeline_interleave': True})
    files_directory = f"./vpp_with_shared_parameter/{test_vpp_with_shared_parameter_stage0.__name__}"
    context.set_context(save_graphs=True, save_graphs_path=files_directory)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    stra1 = ((16, 1), (1, 1))
    stra2 = ((8, 1), (1, 1))
    net = PipelineCell(LazyInlineVppNet(stra1, stra2), 4)
    params = net.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    if os.path.exists(files_directory):
        shutil.rmtree(files_directory)
    model.train(2, dataset, dataset_sink_mode=False)
    file = f"{files_directory}/*validate*.ir"
    para = "pipeline_param"
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "2"
    if os.path.exists(files_directory):
        shutil.rmtree(files_directory)
    context.set_context(save_graphs=False)


def test_vpp_with_shared_parameter_opt_shard_stage0():
    """
    Feature: parallel subgraph inline
    Description: parallel subgraph inline in pipeline parallel mode
    Expectation: success
    """
    context.set_auto_parallel_context(
        device_num=32, global_rank=0, pipeline_stages=2)
    files_directory = f"./vpp_with_shared_parameter/{test_vpp_with_shared_parameter_opt_shard_stage0.__name__}"
    context.set_context(save_graphs=True, save_graphs_path=files_directory)
    context.set_auto_parallel_context(pipeline_config={'pipeline_scheduler': '1f1b', 'pipeline_interleave': True},
                                      enable_parallel_optimizer=True,
                                      parallel_optimizer_config={'optimizer_weight_shard_size': 2,
                                                                 'parallel_optimizer_threshold': 0},
                                      strategy_ckpt_save_file=f"{files_directory}/ckpt_strategy.ckpt")

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    stra1 = ((16, 1), (1, 1))
    stra2 = ((8, 1), (1, 1))
    net = PipelineCell(LazyInlineVppNet(stra1, stra2), 4)
    params = net.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    if os.path.exists(files_directory):
        shutil.rmtree(files_directory)
    model.train(2, dataset, dataset_sink_mode=False)
    file = f"{files_directory}/*validate*.ir"
    para = "pipeline_param"
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "2"
    stra = ms.build_searched_strategy(f"{files_directory}/ckpt_strategy.ckpt")
    stra_keys = list(stra.keys())
    assert len(stra_keys) == 16
    if os.path.exists(files_directory):
        shutil.rmtree(files_directory)
    context.set_context(save_graphs=False)

def test_vpp_with_shared_parameter_stage1():
    """
    Feature: parallel subgraph inline
    Description: parallel subgraph inline in pipeline parallel mode
    Expectation: success
    """
    context.set_auto_parallel_context(
        device_num=32, global_rank=16, pipeline_stages=2)
    files_directory = f"./vpp_with_shared_parameter/{test_vpp_with_shared_parameter_stage1.__name__}"
    context.set_auto_parallel_context(pipeline_config={'pipeline_scheduler': '1f1b', 'pipeline_interleave': True})
    context.set_context(save_graphs=True, save_graphs_path=files_directory)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    stra1 = ((16, 1), (1, 1))
    stra2 = ((8, 1), (1, 1))
    net = PipelineCell(LazyInlineVppNet(stra1, stra2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    if os.path.exists(files_directory):
        shutil.rmtree(files_directory)
    model.train(2, dataset, dataset_sink_mode=False)
    file = f"{files_directory}/*validate*.ir"
    para = "pipeline_param"
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "3"
    if os.path.exists(files_directory):
        shutil.rmtree(files_directory)
    context.set_context(save_graphs=False)


def test_pipeline_split_stage0():
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(Net(strategy1, strategy2), 4)
    params = net.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_stage1():
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(Net(strategy1, strategy2), 4)
    params = net.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_lazy_inline_stage0():
    """
    Feature: share parameter in lazy inline
    Description: two cell share one parameter
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    stra1 = ((16, 1), (1, 1))
    stra2 = ((8, 1), (1, 1))
    net = PipelineCell(LazyInlineNet(stra1, stra2), 4)
    params = net.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_lazy_inline_overlap_grad_comm_nodes_stage0():
    """
    Feature: overlap recompute and grad comm nodes in lazy inline
    Description: test overlap recompute and grad comm nodes in lazy_inline
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(recompute_comm_overlap="AllGather,ReduceScatter")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    stra1 = ((4, 1), (1, 4))
    stra2 = ((4, 1), (1, 4))
    net = PipelineCell(LazyInlineRecomputeNet(stra1, stra2), 4)
    params = net.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_lazy_inline_stage1():
    """
    Feature: share parameter in lazy inline
    Description: two cell share one parameter
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    stra1 = ((16, 1), (1, 1))
    stra2 = ((8, 1), (1, 1))
    net = PipelineCell(LazyInlineNet(stra1, stra2), 4)
    params = net.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_auto_parallel_lazy_inline_stage0():
    """
    Feature: share parameter in lazy inline
    Description: two cell share one parameter
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    stra1 = ((16, 1), (1, 1))
    stra2 = ((8, 1), (1, 1))
    net = PipelineCell(LazyInlineNet(stra1, stra2), 4)
    params = net.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_auto_parallel_lazy_inline_stage1():
    """
    Feature: share parameter in lazy inline
    Description: two cell share one parameter
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    stra1 = ((16, 1), (1, 1))
    stra2 = ((8, 1), (1, 1))
    net = PipelineCell(LazyInlineNet(stra1, stra2), 4)
    params = net.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    model.train(2, dataset, dataset_sink_mode=False)


def test_dump_parallel_info():
    """
    Feature: dump parallel info to json
    Description: dump parallel info with pipeline lazy inline mode
    Expectation: success
    """
    context.set_auto_parallel_context(
        device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    os.environ["DUMP_PARALLEL_INFO"] = "1"
    os.environ["MA_LOG_DIR"] = os.getcwd()
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    stra1 = ((16, 1), (1, 1))
    stra2 = ((8, 1), (1, 1))
    net = PipelineCell(LazyInlineNet(stra1, stra2), 4)
    params = net.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    model.train(2, dataset, dataset_sink_mode=False)
    file = "./dump_parallel_info_0.json"
    para = "\"comm_group_rank_ids\": \"(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)\""
    output = subprocess.check_output(
        ["grep '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "3"
    os.environ["DUMP_PARALLEL_INFO"] = ""
    os.environ["MA_LOG_DIR"] = ""


def test_pipeline_with_begin_end_inline():
    """
    Feature: parallel subgraph inline
    Description: parallel subgraph inline in pipeline parallel mode
    Expectation: success
    """
    context.set_auto_parallel_context(
        device_num=32, global_rank=0, pipeline_stages=2)
    context.set_context(save_graphs=True, save_graphs_path="./pipeline_with_begin_end_inline")
    context.set_context(jit_config={"jit_level": "O2"})
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    if os.path.exists("./speed_up.json"):
        os.remove("./speed_up.json")
    a = {"enable_begin_end_inline_opt": True}
    f = open("speed_up.json", "w")
    f.write(json.dumps(a))
    f.close()
    context.set_context(ascend_config={"parallel_speed_up_json_path": "speed_up.json"})
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    stra1 = ((16, 1), (1, 1))
    stra2 = ((8, 1), (1, 1))
    net = PipelineCell(LazyInlineNet(stra1, stra2), 4)
    params = net.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    if os.path.exists("./pipeline_with_begin_end_inline"):
        shutil.rmtree("./pipeline_with_begin_end_inline")
    model.train(2, dataset, dataset_sink_mode=False)
    file = "./pipeline_with_begin_end_inline/*validate*.ir"
    para = " call @"
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "2"
    if os.path.exists("./pipeline_with_begin_end_inline"):
        shutil.rmtree("./pipeline_with_begin_end_inline")
    if os.path.exists("./speed_up.json"):
        os.remove("./speed_up.json")
    context.set_context(save_graphs=False)


def test_grad_accumulation_with_begin_end_inline():
    """
    Feature: parallel subgraph inline
    Description: parallel subgraph inline in grad parallel
    Expectation: success
    """
    context.set_auto_parallel_context(
        device_num=32, global_rank=0, pipeline_stages=2)
    context.set_context(save_graphs=True, save_graphs_path="./grad_accumulation_with_begin_end_inline")
    context.set_context(jit_config={"jit_level": "O2"})
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    if os.path.exists("./speed_up.json"):
        os.remove("./speed_up.json")
    a = {"enable_begin_end_inline_opt": True}
    f = open("speed_up.json", "w")
    f.write(json.dumps(a))
    f.close()
    context.set_context(ascend_config={"parallel_speed_up_json_path": "speed_up.json"})
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    stra1 = ((16, 1), (1, 1))
    stra2 = ((8, 1), (1, 1))
    net = GradAccumulationCell(LazyInlineNet(stra1, stra2), 4)
    params = net.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    if os.path.exists("./grad_accumulation_with_begin_end_inline"):
        shutil.rmtree("./grad_accumulation_with_begin_end_inline")
    model.train(2, dataset, dataset_sink_mode=False)
    file = "./grad_accumulation_with_begin_end_inline/*validate*.ir"
    para = " call @"
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "2"
    if os.path.exists("./grad_accumulation_with_begin_end_inline"):
        shutil.rmtree("./grad_accumulation_with_begin_end_inline")
    if os.path.exists("./speed_up.json"):
        os.remove("./speed_up.json")
    context.set_context(save_graphs=False)

def test_grad_accumulation_with_cast_head():
    """
    Feature: grad accu cast ahead
    Description: grad accu cast ahead
    Expectation: success
    """
    context.set_auto_parallel_context(
        device_num=32, global_rank=0, pipeline_stages=2)
    context.set_context(save_graphs=True, save_graphs_path="./grad_accumulation_with_cast_ahead")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", enable_parallel_optimizer=True,
                                      parallel_optimizer_config={'parallel_optimizer_threshold': 0,
                                                                 "optimizer_level": "level1"})
    data = Tensor(np.ones([32, 64]), dtype=ms.float16)
    label = Tensor(np.ones([64, 64]), dtype=ms.float16)
    stra1 = ((16, 1), (1, 1))
    stra2 = ((8, 1), (1, 1))
    net = GradAccumulationCell(LazyInlineNet(stra1, stra2, dtype=ms.float32), 4)
    params = net.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optim)
    if os.path.exists("./grad_accumulation_with_cast_ahead"):
        shutil.rmtree("./grad_accumulation_with_cast_ahead")
    model.train(2, dataset, dataset_sink_mode=False)
    file = "./grad_accumulation_with_cast_ahead/*validate*.ir"
    para1 = "instance name: parallel_opt"
    para2 = "Float16"
    output = subprocess.check_output(
        ["grep '%s' %s -A 1|grep '%s'|wc -l" % (para1, file, para2)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert int(out) > 1
    if os.path.exists("./grad_accumulation_with_cast_ahead"):
        shutil.rmtree("./grad_accumulation_with_cast_ahead")
    context.set_context(save_graphs=False)


def test_pipeline_split_stage0_flops():
    """
    Feature: parallel subgraph inline
    Description: parallel subgraph inline in grad parallel
    Expectation: success
    """
    context.set_auto_parallel_context(
        device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(Net(strategy1, strategy2), 4)
    params = net.network.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False, callbacks=[
                FlopsUtilizationCollector(dataset.get_dataset_size())])


def test_pipeline_split_stage0_flops_ma():
    """
    Feature: record flops info with env and callback
    Description: parallel subgraph inline in grad parallel
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=2**16, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    stra1 = ((1, 1), (1, 1))
    stra2 = ((1, 1), (1, 1))
    net = PipelineCell(LargeLazyInlineNet(stra1, stra2), 4)
    params = net.network.trainable_params()
    net.network.cell1.recompute()
    dataset = DatasetLenet(data, label, 3)
    optim = nn.Lamb(params, learning_rate=0.01)
    os.environ["MA_LOG_DIR"] = os.getcwd()
    model = Model(net, optimizer=optim)
    model.train(2, dataset, dataset_sink_mode=False, callbacks=[
                FlopsUtilizationCollector(dataset.get_dataset_size(), enable_ma_collector=True)])
    file = "flops_rank_0.txt"
    para = "flops{type=\"model_flops\", rank_id=\"0\"} 1.680053"
    output = subprocess.check_output(
        ["grep '%s' %s | wc -l" % (para, file)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "1"
    if os.path.exists("time_step_rank_0.txt"):
        os.remove("time_step_rank_0.txt")
    if os.path.exists("flops_rank_0.txt"):
        os.remove("flops_rank_0.txt")
    os.environ["MA_LOG_DIR"] = ""
