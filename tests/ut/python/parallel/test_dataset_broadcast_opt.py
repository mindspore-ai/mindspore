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
# ============================================================================
import os
import re
import subprocess
import mindspore.nn as nn
from mindspore import context
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.train import Model
from mindspore.parallel.shard import Layout
from mindspore.nn import PipelineCell
from mindspore.graph.lazy_inline import lazy_inline
from .test_dynamic_data_sink import GeneratorFakeData
import mindspore.dataset as ds
from mindspore._c_expression import MSContext, ms_ctx_param


class MatMulCell(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        self.param1 = Parameter(initializer("zeros", [64, 64]), name="param1")
        self.matmul = P.MatMul().shard(((2, 1), (1, 1)))
        self.matmul1 = P.MatMul().shard(((4, 1), (1, 1)))

    def construct(self, x):
        out = self.matmul(x, self.param)
        out = self.matmul1(out, self.param1)
        return out


class StageNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.cell1 = MatMulCell()
        self.cell1.pipeline_stage = 0
        self.cell2 = MatMulCell()
        self.cell2.pipeline_stage = 1
        self.cell3 = MatMulCell()
        self.cell3.pipeline_stage = 2
        self.cell4 = MatMulCell()
        self.cell4.pipeline_stage = 3

    def construct(self, x):
        out = self.cell1(x)
        out = self.cell2(out)
        out = self.cell3(out)
        out = self.cell4(out)
        return out


class WithLossCell(nn.Cell):
    @lazy_inline
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label):
        out = self._backbone(data)
        return self._loss_fn(out, label)


def test_dataset_broadcast_pipeline_stage0():
    """
    Feature: opt_level 1 + pipeline parallel
    Description: stage0 used data and last stage used label, test stage0 graph compile
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=4)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    MSContext.get_instance().set_param(ms_ctx_param.dataset_broadcast_opt_level, 1)
    net = StageNet()
    dataset = ds.GeneratorDataset(
        GeneratorFakeData(size=1024, batch_size=8, image_size=(64,),
                          use_parallel=True, num_classes=64), ["data", "label"])
    opt = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    loss = nn.L1Loss()
    loss.pipeline_stage = 3
    loss_cell = WithLossCell(net, loss)
    pp_cell = PipelineCell(loss_cell, 4)
    model = Model(pp_cell, optimizer=opt)
    model.train(2, dataset, dataset_sink_mode=True)


def test_dataset_broadcast_pipeline_stage3():
    """
    Feature: opt_level 1 + pipeline parallel
    Description: stage0 used data and last stage used label, test stage3 graph compile
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=24, pipeline_stages=4)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    MSContext.get_instance().set_param(ms_ctx_param.dataset_broadcast_opt_level, 1)
    net = StageNet()
    dataset = ds.GeneratorDataset(
        GeneratorFakeData(size=1024, batch_size=8, image_size=(64,),
                          use_parallel=True, num_classes=64), ["data", "label"])
    opt = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    loss = nn.L1Loss()
    loss.pipeline_stage = 3
    loss_cell = WithLossCell(net, loss)
    pp_cell = PipelineCell(loss_cell, 4)
    model = Model(pp_cell, optimizer=opt)
    model.train(2, dataset, dataset_sink_mode=True)


def test_dataset_broadcast_pipeline_full_batch_false_stage3():
    """
    Feature: opt_level 1 + pipeline parallel
    Description: stage0 used data and last stage used label, full_batch=False
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=24, pipeline_stages=4)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=False)
    MSContext.get_instance().set_param(ms_ctx_param.dataset_broadcast_opt_level, 1)
    net = StageNet()
    dataset = ds.GeneratorDataset(
        GeneratorFakeData(size=1024, batch_size=8, image_size=(64,),
                          use_parallel=True, num_classes=64), ["data", "label"])
    opt = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    loss = nn.L1Loss()
    loss.pipeline_stage = 3
    loss_cell = WithLossCell(net, loss)
    pp_cell = PipelineCell(loss_cell, 4)
    model = Model(pp_cell, optimizer=opt)
    model.train(2, dataset, dataset_sink_mode=True)


def test_dataset_broadcast_pipeline_set_dataset_strategy_stage0():
    """
    Feature: opt_level 1 + pipeline parallel
    Description: stage0 used data and last stage used label, dataset_strategy=((2, 1), (2, 1))
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=4)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=False)
    MSContext.get_instance().set_param(ms_ctx_param.dataset_broadcast_opt_level, 1)
    net = StageNet()
    dataset = ds.GeneratorDataset(
        GeneratorFakeData(size=1024, batch_size=8, image_size=(64,),
                          use_parallel=True, num_classes=64), ["data", "label"])
    opt = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    loss = nn.L1Loss()
    loss.pipeline_stage = 3
    loss_cell = WithLossCell(net, loss)
    pp_cell = PipelineCell(loss_cell, 4)
    model = Model(pp_cell, optimizer=opt)
    model.train(2, dataset, dataset_sink_mode=True)


class MatMulCell2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        self.matmul = P.MatMul().shard(((2, 1), (1, 1)))
        self.mul = P.Mul()

    def construct(self, x, y):
        out = self.matmul(x, self.param)
        out = self.mul(out, y)
        return out


class StageNet2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.cell1 = MatMulCell()
        self.cell1.pipeline_stage = 0
        self.cell2 = MatMulCell2()
        self.cell2.pipeline_stage = 1
        self.cell3 = MatMulCell2()
        self.cell3.pipeline_stage = 2
        self.cell4 = MatMulCell2()
        self.cell4.pipeline_stage = 3

    def construct(self, x):
        out = self.cell1(x)
        out = self.cell2(x, out)
        out = self.cell3(x, out)
        out = self.cell4(x, out)
        return out


def test_dataset_broadcast_pipeline_multi_stage_used_data_stage0():
    """
    Feature: opt_level 1 + pipeline parallel
    Description: all stage used data and last stage used label, test stage0 graph compile
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=4)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    MSContext.get_instance().set_param(ms_ctx_param.dataset_broadcast_opt_level, 1)
    net = StageNet2()
    dataset = ds.GeneratorDataset(
        GeneratorFakeData(size=1024, batch_size=8, image_size=(64,),
                          use_parallel=True, num_classes=64), ["data", "label"])
    opt = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    loss = nn.L1Loss()
    loss.pipeline_stage = 3
    loss_cell = WithLossCell(net, loss)
    pp_cell = PipelineCell(loss_cell, 4)
    model = Model(pp_cell, optimizer=opt)
    model.train(2, dataset, dataset_sink_mode=True)


def test_dataset_broadcast_pipeline_multi_stage_used_data_stage3():
    """
    Feature: opt_level 1 + pipeline parallel
    Description: all stage used data and last stage used label, test stage0 graph compile
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=24, pipeline_stages=4)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    MSContext.get_instance().set_param(ms_ctx_param.dataset_broadcast_opt_level, 1)
    net = StageNet2()
    dataset = ds.GeneratorDataset(
        GeneratorFakeData(size=1024, batch_size=8, image_size=(64,),
                          use_parallel=True, num_classes=64), ["data", "label"])
    opt = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    loss = nn.L1Loss()
    loss.pipeline_stage = 3
    loss_cell = WithLossCell(net, loss)
    pp_cell = PipelineCell(loss_cell, 4)
    model = Model(pp_cell, optimizer=opt)
    model.train(2, dataset, dataset_sink_mode=True)


def test_dataset_broadcast_full_batch_true():
    """
    Feature: opt_level 2 + full_batch True
    Description: no pipeline, full_batch=True test graph compile
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    MSContext.get_instance().set_param(ms_ctx_param.dataset_broadcast_opt_level, 2)
    net = StageNet2()
    dataset = ds.GeneratorDataset(
        GeneratorFakeData(size=1024, batch_size=8, image_size=(64,),
                          use_parallel=True, num_classes=64), ["data", "label"])
    opt = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    loss = nn.L1Loss()
    loss_cell = WithLossCell(net, loss)
    model = Model(loss_cell, optimizer=opt)
    model.train(2, dataset, dataset_sink_mode=True)


def test_dataset_broadcast_set_dataset_strategy():
    """
    Feature: opt_level 2 + set_dataset_strategy
    Description: no pipeline, dataset_strategy=((2, 1), (2, 1)), test graph compile
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=32, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy=((2, 1), (2, 1)))
    MSContext.get_instance().set_param(ms_ctx_param.dataset_broadcast_opt_level, 2)
    net = StageNet2()
    dataset = ds.GeneratorDataset(
        GeneratorFakeData(size=1024, batch_size=8, image_size=(64,),
                          use_parallel=True, num_classes=64), ["data", "label"])
    opt = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    loss = nn.L1Loss()
    loss_cell = WithLossCell(net, loss)
    model = Model(loss_cell, optimizer=opt)
    model.train(2, dataset, dataset_sink_mode=True)


def test_dataset_broadcast_set_dataset_layout():
    """
    Feature: opt_level 2 + set_dataset_strategy
    Description: no pipeline, dataset_strategy=((2, 1), (2, 1)), test graph compile
    Expectation: success
    """

    def find_graph_file_name(graph_path, file_name_keyword):
        largest_size = 0
        ir_graph_name = None

        for root, _, files in os.walk(graph_path):
            for file in files:
                if file.endswith('.ir') and file_name_keyword in file:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)

                    if file_size > largest_size:
                        largest_size = file_size
                        ir_graph_name = file

        return ir_graph_name

    context.set_context(save_graphs=True, save_graphs_path='./datasink_layout_graph')
    layout = Layout((16, 2), ("remain", "dp"))
    context.set_auto_parallel_context(device_num=32, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      dataset_strategy=(layout("dp", "None"), layout("dp", "None")))
    MSContext.get_instance().set_param(ms_ctx_param.dataset_broadcast_opt_level, 2)
    net = StageNet2()
    dataset = ds.GeneratorDataset(
        GeneratorFakeData(size=1024, batch_size=8, image_size=(64,),
                          use_parallel=True, num_classes=64), ["data", "label"])
    opt = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    loss = nn.L1Loss()
    loss_cell = WithLossCell(net, loss)
    model = Model(loss_cell, optimizer=opt)
    model.train(2, dataset, dataset_sink_mode=True)
    step_parallel_end = find_graph_file_name('./datasink_layout_graph/', 'step_parallel_end')
    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('GetNext', './datasink_layout_graph/' + step_parallel_end)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    match_shape = re.findall(r"shapes: (\(\(\d+, \d+\), \(\d+, \d+\)\))", log_cnt)
    assert match_shape[0] == "((8, 64), (8, 64))"


def test_dataset_broadcast_replace_get_next():
    """
    Feature: opt_level 2 + set_dataset_strategy
    Description: no pipeline, dataset_strategy=((2, 1), (2, 1)), test broadcast input shape and output shape
    Expectation: success
    """

    def find_file_name(graph_path, file_name_keyword):
        largest_size = 0
        ir_name = None

        for root, _, files in os.walk(graph_path):
            for file in files:
                if file.endswith('.ir') and file_name_keyword in file:
                    file_path = os.path.join(root, file)
                    size = os.path.getsize(file_path)

                    if size > largest_size:
                        largest_size = size
                        ir_name = file

        return ir_name

    context.set_context(save_graphs=True, save_graphs_path='./datasink_layout_replace_get_next')
    layout = Layout((16, 2), ("remain", "dp"))
    context.set_auto_parallel_context(device_num=32, global_rank=6)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      dataset_strategy=(layout("dp", "None"), layout("dp", "None")))
    MSContext.get_instance().set_param(ms_ctx_param.dataset_broadcast_opt_level, 2)
    net = StageNet2()
    dataset = ds.GeneratorDataset(
        GeneratorFakeData(size=1024, batch_size=8, image_size=(64,),
                          use_parallel=True, num_classes=64), ["data", "label"])
    opt_lamb = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    loss = nn.L1Loss()
    loss_cell = WithLossCell(net, loss)
    model = Model(loss_cell, optimizer=opt_lamb)
    model.train(2, dataset, dataset_sink_mode=True)
    optimizer_ir = find_file_name('./datasink_layout_replace_get_next/', 'optimizer')
    log_output = subprocess.check_output(
        ["grep -A 1 -r '%s' %s " % ('Broadcast', './datasink_layout_replace_get_next/' + optimizer_ir)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    match_shape = re.findall(r"TupleShape(\(\(\d+, \d+\), \(\d+, \d+\)\))", log_cnt)
    assert match_shape[0] == match_shape[1]
    assert match_shape[0] == "((8, 64), (8, 64))"
