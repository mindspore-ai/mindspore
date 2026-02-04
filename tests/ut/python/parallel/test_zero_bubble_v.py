# Copyright 2025 Huawei Technologies Co., Ltd
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
import shutil
import subprocess
import mindspore.nn as nn
from mindspore import context
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.train import Model
from mindspore.nn import PipelineCell
from mindspore.graph.lazy_inline import lazy_inline
import json
from .test_dynamic_data_sink import GeneratorFakeData
import mindspore.dataset as ds
from mindspore._c_expression import MSContext, ms_ctx_param
from mindspore.parallel.auto_parallel import AutoParallel
import mindspore as ms


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


class MatMulCellWithMixPrecision(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64], dtype=ms.float16), name="param")
        self.param1 = Parameter(initializer("zeros", [64, 64], dtype=ms.float16), name="param1")
        self.matmul = P.MatMul().shard(((2, 1), (1, 1)))
        self.matmul1 = P.MatMul().shard(((4, 1), (1, 1)))
        self.cast = P.Cast()

    def construct(self, x):
        param = self.cast(self.param, ms.float32)
        param1 = self.cast(self.param1, ms.float32)
        out = self.matmul(x, param)
        out = self.matmul1(out, param1)
        return out


class StageNetWithMixPrecision(nn.Cell):
    def __init__(self):
        super().__init__()
        self.cell1 = MatMulCellWithMixPrecision()
        self.cell1.pipeline_stage = 0
        self.cell1.pipeline_segment = 0
        self.cell2 = MatMulCellWithMixPrecision()
        self.cell2.pipeline_stage = 1
        self.cell2.pipeline_segment = 0
        self.cell3 = MatMulCellWithMixPrecision()
        self.cell3.pipeline_stage = 2
        self.cell3.pipeline_segment = 0
        self.cell4 = MatMulCellWithMixPrecision()
        self.cell4.pipeline_stage = 3
        self.cell4.pipeline_segment = 0
        self.cell5 = MatMulCellWithMixPrecision()
        self.cell5.pipeline_stage = 3
        self.cell5.pipeline_segment = 1
        self.cell6 = MatMulCellWithMixPrecision()
        self.cell6.pipeline_stage = 2
        self.cell6.pipeline_segment = 1
        self.cell7 = MatMulCellWithMixPrecision()
        self.cell7.pipeline_stage = 1
        self.cell7.pipeline_segment = 1
        self.cell8 = MatMulCellWithMixPrecision()
        self.cell8.pipeline_stage = 0
        self.cell8.pipeline_segment = 1

    def construct(self, x):
        out = self.cell1(x)
        out = self.cell2(out)
        out = self.cell3(out)
        out = self.cell4(out)
        out = self.cell5(out)
        out = self.cell6(out)
        out = self.cell7(out)
        out = self.cell8(out)
        return out


class StageNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.cell1 = MatMulCell()
        self.cell1.pipeline_stage = 0
        self.cell1.pipeline_segment = 0
        self.cell2 = MatMulCell()
        self.cell2.pipeline_stage = 1
        self.cell2.pipeline_segment = 0
        self.cell3 = MatMulCell()
        self.cell3.pipeline_stage = 2
        self.cell3.pipeline_segment = 0
        self.cell4 = MatMulCell()
        self.cell4.pipeline_stage = 3
        self.cell4.pipeline_segment = 0
        self.cell5 = MatMulCell()
        self.cell5.pipeline_stage = 3
        self.cell5.pipeline_segment = 1
        self.cell6 = MatMulCell()
        self.cell6.pipeline_stage = 2
        self.cell6.pipeline_segment = 1
        self.cell7 = MatMulCell()
        self.cell7.pipeline_stage = 1
        self.cell7.pipeline_segment = 1
        self.cell8 = MatMulCell()
        self.cell8.pipeline_stage = 0
        self.cell8.pipeline_segment = 1

    def construct(self, x):
        out = self.cell1(x)
        out = self.cell2(out)
        out = self.cell3(out)
        out = self.cell4(out)
        out = self.cell5(out)
        out = self.cell6(out)
        out = self.cell7(out)
        out = self.cell8(out)
        return out


class StageNetNewApi(nn.Cell):
    def __init__(self):
        super().__init__()
        self.cell1 = MatMulCell()
        self.cell2 = MatMulCell()
        self.cell3 = MatMulCell()
        self.cell4 = MatMulCell()
        self.cell5 = MatMulCell()
        self.cell6 = MatMulCell()
        self.cell7 = MatMulCell()
        self.cell8 = MatMulCell()

    def construct(self, x):
        out = self.cell1(x)
        out = self.cell2(out)
        out = self.cell3(out)
        out = self.cell4(out)
        out = self.cell5(out)
        out = self.cell6(out)
        out = self.cell7(out)
        out = self.cell8(out)
        return out


class WithLossCell(nn.Cell):
    @lazy_inline
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self._get_attr_from_cell(backbone)

    def construct(self, data, label):
        out = self._backbone(data)
        return self._loss_fn(out, label)
    @property
    def backbone_network(self):
        return self._backbone


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


class StageNetRecompute(nn.Cell):
    def __init__(self):
        super().__init__()
        self.cell1 = MatMulCell()
        self.cell1.pipeline_stage = 0
        self.cell1.pipeline_segment = 0
        self.cell1.recompute()
        self.cell2 = MatMulCell()
        self.cell2.pipeline_stage = 1
        self.cell2.pipeline_segment = 0
        self.cell2.recompute()
        self.cell3 = MatMulCell()
        self.cell3.pipeline_stage = 2
        self.cell3.pipeline_segment = 0
        self.cell3.recompute()
        self.cell4 = MatMulCell()
        self.cell4.pipeline_stage = 3
        self.cell4.pipeline_segment = 0
        self.cell4.recompute()
        self.cell5 = MatMulCell()
        self.cell5.pipeline_stage = 3
        self.cell5.pipeline_segment = 1
        self.cell5.recompute()
        self.cell6 = MatMulCell()
        self.cell6.pipeline_stage = 2
        self.cell6.pipeline_segment = 1
        self.cell6.recompute()
        self.cell7 = MatMulCell()
        self.cell7.pipeline_stage = 1
        self.cell7.pipeline_segment = 1
        self.cell7.recompute()
        self.cell8 = MatMulCell()
        self.cell8.pipeline_stage = 0
        self.cell8.pipeline_segment = 1
        self.cell8.recompute()

    def construct(self, x):
        out = self.cell1(x)
        out = self.cell2(out)
        out = self.cell3(out)
        out = self.cell4(out)
        out = self.cell5(out)
        out = self.cell6(out)
        out = self.cell7(out)
        out = self.cell8(out)
        return out


def test_zero_bubble_v():
    """
    Feature: zerobubblev + 1b1f
    Description: test control edge
    Expectation: success
    """
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = "0"
    graph_root_path = "./graph1"
    rank_graph_path = graph_root_path + "/"
    context.set_context(save_graphs=True, save_graphs_path=graph_root_path)
    context.set_auto_parallel_context(device_num=32, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    pp_config = {"pipeline_interleave": True, "pipeline_scheduler": "zero_bubble_v"}
    context.set_auto_parallel_context(pipeline_config=pp_config, pipeline_stages=4)

    if os.path.exists("./speed_up.json"):
        os.remove("./speed_up.json")
    a = {"pp_1f1b_overlap": "AlltoAll,AlltoAllV"}
    f = open("./speed_up.json", "w")
    f.write(json.dumps(a))
    f.close()
    context.set_context(ascend_config={"parallel_speed_up_json_path": "speed_up.json"})

    MSContext.get_instance().set_param(ms_ctx_param.dataset_broadcast_opt_level, 1)
    net = StageNet()
    dataset = ds.GeneratorDataset(
        GeneratorFakeData(size=1024, batch_size=8, image_size=(64,),
                          use_parallel=True, num_classes=64), ["data", "label"])
    opt = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    loss = nn.L1Loss()
    loss_cell = WithLossCell(net, loss)
    pp_cell = PipelineCell(loss_cell, 8)
    model = Model(pp_cell, optimizer=opt)
    model.train(2, dataset, dataset_sink_mode=True)
    pipeline_scheduler = find_graph_file_name(graph_root_path, 'pipeline_parallel_scheduler')

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('call_call_1f1b', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "call_call_1f1b" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('1b1f_call_call', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "1b1f_call_call" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('input_recv_1f1b', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "input_recv_1f1b" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('send_out_1f1b', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "send_out_1f1b" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('inner_overlap', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "inner_overlap" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('zero_bubble_v_control', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "zero_bubble_v_control" in log_cnt
    if os.path.exists("./speed_up.json"):
        os.remove("./speed_up.json")
    if os.path.exists(graph_root_path):
        shutil.rmtree(graph_root_path)
    del os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"]


def zero_bubble_v_recompute(rank_id):
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = "0"
    graph_root_path = "./graph" + str(rank_id)
    rank_graph_path = graph_root_path + "/"
    context.set_context(save_graphs=True, save_graphs_path=graph_root_path)
    context.set_auto_parallel_context(device_num=32, global_rank=rank_id)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    pp_config = {"pipeline_interleave": True, "pipeline_scheduler": "zero_bubble_v"}
    context.set_auto_parallel_context(pipeline_config=pp_config, pipeline_stages=4)

    if os.path.exists("./speed_up.json"):
        os.remove("./speed_up.json")
    a = {"pp_1f1b_overlap": "AlltoAll,AlltoAllV"}
    f = open("./speed_up.json", "w")
    f.write(json.dumps(a))
    f.close()
    context.set_context(ascend_config={"parallel_speed_up_json_path": "speed_up.json"})

    MSContext.get_instance().set_param(ms_ctx_param.dataset_broadcast_opt_level, 1)
    net = StageNetRecompute()
    dataset = ds.GeneratorDataset(
        GeneratorFakeData(size=1024, batch_size=8, image_size=(64,),
                          use_parallel=True, num_classes=64), ["data", "label"])
    opt = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    loss = nn.L1Loss()
    loss_cell = WithLossCell(net, loss)
    pp_cell = PipelineCell(loss_cell, 8)
    model = Model(pp_cell, optimizer=opt)
    model.train(2, dataset, dataset_sink_mode=True)
    pipeline_scheduler = find_graph_file_name(graph_root_path, 'pipeline_parallel_scheduler')

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('call_call_1f1b', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "call_call_1f1b" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('1b1f_call_call', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "1b1f_call_call" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('input_recv_1f1b', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "input_recv_1f1b" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('send_out_1f1b', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "send_out_1f1b" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('inner_overlap', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "inner_overlap" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('zero_bubble_v_control', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "zero_bubble_v_control" in log_cnt
    if os.path.exists("./speed_up.json"):
        os.remove("./speed_up.json")
    if os.path.exists(graph_root_path):
        shutil.rmtree(graph_root_path)
    del os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"]


def test_zero_bubble_v_recompute_16():
    """
    Feature: zerobubblev + 1b1f + recompute rank16
    Description: test control edge
    Expectation: success
    """
    zero_bubble_v_recompute(16)


def test_zero_bubble_v_recompute_0():
    """
    Feature: zerobubblev + 1b1f + recompute rank0
    Description: test control edge
    Expectation: success
    """
    zero_bubble_v_recompute(0)


def test_zero_bubble_v_new_api():
    """
    Feature: zerobubblev + 1b1f + newapi
    Description: test control edge
    Expectation: success
    """
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = "0"
    graph_root_path = "./graph_new_api"
    rank_graph_path = graph_root_path + "/"
    context.set_context(save_graphs=True, save_graphs_path=graph_root_path)
    context.set_auto_parallel_context(device_num=32, global_rank=0)
    stage_config = {"_backbone.cell1": 0, "_backbone.cell2": 1, "_backbone.cell3": 2,
                    "_backbone.cell4": 3, "_backbone.cell5": 3, "_backbone.cell6": 2,
                    "_backbone.cell7": 1, "_backbone.cell8": 0}
    segment_config = {"_backbone.cell1": 0, "_backbone.cell2": 0, "_backbone.cell3": 0,
                      "_backbone.cell4": 0, "_backbone.cell5": 1, "_backbone.cell6": 1,
                      "_backbone.cell7": 1, "_backbone.cell8": 1}

    if os.path.exists("./speed_up.json"):
        os.remove("./speed_up.json")
    a = {"pp_1f1b_overlap": "AlltoAll,AlltoAllV"}
    f = open("./speed_up.json", "w")
    f.write(json.dumps(a))
    f.close()
    context.set_context(ascend_config={"parallel_speed_up_json_path": "speed_up.json"})

    MSContext.get_instance().set_param(ms_ctx_param.dataset_broadcast_opt_level, 1)
    net = StageNetNewApi()
    dataset = ds.GeneratorDataset(
        GeneratorFakeData(size=1024, batch_size=8, image_size=(64,),
                          use_parallel=True, num_classes=64), ["data", "label"])
    opt = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    loss = nn.L1Loss()
    loss_cell = WithLossCell(net, loss)
    pp_cell = PipelineCell(loss_cell, 8, stage_config, segment_config)
    pp_net_parallel = AutoParallel(pp_cell, parallel_mode="semi_auto")
    pp_net_parallel.pipeline(stages=4, scheduler="zero_bubble_v", interleave=True)
    model = Model(pp_net_parallel, optimizer=opt)
    model.train(2, dataset, dataset_sink_mode=True)
    pipeline_scheduler = find_graph_file_name(graph_root_path, 'pipeline_parallel_scheduler')

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('call_call_1f1b', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "call_call_1f1b" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('1b1f_call_call', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "1b1f_call_call" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('input_recv_1f1b', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "input_recv_1f1b" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('send_out_1f1b', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "send_out_1f1b" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('inner_overlap', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "inner_overlap" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('zero_bubble_v_control', rank_graph_path + pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "zero_bubble_v_control" in log_cnt
    if os.path.exists("./speed_up.json"):
        os.remove("./speed_up.json")
    if os.path.exists(graph_root_path):
        shutil.rmtree(graph_root_path)
    del os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"]


def test_zero_bubble_v_mix_precision():
    """
    Feature: zerobubblev + 1b1f + mix_precision
    Description: test zbv+mix precision
    Expectation: success
    """
    from parallel.auto_parallel_interface._utils import find_ir_file_path
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = "0"
    exec_path = os.path.dirname(os.path.realpath(__file__))
    graph_root_path = os.path.join(exec_path, "graph_mix_precision")
    context.set_context(save_graphs=True, save_graphs_path=graph_root_path)
    context.set_auto_parallel_context(device_num=32, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    pp_config = {"pipeline_interleave": True, "pipeline_scheduler": "zero_bubble_v"}
    context.set_auto_parallel_context(pipeline_config=pp_config, pipeline_stages=4)

    if os.path.exists("./speed_up.json"):
        os.remove("./speed_up.json")
    a = {"pp_1f1b_overlap": "AlltoAll,AlltoAllV"}
    f = open("./speed_up.json", "w")
    f.write(json.dumps(a))
    f.close()
    context.set_context(ascend_config={"parallel_speed_up_json_path": "speed_up.json"})

    MSContext.get_instance().set_param(ms_ctx_param.dataset_broadcast_opt_level, 1)
    net = StageNetWithMixPrecision()
    dataset = ds.GeneratorDataset(
        GeneratorFakeData(size=1024, batch_size=8, image_size=(64,),
                          use_parallel=True, num_classes=64), ["data", "label"])
    opt = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    loss = nn.L1Loss()
    loss_cell = WithLossCell(net, loss)
    pp_cell = PipelineCell(loss_cell, 8)
    model = Model(pp_cell, optimizer=opt)
    model.train(2, dataset, dataset_sink_mode=True)
    pipeline_scheduler = find_ir_file_path(graph_root_path, "pipeline_parallel_scheduler")

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('call_call_1f1b', pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "call_call_1f1b" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('1b1f_call_call', pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "1b1f_call_call" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('input_recv_1f1b', pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "input_recv_1f1b" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('send_out_1f1b', pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "send_out_1f1b" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('inner_overlap', pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "inner_overlap" in log_cnt

    log_output = subprocess.check_output(
        ["grep -r '%s' %s " % ('zero_bubble_v_control', pipeline_scheduler)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert "zero_bubble_v_control" in log_cnt
    if os.path.exists("./speed_up.json"):
        os.remove("./speed_up.json")
    if os.path.exists(graph_root_path):
        shutil.rmtree(graph_root_path)
    del os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"]
