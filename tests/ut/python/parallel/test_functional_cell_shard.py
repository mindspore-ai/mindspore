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
"""Test functional cell shard."""
import numpy as np
import os
import pytest

import mindspore as ms
from mindspore import Tensor, Parameter, context, nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.parallel.shard import Layout
from mindspore.nn import TrainOneStepCell, Momentum
from parallel.utils.utils import ParallelValidator, check_layout_config, compile_net
from tests.ut.python.ops.test_math_ops import VirtualLoss
from .test_pipeline_split import DatasetLenet

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

grad_all = C.GradOperation(get_all=True)

class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, y):
        predict = self.network(y)
        return self.loss(predict)

class GradWrap(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def construct(self, y):
        return grad_all(self.network)(y)

class ShardSubNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight1 = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
        self.weight2 = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
        self.bias = Tensor(np.ones([1024,]), dtype=ms.float32)
        self.w1 = Parameter(self.weight1, "w1")
        self.w2 = Parameter(self.weight2, "w2")
        self.b = Parameter(self.bias, "bias")

        self.matmul1 = P.MatMul()
        self.add = P.Add()
        self.matmul2 = P.MatMul()

    def construct(self, x):
        y = self.matmul1(x, self.w1)
        y = self.add(y, self.b)
        y = self.matmul2(y, self.w2)
        return y

class ShardNet(nn.Cell):
    def __init__(self, in_strategy=None, out_strategy=None, shard_key=None, in_parameter_plan=None):
        super().__init__()
        self.subnet = ShardSubNet()
        if shard_key == "ms":
            self.subnet_shard = ms.shard(self.subnet, in_strategy, out_strategy, parameter_plan=in_parameter_plan)
        else:
            if shard_key == "cell":
                self.subnet.shard(in_strategy, out_strategy, parameter_plan=in_parameter_plan)
            self.subnet_shard = self.subnet
        self.add = P.Add()
        self.matmul = P.MatMul()
        self.relu = nn.ReLU()

    def construct(self, x):
        y = self.subnet_shard(x)
        output = self.relu(y)
        return output


class FuncShardNetWithParam(nn.Cell):
    def __init__(self, in_strategy, out_strategy=None):
        super().__init__()
        self.subnet = ShardSubNet()
        self.add_weight = Parameter(Tensor(1.0, dtype=ms.float32))
        self.shard_func_add = ms.shard(self.func_add, in_strategy, out_strategy)
        self.add = P.Add()
        self.matmul = P.MatMul()
        self.relu = P.ReLU()

    def func_add(self, x):
        return self.add(x, self.add_weight)

    def construct(self, x):
        y = self.subnet(x)
        y = self.shard_func_add(x)
        output = self.relu(y)
        return output


def test_cell_shard_with_layout_be_set_and_propagate():
    """
    Feature: Test cell.shard given layout. The set layout can be seen in shard identity and the next operator.
    Description: dev_num is 8.
    Expectation: compile success, forward reduce_scatter
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    case_name = "test_cell_shard_with_layout_be_set_and_propagate"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    layout = Layout((2, 4, 1), ("dp", "sp", "mp"))
    in_layout1 = (layout("dp", "mp"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(ShardNet(in_layout1, shard_key="cell")))
    compile_net(net, x)
    file = f"{ir_graph_path}/step_parallel_begin_*"
    para1 = "PrimFunc_AShardIdentity(%1)"
    in_layout1 = (
        "in_layout: ({'device_matrix': (2, 4, 1), 'tensor_map': (2, 0), "
        "'interleaved_parallel': false, 'alias_name': (dp, sp, mp)})"
    )
    para2 = "PrimFunc_MatMul(%2"
    in_strategy = "in_strategy: ((2, 1), (1, 4))"
    check_layout_config(para1, file, in_layout1)
    check_layout_config(para2, file, in_strategy)


def test_ms_shard_with_layout_be_set_and_propagate():
    """
    Feature: Test ms.shard given layout. The set layout can be seen in shard identity and the next operator.
    Description: dev_num is 8.
    Expectation: compile success, forward reduce_scatter
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    case_name = "test_ms_shard_with_layout_be_set_and_propagate"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    layout = Layout((2, 4, 1), ("dp", "sp", "mp"))
    in_layout1 = (layout("dp", "mp"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(ShardNet(in_layout1, shard_key="ms")))
    compile_net(net, x)
    file = f"{ir_graph_path}/step_parallel_begin_*"
    para1 = "PrimFunc_AShardIdentity(%1)"
    in_layout1 = (
        "in_layout: ({'device_matrix': (2, 4, 1), 'tensor_map': (2, 0), "
        "'interleaved_parallel': false, 'alias_name': (dp, sp, mp)})"
    )
    para2 = "PrimFunc_MatMul(%2"
    in_strategy = "in_strategy: ((2, 1), (1, 4))"
    check_layout_config(para1, file, in_layout1)
    check_layout_config(para2, file, in_strategy)


def test_ms_shard_with_multi_dim_and_interleaved_parallel_layout():
    """
    Feature: Test ms.shard given layout. The set layout can be seen in shard identity and the next operator.
    Description: dev_num is 16.
    Expectation: compile success, forward reduce_scatter
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=16, global_rank=0)
    case_name = "test_ms_shard_with_multi_dim_and_interleaved_parallel_layout"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    layout = Layout((2, 4, 2, 2), ("dp", "mp", "sp", "interleaved_parallel"))
    in_layout1 = (layout(("dp", "interleaved_parallel", "mp"), "sp"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(ShardNet(in_layout1, shard_key="ms")))
    compile_net(net, x)
    file = f"{ir_graph_path}/step_parallel_begin_*"
    para1 = "PrimFunc_AShardIdentity(%1)"
    in_layout1 = (
        "in_layout: ({'device_matrix': (2, 4, 2, 2), 'tensor_map': ((3, 0, 2), 1), "
        "'interleaved_parallel': true, 'alias_name': (dp, mp, sp, interleaved_parallel)})"
    )
    para2 = "PrimFunc_MatMul(%2"
    in_strategy = "in_strategy: ((8, 2), (2, 1))"
    check_layout_config(para1, file, in_layout1)
    check_layout_config(para2, file, in_strategy)


def test_error_given_illegal_strategy():
    """
    Feature: Test ms.shard given layout. The set layout can be seen in shard identity and the next operator.
    Description: dev_num is 16.
    Expectation: compile success, forward reduce_scatter
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=16, global_rank=0)
    in_layout1 = (([2, 4], 2),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    error_msg = "The tuple strategy for each dimension should be tuple(int)"

    with pytest.raises(Exception) as err:
        net = GradWrap(NetWithLoss(ShardNet(in_layout1, shard_key="ms")))
        compile_net(net, x)
    assert error_msg in str(err.value)


def test_cell_shard_with_out_layout_be_set_and_propagate():
    """
    Feature: Test cell.shard given layout. The set layout can be seen in shard identity and the next operator.
    Description: dev_num is 8.
    Expectation: compile success, forward reduce_scatter
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    case_name = "test_cell_shard_with_out_layout_be_set_and_propagate"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    layout = Layout((2, 4, 1), ("dp", "sp", "mp"))
    in_layout1 = (layout("dp", "mp"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    out_layout1 = (layout("dp", "mp"),)
    parameter_plan = {"self.subnet.w1": layout("mp", "sp")}
    net = GradWrap(NetWithLoss(ShardNet(in_layout1, out_layout1, "cell", parameter_plan)))
    phase = compile_net(net, x)
    file = f"{ir_graph_path}/*_validate_*"
    para1 = "PrimFunc_MatMul(%4"
    in_strategy = "out_strategy: ((2, 1))"
    check_layout_config(para1, file, in_strategy)

    validator = ParallelValidator(net, phase)
    rank_list = {"rank_list": '(0, 1, 2, 3)'}
    assert validator.check_node_attrs('AllReduce-0', rank_list)


def test_cell_shard_with_out_strategy_be_set_and_propagate():
    """
    Feature: Test cell.shard given layout. The set layout can be seen in shard identity and the next operator.
    Description: dev_num is 8.
    Expectation: compile success, forward reduce_scatter
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    case_name = "test_cell_shard_with_out_strategy_be_set_and_propagate"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    layout = Layout((2, 4, 1), ("dp", "sp", "mp"))
    in_layout1 = (layout("dp", "mp"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    out_layout1 = ((2, 1),)
    parameter_plan = {"self.subnet.w1": layout("mp", "sp")}
    net = GradWrap(NetWithLoss(ShardNet(in_layout1, out_layout1, "cell", in_parameter_plan=parameter_plan)))
    phase = compile_net(net, x)
    file = f"{ir_graph_path}/*_validate_*"
    para1 = "PrimFunc_MatMul(%4"
    in_strategy = "out_strategy: ((2, 1))"
    check_layout_config(para1, file, in_strategy)

    validator = ParallelValidator(net, phase)
    rank_list = {"rank_list": '(0, 1, 2, 3)'}
    assert validator.check_node_attrs('AllReduce-0', rank_list)


def test_cell_shard_with_out_strategy_be_set_and_propagate_reduce_scatter():
    """
    Feature: Test cell.shard given layout. The set layout can be seen in shard identity and the next operator.
    Description: dev_num is 8.
    Expectation: compile success, forward reduce_scatter
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    case_name = "test_cell_shard_with_out_strategy_be_set_and_propagate_reduce_scatter"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    layout = Layout((2, 4, 1), ("dp", "sp", "mp"))
    in_layout1 = (layout("dp", "mp"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    out_layout1 = ((8, 1),)
    parameter_plan = {"self.subnet.w1": layout("mp", "sp")}
    net = GradWrap(NetWithLoss(ShardNet(in_layout1, out_layout1, "cell", in_parameter_plan=parameter_plan)))
    phase = compile_net(net, x)
    file = f"{ir_graph_path}/*_validate_*"
    para1 = "PrimFunc_MatMul(%4"
    in_strategy = "out_strategy: ((8, 1))"
    check_layout_config(para1, file, in_strategy)

    validator = ParallelValidator(net, phase)
    rank_list = {"rank_list": '(0, 1, 2, 3)'}
    assert validator.check_node_attrs('ReduceScatter-0', rank_list)


def test_ms_shard_with_out_layout_be_set_and_propagate():
    """
    Feature: Test cell.shard given layout. The set layout can be seen in shard identity and the next operator.
    Description: dev_num is 8.
    Expectation: compile success, forward reduce_scatter
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    case_name = "test_ms_shard_with_out_layout_be_set_and_propagate"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    layout = Layout((2, 4, 1), ("dp", "sp", "mp"))
    in_layout1 = (layout("dp", "mp"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    out_layout1 = (layout("dp", "mp"),)
    parameter_plan = {"self.subnet.w1": layout("mp", "sp")}
    net = GradWrap(NetWithLoss(ShardNet(in_layout1, out_layout1, "ms", in_parameter_plan=parameter_plan)))
    phase = compile_net(net, x)
    file = f"{ir_graph_path}/*_validate_*"
    para1 = "PrimFunc_MatMul(%4"
    in_strategy = "out_strategy: ((2, 1))"
    check_layout_config(para1, file, in_strategy)

    validator = ParallelValidator(net, phase)
    rank_list = {"rank_list": '(0, 1, 2, 3)'}
    assert validator.check_node_attrs('AllReduce-0', rank_list)


@pytest.mark.skip(reason="Disable shard pynative support.")
def test_ms_shard_with_out_strategy_be_set_and_propagate():
    """
    Feature: Test cell.shard given layout. The set layout can be seen in shard identity and the next operator.
    Description: dev_num is 8.
    Expectation: compile success, forward reduce_scatter
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    case_name = "test_ms_shard_with_out_strategy_be_set_and_propagate"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    layout = Layout((2, 4, 1), ("dp", "sp", "mp"))
    in_layout1 = (layout("dp", "mp"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    out_layout1 = ((2, 1),)
    parameter_plan = {"self.subnet.w1": layout("mp", "sp")}
    net = GradWrap(NetWithLoss(ShardNet(in_layout1, out_layout1, "ms", in_parameter_plan=parameter_plan)))
    phase = compile_net(net, x)
    file = f"{ir_graph_path}/*_validate_*"
    para1 = "PrimFunc_MatMul(%4"
    in_strategy = "out_strategy: ((2, 1))"
    check_layout_config(para1, file, in_strategy)

    validator = ParallelValidator(net, phase)
    rank_list = {"rank_list": '(0, 1, 2, 3)'}
    assert validator.check_node_attrs('AllReduce-0', rank_list)


def test_ms_shard_with_out_strategy_be_set_and_propagate_reduce_scatter():
    """
    Feature: Test cell.shard given layout. The set layout can be seen in shard identity and the next operator.
    Description: dev_num is 8.
    Expectation: compile success, forward reduce_scatter
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    case_name = "test_ms_shard_with_out_strategy_be_set_and_propagate_reduce_scatter"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    layout = Layout((2, 4, 1), ("dp", "sp", "mp"))
    in_layout1 = (layout("dp", "mp"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    out_layout1 = ((8, 1),)
    parameter_plan = {"self.subnet.w1": layout("mp", "sp")}
    net = GradWrap(NetWithLoss(ShardNet(in_layout1, out_layout1, "ms", in_parameter_plan=parameter_plan)))
    phase = compile_net(net, x)
    file = f"{ir_graph_path}/*_validate_*"
    para1 = "PrimFunc_MatMul(%4"
    in_strategy = "out_strategy: ((8, 1))"
    check_layout_config(para1, file, in_strategy)

    validator = ParallelValidator(net, phase)
    rank_list = {"rank_list": '(0, 1, 2, 3)'}
    assert validator.check_node_attrs('ReduceScatter-0', rank_list)


@pytest.mark.skip(reason="Disable shard pynative support.")
def test_ms_shard_with_layout_be_set_and_propagate_pynative():
    """
    Feature: Test ms.shard + pynative.
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    context.set_context(mode=ms.PYNATIVE_MODE)
    case_name = "test_ms_shard_with_layout_be_set_and_propagate_pynative"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    layout = Layout((2, 4, 1), ("dp", "sp", "mp"))
    in_layout1 = (layout("dp", "mp"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(ShardNet(in_layout1, shard_key="ms")))
    compile_net(net, x)


@pytest.mark.skip(reason="Disable shard pynative support.")
def test_ms_shard_function_with_parameter_exception():
    """
    Feature: Test ms.shard + pynative + function + parameter.
    Description: dev_num is 8.
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    context.set_context(mode=ms.PYNATIVE_MODE)
    layout = Layout((2, 4, 1), ("dp", "sp", "mp"))
    in_layout1 = (layout("dp", "mp"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(FuncShardNetWithParam(in_layout1)))
    with pytest.raises(RuntimeError):
        compile_net(net, x)


class MatMulNetShard(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = P.MatMul()

    def construct(self, x, w):
        x = self.matmul(x, w)
        return x

class MatMulNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = MatMulNetShard()
        self.matmul_fn = ms.shard(self.matmul, in_strategy=((4, 1), (1, 1)))

    def construct(self, x, w):
        x = self.matmul_fn(x, w)
        return x


class AddNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()

    def construct(self, x, w):
        x = self.add(x, w)
        return x


class AddMatmulNet(nn.Cell):
    def __init__(self, matmul_weight_shape, add_weight_shape, **kwargs):
        super().__init__()
        self.matmul_weight = Parameter(Tensor(np.ones(matmul_weight_shape), ms.float32), name="matmul_weight")
        self.add_weight = Parameter(Tensor(np.ones(add_weight_shape), ms.float32), name="add_weight")
        self.matmul = MatMulNet()
        self.add = AddNet()

    def construct(self, x):
        x = self.matmul(x, self.matmul_weight)
        x = self.add(x, self.add_weight)
        return x


class WithLossCell(nn.Cell):
    @ms.lazy_inline
    def __init__(self, backbone, loss_fn):
        super().__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self._get_attr_from_cell(backbone)

    def construct(self, data, label):
        out = self._backbone(data)
        return self._loss_fn(out, label)


def test_ms_shard_pp_interleave_with_correct_group_rank_ids():
    """
    Feature: Test ms.shard with given layout inside a pp stage and group_rank_ids are correct.
    Description: dev_num is 8.
    Expectation: compile with runtime error raised.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    case_name = "test_ms_shard_pp_interleave_with_correct_group_rank_ids"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    x = Tensor(np.ones([128, 128]), dtype=ms.float32)
    net_2 = AddMatmulNet(matmul_weight_shape=(128, 128), add_weight_shape=(1, 128))
    loss_2 = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    withlosscell = WithLossCell(net_2, loss_2)
    pipeline_cell = nn.PipelineCell(withlosscell, micro_size=4,
                                    stage_config={"_backbone.matmul": 0, "_backbone.add": 1})
    pipeline_net = AutoParallel(pipeline_cell, parallel_mode="sharding_propagation")
    pipeline_net.pipeline(stages=2, interleave=True)
    pipeline_net.dataset_strategy("full_batch")

    y = Tensor(np.ones([128, 128]), dtype=ms.float32)
    dataset = DatasetLenet(x, y, 3)
    params = pipeline_cell.trainable_params()
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = ms.train.Model(pipeline_net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)

    file = f"{ir_graph_path}/*_validate_*"
    para1_str = "= Send("
    group_rank_ids_str = 'group_rank_ids: (0, 4)'
    check_layout_config(para1_str, file, group_rank_ids_str)


def test_cell_shard_with_layout_be_set_and_propagate_defer_inline_0():
    """
    Feature: Test cell.shard given layout. The set layout can be seen in the shard operator.
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    case_name = "test_cell_shard_with_layout_be_set_and_propagate_defer_inline_0"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    layout = Layout((2, 4, 1), ("dp", "sp", "mp"))
    in_layout1 = (layout("dp", "mp"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(ShardNet(in_layout1, shard_key="cell")))
    in_layout2 = (layout("mp", "sp"),)
    net.network.network.relu.shard(in_layout2)
    compile_net(net, x)
    file = f"{ir_graph_path}/05_inline_*"
    para1_str = "= Shard(.*ShardSubNet_construct"
    in_layout1_str = (
        '(((I64(2), I64(4), I64(1)), (I64(2), I64(0)), Bool(0), ("dp", "sp", "mp"))), None'
    )
    para2_str = "= Shard(.*ReLU_construct"
    in_layout2_str = (
        '(((I64(2), I64(4), I64(1)), (I64(0), I64(1)), Bool(0), ("dp", "sp", "mp"))), None'
    )
    check_layout_config(para1_str, file, in_layout1_str)
    check_layout_config(para2_str, file, in_layout2_str)


def test_cell_nested_shard_with_layout_be_set_and_propagate_1():
    """
    Feature: Test cell.shard given nested layout. The set layout can be seen in the shard operator.
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    case_name = "test_cell_nested_shard_with_layout_be_set_and_propagate_1"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    layout = Layout((2, 4, 1), ("dp", "sp", "mp"))
    in_layout1 = (layout("dp", "mp"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(ShardNet(in_layout1, shard_key="cell")))
    in_layout2 = (layout("mp", "sp"),)
    net.network.network.shard(in_layout2)
    compile_net(net, x)
    file = f"{ir_graph_path}/05_inline_*"
    para1_str = "y) = Shard("
    in_layout1_str = (
        '(((I64(2), I64(4), I64(1)), (I64(2), I64(0)), Bool(0), ("dp", "sp", "mp"))), None'
    )
    para2_str = "predict) = Shard("
    in_layout2_str = (
        '(((I64(2), I64(4), I64(1)), (I64(0), I64(1)), Bool(0), ("dp", "sp", "mp"))), None'
    )
    check_layout_config(para1_str, file, in_layout1_str)
    check_layout_config(para2_str, file, in_layout2_str)


def test_cell_nested_shard_with_layout_be_set_and_propagate_2():
    """
    Feature: Test cell.shard given nested layout. The set layout can be seen in the shard operator.
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    case_name = "test_cell_nested_shard_with_layout_be_set_and_propagate_2"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    layout = Layout((2, 4, 1), ("dp", "sp", "mp"))
    in_layout1 = (layout("dp", "mp"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(ShardNet()))
    net.network.network.subnet.shard(in_layout1)
    in_strategy2 = ((1, 8),)
    out_strategy2 = ((4, 2),)
    net.network.network.shard(in_strategy2, out_strategy2)
    compile_net(net, x)
    file = f"{ir_graph_path}/05_inline_*"
    para1_str = "y) = Shard("
    in_layout1_str = (
        '(((I64(2), I64(4), I64(1)), (I64(2), I64(0)), Bool(0), ("dp", "sp", "mp"))), None'
    )
    para2_str = "predict) = Shard("
    in_strategy2_str = (
        '((I64(1), I64(8))), ((I64(4), I64(2)))'
    )
    check_layout_config(para1_str, file, in_layout1_str)
    check_layout_config(para2_str, file, in_strategy2_str)


def test_cell_nested_and_repeated_shard_with_layout_be_set_and_propagate_3():
    """
    Feature: Test cell.shard given nested layout. The set layout can be seen in the shard operator.
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    case_name = "test_cell_nested_and_repeated_shard_with_layout_be_set_and_propagate_3"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    layout = Layout((2, 4, 1), ("dp", "sp", "mp"))
    in_layout1 = (layout("dp", "mp"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(ShardNet()))
    net.network.network.subnet.shard(in_layout1)
    in_layout2 = (layout("mp", "sp"),)
    net.network.network.shard(in_layout2)

    in_layout3 = (layout("sp", "mp"),)
    net.network.network.subnet.shard(in_layout3)

    in_strategy4 = ((1, 8),)
    out_strategy4 = ((4, 2),)
    net.network.network.shard(in_strategy4, out_strategy4)

    compile_net(net, x)
    file = f"{ir_graph_path}/05_inline_*"
    para1_str = "y) = Shard("
    in_layout3_str = (
        '(((I64(2), I64(4), I64(1)), (I64(1), I64(0)), Bool(0), ("dp", "sp", "mp"))), None'
    )
    para2_str = "predict) = Shard("
    in_strategy4_str = (
        '((I64(1), I64(8))), ((I64(4), I64(2)))'
    )
    check_layout_config(para1_str, file, in_layout3_str)
    check_layout_config(para2_str, file, in_strategy4_str)


def test_cell_shard_pp_interleave_4():
    """
    Feature: Test cell.shard with given layout and make shard on a pipeline_cell.
    Description: dev_num is 8.
    Expectation: compile with value error raised.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    setup_function()
    case_name = "test_cell_shard_pp_interleave_4"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    layout = Layout((2, 4, 1), ("dp", "sp", "mp"))
    in_layout1 = (layout("dp", "mp"),)
    withlosscell = NetWithLoss(AddMatmulNet(matmul_weight_shape=(128, 128), add_weight_shape=(1, 128)))
    pipeline_cell = nn.PipelineCell(withlosscell, micro_size=2, stage_config={"network.matmul": 0, "network.add": 1})
    with pytest.raises(ValueError) as error_info:
        pipeline_cell.shard(in_layout1)
    assert "For 'PipelineCell', no 'shard' " in str(error_info.value)


def test_cell_shard_pp_interleave_5():
    """
    Feature: Test cell.shard with given layout and make shard outside a sub cell of pipeline stage.
    Description: dev_num is 8.
    Expectation: compile success.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    case_name = "test_cell_shard_pp_interleave_5"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    x = Tensor(np.ones([128, 128]), dtype=ms.float32)
    net_2 = AddMatmulNet(matmul_weight_shape=(128, 128), add_weight_shape=(1, 128))
    withlosscell = NetWithLoss(net_2)
    pipeline_cell = nn.PipelineCell(withlosscell, micro_size=4, stage_config={"network.matmul": 0, "network.add": 1})
    pipeline_net = AutoParallel(pipeline_cell, parallel_mode="sharding_propagation")
    pipeline_net.pipeline(stages=2, interleave=True)
    pipeline_net.dataset_strategy("full_batch")
    withlosscell.shard(in_strategy=((1, 2),))
    with pytest.raises(RuntimeError) as error_info:
        compile_net(pipeline_net, x)
    assert "For sharded cell " in str(error_info.value)


def test_cell_shard_in_root_graph_00():
    """
    Feature: Test cell.shard with given layout in the root graph, no inlining occurs in the shard expand irpass.
    Description: dev_num is 8.
    Expectation: compile with runtime error raised.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=8, global_rank=0)
    case_name = "test_cell_shard_in_root_graph_00"
    ir_graph_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "layout_ir", case_name)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    layout = Layout((2, 4, 1), ("dp", "sp", "mp"))
    in_layout1 = (layout("dp", "mp"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    net = ShardNet()
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    net.shard(in_layout1)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    compile_net(train_net, x)
