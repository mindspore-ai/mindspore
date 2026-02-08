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

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter, context
from mindspore.graph.api import _cell_graph_executor
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.parallel.shard import Layout
from mindspore.parallel import set_algo_parameters
from parallel.utils.utils import check_layout_config
from tests.ut.python.ops.test_math_ops import VirtualLoss

def compile_net(net, input_x):
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, input_x)
    return phase

grad_all = C.GradOperation(get_all=True)

class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, y):
        predict = self.network(y)
        return self.loss(predict)

class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, y):
        return grad_all(self.network)(y)

class NetTwoMatMul(nn.Cell):
    def __init__(self, weight1, weight2, in_layout1=None, in_layout2=None, out_layout1=None, out_layout2=None):
        super().__init__()
        self.matmul1 = P.MatMul()
        self.matmul2 = P.MatMul()
        if in_layout1:
            self.matmul1 = self.matmul1.shard(in_strategy=in_layout1, out_strategy=out_layout1)
        if in_layout2:
            self.matmul2 = self.matmul2.shard(in_strategy=in_layout2, out_strategy=out_layout2)
        self.relu = P.ReLU()
        self.cast = P.Cast()
        self.gelu = P.GeLU()
        self.depend = P.Depend()
        self.w1 = Parameter(weight1, "w1")
        self.w2 = Parameter(weight2, "w2")

    def construct(self, y):
        y = self.relu(y)
        y_new = self.gelu(y)
        out1 = self.matmul1(y, self.w1)
        out2 = self.matmul2(out1, self.w2)
        return self.relu(out2) + y_new

def test_layout_in_node_prim_attrs_correct():
    """
    Feature: test layout attrs can be extracted from the node
    Description: dev_num is 16.
    Expectation: compile success, attrs contain tensor_map, alias_name, device_matrix...
    """
    case_name = "test_layout_in_node_prim_attrs_correct"
    ir_graph_path = f"./ir/{case_name}"
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=16, global_rank=0)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    layout = Layout((2, 4, 2, 2), ("dp", "mp", "sp", "interleaved_parallel"))
    in_layout1 = (layout(("dp", "interleaved_parallel"), "mp"), layout("mp", "sp"))
    out_layout1 = (layout(("dp", "interleaved_parallel", "mp"), "sp"),)
    in_layout2 = (layout(("dp", "interleaved_parallel", "mp"), "sp"), layout("sp", "None"))
    out_layout2 = (layout(("dp", "interleaved_parallel", "mp", "sp"), "None"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    w1 = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    w2 = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(NetTwoMatMul(w1, w2, in_layout1, in_layout2, out_layout1, out_layout2)))
    compile_net(net, x)
    file = f"{ir_graph_path}/step_auto_parallel_begin_*"
    in_layout_cfg1 = (
        "in_layout: ({'device_matrix': (2, 4, 2, 2), 'tensor_map': ((3, 0), 2), 'interleaved_parallel': true, "
        "'alias_name': (dp, mp, sp, interleaved_parallel)}, {'device_matrix': (2, 4, 2, 2), 'tensor_map': (2, 1), "
        "'interleaved_parallel': true, 'alias_name': (dp, mp, sp, interleaved_parallel)})"
    )
    out_layout_cfg1 = (
        "out_layout: ({'device_matrix': (2, 4, 2, 2), 'tensor_map': ((3, 0, 2), 1), 'interleaved_parallel': true, "
        "'alias_name': (dp, mp, sp, interleaved_parallel)"
    )
    in_layout_cfg2 = (
        "in_layout: ({'device_matrix': (2, 4, 2, 2), 'tensor_map': ((3, 0, 2), 1), 'interleaved_parallel': true, "
        "'alias_name': (dp, mp, sp, interleaved_parallel)}, {'device_matrix': (2, 4, 2, 2), 'tensor_map': (1, -1), "
        "'interleaved_parallel': true, 'alias_name': (dp, mp, sp, interleaved_parallel)})"
    )
    out_layout_cfg2 = (
        "out_layout: ({'device_matrix': (2, 4, 2, 2), 'tensor_map': ((3, 0, 2, 1), -1), "
        "'interleaved_parallel': true, 'alias_name': (dp, mp, sp, interleaved_parallel)})"
    )
    para1 = "%2(out1) = PrimFunc_MatMul"
    para2 = "%4(out2) = PrimFunc_MatMul"
    check_layout_config(para1, file, in_layout_cfg1, out_layout_cfg1)
    check_layout_config(para2, file, in_layout_cfg2, out_layout_cfg2)

def test_layout_propagation_in_two_matmul_net():
    """
    Feature: test layout propagation given a matmul in layout ((2, 4), (4, 2)) and out layout ((2, 2),)
    Description: dev_num is 16.
    Expectation: compile success, the first param strategy of the next matmul should be ((2, 2), (2, 1))
    """
    case_name = "test_layout_propagation_in_two_matmul_net"
    ir_graph_path = f"./ir/{case_name}"
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=16, global_rank=0)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    set_algo_parameters(fully_use_devices=False)
    layout = Layout((2, 4, 2), ("dp", "mp", "sp"))
    in_layout1 = (layout("dp", "mp"), layout("mp", "sp"))
    out_layout1 = (layout("dp", "sp"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    w1 = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    w2 = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(NetTwoMatMul(w1, w2, in_layout1, None, out_layout1)))
    compile_net(net, x)
    file = f"{ir_graph_path}/step_parallel_begin_*"
    in_strategy = "in_strategy: ((2, 2), (2, 1))"
    para1 = "%6(out2) = PrimFunc_MatMul"
    check_layout_config(para1, file, in_strategy)

def test_layout_propagation_with_mixed_strategy():
    """
    Feature: test layout propagation given a matmul in layout ((2, 4), (4, 2)) and the relu strategy ((8, 2), (2, 1))
    Description: dev_num is 16.
    Expectation: compile success, the first param strategy of the next relu should be ((8, 1))
    """
    case_name = "test_layout_propagation_with_mixed_strategy"
    ir_graph_path = f"./ir/{case_name}"
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation",
                                      device_num=16, global_rank=0)
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)
    set_algo_parameters(fully_use_devices=False)
    layout = Layout((2, 4, 2), ("dp", "mp", "sp"))
    in_layout1 = (layout("dp", "mp"), layout("mp", "sp"))
    in_layout2 = ((8, 2), (2, 1))
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    w1 = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    w2 = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(NetTwoMatMul(w1, w2, in_layout1, in_layout2)))
    compile_net(net, x)
    file = f"{ir_graph_path}/step_parallel_begin_*"
    in_strategy = "in_strategy: ((8, 1))"
    para = "$predict) = PrimFunc_ReLU"
    check_layout_config(para, file, in_strategy)
