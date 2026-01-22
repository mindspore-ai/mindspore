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
# limitations under the License

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.graph.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.nn.utils import no_init_parameters
from mindspore.common.initializer import initializer
from tests.ut.python.ops.test_math_ops import VirtualLoss
from parallel.auto_parallel_interface._utils import init_hccl, set_parallel_mode, remove_files, find_ir_file_path,\
    check_node_dependency_backward_search


def setup_function():
    keyword = 'gradient_fp32_sync'
    base_dir = './test_auto_parallel'
    remove_files(keyword, base_dir)


def teardown_function():
    keyword = 'gradient_fp32_sync'
    base_dir = './test_auto_parallel'
    remove_files(keyword, base_dir)


grad_all = C.GradOperation(get_all=True)

class Net(nn.Cell):
    def __init__(self, strategy1):
        super().__init__()
        self.matmul = P.MatMul().shard(strategy1)
        self.cast = P.Cast()
        self.y = ms.Parameter(initializer("ones", [32, 64], dtype=ms.float16), "y")
        self.b = ms.Parameter(initializer("ones", [64, 64], dtype=ms.float32), "b")

    def construct(self, x, y, b):
        out = self.matmul(x, self.y)
        b = self.cast(self.b, ms.float16)
        out = self.matmul(out, b)
        return out


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return grad_all(self.network)(x, y, b)


def test_gradient_fp32_sync_true():
    """
    Feature: test gradient_fp32_sync is True
    Description: PrimFunc_Cast appears after _MirrorOperator in step_parallel_end.ir
    Expectation: compile success
    """
    graph_path = "./test_auto_parallel/test_gradient_fp32_sync_true_graphs"
    context.set_context(save_graphs=True, save_graphs_path=graph_path)

    strategy1 = ((2, 2), (2, 2))
    with no_init_parameters():
        net = GradWrap(NetWithLoss(Net(strategy1)))
    net.set_train()

    # set auto_parallel
    init_hccl(global_rank=0, device_num=8)
    parallel_config = {"parallel_mode": "semi_auto", "dataset_strategy": "full_batch", "gradient_fp32_sync": True}
    net = set_parallel_mode(net, parallel_config)

    # compile_net
    x = Tensor(np.ones([128, 32]), dtype=ms.float16)
    y = Tensor(np.ones([32, 64]), dtype=ms.float16)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    _cell_graph_executor.compile(net, x, y, b)

    # validation
    parm1_dependency_list = ['PrimFunc_Cast', 0, 0, '_MirrorOperator']
    step_parallel_end_path = find_ir_file_path(graph_path, "step_parallel_end")
    check_node_dependency_backward_search(step_parallel_end_path, 50, parm1_dependency_list)


def test_gradient_fp32_sync_false():
    """
    Feature: test gradient_fp32_sync is False
    Description: PrimFunc_Cast appears before _MirrorOperator in step_parallel_end.ir
    Expectation: compile success
    """
    graph_path = "./test_auto_parallel/test_gradient_fp32_sync_false_graphs"
    context.set_context(save_graphs=True, save_graphs_path=graph_path)

    strategy1 = ((2, 2), (2, 2))
    with no_init_parameters():
        net = GradWrap(NetWithLoss(Net(strategy1)))
    net.set_train()

    # set auto_parallel
    init_hccl(global_rank=0, device_num=8)
    parallel_config = {"parallel_mode": "semi_auto", "dataset_strategy": "full_batch", "gradient_fp32_sync": False}
    net = set_parallel_mode(net, parallel_config)

    # compile_net
    x = Tensor(np.ones([128, 32]), dtype=ms.float16)
    y = Tensor(np.ones([32, 64]), dtype=ms.float16)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    _cell_graph_executor.compile(net, x, y, b)

    # validation
    parm1_dependency_list = ['_MirrorOperator', 0, 'PrimFunc_Cast']
    step_parallel_end_path = find_ir_file_path(graph_path, "step_parallel_end")
    check_node_dependency_backward_search(step_parallel_end_path, 50, parm1_dependency_list)
