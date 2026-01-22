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
from mindspore import context, Tensor, nn
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.graph.api import _cell_graph_executor
from tests.ut.python.ops.test_math_ops import VirtualLoss
from parallel.auto_parallel_interface._utils import init_hccl, set_parallel_mode, remove_files, find_ir_file_path,\
    check_node_pairs_num


def setup_function():
    keyword = 'force_fp32_comm'
    base_dir = './test_auto_parallel'
    remove_files(keyword, base_dir)


def teardown_function():
    keyword = 'force_fp32_comm'
    base_dir = './test_auto_parallel'
    remove_files(keyword, base_dir)


grad_all = C.GradOperation(get_all=True)

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


class Net(nn.Cell):
    def __init__(self, matmul_in_strategy, matmul_out_strategy, mul_strategy):
        super().__init__()
        self.matmul = P.MatMul().shard(matmul_in_strategy, matmul_out_strategy)
        self.mul = P.Mul().shard(mul_strategy)

    def construct(self, x, y, b):
        out = self.matmul(x, y)
        out = self.mul(out, b)
        return out


def test_force_fp32_comm_true():
    """
    Feature: test output strategy for matmul operator, force_fp32_comm is True
    Description: transpose_b is false, set output strategy and use reduce scatter
    Expectation: compile success
    """
    graph_path = './test_auto_parallel/test_force_fp32_comm_true_graphs'
    context.set_context(save_graphs=True, save_graphs_path=graph_path)

    matmul_in_strategy = ((2, 2), (2, 2))
    matmul_out_strategy = ((4, 2),)
    mul_strategy = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(matmul_in_strategy, matmul_out_strategy, mul_strategy)))
    net.set_train()

    # set auto_parallel
    init_hccl(global_rank=0, device_num=8)
    parallel_config = {"parallel_mode": "semi_auto", "force_fp32_communication": True, "dataset_strategy": "full_batch"}
    net = set_parallel_mode(net, parallel_config)

    # compile
    x = Tensor(np.ones([128, 32]), dtype=ms.float16)
    y = Tensor(np.ones([32, 64]), dtype=ms.float16)
    b = Tensor(np.ones([128, 64]), dtype=ms.float16)
    _cell_graph_executor.compile(net, x, y, b)

    # validation
    file_path = find_ir_file_path(graph_path, "validate")
    check_pair = {"PrimFunc_Cast": "2"}
    check_node_pairs_num(file_path, check_pair)


def test_force_fp32_comm_false():
    """
    Feature: test output strategy for matmul operator, force_fp32_comm is False
    Description: transpose_b is false, set output strategy and use reduce scatter
    Expectation: compile success
    """
    graph_path = './test_auto_parallel/test_force_fp32_comm_false_graphs'
    context.set_context(save_graphs=True, save_graphs_path=graph_path)

    matmul_in_strategy = ((2, 2), (2, 2))
    matmul_out_strategy = ((4, 2),)
    mul_strategy = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(matmul_in_strategy, matmul_out_strategy, mul_strategy)))
    net.set_train()

    # set auto_parallel
    init_hccl(global_rank=0, device_num=8)
    parallel_config = {"parallel_mode": "semi_auto", "force_fp32_communication": False,
                       "dataset_strategy": "full_batch"}
    net = set_parallel_mode(net, parallel_config)

    # compile
    x = Tensor(np.ones([128, 32]), dtype=ms.float16)
    y = Tensor(np.ones([32, 64]), dtype=ms.float16)
    b = Tensor(np.ones([128, 64]), dtype=ms.float16)
    _cell_graph_executor.compile(net, x, y, b)

    # validation
    file_path = find_ir_file_path(graph_path, "validate")
    check_pair = {"PrimFunc_Cast": "0"}
    check_node_pairs_num(file_path, check_pair)
