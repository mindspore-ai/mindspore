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

import os
import numpy as np

import mindspore as ms
from mindspore import Tensor, Parameter
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.graph.api import _cell_graph_executor
from mindspore.nn.utils import no_init_parameters
from mindspore.common.initializer import initializer
from tests.ut.python.ops.test_math_ops import VirtualLoss
from parallel.auto_parallel_interface._utils import init_hccl, set_parallel_mode, remove_files


def setup_function():
    keyword = 'group_ckpt_save_file'
    base_dir = './test_auto_parallel'
    remove_files(keyword, base_dir)


def teardown_function():
    keyword = 'group_ckpt_save_file'
    base_dir = './test_auto_parallel'
    remove_files(keyword, base_dir)


grad_all = C.GradOperation(get_all=True)

class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x1, x6):
        predict = self.network(x1, x6)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x1, x6):
        return grad_all(self.network)(x1, x6)


class Net(nn.Cell):
    def __init__(self, strategy1, strategy2, strategy3, strategy4, strategy5, strategy6):
        super().__init__()
        self.matmul1 = P.MatMul().shard(strategy1)
        self.matmul2 = P.MatMul().shard(strategy2)
        self.matmul3 = P.MatMul().shard(strategy3)
        self.matmul4 = P.MatMul().shard(strategy4)
        self.matmul5 = P.MatMul().shard(strategy5)
        self.matmul6 = P.MatMul().shard(strategy6)
        self.weight1 = Parameter(initializer("ones", [32, 64], dtype=ms.float32), name="weight1")
        self.weight2 = Parameter(initializer("ones", [64, 64], dtype=ms.float32), name="weight2")
        self.weight3 = Parameter(initializer("ones", [64, 128], dtype=ms.float32), name="weight3")
        self.weight4 = Parameter(initializer("ones", [128, 64], dtype=ms.float32), name="weight4")
        self.weight5 = Parameter(initializer("ones", [64, 128], dtype=ms.float32), name="weight5")
        self.weight6 = Parameter(initializer("ones", [32, 128], dtype=ms.float32), name="weight6")

    def construct(self, x1, x6):
        out = self.matmul1(x1, self.weight1)
        out = self.matmul2(out, self.weight2)
        out = self.matmul3(out, self.weight3)
        out = self.matmul4(out, self.weight4)
        out = self.matmul5(out, self.weight5)
        out = out + self.weight6
        out = self.matmul6(out, x6)
        return out


def test_six_matmul_group_ckpt_save_file_true():
    """
    Feature: test group_ckpt_save_file is set
    Description: group_stage1.ckpt is saved
    Expectation: compile success
    """
    file_path = "./test_auto_parallel/test_group_ckpt_save_file_true/"
    save_strategy_file_path = f"{file_path}/strategy_stage1.ckpt"
    group_ckpt_save_file = f"{file_path}/group_stage1.ckpt"

    strategy1 = ((8, 1), (1, 1))
    strategy2 = ((1, 8), (8, 1))
    strategy3 = ((2, 2), (2, 2))
    strategy4 = ((1, 1), (1, 8))
    strategy5 = ((4, 2), (2, 1))
    strategy6 = ((4, 1), (1, 2))
    with no_init_parameters():
        net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3, strategy4, strategy5, strategy6)))
    x1 = Tensor(np.ones([32, 32]), dtype=ms.float32)
    x6 = Tensor(np.ones([128, 32]), dtype=ms.float32)
    net.set_train()

    # set auto_parallel
    init_hccl(global_rank=0, device_num=8)
    parallel_config = {"parallel_mode": "semi_auto", "save_strategy_file_path": save_strategy_file_path,
                       "group_ckpt_save_file": group_ckpt_save_file, "dataset_strategy": "full_batch"}
    net = set_parallel_mode(net, parallel_config)
    _cell_graph_executor.compile(net, x1, x6)

    # validation
    assert os.path.exists(group_ckpt_save_file) and os.path.isfile(group_ckpt_save_file)


def test_six_matmul_not_group_ckpt_save_file_false():
    """
    Feature: test group_ckpt_save_file is ""
    Description: group_stage1.ckpt does not saved
    Expectation: compile success
    """
    file_path = "./test_auto_parallel/test_group_ckpt_save_file_false/"
    save_strategy_file_path = f"{file_path}/strategy_stage1.ckpt"
    group_ckpt_save_file = ""

    strategy1 = ((8, 1), (1, 1))
    strategy2 = ((1, 8), (8, 1))
    strategy3 = ((2, 2), (2, 2))
    strategy4 = ((1, 1), (1, 8))
    strategy5 = ((4, 2), (2, 1))
    strategy6 = ((4, 1), (1, 2))
    with no_init_parameters():
        net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3, strategy4, strategy5, strategy6)))
    x1 = Tensor(np.ones([32, 32]), dtype=ms.float32)
    x6 = Tensor(np.ones([128, 32]), dtype=ms.float32)
    net.set_train()

    # set auto_parallel
    init_hccl(global_rank=0, device_num=8)
    parallel_config = {"parallel_mode": "semi_auto", "save_strategy_file_path": save_strategy_file_path,
                       "group_ckpt_save_file": group_ckpt_save_file, "dataset_strategy": "full_batch"}
    net = set_parallel_mode(net, parallel_config)
    _cell_graph_executor.compile(net, x1, x6)

    # validation
    assert os.path.exists(group_ckpt_save_file) is False
