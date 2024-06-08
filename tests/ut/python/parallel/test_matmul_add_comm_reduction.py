# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
import subprocess
import shutil
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, w1, w2, b1, b2):
        predict = self.network(x, w1, w2, b1, b2)
        return self.loss(predict)


def compile_net(net, x, w1, w2, b1, b2):
    net.set_train()
    _cell_graph_executor.compile(net, x, w1, w2, b1, b2)


grad_all = C.GradOperation(get_all=True)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, w1, w2, b1, b2):
        return grad_all(self.network)(x, w1, w2, b1, b2)


def test_matmul_add_comm_reduction():
    """
    Feature: test matmul add comm reduction
    Description: change structure like (matmul1+ ... + allreduce) + (matmul2 + ... + allreduce) to
                (matmul1 + ... + matmul2) + allreduce
    Expectation: compile success
    """

    class Net(nn.Cell):
        def __init__(self, matmul_in_strategy, bias_add_strategy, add_strategy):
            super().__init__()
            self.matmul1 = P.MatMul().shard(matmul_in_strategy)
            self.matmul1.add_prim_attr("matmul_add_comm_begin", True)
            self.add1 = P.Add().shard(bias_add_strategy)
            self.matmul2 = P.MatMul().shard(matmul_in_strategy)
            self.matmul2.add_prim_attr("matmul_add_comm_begin", True)
            self.add2 = P.Add().shard(bias_add_strategy)
            self.add3 = P.Add().shard(add_strategy)
            self.add3.add_prim_attr("matmul_add_comm_end", True)

        def construct(self, x, w1, w2, b1, b2):
            mm_out1 = self.matmul1(x, w1)
            add_out1 = self.add1(mm_out1, b1)
            mm_out2 = self.matmul2(x, w2)
            add_out2 = self.add2(mm_out2, b2)
            out = self.add3(add_out1, add_out2)
            return out

    context.set_auto_parallel_context(
        device_num=8, global_rank=0)
    context.set_context(save_graphs=True, save_graphs_path="./graph_comm_reduction")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    matmul_in_strategy = ((2, 4), (4, 1))
    bias_add_strategy = ((8, 1), (1,))
    add_strategy = ((8, 1), (8, 1))
    net = GradWrap(NetWithLoss(Net(matmul_in_strategy, bias_add_strategy, add_strategy)))

    x = Tensor(np.ones([128, 32]), dtype=ms.float16)
    w1 = Tensor(np.ones([32, 64]), dtype=ms.float16)
    w2 = Tensor(np.ones([32, 64]), dtype=ms.float16)
    b1 = Tensor(np.ones([64]), dtype=ms.float16)
    b2 = Tensor(np.ones([64]), dtype=ms.float16)
    if os.path.exists("./graph_comm_reduction/rank_0"):
        shutil.rmtree("./graph_comm_reduction/rank_0")
    # compile
    compile_net(net, x, w1, w2, b1, b2)

    file = "./graph_comm_reduction/rank_0/*validate*.ir"
    prim_name = "AllReduce("
    tag_name = "comm_reduction"
    output = subprocess.check_output(
        ["grep -r '%s' %s | grep '%s' |wc -l" % (prim_name, file, tag_name)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "1"
    if os.path.exists("./graph_comm_reduction/rank_0"):
        shutil.rmtree("./graph_comm_reduction/rank_0")
    context.set_context(save_graphs=False)
