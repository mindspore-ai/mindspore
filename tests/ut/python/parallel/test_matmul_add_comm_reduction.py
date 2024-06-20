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
from mindspore import Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch", parallel_mode="semi_auto_parallel")
    context.set_context(save_graphs=True, save_graphs_path="./graph_comm_reduction")


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


def check_output(num_comm_ops=1):
    file = "./graph_comm_reduction/rank_0/*validate*.ir"
    prim_name = "AllReduce("
    tag_name = "comm_reduction"
    output = subprocess.check_output(
        ["grep -r '%s' %s | grep '%s' |wc -l" % (prim_name, file, tag_name)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == str(num_comm_ops)


def test_matmul_add_comm_reduction_normal():
    """
    Feature: test matmul add comm reduction
    Description: change structure like (matmul1 -> allreduce + ... ) + (matmul2 -> allreduce + ... ) to
                (matmul1 + ...) + (matmul2 + ...) -> allreduce
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

    matmul_in_strategy = ((2, 4), (4, 1))
    bias_add_strategy = ((2, 1), (1,))
    add_strategy = ((2, 1), (2, 1))
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
    check_output()
    context.set_context(save_graphs=False)


def test_matmul_add_comm_reduction_two_matmul_left():
    """
    Feature: test matmul add comm reduction
    Description: do not change structure like (matmul1 -> allreduce + ... + matmul2 -> allreduce + ...) \
                 + (matmul3 -> allreduce + ... )
    Expectation: compile success
    """

    class Net(nn.Cell):
        def __init__(self, matmul_in_strategy, bias_add_strategy, add_strategy, mm_weight_shape):
            super().__init__()
            self.matmul1 = P.MatMul().shard(matmul_in_strategy)
            self.matmul1.add_prim_attr("matmul_add_comm_begin", True)
            self.add1 = P.Add().shard(bias_add_strategy)
            self.matmul2 = P.MatMul().shard(matmul_in_strategy)
            self.matmul2.add_prim_attr("matmul_add_comm_begin", False)
            self.matmul3 = P.MatMul().shard(matmul_in_strategy)
            self.matmul3.add_prim_attr("matmul_add_comm_begin", True)
            self.add2 = P.Add().shard(bias_add_strategy)
            self.add3 = P.Add().shard(add_strategy)
            self.add3.add_prim_attr("matmul_add_comm_end", True)
            self.mm_weight = Parameter(Tensor(np.ones(mm_weight_shape), dtype=ms.float16))

        def construct(self, x, w1, w2, b1, b2):
            mm_out1 = self.matmul1(x, w1)
            add_out1 = self.add1(mm_out1, b1)
            mm_out2 = self.matmul2(add_out1, self.mm_weight)
            mm_out3 = self.matmul2(x, w2)
            add_out3 = self.add2(mm_out3, b2)
            out = self.add3(mm_out2, add_out3)
            return out

    context.set_auto_parallel_context(
        device_num=8, global_rank=0)

    matmul_in_strategy = ((2, 4), (4, 1))
    bias_add_strategy = ((2, 1), (1,))
    add_strategy = ((2, 1), (2, 1))
    mm_weight_shape = [64, 64]
    net = GradWrap(
        NetWithLoss(Net(matmul_in_strategy, bias_add_strategy, add_strategy, mm_weight_shape=mm_weight_shape)))

    x = Tensor(np.ones([128, 32]), dtype=ms.float16)
    w1 = Tensor(np.ones([32, 64]), dtype=ms.float16)
    w2 = Tensor(np.ones([32, 64]), dtype=ms.float16)
    b1 = Tensor(np.ones([64]), dtype=ms.float16)
    b2 = Tensor(np.ones([64]), dtype=ms.float16)
    if os.path.exists("./graph_comm_reduction/rank_0"):
        shutil.rmtree("./graph_comm_reduction/rank_0")
    # compile
    compile_net(net, x, w1, w2, b1, b2)
    check_output(0)
    context.set_context(save_graphs=False)


def test_matmul_add_comm_reduction_one_matmul_one_batch_matmul():
    """
    Feature: test matmul add comm reduction
    Description: change structure like (matmul1 -> allreduce + ... ) + (batch_matmul -> allreduce +  ... ) to
                (matmul1 + ...) + (matmul2 + ...) -> allreduce
    Expectation: compile success
    """

    class Net(nn.Cell):
        def __init__(self, matmul_in_strategy, batch_matmul_in_strategy, bias_add_strategy, add_strategy):
            super().__init__()
            self.matmul1 = P.MatMul().shard(matmul_in_strategy)
            self.matmul1.add_prim_attr("matmul_add_comm_begin", True)
            self.add1 = P.Add().shard(bias_add_strategy)

            self.matmul2 = P.BatchMatMul(transpose_b=True).shard(batch_matmul_in_strategy)
            self.matmul2.add_prim_attr("matmul_add_comm_begin", True)
            self.matmul3 = P.BatchMatMul()

            self.add3 = P.Add().shard(add_strategy)
            self.add3.add_prim_attr("matmul_add_comm_end", True)
            self.reshape = P.Reshape()
            self.shape = P.Shape()
            self.seq_len = 8

        def construct(self, x, w1, w2, b1, w3):
            mm_out1 = self.matmul1(x, w1)
            add_out1 = self.add1(mm_out1, b1)
            bs, _ = self.shape(x)
            reshape_x = self.reshape(x, (bs, self.seq_len, -1))
            mm_out2 = self.matmul2(reshape_x, w2)
            mm_out3 = self.matmul3(mm_out2, w3)
            bmm_out = self.reshape(mm_out3, (bs, -1))
            out = self.add3(add_out1, bmm_out)
            return out

    context.set_auto_parallel_context(
        device_num=8, global_rank=0)

    mp = 8
    dp = 1
    matmul_in_strategy = ((dp, mp), (mp, 1))
    bias_add_strategy = ((dp, 1), (1,))
    add_strategy = ((dp, 1), (dp, 1))
    bmm_in_strategy = ((dp, 1, mp), (dp, 1, mp))
    net = Net(matmul_in_strategy, bmm_in_strategy, bias_add_strategy, add_strategy)
    net.matmul3.shard(((dp, 1, 1), (dp, 1, 1)))
    grad_net = GradWrap(NetWithLoss(net))

    x = Tensor(np.ones([128, 128]), dtype=ms.float16)
    w1 = Tensor(np.ones([128, 64]), dtype=ms.float16)
    w2 = Tensor(np.ones([128, 8, 16]), dtype=ms.float16)
    b1 = Tensor(np.ones([64]), dtype=ms.float16)
    w3 = Tensor(np.ones([128, 8, 8]), dtype=ms.float16)
    if os.path.exists("./graph_comm_reduction/rank_0"):
        shutil.rmtree("./graph_comm_reduction/rank_0")
    # compile
    compile_net(grad_net, x, w1, w2, b1, w3)
    check_output()
    context.set_context(save_graphs=False)
