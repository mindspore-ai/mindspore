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
import json
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

    def construct(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


def compile_net(net, x, y, b):
    net.set_train()
    _cell_graph_executor.compile(net, x, y, b)


grad_all = C.GradOperation(get_all=True)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return grad_all(self.network)(x, y, b)


def test_bias_add_comm_swap():
    """
    Feature: test bias_add comm swap
    Description: change structure like matmul+allreduce/reduce_scatter+bias to matmul+bias+allreduce/reduce_scatter
    Expectation: compile success
    """

    class Net(nn.Cell):
        def __init__(self, matmul_in_strategy, matmul_out_strategy, add_strategy):
            super().__init__()
            self.matmul = P.MatMul().shard(matmul_in_strategy, matmul_out_strategy)
            self.add = P.Add().shard(add_strategy)

        def construct(self, x, w, b):
            out = self.matmul(x, w)
            out = self.add(out, b)
            return out

    context.set_auto_parallel_context(
        device_num=8, global_rank=0)
    context.set_context(save_graphs=True, save_graphs_path="./")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    if os.path.exists("./speed_up.json"):
        os.remove("./speed_up.json")
    a = {"bias_add_comm_swap": True}
    f = open("speed_up.json", "w")
    f.write(json.dumps(a))
    f.close()
    context.set_context(ascend_config={"parallel_speed_up_json_path": "speed_up.json"})

    matmul_in_strategy = ((2, 2), (2, 1))
    matmul_out_strategy = ((4, 1),)
    add_strategy = ((4, 1), (1,))
    net = GradWrap(NetWithLoss(Net(matmul_in_strategy, matmul_out_strategy, add_strategy)))

    x = Tensor(np.ones([128, 32]), dtype=ms.float16)
    w = Tensor(np.ones([32, 64]), dtype=ms.float16)
    b = Tensor(np.ones([64]), dtype=ms.float16)
    if os.path.exists("./rank_0"):
        shutil.rmtree("./rank_0")
    # compile
    compile_net(net, x, w, b)

    file = "./rank_0/*validate*.ir"
    prim_name = "ReduceScatter"
    para = "bias_add_comm_swap"
    output = subprocess.check_output(
        ["grep -r '%s' %s |grep '%s' | wc -l" % (prim_name, file, para)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "1"
    if os.path.exists("./rank_0"):
        shutil.rmtree("./rank_0")
    if os.path.exists("./speed_up.json"):
        os.remove("./speed_up.json")

    # clean env
    a = {"bias_add_comm_swap": False}
    f = open("speed_up.json", "w")
    f.write(json.dumps(a))
    f.close()
    context.set_context(ascend_config={"parallel_speed_up_json_path": "speed_up.json"})
    if os.path.exists("./speed_up.json"):
        os.remove("./speed_up.json")
    context.set_context(save_graphs=False)
