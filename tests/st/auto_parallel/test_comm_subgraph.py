# Copyright 2022 Huawei Technologies Co., Ltd
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
import time
from multiprocessing import Process, Queue
import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.communication.management import init
from mindspore.ops import operations as P
from mindspore.context import ParallelMode


MINDSPORE_HCCL_CONFIG_PATH = "/home/workspace/mindspore_config/hccl/rank_table_8p.json"


class DenseMatMulNet(nn.Cell):
    def __init__(self):
        super(DenseMatMulNet, self).__init__()
        self.matmuls = []
        for _ in range(10):
            self.matmuls.append(P.MatMul().shard(((8, 1), (1, 1))))

    def construct(self, x):
        res = x
        for i in range(10):
            res = self.matmuls[i](x, res)
        return res


def compute_process(q, device_id, device_num, enable_comm_subgraph):
    os.system("mkdir " + str(device_id))
    os.chdir(str(device_id))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(device_id=device_id)
    os.environ['MINDSPORE_HCCL_CONFIG_PATH'] = MINDSPORE_HCCL_CONFIG_PATH
    os.environ['RANK_ID'] = str(device_id)
    os.environ['RANK_SIZE'] = str(device_num)
    if enable_comm_subgraph:
        os.environ['MS_COMM_COMPILER_OPT'] = '-1'
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, device_num=device_num)
    init()
    net = DenseMatMulNet()
    np.random.seed(1)
    x = Tensor(np.random.rand(2, 16).astype(np.float32))
    res = net(x)
    q.put(res.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_comm_subgraph_8p():
    '''
    Feature: extract communication subgraphs and reuse them to replace original communication ops under GRAPH mode
    Description: Test a net that consists of 10 sharded matmul ops
    Expectation: Run success; results before and after enabling this feature should be the same
    '''
    q = Queue()
    device_num = 8
    enable_comm_subgraph = False
    process = []
    for i in range(device_num):
        device_id = i
        process.append(Process(target=compute_process,
                               args=(q, device_id, device_num, enable_comm_subgraph)))

    for i in range(device_num):
        process[i].start()

    print("Waiting for all subprocesses done...")

    for i in range(device_num):
        process[i].join()

    res0 = q.get()
    for i in range(device_num):
        os.system("rm -rf " + str(i))

    time.sleep(10)
    print("End computing...")

    q = Queue()
    enable_comm_subgraph = True
    process = []
    for i in range(device_num):
        device_id = i
        process.append(Process(target=compute_process,
                               args=(q, device_id, device_num, enable_comm_subgraph)))

    for i in range(device_num):
        process[i].start()

    print("Waiting for all subprocesses done...")

    for i in range(device_num):
        process[i].join()

    res1 = q.get()
    for i in range(device_num):
        os.system("rm -rf " + str(i))
    print("End computing...")
    assert res0.all() == res1.all()
