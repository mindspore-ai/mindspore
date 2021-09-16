# Copyright 2020 Huawei Technologies Co., Ltd
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

"""test bert thor performance with 8p on mlperf dataset"""

import os
from multiprocessing import Process, Queue
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import operations as P
import mindspore.communication.management as D
from mindspore import context
from mindspore.context import ParallelMode

MINDSPORE_HCCL_CONFIG_PATH = "/home/workspace/mindspore_config/hccl/rank_table_8p.json"

np.random.seed(1)
os.environ['GLOG_v'] = str(2)

class AllReduceNet(nn.Cell):
    def __init__(self):
        super(AllReduceNet, self).__init__()
        self.all_reduce = P.AllReduce()

    def construct(self, x):
        return self.all_reduce(x)

def train_allreduce_8p(q, device_id, device_num):
    os.system("mkdir " + str(device_id))
    os.chdir(str(device_id))
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=device_id)
    os.environ['MINDSPORE_HCCL_CONFIG_PATH'] = MINDSPORE_HCCL_CONFIG_PATH
    os.environ['RANK_ID'] = str(device_id)
    os.environ['RANK_SIZE'] = str(device_num)
    D.init()
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                      device_num=device_num)

    net = AllReduceNet()
    input_x = np.ones([32, 255, 255, 3]).astype(np.float32)
    except_output = input_x * 8
    output = net(Tensor(input_x, mstype.float32))
    q.put(np.allclose(output.asnumpy(), except_output))

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_pynative_hccl_8p():
    device_num = 8
    process = []
    q = Queue()
    for i in range(device_num):
        device_id = i
        process.append(Process(target=train_allreduce_8p, args=(q, device_id, device_num)))

    for i in range(device_num):
        process[i].start()

    print("Waiting for all subprocesses done...")

    for i in range(device_num):
        process[i].join()

    # check result
    for i in range(device_num):
        assert not q.empty()
        assert q.get()

    for i in range(device_num):
        os.system("rm -rf " + str(i))

    print("End training...")

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_pynative_hccl_8pv2():
    os.environ['GRAPH_OP_RUN'] = str(1)
    device_num = 8
    process = []
    q = Queue()
    for i in range(device_num):
        device_id = i
        process.append(Process(target=train_allreduce_8p, args=(q, device_id, device_num)))

    for i in range(device_num):
        process[i].start()

    print("Waiting for all subprocesses done...")

    for i in range(device_num):
        process[i].join()

    # check result
    for i in range(device_num):
        assert not q.empty()
        assert q.get()

    for i in range(device_num):
        os.system("rm -rf " + str(i))

    print("End training...")
