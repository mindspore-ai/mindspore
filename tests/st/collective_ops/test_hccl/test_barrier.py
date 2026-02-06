# Copyright 2021 Huawei Technologies Co., Ltd
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

"""test hccl reduce with 8p"""

import time
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops.operations import comm_ops
from mindspore.communication.comm_func import barrier
from mindspore.communication.management import init, get_rank

# 'Barrier' operator only supports KernelByKernel mode by now.
np.random.seed(1)
ms.set_context(jit_level='O0')
init()
rank = get_rank()

class BarrierNet(nn.Cell):
    def __init__(self):
        super(BarrierNet, self).__init__()
        self.barrier = comm_ops.Barrier()

    def construct(self):
        self.barrier()

class BarrierFuncNet(nn.Cell):
    def construct(self):
        barrier()

def test_hccl_barrier_8p():
    """
    Feature: test 'Barrier' collective communication operator.
    Description: test 'Barrier' collective communication operator.
    Expectation: all processes in the group synchronize in this operator.
    """
    net = BarrierNet()
    if rank == 3:
        time.sleep(3)
    if rank == 4:
        time.sleep(6)
    print("Process {} start time: {}".format(rank, time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))))
    net()
    print("Process {} end time: {}".format(rank, time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))))

def test_hccl_barrier_func_8p():
    """
    Feature: test 'Barrier' collective communication operator.
    Description: test 'Barrier' collective communication operator.
    Expectation: all processes in the group synchronize in this operator.
    """
    net = BarrierFuncNet()
    start_time = time.time()
    print("Process {} start time: {}".format(rank, time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(start_time))))
    if rank == 3:
        time.sleep(3)
    if rank == 4:
        time.sleep(6)
    net()
    end_time = time.time()
    print("Process {} end time: {}".format(rank, time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(end_time))))
    assert end_time - start_time > 4
