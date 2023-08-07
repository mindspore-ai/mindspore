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

import os
import time
import numpy as np
import mindspore.nn as nn
from mindspore.ops.operations import _inner_ops as inner_p
from mindspore.communication.management import init, get_rank

# 'Barrier' operator only supports KernelByKernel mode by now. So we set 'GRAPH_OP_RUN' to 1.
np.random.seed(1)
os.environ['GRAPH_OP_RUN'] = str(1)
init()
rank = get_rank()

class BarrierNet(nn.Cell):
    def __init__(self):
        super(BarrierNet, self).__init__()
        self.barrier = inner_p.Barrier()

    def construct(self):
        self.barrier()

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
