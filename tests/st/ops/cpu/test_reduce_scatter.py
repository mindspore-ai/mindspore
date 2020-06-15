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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
import mindspore._ms_mpi as mpi
# run comand:
# mpirun -output-filename log -merge-stderr-to-stdout -np 3 python test_reduce_scatter.py

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
context.set_mpi_config(enable_mpi=True)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.op = "sum"

        self.reducescatter = P.HostReduceScatter(op=self.op, group=[0,1,2])

    def construct(self, x):
        return self.reducescatter(x)

class AllGatherNet(nn.Cell):
    def __init__(self):
        super(AllGatherNet, self).__init__()
        self.hostallgather = P.HostAllGather(group=(0, 1, 2))

    def construct(self, x):
        return self.hostallgather(x)  

def test_net_reduce_scatter():
    x = np.arange(12).astype(np.float32) * 0.1
    
    reducescatter = Net()
    rankid = mpi.get_rank_id()
    print("self rankid:", rankid)
    output = reducescatter(Tensor(x, mstype.float32))
    print("output:\n", output)
    if rankid == 0:
        expect_result = np.arange(4).astype(np.float32) * 0.3
    if rankid == 1:
        expect_result = np.arange(4, 8).astype(np.float32) * 0.3
    if rankid == 2:
        expect_result = np.arange(8, 12).astype(np.float32) * 0.3
    diff = abs(output.asnumpy() - expect_result)
    error = np.ones(shape=expect_result.shape) * 1.0e-6
    assert np.all(diff < error)

    allgather = AllGatherNet()
    allgather_output = allgather(output)
    print("allgather result:\n", allgather_output)
    expect_allgather_result =  np.arange(12).astype(np.float32) * 0.3
    diff = abs(allgather_output.asnumpy() - expect_allgather_result)
    error = np.ones(shape=expect_allgather_result.shape) * 1.0e-6
    assert np.all(diff < error)

if __name__ == '__main__':
    test_net_reduce_scatter()
