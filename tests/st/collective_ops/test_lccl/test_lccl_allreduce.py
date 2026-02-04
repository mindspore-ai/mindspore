# Copyright 2024 Huawei Technologies Co., Ltd
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
"""
test lccl allreduce with 8p
"""

import numpy as np

import mindspore as ms
from mindspore import Tensor, nn
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.communication.management import init, HCCL_WORLD_COMM_GROUP, get_rank, get_group_size
from mindspore.ops import operations as P

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", jit_config={"jit_level": "O0", "infer_boost": "on"})

init()
rank = get_rank()
size = get_group_size()
x = np.ones([3, 1, 3, 3]).astype(np.float32) * 0.01 * (rank + 1)

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.x1 = Parameter(initializer(Tensor(x), x.shape), name='x1')
        self.x2 = Parameter(initializer(Tensor(x), x.shape), name='x2')
        self.x3 = Parameter(initializer(Tensor(x), x.shape), name='x3')

        self.op0 = "sum"
        self.op1 = "sum"
        self.op2 = "sum"

        self.all_reduce1 = P.AllReduce(self.op0, group=HCCL_WORLD_COMM_GROUP)
        self.all_reduce2 = P.AllReduce(self.op1, group=HCCL_WORLD_COMM_GROUP)
        self.all_reduce3 = P.AllReduce(self.op2, group=HCCL_WORLD_COMM_GROUP)

    def construct(self):
        return (self.all_reduce1(self.x1),
                self.all_reduce2(self.x2),
                self.all_reduce3(self.x3))


def test_AllReduce():
    """
    Feature: lccl operator test.
    Description: msrun lccl all_reduce 8P case.
    Expectation: success
    """
    all_reduce = Net()
    output = all_reduce()

    expect0 = np.ones([3, 1, 3, 3]).astype(np.float32) * 0
    for i in range(size):
        part = np.ones([3, 1, 3, 3]).astype(np.float32) * 0.01 * (i + 1)
        expect0 += part
    diff0 = output[0].asnumpy() - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output[0].shape == expect0.shape

    expect1 = expect0
    diff1 = output[1].asnumpy() - expect1
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output[1].shape == expect1.shape

    expect2 = expect1
    diff2 = output[2].asnumpy() - expect2
    error2 = np.ones(shape=expect2.shape) * 1.0e-5
    assert np.all(diff2 < error2)
    assert output[2].shape == expect2.shape
