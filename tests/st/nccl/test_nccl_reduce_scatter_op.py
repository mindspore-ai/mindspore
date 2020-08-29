# Copyright 2019 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.communication.management import init, NCCL_WORLD_COMM_GROUP, get_rank, get_group_size
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

init()
rank = get_rank()
size = get_group_size()
x = np.ones([size, 1, 3, 3]).astype(np.float32) * 0.01 * (rank + 1)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.x = Parameter(initializer(Tensor(x), x.shape), name='x')

        self.op0 = "sum"
        self.op1 = "max"
        self.op2 = "min"
        self.op3 = "prod"

        self.reduce_scatter1 = P.ReduceScatter(self.op0, group=NCCL_WORLD_COMM_GROUP)
        self.reduce_scatter2 = P.ReduceScatter(self.op1, group=NCCL_WORLD_COMM_GROUP)
        self.reduce_scatter3 = P.ReduceScatter(self.op2, group=NCCL_WORLD_COMM_GROUP)

    def construct(self):
        return (self.reduce_scatter1(self.x),
                self.reduce_scatter2(self.x),
                self.reduce_scatter3(self.x))


def test_ReduceScatter():
    reduce_scatter = Net()
    output = reduce_scatter()

    sum_ones = np.ones([size, 1, 3, 3]).astype(np.float32) * 0
    for i in range(size):
        sum_ones += np.ones([size, 1, 3, 3]).astype(np.float32) * 0.01 * (i + 1)
    expect0 = sum_ones[rank: rank + 1]
    diff0 = output[0].asnumpy() - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output[0].shape == expect0.shape

    expect1 = np.ones([1, 1, 3, 3]).astype(np.float32) * 0.01 * size
    diff1 = output[1].asnumpy() - expect1
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output[1].shape == expect1.shape

    expect2 = np.ones([1, 1, 3, 3]).astype(np.float32) * 0.01 * 1
    diff2 = output[2].asnumpy() - expect2
    error2 = np.ones(shape=expect2.shape) * 1.0e-5
    assert np.all(diff2 < error2)
    assert output[2].shape == expect2.shape
