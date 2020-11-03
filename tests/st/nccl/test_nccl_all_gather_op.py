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
x = np.ones([1, 1, 3, 3]).astype(np.float32) * 0.01 * (rank + 1)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.all_gather = P.AllGather(group=NCCL_WORLD_COMM_GROUP)
        self.x = Parameter(initializer(Tensor(x), x.shape), name='x')

    def construct(self):
        return self.all_gather(self.x)


def test_AllGather():
    all_gather = Net()
    output = all_gather()

    expect = np.ones([1, 1, 3, 3]).astype(np.float32) * 0.01 * (0 + 1)
    for i in range(size - 1):
        tmp = np.ones([1, 1, 3, 3]).astype(np.float32) * 0.01 * (i + 2)
        expect = np.concatenate((expect, tmp))
    diff = np.absolute(output.asnumpy() - expect)
    error = np.ones(shape=expect.shape) * 1.0e-5
    assert np.all(diff < error)
    assert output.shape == expect.shape
