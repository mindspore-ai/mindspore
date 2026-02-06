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
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

init()
rank = get_rank()
size = get_group_size()

x = np.asarray([1, 1, 1, 1, 1, 1, 1, 1]).astype(np.float32) * (rank + 1)
x1 = np.asarray([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.float32)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.alltoall = P.comm_ops.AlltoAll(split_count=8, split_dim=0, concat_dim=0)

    def construct(self, inputs):
        return self.alltoall(inputs)


def test_AlltoAll():
    alltoall = Net()
    expect0 = np.asarray([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.float32)
    output0 = alltoall(Tensor(x)).asnumpy()
    diff0 = output0 - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape

    expect1 = np.asarray([1, 1, 1, 1, 1, 1, 1, 1]).astype(np.float32) * (rank + 1)
    output1 = alltoall(Tensor(x1)).asnumpy()
    diff1 = output1  - expect1
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape
