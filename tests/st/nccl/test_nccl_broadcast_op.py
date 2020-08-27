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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

init()
rank = get_rank()
size = get_group_size()
x = np.ones([3, 1, 3, 3]).astype(np.float32) * 0.01 * (rank + 1)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.x1 = Parameter(initializer(Tensor(x), x.shape), name='x1')
        self.x2 = Parameter(initializer(Tensor(x), x.shape), name='x2')
        self.x3 = Parameter(initializer(Tensor(x), x.shape), name='x3')

        self.broadcast1 = P.Broadcast(0)
        self.broadcast2 = P.Broadcast(1)
        self.broadcast3 = P.Broadcast(2)

    def construct(self):
        return (self.broadcast1((self.x1,)),
                self.broadcast2((self.x2,)),
                self.broadcast3((self.x3,)))


def test_Broadcast():
    broadcast = Net()
    output = broadcast()

    expect0 = np.ones([3, 1, 3, 3]).astype(np.float32) * 1
    expect1 = np.ones([3, 1, 3, 3]).astype(np.float32) * 2
    expect2 = np.ones([3, 1, 3, 3]).astype(np.float32) * 3

    diff0 = output[0][0].asnumpy() - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output[0][0].shape == expect0.shape

    diff1 = output[1][0].asnumpy() - expect1
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output[1][0].shape == expect1.shape

    diff2 = output[2][0].asnumpy() - expect2
    error2 = np.ones(shape=expect2.shape) * 1.0e-5
    assert np.all(diff2 < error2)
    assert output[2][0].shape == expect2.shape
