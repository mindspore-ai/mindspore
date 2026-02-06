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
from mindspore.communication.management import init, NCCL_WORLD_COMM_GROUP, get_rank, get_group_size
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

init()
rank = get_rank()
size = get_group_size()
if size % 2 != 0:
    raise RuntimeError("Group size should be divided by 2 exactly.")
x = np.ones([3, 3, 3, 3]).astype(np.float32) * 0.01 * (rank + 1)


class SendNet(nn.Cell):
    def __init__(self):
        super(SendNet, self).__init__()
        self.x = Parameter(initializer(Tensor(x), x.shape), name='x')
        self.depend = P.Depend()
        self.send = P.Send(sr_tag=0, dest_rank=rank+size//2, group=NCCL_WORLD_COMM_GROUP)

    def construct(self):
        out = self.depend(self.x, self.send(self.x))
        return out

class RecvNet(nn.Cell):
    def __init__(self):
        super(RecvNet, self).__init__()
        self.recv = P.Receive(sr_tag=0, src_rank=rank-size//2, shape=[3, 3, 3, 3], dtype=mstype.float32,
                              group=NCCL_WORLD_COMM_GROUP)

    def construct(self):
        out = self.recv()
        return out

def test_send_recv():
    if rank < size / 2:
        send_net = SendNet()
        output = send_net()
    else:
        expect_output = np.ones([3, 3, 3, 3]).astype(np.float32) * 0.01 * (rank-size//2 + 1)
        recv_net = RecvNet()
        output = recv_net()

        diff = abs(output.asnumpy() - expect_output)
        error = np.ones(shape=output.shape) * 1.0e-5
        assert np.all(diff < error)
        assert expect_output.shape == output.shape
