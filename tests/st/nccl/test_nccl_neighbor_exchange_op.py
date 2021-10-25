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
import mindspore as ms

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

init()
rank = get_rank()
size = get_group_size()

x = np.asarray([1, 1, 1, 1, 1, 1, 1, 1]).astype(np.float32) * rank


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.neighborexchange = P.comm_ops.NeighborExchange(
            send_rank_ids=[(rank - 1) % 8],
            recv_rank_ids=[(rank + 1) % 8],
            recv_shapes=tuple([[8]]),
            send_shapes=tuple([[8]]),
            recv_type=ms.float32,
            group="nccl_world_group")

    def construct(self, inputs):
        return self.neighborexchange(inputs)


def test_neighborexchange():
    """
    Feature: NeighborExchange operator on GPU
    Description: for each device, send to previous rank and receive from next rank.
                example: rank 0 send to rank 7 and receive from rank 1.
    Expectation: on rank i, result == [1 ,1 ,1, 1, 1, 1, 1, 1] * ((i + 1) % 8)
    """
    neighborexchange = Net()
    expect0 = np.asarray([1, 1, 1, 1, 1, 1, 1, 1]).astype(
        np.float32) * ((rank + 1) % 8)
    inputs = []
    inputs.append(Tensor(x))
    inputs = tuple(inputs)
    output0 = neighborexchange(inputs)[0].asnumpy()
    diff0 = output0 - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape
