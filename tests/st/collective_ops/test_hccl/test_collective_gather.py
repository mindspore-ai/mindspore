# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.communication.management import init, get_rank
from mindspore.ops.operations import comm_ops

np.random.seed(1)
init()
this_rank = get_rank()
dest_rank = 0


class CollectiveGatherNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.collective_gather = comm_ops.CollectiveGather(dest_rank=dest_rank)

    def construct(self, x):
        out = self.collective_gather(x)
        return out


def generate_input(dtype):
    if this_rank == 0:
        return Tensor(np.array([[1, 1, 1]]).astype(dtype))
    if this_rank == 1:
        return Tensor(np.array([[2, 2, 2]]).astype(dtype))
    if this_rank == 2:
        return Tensor(np.array([[3, 3, 3]]).astype(dtype))
    if this_rank == 3:
        return Tensor(np.array([[4, 4, 4]]).astype(dtype))
    return None


def test_hccl_gather_4p_float32():
    """
    Feature: test 'CollectiveGather' communication operator.
    Description: test 'CollectiveGather' communication operator.
    Expectation: expect correct result.
    """
    ms_input = generate_input(np.float32)
    net = CollectiveGatherNet()
    output = net(ms_input)

    if this_rank == dest_rank:
        res = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]).astype(np.float32)
        assert (output.numpy() == res).all()


def test_hccl_gather_4p_float16():
    """
    Feature: test 'CollectiveGather' communication operator.
    Description: test 'CollectiveGather' communication operator.
    Expectation: expect correct result.
    """
    ms_input = generate_input(np.float16)
    net = CollectiveGatherNet()
    output = net(ms_input)

    if this_rank == dest_rank:
        res = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]).astype(np.float16)
        assert (output.numpy() == res).all()
