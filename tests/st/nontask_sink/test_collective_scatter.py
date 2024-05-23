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
src_rank = 0


class CollectiveScatterNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.collective_scatter = comm_ops.CollectiveScatter(src_rank=src_rank)

    def construct(self, x):
        out = self.collective_scatter(x)
        return out


def test_hccl_scatter_4p_float32():
    """
    Feature: test 'CollectiveScatter' communication operator.
    Description: test 'CollectiveScatter' communication operator.
    Expectation: expect correct result.
    """
    ms_input = Tensor(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]).astype(np.float32))
    net = CollectiveScatterNet()
    output = net(ms_input)
    res = np.array([this_rank + 1, this_rank + 1, this_rank + 1]).astype(np.float32)
    assert (output.numpy() == res).all()


def test_hccl_scatter_4p_float16():
    """
    Feature: test 'CollectiveScatter' communication operator.
    Description: test 'CollectiveScatter' communication operator.
    Expectation: expect correct result.
    """
    ms_input = Tensor(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]).astype(np.float16))
    net = CollectiveScatterNet()
    output = net(ms_input)
    res = np.array([this_rank + 1, this_rank + 1, this_rank + 1]).astype(np.float16)
    assert (output.numpy() == res).all()
