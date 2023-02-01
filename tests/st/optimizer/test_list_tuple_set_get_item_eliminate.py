# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.nn import Cell
import pytest


class NetWork(Cell):
    def construct(self, blocks, k, max_iter):
        block_len = len(blocks)
        while k < max_iter:
            for i in range(block_len):
                blocks[i][k] += 1
            k += 1
        return blocks


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_list_getitem_eliminate():
    """
    Feature: optimizer
    Description: Test list_getitem not be replaced as TupleGetItem in pass 'item_tuple_or_list_eliminate'
    Expectation: No exception.
    """
    max_iter = Tensor([1])
    k = Tensor([0])
    inputs = [Tensor([4]), Tensor([5])]
    net = NetWork()
    output = net(inputs, k, max_iter)
    assert output[0] == Tensor([5])
    assert output[1] == Tensor([6])
