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
"""Test Tensor check_input_data_type"""
import pytest

import mindspore as ms
from mindspore import Tensor, nn


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.idx = ms.mutable([2, 1, 0])

    def construct(self, x):
        out = x[self.idx]
        return out


def test_tensor_index_with_mutable():
    """
    Feature: Check the type of index for Tensor.
    Description: Tensor index does not support mutables.
    Expectation: Throw TypeError.
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    net = Net()
    a = Tensor([1, 2, 3, 4, 5])
    with pytest.raises(TypeError):
        net(a)
