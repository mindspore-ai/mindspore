# Copyright 2025 Huawei Technologies Co., Ltd
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
import pytest
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.nn import ReLU
from mindspore.nn import Cell
import numpy as np

from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_tensor_fancy_index_set_item():
    """
    Feature: tensor index
    Description: Verify the result of tensor index error
    Expectation: success
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.b = Tensor([0, 1], mstype.bool_)
            self.value = Tensor(np.arange(8).reshape(2, 4).astype(np.float32))
            self.relu = ReLU()

        def construct(self, input_x):
            input_x[self.b, self.b] = self.value
            out = self.relu(input_x)
            return out

    input_np = np.random.randn(2, 3, 4).astype(np.float32)
    input_me = Tensor(input_np)
    net_me = Net()
    with pytest.raises(ValueError):
        net_me(input_me)
        _pynative_executor.sync()
