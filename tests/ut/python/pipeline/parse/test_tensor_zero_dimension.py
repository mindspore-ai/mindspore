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
import pytest
import mindspore

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P
from mindspore.common.initializer import One

context.set_context(mode=context.GRAPH_MODE)

def test_zero_dimension_list():
    Tensor([])
    with pytest.raises(ValueError) as ex:
        Tensor([[]])
    assert "input_data can not contain zero dimension." in str(ex.value)


def test_zero_dimension_np_array():
    with pytest.raises(ValueError) as ex:
        Tensor(np.ones((1, 0, 3)))
    assert "input_data can not contain zero dimension." in str(ex.value)


def test_zero_dimension_with_zero_shape():
    with pytest.raises(ValueError) as ex:
        Tensor(shape=(1, 0, 3), dtype=mindspore.float32, init=One())
    assert "Shape can not contain zero value." in str(ex.value)


def test_zero_dimension_with_operator():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.strided_slice = P.StridedSlice()

        def construct(self, x):
            a = self.strided_slice(x, (2, 4, 4), (-1, 2, 1), (1, 1, 1))
            return a

    x = Tensor(np.ones((1, 3, 3)))
    net = Net()
    net(x)
