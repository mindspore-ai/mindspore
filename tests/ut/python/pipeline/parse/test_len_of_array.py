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
""" test len of array"""
import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_len_a_3D_tensor():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x, y):
            return len(x), len(y)

    net = Net()
    x = Tensor(np.ones((5, 6, 7)))
    y = Tensor(np.ones((100, 6, 7)))
    ret = net(x, y)
    assert ret == (len(x), len(y)) == (5, 100)


def test_len_a_0D_tensor():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            return len(x)

    net = Net()
    x = Tensor(np.array(100))
    with pytest.raises(TypeError) as err:
        _ = net(x)
    assert "Not support len of a 0-D tensor." in str(err.value)
