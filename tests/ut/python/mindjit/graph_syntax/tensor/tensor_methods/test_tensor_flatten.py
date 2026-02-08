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
""" test flatten"""
import pytest

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_flatten():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[1, 2, 3], [4, 5, 6]], dtype=mstype.float32)

        def construct(self):
            return self.value.flatten()

    net = Net()
    net()


def test_flatten_1():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[1, 2, 3], [4, 5, 6]], dtype=mstype.float32)

        def construct(self):
            return self.value.flatten(order='F')

    net = Net()
    net()


def test_flatten_error():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[1, 2, 3], [4, 5, 6]], dtype=mstype.float32)

        def construct(self):
            return self.value.flatten(order='X')

    net = Net()
    with pytest.raises(ValueError):
        net()


def test_flatten_error_1():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = Tensor([[1, 2, 3], [4, 5, 6]], dtype=mstype.float32)

        def construct(self):
            return self.value.flatten(order=123)

    net = Net()
    with pytest.raises(TypeError):
        net()
