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
""" test use undefined var"""
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_use_undefined_var():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            ret = x + c
            return ret
    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor(np.arange(4)))
    assert "The name 'c' is not defined" in str(err.value)


def test_insert_undefined_var():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            c
            ret = x + x
            return ret
    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor(np.arange(4)))
    assert "The name 'c' is not defined" in str(err.value)


def test_insert_undefined_var_compute():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            c + d
            ret = x + x
            return ret
    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor(np.arange(4)))
    assert "The name 'c' is not defined" in str(err.value)


def test_insert_defined_var():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            x
            ret = x + x
            return ret
    net = Net()
    net(Tensor(np.arange(4)))


def test_insert_defined_var_compute():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            x - x
            ret = x + x
            return ret
    net = Net()
    net(Tensor(np.arange(4)))
