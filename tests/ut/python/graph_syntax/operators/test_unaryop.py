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
""" test_unaryop """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_scalar_usub():
    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            ret = -self.x
            return ret

    net = Net(-2)
    assert net() == 2


def test_tensor_usub():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            ret = -x
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    net = Net()
    net(x)


def test_scalar_uadd():
    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            ret = +self.x
            return ret

    net = Net(-2)
    assert net() == -2
