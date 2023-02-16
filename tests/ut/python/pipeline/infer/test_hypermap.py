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
import numpy as np

from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE)


def test_hypermap_if():
    class Net(Cell):
        """DictNet1 definition"""

        def __init__(self):
            super(Net, self).__init__()
            self.max = P.Maximum()
            self.min = P.Minimum()
            self._list = [1, 2, 3]

        def construct(self, x, y):
            if map(lambda a: a + 1, self._list):
                ret = self.max(x, y)
            else:
                ret = self.min(x, y)
            return ret

    net = Net()
    x = Tensor(np.ones([3, 2, 3], np.float32))
    y = Tensor(np.ones([1, 2, 3], np.float32))
    net(x, y)


def test_hypermap_value():
    class Net(Cell):
        """DictNet1 definition"""

        def __init__(self):
            super(Net, self).__init__()
            self.max = P.Maximum()
            self.min = P.Minimum()
            self._list = [22, 66, 88, 111]

        def construct(self):
            return map(lambda a: a + 1, self._list)

    net = Net()
    assert net() == [23, 67, 89, 112]


def test_hypermap_func_const():
    class NetMap(Cell):
        def __init__(self):
            super(NetMap, self).__init__()

        def double(self, x):
            return 2 * x

        def triple(self, x):
            return 3 * x

        def square(self, x):
            return x * x

        def construct(self):
            _list = [self.double, self.triple, self.square]
            return map(lambda f: f(4), _list)

    net = NetMap()
    assert net() == [8, 12, 16]
