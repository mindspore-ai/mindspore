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
""" test mod"""

import mindspore.nn as nn
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_positive_mod_positive():
    class Mod(nn.Cell):
        def __init__(self, x, y):
            super(Mod, self).__init__()
            self.x = x
            self.y = y

        def construct(self):
            return self.x % self.y
    x = 3.0
    y = 1.3
    mod_net = Mod(x, y)
    expect = x % y
    assert abs(mod_net() - expect) < 0.000001


def test_positive_mod_negative():
    class Mod(nn.Cell):
        def __init__(self, x, y):
            super(Mod, self).__init__()
            self.x = x
            self.y = y

        def construct(self):
            return self.x % self.y
    x = 3.0
    y = -1.3
    mod_net = Mod(x, y)
    expect = x % y
    assert abs(mod_net() - expect) < 0.000001


def test_negative_mod_positive():
    class Mod(nn.Cell):
        def __init__(self, x, y):
            super(Mod, self).__init__()
            self.x = x
            self.y = y

        def construct(self):
            return self.x % self.y
    x = -3.0
    y = 1.3
    mod_net = Mod(x, y)
    expect = x % y
    assert abs(mod_net() - expect) < 0.000001


def test_negative_mod_negative():
    class Mod(nn.Cell):
        def __init__(self, x, y):
            super(Mod, self).__init__()
            self.x = x
            self.y = y

        def construct(self):
            return self.x % self.y
    x = -3.0
    y = -1.3
    mod_net = Mod(x, y)
    expect = x % y
    assert abs(mod_net() - expect) < 0.000001


def test_int_mod_int():
    class Mod(nn.Cell):
        def __init__(self, x, y):
            super(Mod, self).__init__()
            self.x = x
            self.y = y

        def construct(self):
            return self.x % self.y
    x = 3
    y = 2
    mod_net = Mod(x, y)
    expect = x % y
    assert abs(mod_net() - expect) < 0.000001
