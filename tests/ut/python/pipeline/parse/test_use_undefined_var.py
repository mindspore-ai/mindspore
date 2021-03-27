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
            ret = x + a
            return ret

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor(np.arange(4)))
    assert "The name 'a' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_var.py(33)" in str(err.value)
    assert "ret = x + a" in str(err.value)


def test_insert_undefined_var():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            b
            ret = x + x
            return ret

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor(np.arange(4)))
    assert "The name 'b' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_var.py(51)" in str(err.value)


def test_insert_undefined_var_compute():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            c + x
            ret = x + x
            return ret

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor(np.arange(4)))
    assert "The name 'c' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_var.py(69)" in str(err.value)
    assert "c + x" in str(err.value)


def test_insert_undefined_var_in_if():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            if x > 0:
                i
            ret = x + x
            return ret

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor(np.arange(4)))
    assert "The name 'i' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_var.py(89)" in str(err.value)


def test_insert_undefined_var_in_while_inner_if():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            while x > 0:
                if x > 1:
                    j
                x = x - 1
            ret = x + x
            return ret

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor(np.arange(4)))
    assert "The name 'j' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_var.py(109)" in str(err.value)


def test_insert_undefined_var_compute__in_while_inner_if():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            while x > 0:
                if x > 1:
                    p + x
                x = x - 1
            ret = x + x
            return ret

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor(np.arange(4)))
    assert "The name 'p' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_var.py(130)" in str(err.value)
    assert "p + x" in str(err.value)


def test_insert_undefined_var_compute__in_for_inner_if():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            for i in self.value:
                if x > 1:
                    w
                    x = x - i
            ret = x + x
            return ret

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor(np.arange(4)))
    assert "The name 'w' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_var.py(152)" in str(err.value)
    assert "w" in str(err.value)


def test_use_undefined_var_for_inner_if():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            for i in self.value:
                if x > 1:
                    x = x - i + y
            ret = x + x
            return ret

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor(np.arange(4)))
    assert "The name 'y' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_var.py(174)" in str(err.value)
    assert "y" in str(err.value)


def test_use_undefined_var_in_for():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            for i in self.value:
                x = x + d + i
            return x

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor(np.arange(4)))
    assert "The name 'd' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_var.py(194)" in str(err.value)
    assert "x = x + d + i" in str(err.value)


def test_insert_undefined_var_in_for():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            for i in self.value:
                e
                x = x + i
            return x

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor(np.arange(4)))
    assert "The name 'e' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_var.py(213)" in str(err.value)
    assert "e" in str(err.value)


def test_insert_undefined_var_compute_in_for():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            for i in self.value:
                f + i
                x = x + i
            return x

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor(np.arange(4)))
    assert "The name 'f' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_var.py(233)" in str(err.value)
    assert "f + i" in str(err.value)


def test_use_undefined_var_in_while():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x):
            while x < 0:
                x = x - g
            return x

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor(np.arange(4)))
    assert "The name 'g' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_var.py(252)" in str(err.value)
    assert "x = x - g" in str(err.value)


def test_insert_undefined_var_in_while():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            while x < 0:
                h
                x = x - 1
            return x

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor(np.arange(4)))
    assert "The name 'h' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_var.py(271)" in str(err.value)
    assert "h" in str(err.value)


def test_insert_undefined_var_compute_while():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            while x < 0:
                x + i
                x = x - 1
            return x

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor(np.arange(4)))
    assert "The name 'i' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_var.py(291)" in str(err.value)
    assert "x + i" in str(err.value)


def test_call_none_in_if():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            ret = 0
            if self.value:
                ret = self.func(x)
            return ret

    net = Net()
    with pytest.raises(RuntimeError) as err:
        net(Tensor(np.arange(4)))
    assert "Not AbstractFunction: AbstractNone(Value: None)" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_var.py(312)" in str(err.value)
    assert "ret = self.func(x)" in str(err.value)


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
