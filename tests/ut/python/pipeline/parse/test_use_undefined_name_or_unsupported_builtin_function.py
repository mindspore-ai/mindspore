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
""" test use undefined name or unsupported builtin function"""
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, save_graphs=True)


def test_use_undefined_name():
    class Net(nn.Cell):
        def construct(self, x):
            ret = x + a
            return ret

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor([1, 2, 3], mstype.float32))
    assert "The name 'a' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_name_or_unsupported_builtin_function.py(29)" in \
           str(err.value)
    assert "ret = x + a" in str(err.value)


def test_insert_undefined_name():
    class Net(nn.Cell):
        def construct(self, x):
            b
            ret = x + x
            return ret

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor([1, 2, 3], mstype.float32))
    assert "The name 'b' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_name_or_unsupported_builtin_function.py(44)" in \
           str(err.value)


def test_insert_undefined_name_compute():
    class Net(nn.Cell):
        def construct(self, x):
            c + x
            ret = x + x
            return ret

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor([1, 2, 3], mstype.float32))
    assert "The name 'c' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_name_or_unsupported_builtin_function.py(59)" in \
           str(err.value)
    assert "c + x" in str(err.value)


def test_insert_undefined_name_in_if():
    class Net(nn.Cell):
        def construct(self, x):
            if x > 0:
                i
            ret = x + x
            return ret

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor([1, 2, 3], mstype.float32))
    assert "The name 'i' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_name_or_unsupported_builtin_function.py(76)" in \
           str(err.value)


def test_insert_undefined_name_in_while_inner_if():
    class Net(nn.Cell):
        def construct(self, x):
            while x > 0:
                if x > 1:
                    j
                x = x - 1
            ret = x + x
            return ret

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor([1, 2, 3], mstype.float32))
    assert "The name 'j' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_name_or_unsupported_builtin_function.py(93)" in \
           str(err.value)


def test_insert_undefined_name_compute__in_while_inner_if():
    class Net(nn.Cell):
        def construct(self, x):
            while x > 0:
                if x > 1:
                    p + x
                x = x - 1
            ret = x + x
            return ret

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor([1, 2, 3], mstype.float32))
    assert "The name 'p' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_name_or_unsupported_builtin_function.py(111)" in \
           str(err.value)
    assert "p + x" in str(err.value)


def test_insert_undefined_name_compute__in_if_in_for():
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
        net(Tensor([1, 2, 3], mstype.float32))
    assert "The name 'w' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_name_or_unsupported_builtin_function.py(134)" in \
           str(err.value)
    assert "w" in str(err.value)


def test_use_undefined_name_for_inner_if():
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
        net(Tensor([1, 2, 3], mstype.float32))
    assert "The name 'y' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_name_or_unsupported_builtin_function.py(157)" in \
           str(err.value)
    assert "y" in str(err.value)


def test_use_undefined_name_in_for():
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
        net(Tensor([1, 2, 3], mstype.float32))
    assert "The name 'd' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_name_or_unsupported_builtin_function.py(178)" in \
           str(err.value)
    assert "x = x + d + i" in str(err.value)


def test_insert_undefined_name_in_for():
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
        net(Tensor([1, 2, 3], mstype.float32))
    assert "The name 'e' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_name_or_unsupported_builtin_function.py(198)" in \
           str(err.value)
    assert "e" in str(err.value)


def test_insert_undefined_name_compute_in_for():
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
        net(Tensor([1, 2, 3], mstype.float32))
    assert "The name 'f' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_name_or_unsupported_builtin_function.py(219)" in \
           str(err.value)
    assert "f + i" in str(err.value)


def test_use_undefined_name_in_while():
    class Net(nn.Cell):
        def construct(self, x):
            while x < 0:
                x = x - g
            return x

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor([1, 2, 3], mstype.float32))
    assert "The name 'g' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_name_or_unsupported_builtin_function.py(236)" in \
           str(err.value)
    assert "x = x - g" in str(err.value)


def test_insert_undefined_name_in_while():
    class Net(nn.Cell):
        def construct(self, x):
            while x < 0:
                h
                x = x - 1
            return x

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor([1, 2, 3], mstype.float32))
    assert "The name 'h' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_name_or_unsupported_builtin_function.py(252)" in \
           str(err.value)
    assert "h" in str(err.value)


def test_insert_undefined_name_compute_while():
    class Net(nn.Cell):
        def construct(self, x):
            while x < 0:
                x + i
                x = x - 1
            return x

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor([1, 2, 3], mstype.float32))
    assert "The name 'i' is not defined" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_name_or_unsupported_builtin_function.py(269)" in \
           str(err.value)
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
        net(Tensor([1, 2, 3], mstype.float32))
    assert "Not AbstractFunction: AbstractNone(Value: None)" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_name_or_unsupported_builtin_function.py(291)" in \
           str(err.value)
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
    net(Tensor([1, 2, 3], mstype.float32))


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
    net(Tensor([1, 2, 3], mstype.float32))


def test_call_unsupported_builtin_function_in_while():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x, y):
            ret = 0
            number = 5
            while number > 0:
                ret = divmod(x, y)
                number -= 1
            return ret

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor([1, 2, 3]), Tensor([4, 5, 6]))
    assert "The builtin function 'divmod' is not supported in graph mode" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_name_or_unsupported_builtin_function.py(342)" in str(
        err.value)
    assert "ret = divmod(x, y)" in str(err.value)


def test_call_unsupported_builtin_function_in_if_in_for():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [11, 22, 33, 44]

        def construct(self, x):
            for i in self.value:
                if x > 1:
                    x = divmod(x, i)
            ret = x + x
            return ret

    net = Net()
    with pytest.raises(NameError) as err:
        net(Tensor([1, 2, 3], mstype.float32))
    assert "The builtin function 'divmod' is not supported in graph mode" in str(err.value)
    assert "tests/ut/python/pipeline/parse/test_use_undefined_name_or_unsupported_builtin_function.py(364)" in \
           str(err.value)
    assert "x = divmod(x, i)" in str(err.value)


def test_use_defined_class_obj_in_for():
    class Test:
        def __init__(self):
            self.number = 1

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [1, 2, 3]
            self.test = Test()

        def construct(self, x):
            for i in self.value:
                x = i + self.test.number
            ret = x + x
            return ret

    net = Net()
    with pytest.raises(TypeError) as err:
        net(Tensor([1, 2, 3], mstype.float32))
    assert "Invalid object with type" in str(err.value)
