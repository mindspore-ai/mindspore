# Copyright 2022 Huawei Technologies Co., Ltd
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

from functools import partial
import pytest
import mindspore as ms
import tests.st.ms_adapter as adapter
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.GRAPH_MODE)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_abs():
    """
    Feature: MSAdapter
    Description: Test python built-in function abs()
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return abs(x)

    assert type(func(ms.Tensor([-5]))) is ms.Tensor
    assert type(func(adapter.Tensor([-5]))) is adapter.Tensor


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_round():
    """
    Feature: MSAdapter
    Description: Test python built-in function round()
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return round(x)

    assert type(func(ms.Tensor([1.55]))) is ms.Tensor
    assert type(func(adapter.Tensor([1.55]))) is adapter.Tensor


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_map():
    """
    Feature: MSAdapter
    Description: Test python built-in function map()
    Expectation: No exception
    """
    def add(x, y):
        return x + y

    @ms.jit
    def func(x, y):
        return map(add, x, y)

    x = (adapter.Tensor(1), 2)
    y = (adapter.Tensor(2), 4)
    out = func(x, y)
    assert type(out[0]) is adapter.Tensor


@pytest.mark.skip(reason="stub tensor syn will loss adaptive tensor attribute")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_filter():
    """
    Feature: MSAdapter
    Description: Test python built-in function filter()
    Expectation: No exception
    """
    def select_fn(x):
        return True

    @ms.jit
    def func(x):
        return filter(select_fn, x)

    x = (adapter.Tensor(2), 1, 2, 3)
    out = func(x)
    assert type(out[0]) is adapter.Tensor


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_partial():
    """
    Feature: MSAdapter
    Description: Test python built-in function partial()
    Expectation: No exception
    """
    def add(x, y):
        return x + y

    @ms.jit
    def func(data):
        add_ = partial(add, x=2)
        return add_(y=data)

    out = func(adapter.Tensor(1))
    assert type(out) is adapter.Tensor


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_zip():
    """
    Feature: MSAdapter
    Description: Test python built-in function zip()
    Expectation: No exception
    """
    @ms.jit
    def func(x, y):
        return zip(x, y)

    x = (adapter.Tensor(1), 2)
    y = (adapter.Tensor(2), 4)
    out = func(x, y)
    assert type(out[0][0]) is adapter.Tensor
    assert type(out[0][1]) is adapter.Tensor


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_enumerate():
    """
    Feature: MSAdapter
    Description: Test python built-in function enumerate()
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return enumerate(x)

    x = adapter.Tensor([[1, 2], [3, 4], [5, 6]])
    out = func(x)
    assert out[0][0] == 0
    assert type(out[0][1]) is adapter.Tensor


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_isinstance():
    """
    Feature: MSAdapter
    Description: Test python built-in function isinstance()
    Expectation: No exception
    """
    @ms.jit
    def func(x, y):
        a = isinstance(x, ms.Tensor) and not isinstance(x, adapter.Tensor)
        b = isinstance(y, ms.Tensor) and isinstance(y, adapter.Tensor)
        return a, b

    x = ms.Tensor(1)
    y = adapter.Tensor(1)
    a, b = func(x, y)
    assert a and b


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_max():
    """
    Feature: MSAdapter
    Description: Test python built-in function max()
    Expectation: No exception
    """
    @ms.jit
    def func(x, y, z):
        return max(x), max(y, z)

    x = adapter.Tensor([1, 2], dtype=ms.float32)
    y = adapter.Tensor([1], dtype=ms.float32)
    z = adapter.Tensor([2], dtype=ms.float32)
    out = func(x, y, z)
    assert type(out[0]) is adapter.Tensor
    assert type(out[1]) is adapter.Tensor


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_min():
    """
    Feature: MSAdapter
    Description: Test python built-in function min()
    Expectation: No exception
    """
    @ms.jit
    def func(x, y, z):
        return min(x), min(y, z)

    x = adapter.Tensor([1, 2], dtype=ms.float32)
    y = adapter.Tensor([1], dtype=ms.float32)
    z = adapter.Tensor([2], dtype=ms.float32)
    out = func(x, y, z)
    assert type(out[0]) is adapter.Tensor
    assert type(out[1]) is adapter.Tensor


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sum():
    """
    Feature: MSAdapter
    Description: Test python built-in function sum()
    Expectation: No exception
    """
    @ms.jit
    def func(x, y, z):
        return sum(x), sum(y, z)

    x = adapter.Tensor([[1, 2], [3, 4]], dtype=ms.float32)
    y = adapter.Tensor([1, 2, 3], dtype=ms.float32)
    z = adapter.Tensor([4, 5, 6], dtype=ms.float32)
    out = func(x, y, z)
    assert type(out[0]) is adapter.Tensor
    assert type(out[1]) is adapter.Tensor


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_getattr():
    """
    Feature: MSAdapter
    Description: Test python built-in function getattr()
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return getattr(x, "attr")

    x = adapter.Tensor([1, 2, 3])
    assert func(x) == 10


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hasattr():
    """
    Feature: MSAdapter
    Description: Test python built-in function hasattr()
    Expectation: No exception
    """
    @ms.jit
    def func(x, y):
        return hasattr(x, "method"), hasattr(y, "method")

    x = adapter.Tensor([1, 2, 3])
    y = ms.Tensor([1, 2, 3])
    out = func(x, y)
    assert out[0] and not out[1]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_iter():
    """
    Feature: MSAdapter
    Description: Test python built-in function iter()
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return iter(x)[0]

    x = adapter.Tensor([1, 2, 3])
    out = func(x)
    assert type(out) is adapter.Tensor


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_next():
    """
    Feature: MSAdapter
    Description: Test python built-in function next()
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        it = iter(x)
        return next(it)

    x = adapter.Tensor([1, 2, 3])
    out = func(x)
    assert type(out[0]) is adapter.Tensor
    assert type(out[1]) is adapter.Tensor


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_print():
    """
    Feature: MSAdapter
    Description: Test python built-in function print()
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        print(x)
        return x

    func(adapter.Tensor([1, 2, 3]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tuple():
    """
    Feature: MSAdapter
    Description: Test python built-in function tuple()
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return tuple(x)

    x = adapter.Tensor([1, 2, 3])
    out = func(x)
    assert type(out[0]) is adapter.Tensor
    assert type(out[1]) is adapter.Tensor
    assert type(out[2]) is adapter.Tensor


@pytest.mark.skip(reason="stub tensor syn will loss adaptive tensor attribute")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list():
    """
    Feature: MSAdapter
    Description: Test python built-in function list()
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return list(x)

    x = adapter.Tensor([1, 2, 3])
    out = func(x)
    assert type(out[0]) is adapter.Tensor
    assert type(out[1]) is adapter.Tensor
    assert type(out[2]) is adapter.Tensor


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_bool():
    """
    Feature: MSAdapter
    Description: Test python built-in function bool()
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return bool(x)

    x = adapter.Tensor([10])
    out = func(x)
    assert out
