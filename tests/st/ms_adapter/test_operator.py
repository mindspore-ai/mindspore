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

import pytest
import mindspore as ms
import tests.st.ms_adapter as adapter

ms.set_context(mode=ms.GRAPH_MODE)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_arithmetic_operator():
    """
    Feature: MSAdapter
    Description: Test arithmetic operators
    Expectation: No exception
    """
    @ms.jit
    def add_fn(x, y):
        return x + y

    @ms.jit
    def sub_fn(x, y):
        return x - y

    @ms.jit
    def mul_fn(x, y):
        return x * y

    @ms.jit
    def div_fn(x, y):
        return x / y

    @ms.jit
    def floordiv_fn(x, y):
        return x // y

    @ms.jit
    def mod_fn(x, y):
        return x % y

    @ms.jit
    def pow_fn(x, y):
        return x ** y

    def check_output_type(func):
        ms_x = ms.Tensor(1)
        adapter_x = adapter.Tensor(1)
        assert type(func(ms_x, ms_x)) is ms.Tensor
        assert type(func(adapter_x, adapter_x)) is adapter.Tensor      # "Tensor", "Tensor"
        assert type(func(adapter_x, 1)) is adapter.Tensor              # "Tensor", "Number"
        assert type(func(1, adapter_x)) is adapter.Tensor              # "Number", "Tensor"
        assert type(func(adapter_x, (adapter_x,))) is adapter.Tensor   # "Tensor", "Tuple"
        assert type(func((adapter_x,), adapter_x)) is adapter.Tensor   # "Tuple", "Tensor"
        assert type(func(adapter_x, [adapter_x,])) is adapter.Tensor   # "Tensor", "List"
        assert type(func([adapter_x,], adapter_x)) is adapter.Tensor   # "List", "Tensor"

    check_output_type(add_fn)
    check_output_type(sub_fn)
    check_output_type(mul_fn)
    check_output_type(div_fn)
    check_output_type(floordiv_fn)
    check_output_type(mod_fn)
    check_output_type(pow_fn)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_binary_operator():
    """
    Feature: MSAdapter
    Description: Test binary operators
    Expectation: No exception
    """
    @ms.jit
    def equal_fn(x, y):
        return x == y

    @ms.jit
    def not_equal_fn(x, y):
        return x != y

    @ms.jit
    def less_fn(x, y):
        return x < y

    @ms.jit
    def greater_fn(x, y):
        return x > y

    @ms.jit
    def less_equal_fn(x, y):
        return x <= y

    @ms.jit
    def greater_equal_fn(x, y):
        return x >= y

    @ms.jit
    def bitwise_and_fn(x, y):
        return x & y

    @ms.jit
    def bitwise_or_fn(x, y):
        return x | y

    @ms.jit
    def bitwise_xor_fn(x, y):
        return x ^ y

    def check_output_type(func):
        ms_x = ms.Tensor([1, 2, 3])
        ms_y = ms.Tensor([3, 2, 1])
        adapter_x = adapter.Tensor([1, 2, 3], dtype=ms.int32)
        adapter_y = adapter.Tensor([3, 2, 1], dtype=ms.int32)
        assert type(func(ms_x, ms_y)) is ms.Tensor
        assert type(func(adapter_x, adapter_y)) is adapter.Tensor    # "Tensor", "Tensor"
        assert type(func(adapter_x, 1)) is adapter.Tensor            # "Tensor", "Number"
        assert type(func(1, adapter_x)) is adapter.Tensor            # "Number", "Tensor"

    check_output_type(equal_fn)
    check_output_type(not_equal_fn)
    check_output_type(less_fn)
    check_output_type(greater_fn)
    check_output_type(less_equal_fn)
    check_output_type(greater_equal_fn)
    check_output_type(bitwise_and_fn)
    check_output_type(bitwise_or_fn)
    check_output_type(bitwise_xor_fn)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unary_operator():
    """
    Feature: MSAdapter
    Description: Test unary operators
    Expectation: No exception
    """
    @ms.jit
    def positive_fn(x):
        return +x

    @ms.jit
    def negative_fn(x):
        return -x

    ms_x = ms.Tensor([1, -2, 3])
    adapter_x = adapter.Tensor([1, -2, 3])
    assert type(positive_fn(ms_x)) is ms.Tensor
    assert type(negative_fn(ms_x)) is ms.Tensor
    assert type(positive_fn(adapter_x)) is adapter.Tensor
    assert type(negative_fn(adapter_x)) is adapter.Tensor


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logical_operator():
    """
    Feature: MSAdapter
    Description: Test logical operators
    Expectation: No exception
    """
    @ms.jit
    def is_fn(x):
        return x is None

    @ms.jit
    def is_not_fn(x):
        return x is not None

    @ms.jit
    def invert_fn(x):
        return ~x

    @ms.jit
    def logical_not_fn(x):
        return not x

    ms_x = ms.Tensor(True)
    adapter_x = adapter.Tensor(True)
    assert not is_fn(adapter_x)
    assert is_not_fn(adapter_x)
    assert type(invert_fn(ms_x)) is ms.Tensor
    assert type(logical_not_fn(ms_x)) is ms.Tensor
    assert type(invert_fn(adapter_x)) is adapter.Tensor
    assert type(logical_not_fn(adapter_x)) is adapter.Tensor


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_contain_operator():
    """
    Feature: MSAdapter
    Description: Test in / not in
    Expectation: No exception
    """
    @ms.jit
    def in_fn(x, y, z):
        return x in (x, y, z)

    @ms.jit
    def not_in_fn(x, y, z):
        return x not in (x, y, z)

    ms_x = ms.Tensor(2)
    ms_y = ms.Tensor(2)
    ms_z = ms.Tensor(3)
    adapter_x = adapter.Tensor(1)
    adapter_y = adapter.Tensor(2)
    adapter_z = adapter.Tensor(3)
    assert type(in_fn(ms_x, ms_y, ms_z)) is ms.Tensor
    assert type(not_in_fn(ms_x, ms_y, ms_z)) is ms.Tensor
    assert type(in_fn(adapter_x, adapter_y, adapter_z)) is adapter.Tensor
    assert type(not_in_fn(adapter_x, adapter_y, adapter_z)) is adapter.Tensor


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_matmul():
    """
    Feature: MSAdapter
    Description: Test matmul operator
    Expectation: No exception
    """
    @ms.jit
    def func(x, y):
        return x @ y

    ms_x = ms.Tensor([1, 2], ms.float32)
    ms_y = ms.Tensor([3, 4], ms.float32)
    adapter_x = adapter.Tensor([1, 2], dtype=ms.float32)
    adapter_y = adapter.Tensor([3, 4], dtype=ms.float32)
    assert type(func(ms_x, ms_y)) is ms.Tensor
    assert type(func(adapter_x, adapter_y)) is adapter.Tensor


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_getitem():
    """
    Feature: MSAdapter
    Description: Test getietm operator
    Expectation: No exception
    """
    @ms.jit
    def getitem_fn(x, index):
        return x[index]

    @ms.jit
    def getitem_slice_fn(x):
        return x[1:]

    ms_x = ms.Tensor([[1, 2, 3], [4, 5, 6]])
    adapter_x = adapter.Tensor([[1, 2, 3], [4, 5, 6]])
    assert type(getitem_fn(ms_x, 0)) is ms.Tensor
    assert type(getitem_fn(ms_x, None)) is ms.Tensor
    assert type(getitem_fn(ms_x, [0, 1])) is ms.Tensor
    assert type(getitem_fn(ms_x, (0, 1))) is ms.Tensor
    assert type(getitem_slice_fn(ms_x)) is ms.Tensor
    assert type(getitem_fn(adapter_x, 0)) is adapter.Tensor
    assert type(getitem_fn(adapter_x, None)) is adapter.Tensor
    assert type(getitem_fn(adapter_x, [0, 1])) is adapter.Tensor
    assert type(getitem_fn(adapter_x, (0, 1))) is adapter.Tensor
    assert type(getitem_slice_fn(adapter_x)) is adapter.Tensor


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_setitem():
    """
    Feature: MSAdapter
    Description: Test setitem operator
    Expectation: No exception
    """
    @ms.jit
    def setitem_fn(x, index, value):
        x[index] = value
        return x

    ms_x = ms.Tensor([[1, 2, 3], [4, 5, 6]])
    adapter_x = adapter.Tensor([[1, 2, 3], [4, 5, 6]])
    adapter_index = adapter.Tensor([0], dtype=ms.int32)
    adapter_value = adapter.Tensor([7, 8, 9])
    assert type(setitem_fn(adapter_x, adapter_index, adapter_value)) is adapter.Tensor
    assert type(setitem_fn(ms_x, adapter_index, adapter_value)) is ms.Tensor
