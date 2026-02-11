# Copyright 2021-2025 Huawei Technologies Co., Ltd
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
""" test syntax raise in strict mode"""
import os
import pytest
import numpy as np

from mindspore import ops
from mindspore import Tensor
from mindspore import context
from mindspore.graph.api import jit
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_single_string_control_flow():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def foo(x, y):
        if x > y:
            raise ValueError("x is bigger than y")
        return x + y
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([2])
    input_y = Tensor([1])
    with pytest.raises(ValueError) as raise_info:
        ret = foo(input_x, input_y)
        print(ret)
    assert "x is bigger than y" in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_not_raise_single_string_control_flow():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def foo(x, y):
        if x > y:
            raise ValueError("x is bigger than y")
        return x + y
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([0])
    input_y = Tensor([1])
    ret = foo(input_x, input_y)
    assert np.all(ret.asnumpy() == np.array([1]))
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_single_string_control_flow_grad_in_graph():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def foo(x, y):
        if x > y:
            raise ValueError("x is bigger than y")
        return x + y

    @jit
    def grad_foo(foo, x, y):
        return ops.grad(foo)(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([2])
    input_y = Tensor([1])
    with pytest.raises(ValueError) as raise_info:
        ret = grad_foo(foo, input_x, input_y)
        print(ret)
    assert "x is bigger than y" in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_single_string_control_flow_grad_in_pynative():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    Note: foo is renamed to foo_raise_error, since name foo will cause
          test_not_raise_single_string_control_flow_grad_in_pynative failed to find reused graph.
    """
    @jit
    def foo_raise_error(x, y):
        if x > y:
            raise ValueError("x is bigger than y")
        return x + y

    def grad_foo_raise_error(foo, x, y):
        return ops.grad(foo)(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([2])
    input_y = Tensor([1])
    with pytest.raises(ValueError) as raise_info:
        ret = grad_foo_raise_error(foo_raise_error, input_x, input_y)
        print(ret)
    assert "x is bigger than y" in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_not_raise_single_string_control_flow_grad_in_graph():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def foo(x, y):
        if x > y:
            raise ValueError("x is bigger than y")
        return x + y

    @jit
    def grad_foo(foo, x, y):
        return ops.grad(foo)(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([0])
    input_y = Tensor([1])
    ret = grad_foo(foo, input_x, input_y)
    assert np.all(ret.asnumpy() == np.array([1]))
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_not_raise_single_string_control_flow_grad_in_pynative():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def foo(x, y):
        if x > y:
            raise ValueError("x is bigger than y")
        return x + y

    def grad_foo_not_raise_error(foo, x, y):
        return ops.grad(foo)(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([0])
    input_y = Tensor([1])
    ret = grad_foo_not_raise_error(foo, input_x, input_y)
    assert np.all(ret.asnumpy() == np.array([1]))
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_empty_input_control_flow():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def foo(x, y):
        if x > y:
            raise ValueError()
        return x + y
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([2])
    input_y = Tensor([1])
    with pytest.raises(ValueError) as raise_info:
        ret = foo(input_x, input_y)
        print(ret)
    assert str(raise_info.value) == ""
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_not_raise_empty_input_control_flow():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def foo(x, y):
        if x > y:
            raise ValueError()
        return x + y
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([0])
    input_y = Tensor([1])
    ret = foo(input_x, input_y)
    assert np.all(ret.asnumpy() == np.array([1]))
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_multi_input_control_flow():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def foo(x, y):
        if x > y:
            raise ValueError("x is bigger than y", ".")
        return x + y
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([2])
    input_y = Tensor([1])
    with pytest.raises(ValueError) as raise_info:
        ret = foo(input_x, input_y)
        print(ret)
    assert str(raise_info.value) == "('x is bigger than y', '.')"
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_not_raise_multi_input_control_flow():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def foo(x, y):
        if x > y:
            raise ValueError("x is bigger than y", ".")
        return x + y
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([0])
    input_y = Tensor([1])
    ret = foo(input_x, input_y)
    assert np.all(ret.asnumpy() == np.array([1]))
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_format_in_lax():
    """
    Feature: Test raise and format syntax in lax mode.
    Description: Test raise and format syntax in lax mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x, y):
        out = ops.Mul()(x, y)
        if out == Tensor([6]):
            raise AssertionError("The out is {}, y is {}.".format(out, y))
        return out

    input_x = Tensor([2])
    input_y = Tensor([3])
    with pytest.raises(AssertionError) as raise_info:
        ret = func(input_x, input_y)
        print(ret)
    assert str(raise_info.value) == "The out is [6], y is [3]."


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_not_raise_format_in_lax():
    """
    Feature: Test raise and format syntax in lax mode.
    Description: Test raise and format syntax in lax mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x, y):
        out = ops.Mul()(x, y)
        if out == Tensor([6]):
            raise AssertionError("The out is {}, y is {}.".format(out, y))
        return out

    input_x = Tensor([2])
    input_y = Tensor([1])
    ret = func(input_x, input_y)
    print(ret)
    assert ret == Tensor([2])
