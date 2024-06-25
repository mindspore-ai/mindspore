# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Test basic operation with one stage"""
import pytest
import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import jit, jit_class
from tests.mark_utils import arg_mark

cfg = {
    "replace_nncell_by_construct": True,
    "print_after_all": False,
    "compile_by_trace": True,
    "print_bb": False,
    "MAX_INLINE_DEPTH": 10,
    "allowed_inline_modules": ["mindspore"],  # buildsubgraph
}


@jit_class
class UserDefinedNet:
    def __init__(self, val):
        self.val = val
        self.origin = (1, 2, 3, 4)

    def func(self, x):
        return self.val + x


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_jit_class_instance():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def foo(x):
        a = UserDefinedNet(x)
        return a.val

    context.set_context(mode=context.PYNATIVE_MODE)
    ret = foo(1)
    assert ret == 1


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_jit_class_instance_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def foo(x):
        a = UserDefinedNet(x)
        return a.val

    context.set_context(mode=context.PYNATIVE_MODE)
    ret = foo((1, 2, 3, 4))
    assert ret == (1, 2, 3, 4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_jit_class_instance_3():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def foo(x):
        a = UserDefinedNet(x)
        return a.val + 1

    context.set_context(mode=context.PYNATIVE_MODE)
    ret = foo(Tensor([1, 2, 3]))
    assert np.all(ret.asnumpy() == np.array([2, 3, 4]))


@pytest.mark.skip(reason="Fix after adjust guard for getattr")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_jit_class_method():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def foo(x):
        a = UserDefinedNet(x)
        return a.func(x)

    context.set_context(mode=context.PYNATIVE_MODE)
    ret = foo(1)
    assert ret == 2


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_jit_class_method_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def foo(x):
        a = UserDefinedNet(x)
        return a.func(x)

    context.set_context(mode=context.PYNATIVE_MODE)
    ret = foo(Tensor([1, 1, 1]))
    assert np.all(ret.asnumpy() == np.array([2, 2, 2]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_jit_class_in_subgraph():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    def inner_foo(x):
        return UserDefinedNet(x)

    @jit(mode="PIJit", jit_config=cfg)
    def foo(x):
        a = inner_foo(x)
        return a.val

    context.set_context(mode=context.PYNATIVE_MODE)
    ret = foo(1)
    assert ret == 1


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_jit_class_in_subgraph_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    def inner_foo(x):
        return UserDefinedNet(x)

    @jit(mode="PIJit", jit_config=cfg)
    def foo(x):
        a = inner_foo(x)
        return a.val

    context.set_context(mode=context.PYNATIVE_MODE)
    ret = foo((1, 2, 3, 4))
    assert ret == (1, 2, 3, 4)


@jit_class
class UserDefinedTuple(tuple):
    def __repr__(self):
        return "UserDefinedTuple(" + str(list(self)) + ")"


@pytest.mark.skip(reason="Jit handle instance with subclass of tuple wrong, fix later")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_subclass_tuple_jit_class():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def foo():
        a = UserDefinedTuple((1, 2, 3, 4))
        return isinstance(a, tuple)

    context.set_context(mode=context.PYNATIVE_MODE)
    assert foo()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_subclass_tuple_jit_class_2():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def foo():
        a = UserDefinedTuple((1, 2, 3, 4))
        return isinstance(a, UserDefinedTuple)

    context.set_context(mode=context.PYNATIVE_MODE)
    assert foo()
