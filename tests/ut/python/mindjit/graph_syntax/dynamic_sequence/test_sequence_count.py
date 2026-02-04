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
"""test sequence count operation"""
from mindspore.common import mutable
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore import jit
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_sequence_count_dynamic_sequence_const_target():
    """
    Feature: Sequence count operation
    Description: Sequence count operation will use SequenceCount operation
    Expectation: No exception.
    """
    @jit
    def foo():
        x = mutable([1, 2, 3, 4], True)
        ret = x.count(3)
        return isinstance(ret, int), F.isconstant(ret)
    ret1, ret2 = foo()
    assert ret1
    assert not ret2


def test_sequence_count_dynamic_sequence_const_target_2():
    """
    Feature: Sequence count operation
    Description: Sequence count operation will use SequenceCount operation
    Expectation: No exception.
    """
    @jit
    def foo():
        x = mutable((1, 2, 3, 4), True)
        ret = x.count(3)
        return isinstance(ret, int), F.isconstant(ret)
    ret1, ret2 = foo()
    assert ret1
    assert not ret2


def test_sequence_count_dynamic_sequence_variable_target():
    """
    Feature: Sequence count operation
    Description: Sequence count operation will use SequenceCount operation
    Expectation: No exception.
    """
    @jit
    def foo(target):
        x = mutable([Tensor([1]), Tensor([2]), Tensor([3])], True)
        ret = x.count(target)
        return isinstance(ret, int), F.isconstant(ret)
    ret1, ret2 = foo(Tensor([10]))
    assert ret1
    assert not ret2


def test_sequence_count_dynamic_sequence_variable_target_2():
    """
    Feature: Sequence count operation
    Description: Sequence count operation will use SequenceCount operation
    Expectation: No exception.
    """
    @jit
    def foo(target):
        x = mutable((Tensor([1]), Tensor([2]), Tensor([3])), True)
        ret = x.count(target)
        return isinstance(ret, int), F.isconstant(ret)
    ret1, ret2 = foo(Tensor([10]))
    assert ret1
    assert not ret2


def test_sequence_count_constant_sequence_variable_target():
    """
    Feature: Sequence count operation
    Description: Sequence count operation will use SequenceCount operation
    Expectation: No exception.
    """
    @jit
    def foo(target):
        x = [Tensor([1]), Tensor([2]), Tensor([3])]
        ret = x.count(target)
        return isinstance(ret, int), F.isconstant(ret)
    ret1, ret2 = foo(Tensor([10]))
    assert ret1
    assert not ret2


def test_sequence_count_constant_sequence_variable_target_2():
    """
    Feature: Sequence count operation
    Description: Sequence count operation will use SequenceCount operation
    Expectation: No exception.
    """
    @jit
    def foo(target):
        x = (Tensor([1]), Tensor([2]), Tensor([3]))
        ret = x.count(target)
        return isinstance(ret, int), F.isconstant(ret)
    ret1, ret2 = foo(Tensor([10]))
    assert ret1
    assert not ret2
