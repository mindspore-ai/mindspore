# Copyright 2023 Huawei Technologies Co., Ltd
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
"""test mutable with dynamic length"""
from mindspore.common import mutable
from mindspore import Tensor
from mindspore import jit
from mindspore import context
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_generate_mutable_sequence_with_dynamic_length_with_jit():
    """
    Feature: Mutable with dynamic length.
    Description: Generate mutable sequence of dynamic length with in jit.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        output1 = mutable([1, 2, 3, 4], True)
        output2 = mutable([Tensor([1]), Tensor([2]), Tensor([3])], True)
        output3 = mutable([(1, 2, 3), (2, 3, 4), (3, 4, 5)], True)
        return output1, output2, output3
    ret = foo()
    assert len(ret) == 3
    assert ret[0] == [1, 2, 3, 4]
    assert ret[1] == [Tensor([1]), Tensor([2]), Tensor([3])]
    assert ret[2] == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_generate_mutable_sequence_with_dynamic_length_and_shape__with_jit():
    """
    Feature: Mutable with dynamic length.
    Description: Generate mutable sequence of dynamic length and shape with in jit.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    @jit
    def foo():
        output1 = mutable([Tensor([[1, 2, 3]]), Tensor([[2]]), Tensor([[3], [4]])], True)
        output2 = mutable([(1,), (2, 3), (4, 5, 6)], True)
        return output1, output2
    ret = foo()
    assert len(ret) == 2
    assert len(ret[0]) == 3
    expect_tensor = [Tensor([[1, 2, 3]]), Tensor([[2]]), Tensor([[3], [4]])]
    for x in ret[0]:
        for y in expect_tensor:
            assert x.astype("bool").all().asnumpy() == y.astype("bool").all().asnumpy()
    assert ret[1] == [(1,), (2, 3), (4, 5, 6)]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mutable_dynamic_len_with_any():
    """
    Feature: Mutable with dynamic length.
    Description: Generate mutable sequence of dynamic length with in jit.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)

    @jit
    def foo(inputs):
        x = [mutable(inputs), inputs]
        x = mutable(x, True)
        return x

    ret = foo([(1, 2), 2, 2.])
    assert ret == [[(1, 2), 2, 2.], [(1, 2), 2, 2.]]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mutable_dynamic_len_with_any_2():
    """
    Feature: Mutable with dynamic length.
    Description: Generate mutable sequence of dynamic length with in jit.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)

    @jit
    def foo(inputs):
        x = [inputs, mutable(inputs)]
        x = mutable(x, True)
        return x

    ret = foo([(1, 2), 2, 2.])
    assert ret == [[(1, 2), 2, 2.], [(1, 2), 2, 2.]]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mutable_dynamic_len_with_any_3():
    """
    Feature: Mutable with dynamic length.
    Description: Generate mutable sequence of dynamic length with in jit.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)

    @jit
    def foo(inputs, index):
        x = mutable(inputs[mutable(index)], True)
        y = x[1]
        return y

    ret = foo([[1, 2], [4, 5], [2, 2]], 1)
    assert ret == 5
