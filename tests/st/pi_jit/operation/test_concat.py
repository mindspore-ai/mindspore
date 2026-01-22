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
"""Test concat operation"""
import numpy as np
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import Tensor
from mindspore import context
from mindspore import ops
from mindspore.graph.api import jit
from ..share.utils import match_array, assert_executed_by_graph_mode


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cat():
    """
    Feature: ops.cat()
    Description: Test one stage basic operation.
    Expectation: No exception, no graph-break.
    """
    @jit(capture_mode="bytecode")
    def fn(x: Tensor, y: Tensor):
        return ops.cat((x, y))

    context.set_context(mode=context.PYNATIVE_MODE)
    x1 = Tensor(np.array([1, 2, 3]))
    x2 = Tensor(np.array([4, 5, 6]))
    ret = fn(x1, x2)
    match_array(ret.asnumpy(), np.array([1, 2, 3, 4, 5, 6]))
    assert_executed_by_graph_mode(fn)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cat_axis_0():
    """
    Feature: ops.cat()
    Description: Test one stage basic operation.
    Expectation: No exception, no graph-break.
    """
    @jit(capture_mode="bytecode")
    def fn(x: Tensor, y: Tensor):
        return ops.cat((x, y), axis=0)

    context.set_context(mode=context.PYNATIVE_MODE)
    x1 = Tensor(np.array([[0, 1], [2, 1]]))
    x2 = Tensor(np.array([[0, 1], [2, 1]]))
    ret = fn(x1, x2)
    match_array(ret.asnumpy(), np.array([[0, 1], [2, 1], [0, 1], [2, 1]]))
    assert_executed_by_graph_mode(fn)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cat_axis_1():
    """
    Feature: ops.cat()
    Description: Test one stage basic operation.
    Expectation: No exception, no graph-break.
    """
    @jit(capture_mode="bytecode")
    def fn(x, y):
        return ops.cat((x, y), axis=1)

    context.set_context(mode=context.PYNATIVE_MODE)
    x1 = Tensor(np.array([[0, 1], [2, 1]]))
    x2 = Tensor(np.array([[0, 1], [2, 1]]))
    ret = fn(x1, x2)
    match_array(ret.asnumpy(), np.array([[0, 1, 0, 1], [2, 1, 2, 1]]))
    assert_executed_by_graph_mode(fn)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cat_three_tensors_at_axis_0():
    """
    Feature: ops.cat()
    Description: Test one stage basic operation.
    Expectation: No exception, no graph-break.
    """
    @jit(capture_mode="bytecode")
    def fn(x, y, z):
        return ops.cat([x, y, z], axis=0)

    context.set_context(mode=context.PYNATIVE_MODE)
    x1 = Tensor(np.array([[1, 2], [3, 4]]))
    x2 = Tensor(np.array([[5, 6], [7, 8]]))
    x3 = Tensor(np.array([[9, 0]]))
    ret = fn(x1, x2, x3)
    match_array(ret.asnumpy(), np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 0]]))
    assert_executed_by_graph_mode(fn)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cat_three_tensors_at_axis_negative_1():
    """
    Feature: ops.cat()
    Description: Test one stage basic operation.
    Expectation: No exception, no graph-break.
    """
    @jit(capture_mode="bytecode")
    def fn(x, y, z):
        return ops.cat([x, y, z], axis=-1)

    context.set_context(mode=context.PYNATIVE_MODE)
    x1 = Tensor(np.array([[1], [2]]))
    x2 = Tensor(np.array([[2, 2], [3, 3]]))
    x3 = Tensor(np.array([[3, 3, 3], [4, 4, 4]]))
    ret = fn(x1, x2, x3)
    match_array(ret.asnumpy(), np.array([[1, 2, 2, 3, 3, 3], [2, 3, 3, 4, 4, 4]]))
    assert_executed_by_graph_mode(fn)
