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
"""Test call class to create instance"""
import mindspore as ms
from mindspore import Tensor, jit
from mindspore import context
from ..share.utils import match_array, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_tensor():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    def fn():
        a = Tensor([1, 2, 3])
        return a + 1

    context.set_context(mode=context.PYNATIVE_MODE)

    expect = fn()
    actual = jit(fn, capture_mode="bytecode")()
    match_array(actual.asnumpy(), expect.asnumpy())
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_tensor_list():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    def fn():
        t1 = Tensor([1, 1, 1])
        t2 = Tensor([2, 2, 2])
        return [t1, t2]

    context.set_context(mode=context.PYNATIVE_MODE)

    expect = fn()
    actual = jit(fn, capture_mode="bytecode")()

    assert isinstance(actual, list)
    assert len(actual) == 2
    match_array(actual[0].asnumpy(), expect[0].asnumpy())
    match_array(actual[1].asnumpy(), expect[1].asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_create_tensor_by_ms_api():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    def fn():
        a = ms.tensor([1, 2, 3])
        return a + 1

    context.set_context(mode=context.PYNATIVE_MODE)
    expect = fn()

    fn = jit(fn, capture_mode="bytecode")
    actual = fn()

    match_array(actual.asnumpy(), expect.asnumpy())
    assert_executed_by_graph_mode(fn)
