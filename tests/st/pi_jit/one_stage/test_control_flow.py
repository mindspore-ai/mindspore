# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test control flow in pijit"""
from mindspore import Tensor
from mindspore.graph.api import jit
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_if_break_in_condition():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(capture_mode="bytecode")
    def func(x):
        a = 0
        m = x * 3
        if m == 1:
            a = a + 10
        return a + 1

    x = Tensor([1])
    ret = func(x)
    assert ret == 1


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_for_break_in_condition():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(capture_mode="bytecode")
    def func(x):
        a = 0
        m = x * 3
        for _ in range(m):
            a = a + 1
        return a

    x = Tensor([1])
    ret = func(x)
    assert ret == 3


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_while_break_in_condition():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(capture_mode="bytecode")
    def func(x):
        a = 0
        m = x * 3
        i = 0
        while i < m:
            a = a + 1
            i = i + 1
        return a

    x = Tensor([1])
    ret = func(x)
    assert ret == 3
