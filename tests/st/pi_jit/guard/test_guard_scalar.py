# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
"""Test guard scalar"""

import pytest
import mindspore as ms
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import assert_graph_compile_status, reset_func


@ms.jit(capture_mode="bytecode", fullgraph=True)
def func(x, y=ms.Tensor([1])): # Tensor as input to ensure graph compile.
    return x, y + 1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("value", [1, 2.0, 'str', b'bytes', [1 + 1j]])
def test_guard_scalar(value):
    """
    Feature: Guard
    Description: Test guard for scalar.
    Expectation: No exception
    """
    out1, _ = func(value)
    # guard success
    out2, _ = func(value)
    assert_graph_compile_status(func, 0, 2, 1) # break_count: 0, call_count: 2, compile_count: 1
    expect = value if not isinstance(value, bytes) else str(value, encoding='utf-8')
    assert out1 == expect and out2 == expect
    # guard fail
    out3, _ = func(value * 2)
    assert_graph_compile_status(func, 0, 1, 2) # break_count: 0, call_count: 1, compile_count: 2
    assert out3 == expect * 2
    reset_func(func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("value", [True, False])
def test_guard_bool(value):
    """
    Feature: Guard
    Description: Test guard for bool, 
    Expectation: No exception
    """
    out1, _ = func(value)
    # guard success
    out2, _ = func(value)
    assert_graph_compile_status(func, 0, 2, 1)
    assert out2 == out1
    # guard fail
    out3, _ = func(not value)
    assert_graph_compile_status(func, 0, 1, 2)
    assert out3 != out1
    reset_func(func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("value", [1, 1.0])
def test_guard_mutable_scalar(value):
    """
    Feature: Guard
    Description: Test guard for mutable scalar.
    Expectation: No exception
    """
    func(ms.mutable(value))
    func(value * 2)
    func(value * 3)
    assert_graph_compile_status(func, 0, 3, 1)
    reset_func(func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_guard_mutable_bool():
    """
    Feature: Guard
    Description: Test guard for mutable bool.
    Expectation: No exception
    """
    func(ms.mutable(True))
    func(True)
    func(False)
    assert_graph_compile_status(func, 0, 3, 1)
    reset_func(func)
