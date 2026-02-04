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
"""Test Tensor methods"""

import pytest

import mindspore
from mindspore import context, ops, jit

import pytest
from tests.st.pi_jit.share.utils import assert_equal, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark


@pytest.mark.skip
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_type_guard():
    """
    Feature: Test Tensor TypeGuard.
    Description: Calling x.squeeze() will add a TypeGuard on x.squeeze.__self__.
    Expectation: No guard checking failure, no exception, no graph break.
    """

    def fn(x):
        return ops.add(x.squeeze(), 1)

    context.set_context(mode=context.PYNATIVE_MODE)
    x = ops.arange(0, 4)
    o1 = fn(x)

    compiled_fn = jit(fn, capture_mode='bytecode')
    x = ops.arange(0, 4)  # It is a StubTensor
    o2 = compiled_fn(x)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)


@pytest.mark.skip
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_stub_tensor_load_method():
    """
    Feature: Test Tensor LOAD_METHOD.
    Description: Properly handle the self parameter in Tensor methods to avoid treating self as a constant.
    Expectation: no exception, no graph break.
    """

    def fn(x):
        y = x.permute((0, 2, 1))
        return y + 1

    context.set_context(mode=context.PYNATIVE_MODE)
    x1 = ops.randn(2, 3, 4)  # It is a StubTensor
    o1 = fn(x1)

    compiled_fn = jit(fn, capture_mode='bytecode')
    o2 = compiled_fn(x1)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(fn)

    x2 = ops.randn(2, 3, 4)  # It is a StubTensor
    o1 = fn(x2)

    compiled_fn = jit(fn, capture_mode='bytecode')
    o2 = compiled_fn(x2)

    assert_equal(o1, o2)
    # Should not treat argument x as a constant tensor, should reuse the first graph.
    assert_executed_by_graph_mode(fn, call_count=2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_getitem_setitem_by_slice_v1():
    """
    Feature: Test Tensor getitem/setitem.
    Description: Test two tensors getitem by slice, and the slice is a variable.
    Expectation: no exception, no graph break.
    """

    def fn(x, kv_cache, pe_cache, start: int):
        B = x.shape[0]
        y = x + 1
        # start is mutable, so the slice is variable, not constant.
        kv_cache[:B, start : start + 1] = y
        pe_cache[:B, start : start + 1] = y + 1
        return kv_cache + pe_cache

    x = ops.randn(2, 1, 2, dtype=mindspore.float32)
    kv_cache1 = ops.zeros((2, 2, 2), dtype=mindspore.float32)
    pe_cache1 = ops.zeros((2, 2, 2), dtype=mindspore.float32)
    start = mindspore.mutable(1)  # Use mutable to ensure the slice is variable

    o1 = fn(x, kv_cache1, pe_cache1, start)

    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)
    kv_cache2 = ops.zeros((2, 2, 2), dtype=mindspore.float32)
    pe_cache2 = ops.zeros((2, 2, 2), dtype=mindspore.float32)
    o2 = compiled_fn(x, kv_cache2, pe_cache2, start)

    assert_equal(o1, o2)
    assert_equal(kv_cache1, kv_cache2)
    assert_equal(pe_cache1, pe_cache2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_getitem_setitem_by_slice_v2():
    """
    Feature: Test Tensor getitem/setitem.
    Description: Test two tensors getitem by slice, and the slice is a variable.
    Expectation: no exception, no graph break.
    """

    def fn(x, kv_cache, pe_cache, start: int):
        y = x + 1
        # start is mutable, so the slice is variable, not constant.
        kv_cache[start : start + 1] = y
        pe_cache[start : start + 1] = y + 1
        return kv_cache + pe_cache

    x = ops.randn(1, 2, dtype=mindspore.float32)
    kv_cache1 = ops.zeros((2, 2), dtype=mindspore.float32)
    pe_cache1 = ops.zeros((2, 2), dtype=mindspore.float32)
    start = mindspore.mutable(1)  # Use mutable to ensure the slice is variable

    o1 = fn(x, kv_cache1, pe_cache1, start)

    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)
    kv_cache2 = ops.zeros((2, 2), dtype=mindspore.float32)
    pe_cache2 = ops.zeros((2, 2), dtype=mindspore.float32)
    o2 = compiled_fn(x, kv_cache2, pe_cache2, start)

    assert_equal(o1, o2)
    assert_equal(kv_cache1, kv_cache2)
    assert_equal(pe_cache1, pe_cache2)
    assert_executed_by_graph_mode(compiled_fn)
