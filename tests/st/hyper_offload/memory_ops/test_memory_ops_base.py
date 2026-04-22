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
"""Base tests for HyperOffload memory operators."""

import os
from contextlib import contextmanager

import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, jit, ops

from tests.mark_utils import arg_mark


@contextmanager
def _hyper_offload_guard(enabled="1"):
    """Temporarily set HyperOffload env and restore it after test."""
    old_value = os.environ.get("MS_DEV_HYPER_OFFLOAD")
    os.environ["MS_DEV_HYPER_OFFLOAD"] = enabled
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("MS_DEV_HYPER_OFFLOAD", None)
        else:
            os.environ["MS_DEV_HYPER_OFFLOAD"] = old_value


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    mem_peak=4.0)
def test_ops_get_data():
    """
    Feature: Hyper offload base operator
    Description: Verify GetData handles in-place updates correctly.
    Expectation: No Exception.
    """
    with _hyper_offload_guard("1"):

        @jit
        def foo(x):
            return ops.auto_generate.GetData()(x)

        x = Tensor([1, 2, 3, 4])
        result = foo(x)
        assert np.all(result.asnumpy() == np.array((1, 2, 3, 4)))

        @jit
        def foo1(x):
            y = ops.auto_generate.GetData()(x)
            x.add_(1)
            return y

        x = Tensor([1, 2, 3, 4])
        result = foo1(x)
        assert np.all(result.asnumpy() == np.array((2, 3, 4, 5)))

        @jit
        def foo2(x):
            y = ops.auto_generate.GetData()(x)
            x.add_(1)
            y.add_(1)
            return y

        x = Tensor([1, 2, 3, 4])
        result = foo2(x)
        assert np.all(result.asnumpy() == np.array((3, 4, 5, 6)))


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    mem_peak=2.0)
def test_remote_ops_copy_to_host():
    """
    Feature: Hyper offload base operator
    Description: Verify CopyToHost output is CPU tensor.
    Expectation: No Exception.
    """
    with _hyper_offload_guard("1"):

        @jit
        def foo(x):
            return ops.auto_generate.CopyToHost()(x)

        x = Tensor([1, 2, 3, 4])
        result = foo(x)
        assert np.all(result.asnumpy() == np.array((1, 2, 3, 4)))
        assert result.device == "CPU"


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    mem_peak=0.01)
def test_copy_to_host():
    """
    Feature: Hyper offload base operator
    Description: Verify in-place update on host copied tensor raises runtime error.
    Expectation: Throw RuntimeError.
    """
    with _hyper_offload_guard("1"):

        @jit
        def foo(x):
            copied = ops.auto_generate.CopyToHost()(x)
            copied.add_(1)
            return copied

        with pytest.raises(RuntimeError) as error_info:
            x = Tensor([1, 2, 3, 4])
            foo(x)
        assert "Invalid input device tensor type for kernel" in str(error_info.value)


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    mem_peak=4.0)
def test_memory_ops_set_data():
    """
    Feature: Hyper offload base operator
    Description: Verify SetData alias behavior in different update patterns.
    Expectation: No Exception.
    """
    with _hyper_offload_guard("1"):

        @jit
        def foo1(x, a):
            ops.auto_generate.SetData()(x, a)
            return x

        x = Tensor([1, 2, 3, 4])
        a = Tensor([5, 6, 7, 8])
        result = foo1(x, a)
        assert np.all(result.asnumpy() == np.array((5, 6, 7, 8)))

        @jit
        def foo2(x, a):
            ops.auto_generate.SetData()(x, a)
            a.add_(1)
            return x

        x = Tensor([1, 2, 3, 4])
        a = Tensor([5, 6, 7, 8])
        result = foo2(x, a)
        assert np.all(result.asnumpy() == np.array((6, 7, 8, 9)))

        @jit
        def foo3(x, a):
            ops.auto_generate.SetData()(x, a)
            a.add_(1)
            x.add_(1)
            return a

        x = Tensor([1, 2, 3, 4])
        a = Tensor([5, 6, 7, 8])
        result = foo3(x, a)
        assert np.all(result.asnumpy() == np.array((7, 8, 9, 10)))

        @jit
        def foo4(x, a):
            b = a + 1
            y = x * 5
            ops.auto_generate.SetData()(y, b)
            y.add_(1)
            b.mul_(2)
            return y

        x = Tensor([1, 1, 1])
        a = Tensor([2, 2, 2])
        result = foo4(x, a)
        assert np.all(result.asnumpy() == np.array((8, 8, 8)))


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    mem_peak=4.0)
def test_remote_ops_copy_to():
    """
    Feature: Hyper offload base operator
    Description: Verify CopyToHost + CopyToDevice keeps values and device placement.
    Expectation: No Exception.
    """
    with _hyper_offload_guard("1"):
        ms.set_context(device_id=0)

        @jit
        def foo(x):
            x = ops.auto_generate.CopyToHost()(x)
            return ops.auto_generate.CopyToDevice()(x)

        x = Tensor([1, 2, 3, 4])
        result = foo(x)
        assert np.all(result.asnumpy() == np.array((1, 2, 3, 4)))
        assert result.device == "Ascend:0"


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    mem_peak=4.0)
def test_ops_to_host_free_set_data():
    """
    Feature: Hyper offload base operator
    Description: Verify CopyToHost + Free + SetData flow keeps tensor values.
    Expectation: No Exception.
    """
    with _hyper_offload_guard("1"):
        ms.set_context(device_id=0)

        @jit
        def foo(x):
            host_tensor = ops.auto_generate.CopyToHost()(x)
            freed_tensor = ops.auto_generate.Free()(x)
            ops.auto_generate.SetData()(freed_tensor, host_tensor)
            return freed_tensor

        x = Tensor([1, 2, 3, 4])
        result = foo(x)
        assert np.all(result.asnumpy() == np.array((1, 2, 3, 4)))


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    mem_peak=2.0)
def test_remote_ops_copy_to_and_free():
    """
    Feature: Hyper offload base operator
    Description: Verify CopyToHost on released tensor raises runtime error.
    Expectation: Throw RuntimeError.
    """
    with _hyper_offload_guard("1"):
        ms.set_context(device_id=0)

        @jit
        def foo(x):
            copied = ops.auto_generate.CopyToHost()(x)
            ops.auto_generate.Free()(x)
            recopy = ops.auto_generate.CopyToHost()(x)
            return copied, recopy

        x = Tensor([1, 2, 3, 4])
        result = foo(x)
        with pytest.raises(RuntimeError) as error_info:
            _ = result[0].asnumpy()
        assert "The first input of Primitive 'CopyToHost' has been released before." in str(error_info.value)
