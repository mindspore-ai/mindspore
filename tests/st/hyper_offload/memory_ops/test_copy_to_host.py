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
"""Tests for HyperOffload CopyToHost operator."""

import numpy as np
import mindspore as ms
from mindspore import Tensor, jit, ops

from tests.mark_utils import arg_mark


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=2.00,
)
def test_remote_ops_copy_to_host():
    """
    Feature: Hyper offload base operator
    Description: Verify CopyToHost keeps values and output is on CPU.
    Expectation: No Exception.
    """

    @jit
    def foo(x):
        return ops.auto_generate.CopyToHost()(x)

    x = Tensor([1, 2, 3, 4])
    result = foo(x)
    assert np.all(result.asnumpy() == np.array((1, 2, 3, 4)))
    assert result.device == "CPU"


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=4.00,
)
def test_remote_ops_copy_to_host_and_element_wise():
    """
    Feature: Hyper offload base operator
    Description: Verify element-wise result remains correct after CopyToHost.
    Expectation: No Exception.
    """

    @jit
    def foo(x, y):
        mul = ops.Mul()
        return ops.auto_generate.CopyToHost()(mul(x, y))

    input_x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    input_y = Tensor(np.array([4.0, 5.0, 6.0]), ms.float32)
    result = foo(input_x, input_y)
    assert np.all(result.asnumpy() == np.array((4, 10, 18)))
    assert result.device == "CPU"


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=4.00,
)
def test_remote_ops_copy_to_host_and_squeeze():
    """
    Feature: Hyper offload base operator
    Description: Verify squeeze output remains correct after CopyToHost.
    Expectation: No Exception.
    """

    @jit
    def foo(x):
        squeeze = ops.Squeeze(2)
        return ops.auto_generate.CopyToHost()(squeeze(x))

    input_tensor = Tensor(np.ones(shape=[3, 2, 1]), ms.float32)
    result = foo(input_tensor)
    assert np.all(result.asnumpy() == np.array([[1, 1], [1, 1], [1, 1]]))
    assert result.device == "CPU"


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=2.00,
)
def test_remote_ops_copy_to_host_and_broadcast_to():
    """
    Feature: Hyper offload base operator
    Description: Verify broadcast output remains correct after CopyToHost.
    Expectation: No Exception.
    """

    @jit
    def foo(x, shape):
        broadcast_to = ops.BroadcastTo(shape)
        broadcast_to(x)
        return ops.auto_generate.CopyToHost()(x)

    shape = (2, 3)
    input_x = Tensor(np.array([1, 2, 3]).astype(np.float32))
    result = foo(input_x, shape)
    assert np.all(result.asnumpy() == np.array([[1, 2, 3], [1, 2, 3]]))
    assert result.device == "CPU"


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=4.00,
)
def test_remote_ops_copy_to_host_and_smooth_l1_loss():
    """
    Feature: Hyper offload base operator
    Description: Verify SmoothL1Loss output remains correct after CopyToHost.
    Expectation: No Exception.
    """

    @jit
    def foo(input_data, target_data):
        loss = ops.SmoothL1Loss()
        output = loss(input_data, target_data)
        return ops.auto_generate.CopyToHost()(output)

    input_data = Tensor(np.array([1, 2, 3]), ms.float32)
    target_data = Tensor(np.array([1, 2, 2]), ms.float32)
    result = foo(input_data, target_data)
    assert np.all(result.asnumpy() == np.array([0, 0, 0.5]))
    assert result.device == "CPU"


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=4.00,
)
def test_remote_ops_copy_to_host_and_add():
    """
    Feature: Hyper offload base operator
    Description: Verify add output remains correct after CopyToHost.
    Expectation: No Exception.
    """

    @jit
    def foo(x, y):
        add = ops.Add()
        return ops.auto_generate.CopyToHost()(add(x, y))

    x = Tensor([1, 2, 3, 4])
    y = Tensor([1, 2, 3, 4])
    result = foo(x, y)
    assert np.all(result.asnumpy() == np.array((2, 4, 6, 8)))
    assert result.device == "CPU"
