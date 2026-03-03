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
"""Tests for HyperOffload CopyToDevice operator."""

import os

import numpy as np
import mindspore as ms
from mindspore import Tensor, context, jit, ops

from tests.mark_utils import arg_mark


os.environ["DEVICE_ID"] = "0"
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=4.00,
)
def test_remote_ops_copy_to_device():
    """
    Feature: Device memory base operator
    Description: Verify CopyToHost + CopyToDevice keeps values and device placement.
    Expectation: No Exception.
    """

    @jit
    def foo(x):
        x = ops.auto_generate.CopyToHost()(x)
        x = ops.auto_generate.CopyToDevice()(x)
        return x

    x = Tensor([1, 2, 3, 4])
    result = foo(x)
    assert np.all(result.asnumpy() == np.array((1, 2, 3, 4)))
    assert result.device == "Ascend:0"


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=4.00,
)
def test_remote_ops_copy_to_device_and_element_wise():
    """
    Feature: Device memory base operator
    Description: Verify element-wise op result remains correct after host/device copy.
    Expectation: No Exception.
    """

    @jit
    def foo(x, y):
        mul = ops.Mul()
        result = mul(x, y)
        result = ops.auto_generate.CopyToHost()(result)
        return ops.auto_generate.CopyToDevice()(result)

    input_x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    input_y = Tensor(np.array([4.0, 5.0, 6.0]), ms.float32)
    result = foo(input_x, input_y)
    assert np.all(result.asnumpy() == np.array((4, 10, 18)))
    assert result.device == "Ascend:0"


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=4.00,
)
def test_remote_ops_copy_to_device_and_squeeze():
    """
    Feature: Device memory base operator
    Description: Verify squeeze output remains correct after host/device copy.
    Expectation: No Exception.
    """

    @jit
    def foo(x):
        squeeze = ops.Squeeze(2)
        squeezed = squeeze(x)
        squeezed = ops.auto_generate.CopyToHost()(squeezed)
        return ops.auto_generate.CopyToDevice()(squeezed)

    input_tensor = Tensor(np.ones(shape=[3, 2, 1]), ms.float32)
    result = foo(input_tensor)
    assert np.all(result.asnumpy() == np.array([[1, 1], [1, 1], [1, 1]]))
    assert result.device == "Ascend:0"


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=4.00,
)
def test_remote_ops_copy_to_device_and_broadcast_to():
    """
    Feature: Device memory base operator
    Description: Verify broadcast result remains correct after host/device copy.
    Expectation: No Exception.
    """

    @jit
    def foo(x, shape):
        broadcast_to = ops.BroadcastTo(shape)
        broadcast_to(x)
        x = ops.auto_generate.CopyToHost()(x)
        return ops.auto_generate.CopyToDevice()(x)

    shape = (2, 3)
    input_x = Tensor(np.array([1, 2, 3]).astype(np.float32))
    result = foo(input_x, shape)
    assert np.all(result.asnumpy() == np.array([[1, 2, 3], [1, 2, 3]]))
    assert result.device == "Ascend:0"


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=4.00,
)
def test_remote_ops_copy_to_device_and_smooth_l1_loss():
    """
    Feature: Device memory base operator
    Description: Verify SmoothL1Loss result remains correct after host/device copy.
    Expectation: No Exception.
    """

    @jit
    def foo(input_data, target_data):
        loss = ops.SmoothL1Loss()
        output = loss(input_data, target_data)
        output = ops.auto_generate.CopyToHost()(output)
        return ops.auto_generate.CopyToDevice()(output)

    input_data = Tensor(np.array([1, 2, 3]), ms.float32)
    target_data = Tensor(np.array([1, 2, 2]), ms.float32)
    result = foo(input_data, target_data)
    assert np.all(result.asnumpy() == np.array([0, 0, 0.5]))
    assert result.device == "Ascend:0"


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=4.00,
)
def test_remote_ops_copy_to_device_and_add():
    """
    Feature: Device memory base operator
    Description: Verify add result remains correct after host/device copy.
    Expectation: No Exception.
    """

    @jit
    def foo(x, y):
        add = ops.Add()
        result = add(x, y)
        result = ops.auto_generate.CopyToHost()(result)
        return ops.auto_generate.CopyToDevice()(result)

    x = Tensor([1, 2, 3, 4])
    y = Tensor([1, 2, 3, 4])
    result = foo(x, y)
    assert np.all(result.asnumpy() == np.array((2, 4, 6, 8)))
    assert result.device == "Ascend:0"
