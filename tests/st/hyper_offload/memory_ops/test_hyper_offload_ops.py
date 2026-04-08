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
"""Tests for HyperOffload runtime operators."""

import os
from contextlib import contextmanager

import numpy as np
import mindspore as ms
from mindspore import Parameter, Tensor, context, jit, mutable, ops
from mindspore.common.initializer import One
from mindspore.nn import Cell

from tests.mark_utils import arg_mark


os.environ["DEVICE_ID"] = "0"
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)


def _restore_env_var(env_name, env_value):
    """Restore env variable to old value."""
    if env_value is None:
        os.environ.pop(env_name, None)
    else:
        os.environ[env_name] = env_value


@contextmanager
def _hyper_offload_guard(enabled="1"):
    """Temporarily set HyperOffload env value and restore after exit."""
    old_value = os.environ.get("MS_DEV_HYPER_OFFLOAD")
    os.environ["MS_DEV_HYPER_OFFLOAD"] = enabled
    try:
        yield
    finally:
        _restore_env_var("MS_DEV_HYPER_OFFLOAD", old_value)


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    mem_peak=4.00)
def test_hiar_addr_to_remote():
    """
    Feature: Hyper offload base operator
    Description: Verify UpdateToRemote keeps values and tensors stay on Ascend.
    Expectation: No Exception.
    """

    @jit
    def foo(x):
        return ops.auto_generate.UpdateToRemote()(x)

    x = Tensor([1, 2, 3, 4])
    result = foo(x)
    assert np.all(result.asnumpy() == np.array((1, 2, 3, 4)))
    assert result.device == "Ascend:0"

    x = Tensor(shape=(6), dtype=ms.float32, init=One())
    result = foo(x)
    assert np.all(result.asnumpy() == np.ones((6)))
    assert result.device == "Ascend:0"

    x = Tensor(shape=(6, 7), dtype=ms.float32, init=One())
    result = foo(x)
    assert np.all(result.asnumpy() == np.ones((6, 7)))
    assert result.device == "Ascend:0"

    x = Tensor(shape=(6, 7, 8), dtype=ms.float32, init=One())
    result = foo(x)
    assert np.all(result.asnumpy() == np.ones((6, 7, 8)))
    assert result.device == "Ascend:0"


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    mem_peak=4.00)
def test_hiar_addr_update_to_device():
    """
    Feature: Hyper offload base operator
    Description: Verify UpdateToDevice keeps values and tensors stay on Ascend.
    Expectation: No Exception.
    """
    with _hyper_offload_guard("1"):

        @jit
        def foo(x):
            return ops.auto_generate.UpdateToDevice()(x)

        x = Tensor([1, 2, 3, 4])
        result = foo(x)
        assert np.all(result.asnumpy() == np.array((1, 2, 3, 4)))

        x = Tensor(shape=(6), dtype=ms.float32, init=One())
        result = foo(x)
        assert np.all(result.asnumpy() == np.ones((6)))
        assert result.device == "Ascend:0"

        x = Tensor(shape=(6, 7), dtype=ms.float32, init=One())
        result = foo(x)
        assert np.all(result.asnumpy() == np.ones((6, 7)))
        assert result.device == "Ascend:0"

        x = Tensor(shape=(6, 7, 8), dtype=ms.float32, init=One())
        result = foo(x)
        assert np.all(result.asnumpy() == np.ones((6, 7, 8)))
        assert result.device == "Ascend:0"


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    mem_peak=4.00)
def test_hiar_addr_detach():
    """
    Feature: Hyper offload base operator
    Description: Verify Detach keeps values stable for different tensor ranks.
    Expectation: No Exception.
    """
    with _hyper_offload_guard("1"):

        @jit
        def foo(x):
            x = ops.auto_generate.UpdateToRemote()(x)
            x = ops.auto_generate.Detach()(x)
            return ops.auto_generate.Detach()(x)

        x = Tensor([1, 2, 3, 4])
        result = foo(x)
        assert np.all(result.asnumpy() == np.array((1, 2, 3, 4)))

        x = Tensor(shape=(6), dtype=ms.float32, init=One())
        result = foo(x)
        assert np.all(result.asnumpy() == np.ones((6)))

        x = Tensor(shape=(6, 7), dtype=ms.float32, init=One())
        result = foo(x)
        assert np.all(result.asnumpy() == np.ones((6, 7)))

        x = Tensor(shape=(6, 7, 8), dtype=ms.float32, init=One())
        result = foo(x)
        assert np.all(result.asnumpy() == np.ones((6, 7, 8)))


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    mem_peak=4.00)
def test_hiar_addr_to_remote_and_detach():
    """
    Feature: Hyper offload base operator
    Description: Verify UpdateToRemote and Detach sequence keeps result value.
    Expectation: No Exception.
    """
    with _hyper_offload_guard("1"):

        @jit
        def foo(x):
            ops.auto_generate.UpdateToRemote()(x)
            ops.auto_generate.Detach()(x)
            return x

        x = Tensor([1, 2, 3, 4])
        result = foo(x)
        assert np.all(result.asnumpy() == np.array((1, 2, 3, 4)))


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    mem_peak=4.00)
def test_hiar_addr_in_for_variable_loop():
    """
    Feature: Hyper offload base operator
    Description: Verify HyperOffload operators in variable loop control flow.
    Expectation: No Exception.
    """
    with _hyper_offload_guard("1"):

        class Net(Cell):
            """Network using update-to-device and detach in loop."""

            def __init__(self):
                super().__init__()
                self.param_a = Parameter(Tensor([1, 1, 1], dtype=ms.int32), name="param_a")
                self.param_b = Parameter(Tensor([1, 1, 1], dtype=ms.int32), name="param_b")
                self.param_c = Parameter(Tensor([1, 1, 1], dtype=ms.int32), name="param_c")
                self.params = self.trainable_params()
                self.prefetch = ops.auto_generate.UpdateToDevice()
                self.depend = ops.Depend()
                self.detach = ops.auto_generate.Detach()

            @jit
            def construct(self, a, b, c, d):
                """Execute update and detach in a variable for-loop."""
                inputs = (a, b, c)
                result = Tensor([0, 0, 0])
                for i in range(d):
                    prefetch_result = self.prefetch(self.params[i], sync=False)
                    current = self.depend(inputs[i], prefetch_result)
                    result = result + current + self.params[i]
                    detach_result = self.detach(inputs[i], sync=False)
                    result = self.depend(result, detach_result)
                return result

        x = Tensor([1, 1, 1], dtype=ms.int32)
        y = Tensor([1, 1, 1], dtype=ms.int32)
        z = Tensor([1, 1, 1], dtype=ms.int32)
        loop_count = mutable(3)
        net = Net()
        result = net(x, y, z, loop_count)
        assert np.all(result.asnumpy() == np.array((6, 6, 6)))
