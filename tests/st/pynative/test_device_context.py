# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Test DeviceCtx on real devices."""

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, context, mint, ops
from mindspore._c_expression import DeviceManagerConf
from mindspore.graph.api import _pynative_executor
from mindspore.ops.primitive import _run_op_sync
from mindspore.utils.ms_dispatch import MsDispatchMode
from tests.mark_utils import arg_mark


_DISPATCHED_OPS = []


class _RecordDispatchMode(MsDispatchMode):
    def __ms_dispatch__(self, func, args=(), kwargs=None):
        _DISPATCHED_OPS.append(func.name)
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


def _assert_current_device(device_target):
    """Check public and internal current-device targets are consistent."""
    current_device = ms.get_current_device()
    assert current_device.device_target == device_target
    assert context.get_context("device_target") == device_target
    assert DeviceManagerConf.get_instance().get_device_target() == device_target


def _assert_tensor_device(tensor, expected_device):
    if expected_device == "CPU":
        assert tensor.device == "CPU"
    else:
        assert tensor.device.startswith(expected_device)


def _assert_tensor_result(tensor, expected, expected_device):
    _pynative_executor.sync()
    assert np.allclose(tensor.asnumpy(), expected)
    _assert_tensor_device(tensor, expected_device)


def _make_pair():
    x = Tensor(np.array([1.0, 2.0, 3.0], np.float32))
    y = Tensor(np.array([4.0, 5.0, 6.0], np.float32))
    return x, y


def _make_tensor_on_device(array, expected_device):
    tensor = Tensor(array)
    if expected_device != "CPU":
        tensor = mint.add(tensor, Tensor(np.zeros_like(array)))
        _assert_tensor_device(tensor, expected_device)
    return tensor


def _run_add_and_check(expected_device):
    x, y = _make_pair()
    out = mint.add(x, y)
    _assert_tensor_result(out, np.array([5.0, 7.0, 9.0], np.float32), expected_device)


def _run_ops_add_and_check(expected_device):
    x, y = _make_pair()
    out = ops.add(x, y)
    _assert_tensor_result(out, np.array([5.0, 7.0, 9.0], np.float32), expected_device)


def _run_addn_primitive_and_check(expected_device):
    x, y = _make_pair()
    out = ops.AddN()((x, y))
    _assert_tensor_result(out, np.array([5.0, 7.0, 9.0], np.float32), expected_device)


def _run_addn_sync_and_check(expected_device):
    x, y = _make_pair()
    addn = ops.AddN()
    out = _run_op_sync(addn, addn.name, ((x, y),))
    _assert_tensor_result(out, np.array([5.0, 7.0, 9.0], np.float32), expected_device)


def _run_tensor_overload_and_check(expected_device):
    x, y = _make_pair()
    function_out = mint.add(x, y)
    _assert_tensor_result(function_out, np.array([5.0, 7.0, 9.0], np.float32), expected_device)

    x, y = _make_pair()
    ops_function_out = ops.add(x, y)
    _assert_tensor_result(ops_function_out, np.array([5.0, 7.0, 9.0], np.float32), expected_device)

    x, y = _make_pair()
    method_out = x.add(y)
    _assert_tensor_result(method_out, np.array([5.0, 7.0, 9.0], np.float32), expected_device)

    x, y = _make_pair()
    binary_out = x + y
    _assert_tensor_result(binary_out, np.array([5.0, 7.0, 9.0], np.float32), expected_device)


def _run_tensor_getitem_setitem_and_check(expected_device):
    base_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    getitem_cases = (
        (lambda x: x[True], base_np[True]),
        (lambda x: x[1], base_np[1]),
        (lambda x: x[0:2:1], base_np[0:2:1]),
        (lambda x: x[None], base_np[None]),
        (lambda x: x[...], base_np[...]),
        (lambda x: x[1, :, 2], base_np[1, :, 2]),
        (lambda x: x[..., 1, None], base_np[..., 1, None]),
        (lambda x: x[[0, 1], [1, 2], [2, 3]], base_np[[0, 1], [1, 2], [2, 3]]),
        (lambda x: x[Tensor(1)], base_np[1]),
    )
    for getter, expected in getitem_cases:
        x = _make_tensor_on_device(base_np, expected_device)
        _assert_tensor_result(getter(x), expected, expected_device)

    zero_np = np.zeros((2, 3, 4), np.float32)
    one_np = np.ones((2, 3, 4), np.float32)
    setitem_cases = (
        (lambda x: x.__setitem__(True, True), one_np),
        (lambda x: x.__setitem__(False, 7.0), zero_np),
        (lambda x: x.__setitem__(None, 2.0), np.full((2, 3, 4), 2.0, np.float32)),
        (lambda x: x.__setitem__(Ellipsis, 3), np.full((2, 3, 4), 3.0, np.float32)),
        (
            lambda x: x.__setitem__(1, Tensor(np.ones((3, 4), np.float32) * 4)),
            np.stack((np.zeros((3, 4), np.float32), np.ones((3, 4), np.float32) * 4)),
        ),
        (lambda x: x.__setitem__(slice(0, 2), 5.0), np.full((2, 3, 4), 5.0, np.float32)),
        (
            lambda x: x.__setitem__((1, slice(None), 2), 6.0),
            np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                      [[0, 0, 6, 0], [0, 0, 6, 0], [0, 0, 6, 0]]], np.float32),
        ),
        (
            lambda x: x.__setitem__(([0, 1], [1, 2], [2, 3]), 9.0),
            np.array([[[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 0, 0]],
                      [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 9]]], np.float32),
        ),
        (
            lambda x: x.__setitem__(Tensor(1), Tensor(np.ones((3, 4), np.float32) * 8)),
            np.stack((np.zeros((3, 4), np.float32), np.ones((3, 4), np.float32) * 8)),
        ),
    )
    for setter, expected in setitem_cases:
        x = _make_tensor_on_device(zero_np, expected_device)
        setter(x)
        _assert_tensor_result(x, expected, expected_device)


def _run_ms_dispatch_and_check(expected_device):
    global _DISPATCHED_OPS
    _DISPATCHED_OPS = []
    x, y = _make_pair()
    with _RecordDispatchMode():
        out = mint.add(x, y)
    _assert_tensor_result(out, np.array([5.0, 7.0, 9.0], np.float32), expected_device)
    assert _DISPATCHED_OPS == ["AddExt"]


def _run_backward_and_check(expected_device):
    x = Tensor(np.array([1.0, 2.0], np.float32))
    y = Tensor(np.array([3.0, 4.0], np.float32))

    def forward_fn(input_x, input_y):
        return (mint.add(input_x, input_y) * input_x).sum()

    value, grads = ms.value_and_grad(forward_fn, grad_position=(0, 1))(x, y)
    _assert_tensor_result(value, np.array(16.0, np.float32), expected_device)
    _assert_tensor_result(grads[0], np.array([5.0, 8.0], np.float32), expected_device)
    _assert_tensor_result(grads[1], np.array([1.0, 2.0], np.float32), expected_device)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', mem_peak=0.01)
def test_device_ctx_all():
    """
    Feature: DeviceCtx.
    Description: Run PyNative paths on Ascend CI and switch devices only through DeviceCtx.
    Expectation: DeviceCtx routes ops to the active device and restores the previous device.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    origin_device = ms.get_current_device()
    current_device_id = context.get_context("device_id")

    with pytest.raises(ValueError):
        with ms.DeviceCtx("Davinci"):
            pass

    with pytest.raises(ValueError):
        with ms.DeviceCtx("Ascend", -1):
            pass

    with ms.DeviceCtx("Ascend"):
        _assert_current_device("Ascend")
        _run_add_and_check("Ascend")
        _run_ops_add_and_check("Ascend")

    with ms.DeviceCtx("Ascend", current_device_id):
        _assert_current_device("Ascend")
        _run_add_and_check("Ascend")
        _run_ops_add_and_check("Ascend")

    with ms.DeviceCtx("Ascend"):
        _assert_current_device("Ascend")
        _run_addn_primitive_and_check("Ascend")
        _run_addn_sync_and_check("Ascend")
        _run_tensor_overload_and_check("Ascend")
        _run_tensor_getitem_setitem_and_check("Ascend")
        _run_ms_dispatch_and_check("Ascend")
        _run_backward_and_check("Ascend")
        pending_np = np.arange(1024, dtype=np.float32)
        pending_ascend = mint.add(Tensor(pending_np), Tensor(pending_np))
        _assert_tensor_device(pending_ascend, "Ascend")

        with ms.DeviceCtx("CPU"):
            _assert_current_device("CPU")
            _run_add_and_check("CPU")
            _run_ops_add_and_check("CPU")
            _run_addn_primitive_and_check("CPU")
            _run_addn_sync_and_check("CPU")
            _run_tensor_overload_and_check("CPU")
            _run_tensor_getitem_setitem_and_check("CPU")
            _run_ms_dispatch_and_check("CPU")
            _run_backward_and_check("CPU")

            with ms.DeviceCtx("meta"):
                _assert_current_device("meta")
                tensor = mint.empty((2, 3), dtype=ms.float32)
                assert tensor.device == "meta"
                assert tensor.data_ptr() == 0

            _assert_current_device("CPU")
            _run_add_and_check("CPU")

        _assert_current_device("Ascend")
        _assert_tensor_result(pending_ascend, pending_np + pending_np, "Ascend")
        _run_add_and_check("Ascend")
        _run_tensor_getitem_setitem_and_check("Ascend")

    _assert_current_device(origin_device.device_target)
