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
"""Shared fixtures for Cell.to / Cell.to_empty tests."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, Parameter, nn, dtype as mstype
from mindspore._c_expression import NoFallbackGuard


@pytest.fixture(autouse=True)
def pynative_mode():
    ms.set_context(mode=ms.PYNATIVE_MODE)
    yield


class MinimalTraceableTensor(Tensor):
    """
    Minimal traceable Tensor subclass for testing Cell._apply strategy A path.

    The payload slot is observable Python state that later tests can use to
    verify subclass state is preserved or swapped correctly.
    """
    __slots__ = ("_payload_slot",)

    def __new__(cls, data, payload=None):
        if isinstance(data, MinimalTraceableTensor) and payload is None:
            payload = data._payload_slot
        if not isinstance(data, Tensor):
            data = Tensor(data)
        tensor = Tensor._make_subclass(cls, data)
        tensor._payload_slot = payload
        return tensor

    def __tensor_flatten__(self):
        return [], {"payload": self._payload_slot}

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        raise NotImplementedError("MinimalTraceableTensor is only used for Cell._apply protocol checks.")

    def __fallback__(self, func, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        with NoFallbackGuard():
            result = func(*args, **kwargs)
        if isinstance(result, Tensor) and result.__class__ is Tensor:
            return MinimalTraceableTensor(result, payload=self._payload_slot)
        return result


@pytest.fixture
def simple_net():
    """Single-layer Cell with one float Parameter and one int Buffer."""
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor(np.ones((4, 4)), mstype.float32), name="w")
            self.register_buffer("buf", Tensor(np.array([1, 2, 3]), mstype.int64))

        def construct(self, x):
            return x @ self.w

    return Net()


@pytest.fixture
def nested_net():
    """Three-level nested Cell for recurse testing."""
    class Inner(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w_inner = Parameter(Tensor(np.ones((2, 2)), mstype.float32), name="w_inner")

        def construct(self, x):
            return x @ self.w_inner

    class Middle(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w_mid = Parameter(Tensor(np.ones((3, 3)), mstype.float32), name="w_mid")
            self.inner = Inner()

        def construct(self, x):
            return x @ self.w_mid

    class Outer(nn.Cell):
        def __init__(self):
            super().__init__()
            self.w_out = Parameter(Tensor(np.ones((4, 4)), mstype.float32), name="w_out")
            self.middle = Middle()

        def construct(self, x):
            return x @ self.w_out

    return Outer()


def assert_on_device(tensor, device):
    """Assert tensor's device starts with given string, tolerating :0 suffix."""
    actual = tensor.device
    assert actual == device or actual.startswith(device + ":"), \
        f"Expected device starts with {device!r}, got {actual!r}"


def allocated_bytes(tensor):
    """Return actual allocated bytes; 0 for meta tensor."""
    if tensor.is_meta:
        return 0
    return tensor.nbytes
