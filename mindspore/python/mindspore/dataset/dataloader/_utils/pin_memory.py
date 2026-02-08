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
# ==============================================================================
"""
Pin memory module.
"""

import collections
import copy
from queue import Empty

import mindspore
from . import WORKER_TIME_OUT
from .worker import WorkerException


def pin_worker_fn(in_queue, out_queue, device_id, pin_memory_done):
    """
    Pin memory worker function.
    """

    while not pin_memory_done.is_set():
        try:
            order_index, data = in_queue.get(WORKER_TIME_OUT)
        except Empty:
            continue

        if not isinstance(data, WorkerException):
            try:
                data = pin_memory_fn(data)
            except Exception:  # pylint: disable=W0703
                data = WorkerException()

        out_queue.put((order_index, data))


def pin_memory_fn(data):
    """
    Pin memory function.
    """

    if isinstance(data, mindspore.Tensor):
        return data.pin_memory()
    if isinstance(data, (str, bytes)):
        return data
    if isinstance(data, collections.abc.Mapping):
        try:
            if isinstance(data, collections.abc.MutableMapping):
                # The sequence type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new sequence.
                # Create a clone and update it if the sequence type is mutable.
                clone = copy.copy(data)
                clone.update({k: pin_memory_fn(sample) for k, sample in data.items()})
                return clone
            return type(data)({k: pin_memory_fn(sample) for k, sample in data.items()})  # type: ignore[call-arg]
        except TypeError:
            # The mapping type may not support `copy()` / `update(mapping)`
            # or `__init__(iterable)`.
            return {k: pin_memory_fn(sample) for k, sample in data.items()}
    if isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return type(data)(*(pin_memory_fn(sample) for sample in data))
    if isinstance(data, tuple):
        return [pin_memory_fn(sample) for sample in data]  # Backwards compatibility.
    if isinstance(data, collections.abc.Sequence):
        try:
            if isinstance(data, collections.abc.MutableSequence):
                # The sequence type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new sequence.
                # Create a clone and update it if the sequence type is mutable.
                clone = copy.copy(data)  # type: ignore[arg-type]
                for i, item in enumerate(data):
                    clone[i] = pin_memory_fn(item)
                return clone
            return type(data)([pin_memory_fn(sample) for sample in data])  # type: ignore[call-arg]
        except TypeError:
            # The sequence type may not support `copy()` / `__setitem__(index, item)`
            # or `__init__(iterable)` (e.g., `range`).
            return [pin_memory_fn(sample) for sample in data]
    if hasattr(data, "pin_memory"):
        return data.pin_memory()
    return data
