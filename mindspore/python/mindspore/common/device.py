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
""" device class """

import threading
from mindspore.device_manager import get_current_device
from mindspore._c_expression import DeviceManagerConf, MSContext, _wait_pipeline


class _ThreadLocalStack:
    """ Create thread local stack to store device info. """
    def __init__(self):
        self._local = threading.local()

    def get_stack(self):
        """Get the thread-local device stack, creating it if needed."""
        if not hasattr(self._local, 'stack'):
            self._local.stack = []
        return self._local.stack

    def push(self, item):
        """Push a device info onto the stack."""
        self.get_stack().append(item)

    def pop(self):
        """Pop and return the top device info from the stack."""
        stack = self.get_stack()
        if not stack:
            raise RuntimeError("Device stack is empty")
        return stack.pop()

    def is_empty(self):
        """Return True if the device stack is empty."""
        return len(self.get_stack()) == 0


_DEVICE_STACK = _ThreadLocalStack()


def _is_same_device(device_info, device_target, device_id):
    """Return True if the device info matches the target device."""
    return device_info.device_target == device_target and device_info.device_id == device_id


class DeviceCtx:
    """ Device context. """
    def __init__(self, name: str, device_id=None):
        self.name = name
        self.device_id = device_id

    def __enter__(self):
        valid_targets = ["CPU", "GPU", "Ascend", "meta"]
        if self.name not in valid_targets:
            raise ValueError(f"For DeviceCtx, the name must be one of {valid_targets}, but got {self.name}.")
        if self.device_id is not None and self.device_id < 0:
            raise ValueError("For DeviceCtx, the device_id must be greater than or equal to 0.")
        device_info = get_current_device()
        is_default = False
        if self.device_id is None:
            self.device_id = 0
            is_default = True
        if not _is_same_device(device_info, self.name, self.device_id):
            _wait_pipeline()
        MSContext.get_instance().set_device_target_inner(self.name)
        DeviceManagerConf.get_instance().set_device(self.name, self.device_id, is_default)
        # Push only after the device is switched, otherwise a failed __enter__ would leave a stale entry behind
        # and __exit__ of an enclosing context would restore the wrong device.
        _DEVICE_STACK.push(device_info)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        device_info = _DEVICE_STACK.pop()
        try:
            current_device = get_current_device()
            if not _is_same_device(current_device, device_info.device_target, device_info.device_id):
                _wait_pipeline()
        finally:
            MSContext.get_instance().set_device_target_inner(device_info.device_target)
            DeviceManagerConf.get_instance().set_device(device_info.device_target, device_info.device_id, False)
