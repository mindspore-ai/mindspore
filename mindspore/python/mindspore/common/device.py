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
from mindspore.device_manager import get_current_device, set_device
from mindspore._c_expression import DeviceManagerConf, MSContext


class _ThreadLocalStack:
    """ Create thread local stack to store device info. """
    def __init__(self):
        self._local = threading.local()

    def get_stack(self):
        if not hasattr(self._local, 'stack'):
            self._local.stack = []
        return self._local.stack

    def push(self, item):
        self.get_stack().append(item)

    def pop(self):
        stack = self.get_stack()
        if not stack:
            raise RuntimeError("Device stack is empty")
        return stack.pop()

    def is_empty(self):
        return len(self.get_stack()) == 0


_DEVICE_STACK = _ThreadLocalStack()


class DeviceCtx:
    """ Device context. """
    def __init__(self, name: str, device_id=None):
        self.name = name
        self.device_id = device_id

    def __enter__(self):
        if self.name != "meta":
            raise ValueError(f"For DeviceCtx, the name only support 'meta' currently, but got {self.name}")
        device_info = get_current_device()
        _DEVICE_STACK.push(device_info)
        is_default = False
        if self.device_id is None:
            self.device_id = 0
            is_default = True
        MSContext.get_instance().set_device_target_inner(self.name)
        DeviceManagerConf.get_instance().set_device(self.name, self.device_id, is_default)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        device_info = _DEVICE_STACK.pop()
        MSContext.get_instance().set_device_target_inner(device_info.device_target)
        DeviceManagerConf.get_instance().set_device(device_info.device_target, device_info.device_id, False)
