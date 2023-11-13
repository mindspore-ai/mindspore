# Copyright 2023 Huawei Technologies Co., Ltd
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

"""Hal base class."""
from mindspore._c_expression import MSContext, DeviceContextManager

_context_handle = MSContext.get_instance()
_device_context_mgr = DeviceContextManager.get_instance()


class _HalBase():
    """
    Hal base class. Different backends inherit from this class and wrap around hardware interfaces.
    """
    def __init__(self, backend=None):
        valid_backends = ["CPU", "GPU", "Ascend"]
        if backend not in valid_backends:
            raise ValueError(f"For '_HalBase', the argument 'backend' must be one of "
                             f"{valid_backends}, but got {backend}.")
        self.backend = backend

    def device_count(self):
        """
        Return device count of this backend.
        """

    def get_device_capability(self, device_id):
        """
        Get device capability of the specified device id.
        """

    def get_device_properties(self, device_id):
        """
        Get device properties of the specified device id.
        """

    def get_device_name(self, device_id):
        """
        Get device name of the specified device id.
        """

    def get_arch_list(self):
        """
        Get the architecture list this MindSpore was compiled for.
        """
