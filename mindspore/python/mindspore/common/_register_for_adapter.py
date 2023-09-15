# Copyright 2022-2023 Huawei Technologies Co., Ltd
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

"""Registry MSAdapter config."""


class Registry:
    """Registry class for ms adapter."""

    def __init__(self):
        self.is_registered = False
        self._tensor = None
        self._parameter = None
        self._convert_map = {}

    @property
    def tensor(self):
        """Return the registered tensor."""
        if self._tensor is None:
            raise ValueError("Before using Tensor in MSAdapter, please call 'set_adapter_config'.")
        return self._tensor

    @property
    def parameter(self):
        """Return the registered parrameter."""
        if self._parameter is None:
            raise ValueError("Before using Parameter in MSAdapter, please call 'set_adapter_config'.")
        return self._parameter

    @property
    def convert_map(self):
        """Return the registered convert map."""
        return self._convert_map

    def register_tensor(self, value):
        """Register the tensor of ms adapter."""
        if self._tensor is not None:
            raise ValueError("Repeated registration of tensor in ms adapter config.")
        self._tensor = value
        self.is_registered = True

    def register_parameter(self, value):
        """Register the parameter of ms adapter."""
        if self._parameter is not None:
            raise ValueError("Repeated registration of Parameter in ms adapter config.")
        self._parameter = value

    def register_convert_map(self, value):
        """Register the convert map of ms adapter."""
        if not isinstance(value, dict):
            raise ValueError(f"Expect a dict type, but got {type(value)}.")
        self._convert_map = value


ms_adapter_registry = Registry()
