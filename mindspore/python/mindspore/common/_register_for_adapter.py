# Copyright 2022 Huawei Technologies Co., Ltd
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

from mindspore.common.tensor import Tensor


class Registry:
    """Registry class for ms adapter."""

    def __init__(self):
        self._tensor = None
        self._convert_map = {}

    @property
    def tensor(self):
        if self._tensor is None:
            raise ValueError("Before using Tensor in MSAdapter, please call 'set_adapter_config'.")
        return self._tensor

    @property
    def convert_map(self):
        return self._convert_map

    def register_tensor(self, value):
        if self._tensor is not None:
            raise ValueError("Repeated registration of tensor in ms adapter config.")
        if not issubclass(value, Tensor):
            raise ValueError(f"The tensor definition here should be a subclass of ms.Tensor, but got {value}.")
        self._tensor = value

    def register_convert_map(self, value):
        if not isinstance(value, dict):
            raise ValueError(f"Expect a dict type, but got {type(value)}.")
        self._convert_map = value


ms_adapter_registry = Registry()
