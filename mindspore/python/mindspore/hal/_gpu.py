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

"""Hal gpu interfaces."""
from mindspore._c_expression import gpu_get_device_count, gpu_get_device_name, gpu_get_device_capability, \
     gpu_get_device_properties, gpu_get_arch_list
from ._base import _HalBase


class _HalGPU(_HalBase):
    """
    Hal gpu interfaces.
    """
    def __init__(self):
        super(_HalGPU, self).__init__("GPU")

    def device_count(self):
        """
        Return GPU device count.
        """
        return gpu_get_device_count()

    def get_device_capability(self, device_id):
        """
        Get GPU capability of the specified device id.
        """
        return gpu_get_device_capability(device_id)

    def get_device_properties(self, device_id):
        """
        Get GPU properties of the specified device id.
        """
        return gpu_get_device_properties(device_id)

    def get_device_name(self, device_id):
        """
        Get GPU name of the specified device id.
        """
        return gpu_get_device_name(device_id)

    def get_arch_list(self):
        """
        Get the architecture list this MindSpore was compiled for.
        """
        return gpu_get_arch_list()
