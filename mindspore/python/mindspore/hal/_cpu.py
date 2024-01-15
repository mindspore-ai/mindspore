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

"""Hal cpu interfaces."""
from mindspore import log as logger
from ._base import _HalBase


class _HalCPU(_HalBase):
    """
    Hal cpu interfaces.
    """
    def __init__(self):
        super(_HalCPU, self).__init__("CPU")

    def device_count(self):
        """
        Return CPU device count. Always return 1 for CPU.
        """
        return 1

    def get_device_capability(self, device_id):
        """
        Get CPU capability of the specified device id. Not implemented for CPU.
        """
        logger.warning("'get_device_capability' for CPU is not implemented. Return None.")

    def get_device_properties(self, device_id):
        """
        Get CPU properties of the specified device id. Not implemented for CPU.
        """
        logger.warning("'get_device_properties' for CPU is not implemented. Return None.")

    def get_device_name(self, device_id):
        """
        Get CPU name of the specified device id. Always return "CPU".
        """
        return "CPU"

    def get_arch_list(self):
        """
        Get the architecture list this MindSpore was compiled for. Not implemented for CPU.
        """
        logger.warning("'get_arch_list' for CPU is not implemented. Return None.")
