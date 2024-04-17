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

"""Hal ascend interfaces."""
from mindspore._c_expression import ascend_get_device_count, ascend_get_device_name, ascend_get_device_properties
from mindspore import log as logger
from ._base import _HalBase


class _HalAscend(_HalBase):
    """
    Hal ascend interfaces.
    """
    def __init__(self):
        super(_HalAscend, self).__init__("Ascend")

    def device_count(self):
        """
        Return Ascend device count.
        """
        return ascend_get_device_count()

    def get_device_capability(self, device_id):
        """
        Get Ascend capability of the specified device id. Not implemented for Ascend.
        """
        logger.warning("'get_device_capability' for Ascend is not implemented. Return None.")

    def get_device_properties(self, device_id):
        """
        Get Ascend properties of the specified device id.
        """
        return ascend_get_device_properties(device_id)

    def get_device_name(self, device_id):
        """
        Get Ascend name of the specified device id.
        """
        return ascend_get_device_name(device_id)

    def get_arch_list(self):
        """
        Get the architecture list this MindSpore was compiled for. Not implemented for Ascend yet.
        """
        logger.warning("'get_arch_list' for Ascend is not implemented. Return None.")
