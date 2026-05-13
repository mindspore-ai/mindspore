# Copyright 2024 Huawei Technologies Co., Ltd
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

"""Device context ascend interfaces"""
import mindspore as ms
from mindspore._c_expression import MSContext
from mindspore import log as logger

try:
    from mindspore._c_expression import ascend_get_device_count
except ImportError:
    pass


def device_count():
    """
    Returns the compute-capable device count of Ascend.

    Returns:
        int, the number of compute-capable Ascend devices.

    Examples:
        >>> import mindspore as ms
        >>> print(ms.device_context.ascend.device_count())
        8
    """
    if not MSContext.get_instance().is_pkg_support_device("Ascend") or not is_available():
        raise RuntimeError("Device Ascend not exist.")
    return ascend_get_device_count()


def is_available():
    """
    Returns whether the Ascend backend is available.

    Returns:
        bool, whether the Ascend backend is available for this MindSpore package.

    Examples:
        >>> import mindspore as ms
        >>> print(ms.device_context.ascend.is_available())
        True
    """
    # MindSpore will try to load plugins in "import mindspore", and availability status will be stored.
    if not MSContext.get_instance().is_pkg_support_device("Ascend"):
        logger.warning("Device Ascend is not available.")
        load_plugin_error = MSContext.get_instance().load_plugin_error()
        if load_plugin_error != "":
            logger.warning("Here's error when loading plugin for MindSpore package."
                           f"Error message: {load_plugin_error}")
        return False
    return True


def _is_supported():
    device_target = ms.context.get_context("device_target")
    if device_target in ['CPU', 'GPU']:
        logger.error(f"{device_target} device is not supported. Please use correct device")
        return False
    return True
