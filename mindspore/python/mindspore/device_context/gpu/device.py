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

"""Device context GPU interfaces"""
import mindspore as ms
from mindspore import log as logger
from mindspore._c_expression import MSContext
try:
    from mindspore._c_expression import gpu_get_device_count
except ImportError:
    pass


def device_count():
    """
    Returns the number of GPUs available.

    Returns:
        int, the number of GPUs available.

    Examples:
        >>> import mindspore as ms
        >>> ms.device_context.gpu.device.device_count()
    """
    if not MSContext.get_instance().is_pkg_support_device("GPU") or not is_available():
        raise RuntimeError("Device_target GPU not exist.")

    return gpu_get_device_count()


def is_available():
    """
    Returns a bool indicating if CUDA is currently available.

    Returns:
        bool, indicating if CUDA is currently available.

    Examples:
        >>> import mindspore as ms
        >>> ms.device_context.gpu.device.is_available()
    """
    # MindSpore will try to load plugins in "import mindspore", and availability status will be stored.
    if not MSContext.get_instance().is_pkg_support_device("GPU"):
        logger.warning("Device GPU is not available.")
        load_plugin_error = MSContext.get_instance().load_plugin_error()
        if load_plugin_error != "":
            logger.warning("Here's error when loading plugin for MindSpore package."
                           f"Error message: {load_plugin_error}")
        return False
    return True


def _is_supported():
    device_target = ms.context.get_context("device_target")
    if device_target in ['CPU', 'Ascend']:
        logger.error(f"{device_target} device is not supported. Please use correct device")
        return False
    return True
