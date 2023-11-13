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

"""Hardware device interfaces."""
from mindspore._c_expression import MSContext, DeviceContextManager
from mindspore import log as logger
from mindspore import context

try:
    from ._cpu import _HalCPU
    CPU_AVAILABLE = True
except ImportError:
    pass

try:
    from ._gpu import _HalGPU
    GPU_AVAILABLE = True
except ImportError:
    pass

try:
    from ._ascend import _HalAscend
    ASCEND_AVAILABLE = True
except ImportError:
    pass

_context_handle = MSContext.get_instance()
_device_context_mgr = DeviceContextManager.get_instance()

hal_instances = {}
valid_targets = ["CPU", "GPU", "Ascend"]
# Create hal instance as soon as module is imported.
for target in valid_targets:
    if _context_handle.is_pkg_support_device(target):
        if target == "CPU" and CPU_AVAILABLE:
            hal_instances["CPU"] = _HalCPU()
        elif target == "GPU" and GPU_AVAILABLE:
            hal_instances["GPU"] = _HalGPU()
        elif target == "Ascend" and ASCEND_AVAILABLE:
            hal_instances["Ascend"] = _HalAscend()
        else:
            pass


def is_initialized(device_target):
    """
    Return whether specified backend is initialized.
    Note:
        MindSpore's backends "CPU", "GPU" and "Ascend" will be initialized in the following scenarios:
        - For distributed job, backend will be initialized after `mindspore.communication.init` is called.
        - For graph mode, backend is initialized after graph compiling phase.
        - For PyNative mode, backend is initialized when creating the first tensor.
    """
    if device_target not in valid_targets:
        raise ValueError(f"For 'hal.is_initialized', the argument 'device_target' must be one of "
                         f"{valid_targets}, but got {device_target}.")
    _device_context = _device_context_mgr.get_device_context(device_target)
    if _device_context is None:
        logger.warning(f"Backend {device_target} is not created yet.")
        return False
    return _device_context.initialized()


def is_available(device_target):
    """
    Return whether specified backend is available.
    All dependent libraries should be successfully loaded if this backend is available.
    """
    if device_target not in valid_targets:
        raise ValueError(f"For 'hal.is_available', the argument 'device_target' must be one of "
                         f"{valid_targets}, but got {device_target}.")

    # MindSpore will try to load plugins in "import mindspore", and availability status will be stored.
    if not _context_handle.is_pkg_support_device(device_target):
        logger.warning(f"Backend {device_target} is not available.")
        loading_plugin_error = _context_handle.loading_plugin_error()
        if loading_plugin_error != "":
            logger.warning(f"Here's error when loading plugin for MindSpore package."
                           f"Error message: {loading_plugin_error}")
        return False
    return True


def device_count(device_target=None):
    """
    Return device count of currently used backend.
    Note:
        If `device_target` is not specified, get the device count of the current backend set by context.
    """
    if device_target is None:
        device_target = context.get_context("device_target")
    if device_target not in hal_instances:
        logger.warning(f"{device_target} backend is not available for this MindSpore package."
                       "You can call hal.is_available to check the reason.")
    return hal_instances[device_target].device_count()


def get_device_capability(device_id, device_target=None):
    """
    Get specified device's capability.
    """
    if device_target is None:
        device_target = context.get_context("device_target")
    if device_target not in hal_instances:
        logger.warning(f"{device_target} backend is not available for this MindSpore package."
                       "You can call hal.is_available to check the reason.")
    return hal_instances[device_target].get_device_capability(device_id)


def get_device_properties(device_id, device_target=None):
    """
    Get specified device's properties.
    """
    if device_target is None:
        device_target = context.get_context("device_target")
    if device_target not in hal_instances:
        logger.warning(f"{device_target} backend is not available for this MindSpore package."
                       "You can call hal.is_available to check the reason.")
    return hal_instances[device_target].get_device_properties(device_id)


def get_device_name(device_id, device_target=None):
    """
    Get specified device's name.
    """
    if device_target is None:
        device_target = context.get_context("device_target")
    if device_target not in hal_instances:
        logger.warning(f"{device_target} backend is not available for this MindSpore package."
                       "You can call hal.is_available to check the reason.")
    return hal_instances[device_target].get_device_name(device_id)


def get_arch_list():
    """
    Get the architecture list this MindSpore was compiled for.
    """
    return
