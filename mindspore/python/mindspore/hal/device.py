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
import inspect
import functools
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


def _check_inputs_validation(fn):
    """
    Decorator to check whether the device target validation.
    If device target's hal instance is not created, throw an exception.
    """
    @functools.wraps(fn)
    def deco(*args, **kwargs):
        bound_args = inspect.signature(fn).bind(*args, **kwargs)
        bound_args.apply_defaults()
        params = bound_args.arguments
        if "device_target" in params:
            device_target = params["device_target"]
            if device_target is None:
                device_target = context.get_context("device_target")
                params["device_target"] = device_target
            if device_target not in valid_targets:
                raise ValueError(f"The argument 'device_target' must be one of "
                                 f"{valid_targets}, but got {device_target}.")
            if device_target not in hal_instances:
                raise ValueError(f"{device_target} backend is not available for this MindSpore package."
                                 "You can call hal.is_available to check the reason.")

        if "device_id" in params:
            device_id = params["device_id"]
            if device_id < 0:
                raise ValueError(f"`device_id` should not be negative, but got {device_id}.")

        return fn(*bound_args.args, **bound_args.kwargs)
    return deco


def is_initialized(device_target):
    """
    Return whether specified backend is initialized.

    Note:
        MindSpore's backends "CPU", "GPU" and "Ascend" will be initialized in the following scenarios:
        - For distributed job, backend will be initialized after `mindspore.communication.init` method is called.
        - For graph mode, backend is initialized after graph compiling phase.
        - For PyNative mode, backend is initialized when running the first operator.

    Args:
        device_target (str): The device name of backend, should be one of "CPU", "GPU" and "Ascend".

    Return:
        Bool, whether the specified backend is initialized.
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

    Args:
        device_target (str): The device name of backend, should be one of "CPU", "GPU" and "Ascend".

    Return:
        Bool, whether the specified backend is available for this MindSpore package.
    """
    if device_target not in valid_targets:
        raise ValueError(f"For 'hal.is_available', the argument 'device_target' must be one of "
                         f"{valid_targets}, but got {device_target}.")

    # MindSpore will try to load plugins in "import mindspore", and availability status will be stored.
    if not _context_handle.is_pkg_support_device(device_target):
        logger.warning(f"Backend {device_target} is not available.")
        load_plugin_error = _context_handle.load_plugin_error()
        if load_plugin_error != "":
            logger.warning(f"Here's error when loading plugin for MindSpore package."
                           f"Error message: {load_plugin_error}")
        return False
    return True


@_check_inputs_validation
def device_count(device_target=None):
    """
    Return device count of specified backend.

    Note:
        If `device_target` is not specified, get the device count of the current backend set by context.
        For CPU backend, this method always returns 1.

    Args:
        device_target (str): The device name of backend, should be one of "CPU", "GPU" and "Ascend".

    Return:
        int.
    """
    return hal_instances[device_target].device_count()


@_check_inputs_validation
def get_device_capability(device_id, device_target=None):
    """
    Get specified device's capability.

    Note:
        If `device_target` is not specified, get the device capability of the current backend set by context.

    Args:
        device_id (int): The device id of which the capability will be returned.
        device_target (str): The device name of backend, should be one of "CPU", "GPU" and "Ascend".

    Return:
        tuple(int, int) for GPU.
        None for Ascend and CPU.
    """
    return hal_instances[device_target].get_device_capability(device_id)


@_check_inputs_validation
def get_device_properties(device_id, device_target=None):
    """
    Get specified device's properties.

    Note:
        If `device_target` is not specified, get the device properties of the current backend set by context.
        For Ascend, backend must be initialized before calling this method,
        or `total_memory` and `free_memory` will be 0,
        and `device_id` will be ignored since this method only returns current device's properties.

    Args:
        device_id (int): The device id of which the properties will be returned.
        device_target (str): The device name of backend, should be one of "CPU", "GPU" and "Ascend".

    Return:
        `cudaDeviceProp` for GPU.
        `AscendDeviceProperties` for Ascend:
        AscendDeviceProperties {
            std::string name;
            size_t total_memory;
            size_t free_memory;
        }.
        None for CPU.
    """
    return hal_instances[device_target].get_device_properties(device_id)


@_check_inputs_validation
def get_device_name(device_id, device_target=None):
    """
    Get specified device's name.

    Note:
        If `device_target` is not specified, get the device name of the current backend set by context.
        This method always returns "CPU" for CPU backend.

    Args:
        device_id (int): The device id of which the name will be returned.
        device_target (str): The device name of backend, should be one of "CPU", "GPU" and "Ascend".

    Return:
        str.
    """
    return hal_instances[device_target].get_device_name(device_id)


@_check_inputs_validation
def get_arch_list(device_target=None):
    """
    Get the architecture list this MindSpore was compiled for.

    Note:
        If `device_target` is not specified, get the device name of the current backend set by context.

    Args:
        device_target (str): The device name of backend, should be one of "CPU", "GPU" and "Ascend".

    Return:
        str for GPU.
        None for Ascend and CPU.
    """
    return hal_instances[device_target].get_arch_list()
