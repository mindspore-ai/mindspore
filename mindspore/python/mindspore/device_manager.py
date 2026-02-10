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

"""Device manager interfaces."""

__all__ = ['set_device', 'set_deterministic', 'get_current_device']

import os
from mindspore import log as logger
from mindspore._c_expression import DeviceManagerConf, DeviceContextManager, MSContext, CollectiveManager
from mindspore._checkparam import args_type_check
from mindspore.parallel._ps_context import _need_reset_device_target_for_ps

class DeviceInfo(tuple):
    """
    DeviceInfo class. Store the current device target and the corresponding device id.
    """
    def __new__(cls, device_target, device_id):
        return super().__new__(cls, (device_target, device_id))

    @property
    def device_target(self):
        return self[0]

    @property
    def device_id(self):
        return self[1]


@args_type_check(device_target=str, device_id=int)
def set_device(device_target, device_id=None):
    """
    Set the device target and device id for the running environment.

    Note:
        - The `device_target` must be set in the ["CPU", "GPU", "Ascend"], there is no default value.
        - Suggest setting `device_target` and `device_id` before calling :func:`mindspore.communication.init`.

    Args:
        device_target (str): The target device to run, only support ``"Ascend"``, ``"GPU"``, and ``"CPU"``.
        device_id (int, optional): ID of the target device, the value must be in [0, device_num_per_host-1],
            where device_num_per_host refers to the total number of devices on the host. Default: ``None`` .
            The framework sets different default behaviours according to the scenario:
            if it is a single-card scenario, the framework will be set to ``0``.
            In a distributed scenario where msrun is started, the framework will
            automatically negotiates the available `device_id` values.
            In a distributed scenario with other startup methods, the framework is set to ``0``.

    Examples:
        >>> import mindspore as ms
        >>> ms.set_device("Ascend", 1)
    """
    valid_targets = ["CPU", "GPU", "Ascend"]
    if device_target not in valid_targets:
        raise ValueError(f"The argument 'device_target' must be one of {valid_targets}, but got {device_target}.")
    # If in Parameter Server mode, Ascend card should not be used by server and scheduler.
    if _need_reset_device_target_for_ps(device_target):
        logger.info("Reset device target to CPU when set_device.")
        device_target = "CPU"

    is_default = False
    if device_id is None:
        device_id = 0
        is_default = True
    if device_id < 0:
        raise ValueError("The device id must bigger than or equal to 0.")

    MSContext.get_instance().set_device_target_inner(device_target)

    if DeviceManagerConf.get_instance().is_device_enable():
        old_device_target = DeviceManagerConf.get_instance().get_device_target()
        old_device_id = DeviceManagerConf.get_instance().get_device_id()
        if old_device_target != device_target or old_device_id != device_id:
            raise RuntimeError("The 'mindspore.set_device' can not be modified.")
        return

    device_context = DeviceContextManager.get_instance().get_device_context(device_target)
    if device_context is not None and device_context.initialized():
        raise RuntimeError("The runtime has been initialized, please set it before the kernel is executed, "
                           "or before calling 'mindspore.communication.init()'. "
                           "Suggest setting it as early as possible.")
    DeviceManagerConf.get_instance().set_device(device_target, device_id, is_default)


def get_current_device():
    """
    Get device target and device id in the current running environment.

    Examples:
        >>> import mindspore as ms
        >>> ms.set_device("Ascend", 1)
        >>> ms.get_current_device()
        ('Ascend', 1)
        >>> ms.get_current_device().device_target
        'Ascend'
        >>> ms.get_current_device().device_id
        1
    """
    device_target = DeviceManagerConf.get_instance().get_device_target()
    device_id = DeviceManagerConf.get_instance().get_device_id()
    return DeviceInfo(device_target, device_id)


@args_type_check(deterministic=bool)
def set_deterministic(deterministic):
    """
    Enables or disables deterministic computing.

    This configuration is a global configuration, and once enabled, subsequent calculation operations
    will follow the configuration setting. When deterministic computing is enabled, the same output
    is generated if an operator is executed multiple times with the same hardware and input. This often
    slows down operator execution.

    The framework does not enable deterministic computation by default.

    Note:
        - In a distributed scenario, it is suggested to set deterministic computing before
          calling :func:`mindspore.communication.init` to enable deterministic operation for
          communication operators in the global communication group.
        - The fixed method for deterministic calculation must be in the same main process as the network,
          operator, etc. Only one deterministic state can be set in the same thread, and it is not recommended
          to set deterministic state multiple times in one thread.

    Args:
        deterministic (bool): Whether to enable deterministic computing.

    Examples:
        >>> import mindspore as ms
        >>> ms.set_deterministic(True)
    """
    # Check whether the configuration environment is valid.
    if DeviceManagerConf.get_instance().is_deterministic_configured():
        raise RuntimeError("The 'mindspore.set_deterministic' can not be set repeatedly.")

    logger.info(f"Set deterministic setting to '{deterministic}'.")

    # Must wait for all async created groups to be initialized so that
    # deterministic feature could be consistent between all processes.
    CollectiveManager.get_instance().wait_all_comm_init()

    # Check the hccl_deterministic and te_parallel_compiler.
    hccl_deterministic = os.getenv("HCCL_DETERMINISTIC")
    te_parallel_compiler = os.getenv("TE_PARALLEL_COMPILER")
    if deterministic:
        if hccl_deterministic == "strict":
            os.environ["HCCL_DETERMINISTIC"] = "strict"
        else:
            if hccl_deterministic and hccl_deterministic != "true":
                logger.warning(f"Environment 'HCCL_DETERMINISTIC' should be 'true' or 'strict' "
                               f"when set deterministic='True', but got '{hccl_deterministic}'. "
                               f"'HCCL_DETERMINISTIC' will be set to 'true'.")
            os.environ["HCCL_DETERMINISTIC"] = "true"
        if te_parallel_compiler and te_parallel_compiler != "1":
            logger.warning(f"Environment 'TE_PARALLEL_COMPILER' should be '1' when set deterministic='True', but "
                           f"got '{te_parallel_compiler}'. 'TE_PARALLEL_COMPILER' will be set to '1'.")
        os.environ["TE_PARALLEL_COMPILER"] = "1"
    else:
        if hccl_deterministic and hccl_deterministic != "false":
            logger.warning(f"Environment 'HCCL_DETERMINISTIC' should not be set or be 'false' when set "
                           f"deterministic='False', but got '{hccl_deterministic}'. 'HCCL_DETERMINISTIC' "
                           f"will be unset.")
            del os.environ["HCCL_DETERMINISTIC"]
        if te_parallel_compiler and te_parallel_compiler != "0":
            logger.warning(f"Environment 'TE_PARALLEL_COMPILER' should not be set or be '0' when set "
                           f"deterministic='False', but got '{te_parallel_compiler}'. 'TE_PARALLEL_COMPILER' "
                           f"will be unset.")
            del os.environ["TE_PARALLEL_COMPILER"]

    DeviceManagerConf.get_instance().set_deterministic(deterministic)
