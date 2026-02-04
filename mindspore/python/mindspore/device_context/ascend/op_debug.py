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

"""Op debug interfaces."""
from mindspore._checkparam import args_type_check
from .device import _is_supported
try:
    from mindspore._c_expression import AscendOpDebugConf
except ImportError:
    pass

function_status = {'execute_timeout': False, 'debug_option': False}


@args_type_check(op_timeout=int)
def execute_timeout(op_timeout):
    """
    Set the maximum duration of executing an operator in seconds. The framework operator execution timeout time
    is ``900`` by default.
    please refer to `Ascend Community document about aclrtSetOpExecuteTimeOut
    <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/API/appdevgapi/aclcppdevg_03_0132.html>`_.

    Args:
        op_timeout (int): Set the maximum duration of executing an operator in seconds.
          If the execution time exceeds this value, system will terminate the task.
          0 means endless wait. The defaults for AI Core and AI CPU operators vary on different hardware.

    Examples:
        >>> import mindspore as ms
        >>> ms.device_context.ascend.op_debug.execute_timeout(100)
    """
    if not function_status['execute_timeout']:
        function_status['execute_timeout'] = True
        if not _is_supported():
            return
    if op_timeout == AscendOpDebugConf.get_instance().execute_timeout():
        return
    # Check the configuration environment whether valid
    if AscendOpDebugConf.get_instance().is_execute_timeout_configured():
        raise RuntimeError("The 'execute_timeout' can not be set repeatedly.")
    if op_timeout < 0:
        raise ValueError("The num of execute_timeout must bigger than or equal to 0.")
    AscendOpDebugConf.get_instance().set_execute_timeout(op_timeout)


def debug_option(option_value):
    """
    Enable debugging options for Ascend operators, default not enabled.

    Args:
        option_value(str): Ascend operators debugging configuration. Currently, only memory
            access violation detection is supported.
            The value currently only supports being set to ``"oom"``.

            - ``"oom"``: When there is a memory out of bounds during the execution of an operator,
              AscendCL will return an error code of ``EZ9999``.

    Examples:
        >>> import mindspore as ms
        >>> ms.device_context.ascend.op_debug.debug_option("oom")
    """
    if not function_status['debug_option']:
        function_status['debug_option'] = True
        if not _is_supported():
            return
    if option_value == AscendOpDebugConf.get_instance().debug_option():
        return
    # Check the configuration environment whether valid
    if AscendOpDebugConf.get_instance().is_debug_option_configured():
        raise RuntimeError("The 'debug_option' can not be set repeatedly.")
    valid_order = {"oom"}
    if not isinstance(option_value, str):
        raise TypeError(
            f"For 'device_context.ascend.op_debug.debug_option(option_value)', the type of 'option_value' must be str, "
            f"but got {type(option_value)}."
        )
    if option_value not in valid_order:
        raise ValueError(
            f"For 'device_context.ascend.op_debug.debug_option(option_value)', the 'option_value' supports being set "
            f"to 'oom' currently, but got {option_value}."
        )
    AscendOpDebugConf.get_instance().set_debug_option(option_value)


def aclinit_config(config):
    """
    Configure the configuration items for the aclInit interface.
    please refer to `Ascend Community document about aclInit.
    <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/appdevgapi/aclcppdevg_03_0022.html>`_.

    Args:
        config(dict): When initializing AscendCL, you can enable or configure the
            following features through this configuration interface.

            - ``"max_opqueue_num"``: When executing using the single-operator model method, to save memory and balance
              the performance of calls, you can configure the maximum length of the single-operator model mapping
              queue through the max_opqueue_num parameter. If the length reaches the maximum, the system will first
              delete the mapping information that has not been used for a long time and the cached single-operator
              model, and then load the latest mapping information and the corresponding single-operator model.
              If the maximum length of the mapping queue is not configured, the default maximum length is 20,000.
            - ``"err_msg_mode"``: This parameter is used to control the level at which error information is retrieved,
              either by process or by thread. The default level is by process. "0" indicating that error information
              is retrieved by thread.
              "1" is the default value, indicates that error information is retrieved by process.
            - ``"dump"``: This parameter is used to enable exception dump for Ascend operators. The value can be set to
              {"dump_scene": "lite_exception"}, {"dump_scene": "lite_exception:disable"}.
              {"dump_scene": "lite_exception"} indicates that the exception dump is enabled.
              {"dump_scene": "lite_exception:disable"} indicates that the exception dump is disabled.
              {"dump_scene": "lite_exception"} is the default value, indicates that the exception dump is enabled.

    Examples:
        >>> import mindspore as ms
        >>> ms.set_device("Ascend", 0)
        >>> ms.device_context.ascend.op_debug.aclinit_config({"max_opqueue_num": "20000", "err_msg_mode": "1",
        ...                                                   "dump": {"dump_scene": "lite_exception"}})
    """
    aclinit_cfg_modes = {
        "max_opqueue_num": (str,),
        "err_msg_mode": ['0', '1'],
        "dump": [{"dump_scene": "lite_exception"}, {"dump_scene": "lite_exception:disable"}],
    }
    instance = AscendOpDebugConf.get_instance()
    aclinit_cfg_setters = {
        "max_opqueue_num": instance.set_max_opqueue_num,
        "err_msg_mode": instance.set_err_msg_mode,
        "dump": instance.set_lite_exception_dump
    }
    aclinit_cfg_set = tuple(aclinit_cfg_modes.keys())
    for key, value in config.items():
        if key not in aclinit_cfg_set:
            raise ValueError(f"For 'ms.device_context.ascend.op_debug.aclinit_config', the key must be one of "
                             f"{aclinit_cfg_set}, but got {key}.")
        supported_modes = aclinit_cfg_modes.get(key)
        if isinstance(supported_modes, list) and value not in supported_modes:
            raise ValueError(f"For 'ms.device_context.ascend.op_debug.aclinit_config', the value of argument {key} "
                             f"must be one of {supported_modes}, but got {value}.")
        if isinstance(supported_modes, tuple) and not isinstance(value, supported_modes):
            raise TypeError(f"For 'ms.device_context.ascend.op_debug.aclinit_config', the type of argument {key} "
                            f"must be one of {supported_modes}, but got {type(value)}.")
        cfg_setter = aclinit_cfg_setters.get(key)
        cfg_setter(value)
