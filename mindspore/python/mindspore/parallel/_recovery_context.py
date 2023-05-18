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
"""Context for recovery"""

from mindspore import _checkparam as Validator
from mindspore._c_expression import RecoveryContext

RECOVERY_CONTEXT = None


def recovery_context():
    """
    Get the global RECOVERY_CONTEXT, if it is not created, create a new one.

    Returns:
        RECOVERY_CONTEXT, the global recovery context.
    """

    global RECOVERY_CONTEXT
    if RECOVERY_CONTEXT is None:
        RECOVERY_CONTEXT = RecoveryContext.get_instance()
    return RECOVERY_CONTEXT

_set_recovery_context_func_map = {
    "ckpt_path": recovery_context().set_ckpt_path,
    "need_reset": recovery_context().set_need_reset
}

_get_recovery_context_func_map = {
    "enable_recovery": recovery_context().enable_recovery,
    "latest_ckpt_file": recovery_context().latest_ckpt_file,
    "latest_ckpt_epoch": recovery_context().latest_ckpt_epoch,
    "latest_ckpt_step": recovery_context().latest_ckpt_step,
    "need_reset": recovery_context().need_reset,
    "recovery_path": recovery_context().recovery_path,
    "ckpt_path": recovery_context().ckpt_path
}

_check_bool_keys = ["need_reset"]


def _set_recovery_context(**kwargs):
    """
    Set recovery context value.

    Note:
        Some other environment variables should also be set for recovery.
        These environment variables are listed below:

            MS_ENABLE_RECOVERY    # Enable recovery
            MS_RECOVERY_PATH      # The persistent path for recovery
            MS_RECOVERY_INTERVAL  # The persistent interval for recovery

    Args:
        ckpt_path (string): Set the recovery path used to save checkpoint. Default: ''.
        need_reset (bool): Set whether should call reset minddata and load ckpt for disaster recovery.
            Default: ``False``.

    Raises:
        ValueError: If input key is not the attribute in recovery context.
    """

    for key, value in kwargs.items():
        if key not in _set_recovery_context_func_map:
            raise ValueError("Set recovery context keyword %s is not recognized!" % key)
        _check_value(key, value)
        set_func = _set_recovery_context_func_map[key]
        set_func(value)


def _get_recovery_context(attr_key):
    """
    Get recovery context attribute value according to the key.

    Args:
        attr_key (str): The key of the attribute.

    Returns:
        Returns attribute value according to the key.

    Raises:
        ValueError: If input key is not attribute in revovery context.
    """

    if attr_key not in _get_recovery_context_func_map:
        raise ValueError("Get recovery context keyword %s is not recognized!" % attr_key)
    get_func = _get_recovery_context_func_map[attr_key]
    value = get_func()
    return value


def _check_value(key, value):
    """
    Validate the value for recovery context keys.
    """

    if key in _check_bool_keys:
        Validator.check_bool(value, key)
