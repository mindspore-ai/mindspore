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

"""Op tuning interfaces."""
try:
    from mindspore._c_expression import AscendOpTuningConf
except ImportError:
    pass
from .device import _is_supported

function_status = {'op_compile': False, 'aoe_tune_mode': False,
                   'aoe_job_type': False, 'aclnn_cache': False}


def op_compile(value):
    """
    Whether to select online compilation. The default settings of the framework are online compilation for static
    shape, and compiled operator binary files for dynamic shape. The default settings may change in the future.
    For detailed information, please refer to `Ascend community
    <https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/apiref/appdevgapi/aclcppdevg_03_1371.html>`_ .

    Args:
        value (bool): Whether to select online compilation or not.

            - ``True``: Online compilation is prioritized.
            - ``False``: Compiled operator binary files are prioritized to improve compilation performance.

    Examples:
        >>> import mindspore as ms
        >>> ms.device_context.ascend.op_tuning.op_compile(True)
    """
    if not function_status['op_compile']:
        function_status['op_compile'] = True
        if not _is_supported():
            return
    if value == AscendOpTuningConf.get_instance().jit_compile():
        return
    # Check the configuration environment whether valid
    if AscendOpTuningConf.get_instance().is_jit_compile_configured():
        raise RuntimeError("The 'op_compile' can not be set repeatedly.")
    supported_modes = [True, False]
    if value not in supported_modes:
        raise TypeError(f"For 'op_compile', the type of input value must be one of "
                        f"{supported_modes}, but got {value}.")
    is_enable = "1" if value else "0"
    AscendOpTuningConf.get_instance().set_jit_compile(is_enable)


def aoe_tune_mode(tune_mode):
    """
    AOE tuning mode setting, which is not set by default.

    Args:
        tune_mode (str): The AOE tuning mode to set.

            - ``"online"``: The online tuning function is turned on.
            - ``"offline"``: The GE graph will be saved for offline tuning.

    Examples:
        >>> import mindspore as ms
        >>> ms.device_context.ascend.op_tuning.aoe_tune_mode("online")
    """
    if not function_status['aoe_tune_mode']:
        function_status['aoe_tune_mode'] = True
        if not _is_supported():
            return
    if tune_mode == AscendOpTuningConf.get_instance().aoe_tune_mode():
        return
    # Check the configuration environment whether valid
    if AscendOpTuningConf.get_instance().is_aoe_tune_mode_configured():
        raise RuntimeError("The 'aoe_tune_mode' can not be set repeatedly.")
    candidate = ["online", "offline"]
    if tune_mode not in candidate:
        raise ValueError(
            f"For 'device_context.ascend.op_tuning.aoe_tune_mode', the argument 'tune_mode' must be in "
            f"['online', 'offline'], but got {tune_mode}."
        )
    AscendOpTuningConf.get_instance().set_aoe_tune_mode(tune_mode)


def aoe_job_type(config):
    """
    Set the parameters specific to Ascend Optimization Engine. It needs to be used in
    conjunction with :func:`mindspore.device_context.ascend.op_tuning.aoe_tune_mode`.
    The default value set by the framework is ``"2"``.

    Args:
        config (str): Choose the tuning type.

            - ``"1"``: Set to subgraph tuning.
            - ``"2"``: Set to operator tuning.

    Examples:
        >>> import mindspore as ms
        >>> ms.device_context.ascend.op_tuning.aoe_job_type("1")
    """
    if not function_status['aoe_job_type']:
        function_status['aoe_job_type'] = True
        if not _is_supported():
            return
    if config == AscendOpTuningConf.get_instance().aoe_job_type():
        return
    # Check the configuration environment whether valid
    if AscendOpTuningConf.get_instance().is_aoe_job_type_configured():
        raise RuntimeError("The 'aoe_job_type' can not be set repeatedly.")
    aoe_cfgs = ["1", "2"]
    if config not in aoe_cfgs:
        raise ValueError(
            f"For 'aoe_job_type', the config must be one of {aoe_cfgs}, but got {config}."
        )
    AscendOpTuningConf.get_instance().set_aoe_job_type(config)


def aclnn_cache(enable_global_cache=False, cache_queue_length=10000):
    """
    Configure aclnn cache parameters.

    Args:
        enable_global_cache (bool, optional): Whether to set the aclnn cache to
            global when in GRAPH_MODE. Default: ``False``.
        cache_queue_length (int, optional): The cache queue length to set.
            Default: ``10000``.

    Examples:
        >>> import mindspore as ms
        >>> ms.device_context.ascend.op_tuning.aclnn_cache(True, 10000)
    """
    if not function_status['aclnn_cache']:
        function_status['aclnn_cache'] = True
        if not _is_supported():
            return
    # Check the configuration environment whether valid
    if AscendOpTuningConf.get_instance().is_aclnn_cache_configured():
        raise RuntimeError("The 'aclnn_cache' can not be set repeatedly.")
    cache_cfgs = [True, False]
    if enable_global_cache not in cache_cfgs:
        raise ValueError(
            f"For 'aclnn_cache', the config must be one of {cache_cfgs}, but got {enable_global_cache}."
        )
    AscendOpTuningConf.get_instance().set_aclnn_global_cache(enable_global_cache)
    if cache_queue_length < 0:
        raise ValueError(
            f"For 'aclnn_cache', the config must greater than 0, but got {enable_global_cache}."
        )
    AscendOpTuningConf.get_instance().set_cache_queue_length(cache_queue_length)
