# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
"""
The context of mindspore, used to configure the current execution environment,
includes the execution mode, execution backend and other feature switches.
"""
from __future__ import absolute_import

import json
import os
import time
import threading
from collections import namedtuple
from types import FunctionType

from mindspore import log as logger
from mindspore._c_expression import MSContext, ms_ctx_param, CollectiveManager
from mindspore import _checkparam as Validator
from mindspore._checkparam import args_type_check
from mindspore.parallel._auto_parallel_context import _set_auto_parallel_context, _get_auto_parallel_context, \
    _reset_auto_parallel_context
from mindspore.parallel._ps_context import _set_ps_context, _get_ps_context, _reset_ps_context, \
    _need_reset_device_target_for_ps
from mindspore.hal.device import is_initialized
from mindspore.graph import api

__all__ = ['GRAPH_MODE', 'PYNATIVE_MODE', 'STRICT', 'COMPATIBLE', 'LAX', 'set_context', 'get_context',
           'set_auto_parallel_context', 'get_auto_parallel_context', 'reset_auto_parallel_context', 'ParallelMode',
           'set_ps_context', 'get_ps_context']

GRAPH_MODE = 0
PYNATIVE_MODE = 1
_DEVICE_APP_MEMORY_SIZE = 31  # The max memory size of graph plus variable.
_RE_PATTERN = r'[1-9][0-9]*(\.)?[0-9]*GB|0\.[0-9]*GB'
K_CONTEXT = None

# Enumerate for the property 'jit_syntax_level'.
STRICT = 0
COMPATIBLE = 1
LAX = 2

# Enumerate for the property 'debug_level'.
RELEASE = 0
DEBUG = 1


def _make_directory(path):
    """Make directory."""
    if path is None or not isinstance(path, str) or path.strip() == "":
        raise ValueError(f"For 'context.set_context', the 'save_graphs_path' or the 'print_file_path' is invalid "
                         f"type, it should be Non-empty string, but got '{path}'.")

    path = os.path.realpath(path)
    logger.debug("The absolute path is %r", path)

    if not os.path.exists(path):
        logger.debug("The directory(%s) doesn't exist, will create it", path)
        try:
            os.makedirs(path, mode=0o700)
        except FileExistsError:
            logger.debug("The directory(%s) already exist.", path)
        except PermissionError as e:
            logger.critical(f"No write permission on the directory '{path}'', error = {e}")
            raise ValueError(e.__str__() + f"\nNo write permission on the directory '{path}'.") from e
    return path


def _get_print_file_name(file_name):
    """Add timestamp suffix to file name. Rename the file name:  file_name + "." + time(seconds)."""
    time_second = str(int(time.time()))
    file_name = file_name + "." + time_second
    if os.path.exists(file_name):
        raise ValueError("For 'context.set_context', the argument 'print_file_path' {} already exists, "
                         "please check it".format(file_name))
    return file_name


class _ThreadLocalInfo(threading.local):
    """
    Thread local Info used for store thread local attributes.
    """

    def __init__(self):
        super().__init__()
        self._reserve_class_name_in_scope = True
        self.debug_runtime = False

    @property
    def reserve_class_name_in_scope(self):
        """Get whether to save the network class name in the scope."""
        return self._reserve_class_name_in_scope

    @reserve_class_name_in_scope.setter
    def reserve_class_name_in_scope(self, reserve_class_name_in_scope):
        """Set whether to save the network class name in the scope."""
        self._reserve_class_name_in_scope = reserve_class_name_in_scope


_ContextRecord = namedtuple(
    "_ContextRecord", ["is_pynative_mode", "switch_context_fn"])


class _ContextSwitchInfo(threading.local):
    """
    Record of context switch information.

    Args:
        is_pynative (bool): Whether to adopt the PyNative mode.
    """

    def __init__(self, is_pynative):
        super().__init__()
        self.context_stack = []
        if is_pynative:
            self.push(True, None)

    def push(self, is_pynative, switch_context_fn):
        """
        Push a context switch record onto the stack.

        Args:
            is_pynative (bool): Whether context switch to PyNative mode.
            switch_context_fn (Function): A callable that executes the context switch.
        """
        if isinstance(switch_context_fn, FunctionType):
            switch_context_fn()
        self.context_stack.append(
            _ContextRecord(is_pynative, switch_context_fn))

    def pop(self):
        self.context_stack.pop()


class _Context:
    """
    _Context is the environment in which operations are executed

    Note:
        Create a context through instantiating Context object is not recommended.
        should use context() to get the context since Context is a singleton.
    """
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._instance_lock:
                cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self):
        self._thread_local_info = _ThreadLocalInfo()
        self._context_switches = _ContextSwitchInfo(False)
        self._context_handle = MSContext.get_instance()
        self._support_binary = False
        self.enable_compile_cache = None
        self._mode = PYNATIVE_MODE
        self.aoe_config = {}
        self.jit_config = {}
        self.ascend_config = {}
        self.gpu_config = {}

    def __getattribute__(self, attr):
        value = object.__getattribute__(self, attr)
        if attr == "_context_handle" and value is None:
            raise ValueError("Get {} failed, please check whether 'env_config_path' is correct.".format(attr))
        return value

    def get_param(self, param):
        return self._context_handle.get_param(param)

    def set_param(self, param, value):
        self._context_handle.set_param(param, value)

    def get_mode(self):
        """Get current mode."""
        return self._mode

    def get_jit_config(self):
        """Get current jit_config."""
        return self.jit_config

    def set_mode(self, mode):
        """
        Switch between Graph mode and PyNative mode.

        Args:
            mode (int): GRAPH_MODE or PYNATIVE_MODE.
        """
        if mode == PYNATIVE_MODE:
            if self.enable_debug_runtime:
                self.set_backend_policy("vm")
            self._context_switches.push(True, None)
        elif mode == GRAPH_MODE:
            if self.enable_debug_runtime:
                self.set_backend_policy("ge")
            self._context_switches.push(False, None)
        else:
            raise ValueError(f"For 'context.set_context', the argument 'mode' should be context.GRAPH_MODE (0) "
                             f"or context.PYNATIVE_MODE (1), but got {mode}.")
        self.set_param(ms_ctx_param.mode, mode)
        self._mode = mode

    def set_jit_syntax_level(self, level):
        """"Set the JIT syntax level for graph compiling"""
        if level != STRICT and level != COMPATIBLE and level != LAX:
            raise ValueError(f"For 'context.set_jit_syntax_level', the argument 'level' should be context.STRICT "
                             f"or context.LAX, but got {level}.")
        self.set_param(ms_ctx_param.jit_syntax_level, level)

    def set_debug_level(self, level):
        """"Set the debug level for graph compiling"""
        if level != RELEASE and level != DEBUG:
            raise ValueError(f"For 'context.set_debug_level', the argument 'level' should be context.RELEASE "
                             f"or context.DEBUG, but got {level}.")
        self.set_param(ms_ctx_param.debug_level, level)

    def set_memory_optimize_level(self, memory_optimize_level):
        """
        The memory optimize level, support "O0", "O1".

        Args:
            target (str): "O0", "O1"
        """
        memory_optimize_levels = ["O0", "O1"]
        if memory_optimize_level not in memory_optimize_levels:
            raise ValueError(f"For 'context.set_context', the argument 'memory_optimize_level' must be one of "
                             f"{memory_optimize_levels}, but got {memory_optimize_level}.")
        if memory_optimize_level == "O0":
            self.set_param(ms_ctx_param.memory_optimize_level, 0)
        else:
            self.set_param(ms_ctx_param.memory_optimize_level, 1)

    def set_exec_order(self, exec_order):
        """
        The execution order mode, support "bfs", "dfs".
        """
        exec_order_modes = ["bfs", "dfs"]
        if exec_order not in exec_order_modes:
            raise ValueError(f"For 'context.set_context', the argument 'exec_order' must be one of "
                             f"{exec_order_modes}, but got {exec_order}.")
        self.set_param(ms_ctx_param.exec_order, exec_order)

    def set_deterministic(self, deterministic):
        """
        Enable model run in deterministic, and support the values "ON" and "OFF".

        Args:
            deterministic (str): "ON", "OFF"
        """
        deterministic_options = ["ON", "OFF"]
        if deterministic not in deterministic_options:
            raise ValueError(f"For 'context.set_context', the argument 'deterministic' must be one of "
                             f"{deterministic_options}, but got {deterministic}.")

        logger.info(f"Set deterministic setting to '{deterministic}'.")

        # Must wait for all async created groups to be initialized so that
        # deterministic feature could be consistent between all processes.
        CollectiveManager.get_instance().wait_all_comm_init()

        self.set_param(ms_ctx_param.deterministic, deterministic)

        hccl_deterministic = os.getenv("HCCL_DETERMINISTIC")
        te_parallel_compiler = os.getenv("TE_PARALLEL_COMPILER")
        if deterministic == "ON":
            if hccl_deterministic and hccl_deterministic != "true":
                logger.warning(f"Environment 'HCCL_DETERMINISTIC' should be 'true' when set deterministic='ON', but "
                               f"got '{hccl_deterministic}'. 'HCCL_DETERMINISTIC' will be set to 'true'.")
            if te_parallel_compiler and te_parallel_compiler != "1":
                logger.warning(f"Environment 'TE_PARALLEL_COMPILER' should be '1' when set deterministic='ON', but "
                               f"got '{te_parallel_compiler}'. 'TE_PARALLEL_COMPILER' will be set to '1'.")
            os.environ["HCCL_DETERMINISTIC"] = "true"
            os.environ["TE_PARALLEL_COMPILER"] = "1"
        if deterministic == "OFF":
            if hccl_deterministic and hccl_deterministic != "false":
                logger.warning(f"Environment 'HCCL_DETERMINISTIC' should not be set or be 'false' when set "
                               f"deterministic='OFF', but got '{hccl_deterministic}'. 'HCCL_DETERMINISTIC' "
                               f"will be unset.")
                del os.environ["HCCL_DETERMINISTIC"]
            if te_parallel_compiler and te_parallel_compiler != "0":
                logger.warning(f"Environment 'TE_PARALLEL_COMPILER' should not be set or be '0' when set "
                               f"deterministic='OFF', but got '{te_parallel_compiler}'. 'TE_PARALLEL_COMPILER' "
                               f"will be unset.")
                del os.environ["TE_PARALLEL_COMPILER"]

    def set_ascend_config(self, ascend_config):
        """
        Enable ascend config.

        Args:
            ascend_config (dict):
                - precision_mode (str): "force_fp16", "allow_fp32_to_fp16", "allow_mix_precision",
                            "must_keep_origin_dtype", "force_fp32", "allow_fp32_to_bf16",
                            "allow_mix_precision_fp16" and "allow_mix_precision_bf16".
                - jit_compile (bool): ``False`` and ``True``.
                - atomic_clean_policy (int): ``0`` and ``1``. Default: ``1`` .
                - op_precision_mode (str): precision mode config file path.
                - op_debug_option (str): Enable debugging options for Ascend operators,
                  default not enabled, only supports ``"oom"`` currently.
                  ``"oom"``: Detect memory out of bounds.
                - ge_options (dict): Global or session CANN options.
                - exception_dump (str): Has been deprecated since MindSpore 2.6. Please use
                  api :func:`mindspore.device_context.ascend.op_debug.aclinit_config` instead.
                - parallel_speed_up_json_path(Union[str, None]): The path to the parallel speed up json file.
                  If its value is None or '', it does not take effect. Default None.
                - host_scheduling_max_threshold(int): The host scheduling max threshold.
                - hccl_watchdog (bool): Enable a thread to monitor the failure of collective communication.
                  Default: ``True`` .
        """
        ascend_cfg_modes = {
            'precision_mode': ["force_fp16", "allow_fp32_to_fp16", "allow_mix_precision", "must_keep_origin_dtype",
                               "force_fp32", "allow_fp32_to_bf16", "allow_mix_precision_fp16",
                               "allow_mix_precision_bf16"],
            'jit_compile': [True, False],
            'atomic_clean_policy': [0, 1],
            'matmul_allow_hf32': [True, False],
            'conv_allow_hf32': [True, False],
            'exception_dump': ["0", "1", "2"],
            'op_precision_mode': (str,),
            'ge_options': (dict,),
            'parallel_speed_up_json_path': (str, None),
            'host_scheduling_max_threshold': (int,),
            'cur_step_num': (int,),
            'save_checkpoint_steps': (int,),
            'need_ckpt': (bool,),
            'last_triggered_step': (int,),
            'hccl_watchdog': (bool,),
            'topo_order': (dict,),
            'op_debug_option': (str, None),
        }
        ascend_cfg_setters = {
            'precision_mode': self._get_ascend_config_setter('precision_mode'),
            'jit_compile': self._get_ascend_config_setter('jit_compile', lambda v: "1" if v else "0"),
            'atomic_clean_policy': self._get_ascend_config_setter('atomic_clean_policy', str),
            'matmul_allow_hf32': self._get_ascend_config_setter('matmul_allow_hf32', lambda v: "1" if v else "0"),
            'conv_allow_hf32': self._get_ascend_config_setter('conv_allow_hf32', lambda v: "1" if v else "0"),
            'exception_dump': lambda x: x,
            'op_debug_option': self._set_op_debug_option,
            'op_precision_mode': self._set_op_precision_mode,
            'ge_options': self._set_ge_options,
            'parallel_speed_up_json_path': self._set_speedup_config_path,
            'host_scheduling_max_threshold': self._get_ascend_config_setter('host_scheduling_max_threshold', str),
            'cur_step_num': self._set_cur_step_num,
            'save_checkpoint_steps': self._set_save_checkpoint_steps,
            'need_ckpt': self._set_need_ckpt,
            'last_triggered_step': self._set_last_triggered_step,
            'hccl_watchdog': self._set_hccl_watchdog,
            'topo_order': self._set_topo_order
        }
        invalid_context_dict = {
            'exception_dump': {'version': '2.6', 'interface': 'device_context.ascend.op_debug.aclinit_config()'}
        }
        ascend_cfg_set = tuple(ascend_cfg_modes.keys())
        for ascend_key, ascend_value in ascend_config.items():
            if ascend_key not in ascend_cfg_set:
                raise ValueError(f"For 'context.set_context', the key of argument 'ascend_config' must be one of "
                                 f"{ascend_cfg_set}, but got {ascend_key}.")
            if ascend_key in invalid_context_dict:
                key = invalid_context_dict.get(ascend_key)
                deprecated_version, new_interface = key.get('version'), key.get('interface')
                log = (
                    f"For 'ascend_config', the parameter '{ascend_key}' has been removed"
                    f" since MindSpore {deprecated_version} version."
                )
                if new_interface:
                    log += f" Please use the {new_interface} instead."
                raise ValueError(log)
            supported_modes = ascend_cfg_modes.get(ascend_key)
            if isinstance(supported_modes, list) and ascend_value not in supported_modes:
                raise ValueError(f"For 'ascend_config', the value of argument {ascend_key} must be one of "
                                 f"{supported_modes}, but got {ascend_value}.")
            if isinstance(supported_modes, tuple) and not isinstance(ascend_value, supported_modes):
                raise TypeError(f"For 'ascend_config', the type of argument {ascend_key} must be one of "
                                f"{supported_modes}, but got {type(ascend_value)}.")
            cfg_setter = ascend_cfg_setters.get(ascend_key)
            cfg_setter(ascend_value)
        self.ascend_config = ascend_config

    def set_gpu_config(self, gpu_config):
        """
        Enable gpu config.

        Args:
            gpu_config (dict):

                - conv_fprop_algo (str): "normal", "performance" or user specifies conv forward algorithm directly.
                - conv_dgrad_algo (str): "normal", "performance" or user specifies conv data grad algorithm directly.
                - conv_wgrad_algo (str): "normal", "performance" or user specifies conv weight grad algorithm directly.
                - conv_allow_tf32 (bool): ``False`` and ``True``.
                - matmul_allow_tf32 (bool): ``False`` and ``True``.
        """

        gpu_cfgs = {'conv_fprop_algo': ["normal", "performance", "implicit_gemm", "precomp_gemm", "gemm", "direct",
                                        "fft", "fft_tiling", "winograd", "winograd_nonfused"],
                    'conv_dgrad_algo': ["normal", "performance", "algo_0", "algo_1", "fft", "fft_tiling", "winograd",
                                        "winograd_nonfused"],
                    'conv_wgrad_algo': ["normal", "performance", "algo_0", "algo_1", "fft", "algo_3", "fft_tiling",
                                        "winograd_nonfused"],
                    'conv_allow_tf32': [True, False],
                    'matmul_allow_tf32': [True, False]}
        for gpu_key in gpu_config:
            if gpu_key not in gpu_cfgs:
                raise ValueError(f"For 'context.set_context', the key of argument 'gpu_config' must be one of "
                                 f"{gpu_cfgs}, but got {gpu_key}.")
            supported_value = gpu_cfgs.get(gpu_key)
            if gpu_config[gpu_key] not in supported_value:
                raise ValueError(f"For 'gpu_config', the value of argument {gpu_key} must be one of "
                                 f"{supported_value}, but got {gpu_config[gpu_key]}.")
            if gpu_key == 'conv_fprop_algo':
                self.set_param(ms_ctx_param.conv_fprop_algo, gpu_config[gpu_key])
            if gpu_key == 'conv_dgrad_algo':
                self.set_param(ms_ctx_param.conv_dgrad_algo, gpu_config[gpu_key])
            if gpu_key == 'conv_wgrad_algo':
                self.set_param(ms_ctx_param.conv_wgrad_algo, gpu_config[gpu_key])
            if gpu_key == 'conv_allow_tf32':
                self.set_param(ms_ctx_param.conv_allow_tf32, gpu_config[gpu_key])
            if gpu_key == 'matmul_allow_tf32':
                self.set_param(ms_ctx_param.matmul_allow_tf32, gpu_config[gpu_key])
            self.gpu_config = gpu_config

    def set_jit_config(self, jit_config):
        """
        Enable jit config.

        Args:
            jit_config (dict):

                - jit_level (str): "O0", "O1" or "O2" to control the compilation optimization level.
        """
        jit_cfgs = {'jit_level': ["O0", "O1", "O2"], 'infer_boost': ["on", "off"]}
        key_args_map = {'jit_level': ms_ctx_param.jit_level, 'infer_boost': ms_ctx_param.infer_boost}
        for jit_key in jit_config:
            if jit_key not in jit_cfgs:
                raise ValueError(f"For 'context.set_context', the key of argument 'jit_config' must be one of "
                                 f"{jit_cfgs}, but got {jit_key}.")
            supported_value = jit_cfgs.get(jit_key)
            if jit_config[jit_key] not in supported_value:
                raise ValueError(f"For 'jit_config', the value of argument {jit_key} must be one of "
                                 f"{supported_value}, but got {jit_config[jit_key]}.")
            self.set_param(key_args_map[jit_key], jit_config[jit_key])
        self.jit_config = jit_config

        jit_level = jit_config.get("jit_level", None)
        if jit_config.get("infer_boost", None) == "on" and (jit_level == "O1" or jit_level == "O2"):
            raise ValueError("Only jit_level set O0 can set infer_boost to on.")

    def set_backend_policy(self, policy):
        success = self._context_handle.set_backend_policy(policy)
        if not success:
            raise RuntimeError("Backend policy must be one of values in ['ge', 'vm', 'ms']. "
                               "But got {}.".format(policy))

    def set_save_graphs_path(self, save_graphs_path):
        self.set_param(ms_ctx_param.save_graphs_path, _make_directory(save_graphs_path))

    def set_device_target(self, target):
        """
        The target device to run, support "Ascend", "GPU", and "CPU".

        Args:
            target (str): "Ascend", "GPU", and "CPU".
        """
        valid_targets = ["CPU", "GPU", "Ascend", "Davinci"]
        if target not in valid_targets:
            raise ValueError(f"For 'context.set_context', the argument 'device_target' must be one of "
                             f"{valid_targets}, but got {target}.")
        if target == "Davinci":
            target = "Ascend"
            logger.warning("The device 'Davinci' is deprecated and will be removed in the next version. "
                           "For 'context.set_context', please set the argument 'device_target' "
                           "to 'CPU', 'GPU' or 'Ascend',if you set it to 'Davinci', it will be automatically "
                           "changed to 'Ascend'.")
        # If in Parameter Server mode, Ascend card should not be used by server and scheduler.
        if _need_reset_device_target_for_ps(target):
            logger.info("Reset device target to CPU when set_device_target.")
            target = "CPU"
        self.set_param(ms_ctx_param.device_target, target)
        if self.enable_debug_runtime and target == "CPU":
            self.set_backend_policy("vm")

    def set_aoe_tune_mode(self, tune_mode):
        """
        Set aoe tune mode, support "online" and "offline".

        Args:
            tune_mode (str): "online" and "offline".
        """
        candidate = ["online", "offline"]
        if tune_mode in candidate:
            self.set_param(ms_ctx_param.aoe_tune_mode, tune_mode)
        else:
            raise ValueError(f"For 'context.set_context', the argument 'aoe_tune_mode' must be in "
                             f"['online', 'offline'], but got {tune_mode}.")

    def set_aoe_config(self, aoe_config):
        """
        Enable aoe config.

        Args:
            aoe_config (dict):
                - job_type (str): ``"1"``, ``"2"``. Default: ``"2"`` .
                  - ``"1"``: subgraph tuning.
                  - ``"2"``: operator tuning.
        """

        aoe_cfgs = {'job_type': ["1", "2"]}
        for aoe_config_key in aoe_config:
            if aoe_config_key not in aoe_cfgs:
                raise ValueError(f"For 'context.set_context', the key of argument 'aoe_config' must be one of "
                                 f"{aoe_cfgs}, but got {aoe_config_key}.")
            supported_value = aoe_cfgs.get(aoe_config_key)
            if aoe_config[aoe_config_key] not in supported_value:
                raise ValueError(f"For 'aoe_config', the value of argument {aoe_config_key} must be one of "
                                 f"{supported_value}, but got {aoe_config[aoe_config_key]}.")
            if aoe_config_key == 'job_type':
                self.set_param(ms_ctx_param.aoe_job_type, aoe_config[aoe_config_key])
        self.aoe_config = aoe_config

    def set_device_id(self, device_id):
        if device_id < 0 or device_id > 4095:
            raise ValueError(f"For 'context.set_context', the argument 'device_id' must be in range [0, 4095], "
                             f"but got {device_id}.")
        self.set_param(ms_ctx_param.device_id, device_id)

    def set_max_call_depth(self, max_call_depth):
        if max_call_depth <= 0:
            raise ValueError(f"For 'context.set_context', the argument 'max_call_depth' must be greater than 0, "
                             f"but got {max_call_depth}.")
        self.set_param(ms_ctx_param.max_call_depth, max_call_depth)

    def set_profiling_options(self, option):
        if not isinstance(option, str):
            raise TypeError("For 'context.set_context', the argument 'profiling_option' must be string, "
                            "but got {}.".format(type(option)))
        self.set_param(ms_ctx_param.profiling_options, option)

    def set_variable_memory_max_size(self, variable_memory_max_size):
        """set values of variable_memory_max_size and graph_memory_max_size"""
        logger.warning("For 'context.set_context', the parameter 'variable_memory_max_size' is deprecated, "
                       "and will be removed in a future "
                       "version. Please use parameter 'max_device_memory' instead.")
        if not Validator.check_str_by_regular(variable_memory_max_size, _RE_PATTERN):
            raise ValueError("For 'context.set_context', the argument 'variable_memory_max_size' should be in correct"
                             " format! It must be a string ending with 'GB', in addition to that, it must contain "
                             "only numbers or decimal points, such as \"5GB\" or \"3.5GB\", but got {}GB."
                             .format(variable_memory_max_size))
        if float(variable_memory_max_size[:-2]) > _DEVICE_APP_MEMORY_SIZE:
            raise ValueError("For 'context.set_context', the argument 'variable_memory_max_size' should not be "
                             "greater than 31GB, but got {}GB.".format(variable_memory_max_size))
        variable_memory_max_size_ = variable_memory_max_size[:-2] + " * 1024 * 1024 * 1024"
        graph_memory_max_size = _DEVICE_APP_MEMORY_SIZE - int(variable_memory_max_size[:-2])
        graph_memory_max_size_ = str(graph_memory_max_size) + " * 1024 * 1024 * 1024"
        self.set_param(ms_ctx_param.variable_memory_max_size, variable_memory_max_size_)
        self.set_param(ms_ctx_param._graph_memory_max_size, graph_memory_max_size_)

    def set_max_device_memory(self, max_device_memory):
        if not Validator.check_str_by_regular(max_device_memory, _RE_PATTERN):
            raise ValueError("For 'context.set_context', the argument 'max_device_memory' should be in correct "
                             " format! It must be a string ending with 'GB', in addition to that, it must contain "
                             "only numbers or decimal points, such as \"5GB\" or \"3.5GB\", but got {}."
                             .format(max_device_memory))
        max_device_memory_value = float(max_device_memory[:-2])
        if max_device_memory_value == 0:
            raise ValueError("For 'context.set_context', the argument 'max_device_memory' should not be \"0GB\".")
        self.set_param(ms_ctx_param.max_device_memory, max_device_memory_value)

    def set_mempool_block_size(self, mempool_block_size):
        """Set the block size of memory pool."""
        global_jit_config = get_jit_config()
        is_ge = False
        if global_jit_config:
            is_ge = global_jit_config.get('backend') == "GE" or global_jit_config.get('jit_level') == "O2"
        if is_ge:
            logger.warning("GE doesn't support to set parameter 'mempool_block_size' of context currently, "
                           "you can use pynative mode or set jit_level=O0/O1.")
            return
        if not Validator.check_str_by_regular(mempool_block_size, _RE_PATTERN):
            raise ValueError("For 'context.set_context', the argument 'mempool_block_size' should be in "
                             "correct format! Such as \"10GB\", "
                             "but got {}".format(mempool_block_size))
        mempool_block_size_value = float(mempool_block_size[:-2])
        if mempool_block_size_value < 1.0:
            raise ValueError("For 'context.set_context',  the argument 'mempool_block_size' should be "
                             "greater or equal to \"1GB\", "
                             "but got {}GB".format(float(mempool_block_size[:-2])))
        self.set_param(ms_ctx_param.mempool_block_size, mempool_block_size_value)

    def set_print_file_path(self, file_path):
        """Add timestamp suffix to file name. Sets print file path."""
        print_file_path = os.path.realpath(file_path)
        if os.path.isdir(print_file_path):
            raise IOError("For 'context.set_context', the argument 'print_file_path' should be file path, "
                          "but got directory {}.".format(file_path))

        if os.path.exists(print_file_path):
            _path, _file_name = os.path.split(print_file_path)
            path = _make_directory(_path)
            file_name = _get_print_file_name(_file_name)
            full_file_name = os.path.join(path, file_name)
        else:
            full_file_name = print_file_path
        self.set_param(ms_ctx_param.print_file_path, full_file_name)

    def set_env_config_path(self, env_config_path):
        """Check and set env_config_path."""
        if not self._context_handle.enable_dump_ir():
            raise ValueError("For 'context.set_context', the argument 'env_config_path' is not supported, please "
                             "enable ENABLE_DUMP_IR with '-D on' and recompile source firstly.")
        env_config_path = os.path.realpath(env_config_path)
        if not os.path.isfile(env_config_path):
            raise ValueError("For 'context.set_context', the 'env_config_path' file %r is not exists, "
                             "please check whether 'env_config_path' is correct." % env_config_path)
        try:
            with open(env_config_path, 'r', encoding='utf-8') as f:
                json.load(f)
        except (TypeError, ValueError) as exo:
            raise ValueError(str(exo) + "\nFor 'context.set_context', open or load the 'env_config_path' file {} "
                                        "failed, please check whether 'env_config_path' is json file and correct, "
                                        "or may not have permission to read it.".format(env_config_path)) from exo
        self.set_param(ms_ctx_param.env_config_path, env_config_path)

    def set_runtime_num_threads(self, runtime_num_threads):
        """Check and set runtime_num_threads."""
        if runtime_num_threads < 0:
            raise ValueError("The num of thread must bigger than or equal to 0.")
        self.set_param(ms_ctx_param.runtime_num_threads, runtime_num_threads)

    def set_op_timeout(self, op_timeout):
        """Set the maximum duration of executing an operator in seconds."""
        if op_timeout < 0:
            raise ValueError("The num of op exe timeout must bigger than or equal to 0.")
        self.set_param(ms_ctx_param.op_timeout, op_timeout)

    def set_inter_op_parallel_num(self, inter_op_parallel_num):
        """Check and set inter_op_parallel_num."""
        if inter_op_parallel_num < 0:
            raise ValueError("The num of parallel thread must bigger than or equal to 0.")
        self.set_param(ms_ctx_param.inter_op_parallel_num, inter_op_parallel_num)

    setters = {
        'mode': set_mode,
        'save_graphs_path': set_save_graphs_path,
        'device_target': set_device_target,
        'aoe_tune_mode': set_aoe_tune_mode,
        'device_id': set_device_id,
        'max_call_depth': set_max_call_depth,
        'profiling_options': set_profiling_options,
        'variable_memory_max_size': set_variable_memory_max_size,
        'max_device_memory': set_max_device_memory,
        'mempool_block_size': set_mempool_block_size,
        'print_file_path': set_print_file_path,
        'env_config_path': set_env_config_path,
        'inter_op_parallel_num': set_inter_op_parallel_num,
        'runtime_num_threads': set_runtime_num_threads,
        'memory_optimize_level': set_memory_optimize_level,
        'exec_order': set_exec_order,
        'op_timeout': set_op_timeout,
        'deterministic': set_deterministic,
        'ascend_config': set_ascend_config,
        'jit_syntax_level': set_jit_syntax_level,
        'debug_level': set_debug_level,
        'gpu_config': set_gpu_config,
        'aoe_config': set_aoe_config,
        'jit_config': set_jit_config,
    }

    @property
    def reserve_class_name_in_scope(self):
        """Get whether to save the network class name in the scope."""
        return self._thread_local_info.reserve_class_name_in_scope

    @reserve_class_name_in_scope.setter
    def reserve_class_name_in_scope(self, reserve_class_name_in_scope):
        """Set whether to save the network class name in the scope."""
        if not isinstance(reserve_class_name_in_scope, bool):
            raise ValueError("For 'context.set_context', the type of the property 'reserve_class_name_in_scope' must "
                             "be bool, but got {}.".format(type(reserve_class_name_in_scope)))
        self._thread_local_info.reserve_class_name_in_scope = reserve_class_name_in_scope

    @property
    def enable_ge(self):
        return self._context_handle.get_backend_policy() == 'ge'

    @property
    def enable_debug_runtime(self):
        return self._thread_local_info.debug_runtime

    @enable_debug_runtime.setter
    def enable_debug_runtime(self, enable):
        thread_info = self._thread_local_info
        thread_info.debug_runtime = enable

    @property
    def support_binary(self):
        """Whether support run .pyc or .so in graph mode."""
        return self._support_binary

    @support_binary.setter
    def support_binary(self, support: bool):
        if not isinstance(support, bool):
            raise TypeError(f"The attribute 'support_binary' should be a bool, but got {type(support)}.")
        self._support_binary = support

    def _get_ascend_config_setter(self, ascend_key, trans_fn=None):
        def _config_setter(ascend_value):
            self.set_param(ms_ctx_param.__members__[ascend_key], trans_fn(ascend_value))

        if trans_fn is None:
            def trans_fn(x):
                return x
        return _config_setter

    def _set_op_debug_option(self, option_value):
        valid_order = {'oom'}
        if not isinstance(option_value, str):
            raise TypeError(f"For 'ascend_config', the type of 'op_debug_option' must be str, "
                            f"but got {type(option_value)}.")
        if option_value not in valid_order:
            raise ValueError(f"For 'ascend_config', the 'op_debug_option' supports being set to 'oom' currently, "
                             f"but got {option_value}.")
        self.set_param(ms_ctx_param.op_debug_option, option_value)

    def _set_op_precision_mode(self, ascend_value):
        op_precision_path = ascend_value
        real_path = os.path.realpath(op_precision_path)
        if not os.path.exists(real_path):
            raise ValueError(f"For 'ascend_config', the 'op_precision_mode' is invalid path, "
                             f"got '{op_precision_path}'.")
        self.set_param(ms_ctx_param.op_precision_mode, ascend_value)

    def _set_ge_options(self, ge_options):
        """Set ge options."""
        for level, options in ge_options.items():
            if level not in ['global', 'session']:
                raise ValueError(f"For 'ascend_config', the key of ge_options must be one of "
                                 f"('global', 'session'), but got {level}.")

            if not isinstance(options, dict):
                raise TypeError(f"For 'ge_options', the type of {level} options must be dict, "
                                f"but got {type(options)}. The error options: {options}.")

            for key, value in options.items():
                if not isinstance(key, str):
                    raise TypeError(f"For 'ge_options', the type of key and value must be str, "
                                    f"but got {type(key)}. The error key is {key}.")
                if not isinstance(value, str):
                    raise TypeError(f"For 'ge_options', the type of key and value must be str, "
                                    f"but got {type(value)}. The error value is {value}")

        options_str = json.dumps(ge_options)
        self.set_param(ms_ctx_param.ge_options, options_str)

    def _set_topo_order(self, topo_order):
        """
        Set topo order.

        Args:
            topo_order (dict):
                key: str, the name of the graph.
                value: str, the topo order of the graph, should be one of 'dfs', 'bfs', 'rdfs'.
        """
        valid_order = {'dfs', 'bfs', 'rdfs'}
        if not isinstance(topo_order, dict):
            raise TypeError(f"For 'ascend_config', the 'topo_order' should be a dict, "
                            f"got '{type(topo_order)}'.")
        for k, v in topo_order.items():
            if not isinstance(k, str):
                raise TypeError("key {} is not a str".format(k))
            if v not in valid_order:
                raise ValueError("value {} should be one of {}.".format(v, valid_order))

        options_str = json.dumps(topo_order)
        self.set_param(ms_ctx_param.topo_order, options_str)

    def _set_hccl_watchdog(self, flag):
        """set hccl watchdog"""
        if not isinstance(flag, bool):
            raise TypeError(f"For 'ascend_config', the type of 'hccl_watchdog' must be bool, but got {type(flag)}.")
        self.set_param(ms_ctx_param.hccl_watchdog, flag)

    def _set_need_ckpt(self, need_ckpt):
        """Set need ckpt flag"""
        if not isinstance(need_ckpt, bool):
            raise TypeError(f"For step num, the value type should be int, but got {type(need_ckpt)}, {need_ckpt}")
        self.set_param(ms_ctx_param.need_ckpt, need_ckpt)

    def _set_cur_step_num(self, step_num):
        """set current step num at every step begin"""
        if not isinstance(step_num, int):
            raise TypeError(f"For step num, the value type should be int, but got {type(step_num)}, {step_num}")
        self.set_param(ms_ctx_param.cur_step_num, step_num)

    def _set_save_checkpoint_steps(self, steps):
        """set save checkpoint steps before run"""
        if not isinstance(steps, int):
            raise TypeError(f"For step num, the value type should be int, but got {type(steps)}, {steps}")
        self.set_param(ms_ctx_param.save_checkpoint_steps, steps)

    def _set_last_triggered_step(self, step):
        """set last triggered save ckpt steps before run"""
        if not isinstance(step, int):
            raise TypeError(f"For step num, the value type should be int, but got {type(step)}, {step}")
        self.set_param(ms_ctx_param.last_triggered_step, step)

    @staticmethod
    def _check_speedup_config_str_value(key, value):
        """check speedup config str value"""
        if key in ["pp_1f1b_overlap", "recompute_comm_overlap", "recomputation_communication_overlap",
                   "matmul_grad_comm_overlap", "grad_matmul_communication_overlap"]:
            if isinstance(value, str):
                values = value.split(",")
                for v in values:
                    if v not in ['AlltoAll', 'AlltoAllV', 'MorphAllGather', 'AllReduce',
                                 'AllGather', 'ReduceScatter', 'MorphReduceScatter', '']:
                        raise ValueError("{} 's value should be subset of ['AlltoAll', 'AlltoAllV',"
                                         " 'MorphAllGather', 'AllGather', 'ReduceScatter',"
                                         " 'MorphReduceScatter', 'AllReduce'].".format(key))
                return value
            if value:
                return "AlltoAll,AlltoAllV,AllGather,ReduceScatter,AllReduce"
            return ""

        return value

    def _set_speedup_config_path(self, speedup_config_path):
        """"Check and set speedup config for auto parallel."""
        if speedup_config_path is None or speedup_config_path == "":
            return
        speedup_config_real_path = os.path.realpath(speedup_config_path)
        if not os.path.exists(speedup_config_real_path):
            raise ValueError(f"For 'ascend_config', the path to parallel_speed_up_json: "
                             f"{speedup_config_real_path} does not exist, please check whether the "
                             f"'parallel_speed_up_json_path' is correct.")
        try:
            valid_option = {"recompute_comm_overlap": (ms_ctx_param.recompute_comm_overlap, str),
                            "recomputation_communication_overlap": (ms_ctx_param.recompute_comm_overlap, str),
                            "matmul_grad_comm_overlap": (ms_ctx_param.matmul_grad_comm_overlap, (bool, str)),
                            "grad_matmul_communication_overlap": (ms_ctx_param.matmul_grad_comm_overlap, (bool, str)),
                            "enable_task_opt": (ms_ctx_param.enable_task_opt, bool),
                            "enable_communication_fusion": (ms_ctx_param.enable_task_opt, bool),
                            "enable_grad_comm_opt": (ms_ctx_param.enable_grad_comm_opt, bool),
                            "grad_computation_allreduce_overlap": (ms_ctx_param.enable_grad_comm_opt, bool),
                            "recompute_allgather_overlap_fagrad":
                                (ms_ctx_param.recompute_allgather_overlap_fagrad, bool),
                            "grad_fa_allgather_overlap":
                                (ms_ctx_param.recompute_allgather_overlap_fagrad, bool),
                            "interleaved_matmul_comm": (ms_ctx_param.interleaved_matmul_comm, bool),
                            "bias_add_comm_swap": (ms_ctx_param.bias_add_comm_swap, bool),
                            "allreduce_and_biasadd_swap": (ms_ctx_param.bias_add_comm_swap, bool),
                            "enable_opt_shard_comm_opt": (ms_ctx_param.enable_opt_shard_comm_opt, bool),
                            "computation_allgather_overlap": (ms_ctx_param.enable_opt_shard_comm_opt, bool),
                            "enable_begin_end_inline_opt": (ms_ctx_param.enable_begin_end_inline_opt, bool),
                            "enable_concat_eliminate_opt": (ms_ctx_param.enable_concat_eliminate_opt, bool),
                            "interleaved_layernorm_comm": (ms_ctx_param.interleaved_layernorm_comm, bool),
                            "enable_allreduce_slice_to_reducescatter":
                                (ms_ctx_param.enable_allreduce_slice_to_reducescatter, bool),
                            "enable_interleave_split_concat_branch":
                                (ms_ctx_param.enable_interleave_split_concat_branch, bool),
                            "enable_interleave_parallel_branch":
                                (ms_ctx_param.enable_interleave_parallel_branch, bool),
                            "enable_offloading_packed_experts": (ms_ctx_param.enable_offloading_packed_experts, bool),
                            "compute_communicate_fusion_level":
                                (ms_ctx_param.compute_communicate_fusion_level, int),
                            "computation_communication_fusion_level":
                                (ms_ctx_param.compute_communicate_fusion_level, int),
                            "enable_flash_attention_load_balance":
                                (ms_ctx_param.enable_flash_attention_load_balance, bool),
                            "pp_1f1b_overlap":
                                (ms_ctx_param.pp_1f1b_overlap, str),
                            "dataset_broadcast_opt_level":
                                (ms_ctx_param.dataset_broadcast_opt_level, int)}
            name_replace = {
                "recompute_comm_overlap": "recomputation_communication_overlap",
                "matmul_grad_comm_overlap": "grad_matmul_communication_overlap",
                "recompute_allgather_overlap_fagrad": "grad_fa_allgather_overlap",
                "enable_task_opt": "enable_communication_fusion",
                "enable_grad_comm_opt": "grad_computation_allreduce_overlap",
                "enable_opt_shard_comm_opt": "computation_allgather_overlap",
                "compute_communicate_fusion_level": "computation_communication_fusion_level",
                "dataset_broadcast_opt_level": "dataset_broadcast_opt_level",
                "bias_add_comm_swap": "allreduce_and_biasadd_swap"}
            with open(speedup_config_real_path, 'r', encoding='utf-8') as f:
                speedup_config = json.load(f)
                for key, value in speedup_config.items():
                    if not isinstance(key, str):
                        raise TypeError("key {} is not a str".format(key))
                    if key not in valid_option:
                        raise ValueError("key {} should be one of {}.".format(key, valid_option.keys()))
                    if key in name_replace:
                        logger.warning(f"For 'context.set_context', '{key}' parameter is deprecated, "
                                       "and will be removed in the next version, "
                                       f"Please use '{name_replace.get(key)}' instead.")
                    set_func, valid_type = valid_option.get(key)
                    if not isinstance(value, valid_type):
                        if not ((key == "recompute_comm_overlap" or key == "recomputation_communication_overlap")
                                and isinstance(value, bool)):
                            raise TypeError(f"The value type of {key} must be {valid_type}, "
                                            f"but got value is {value} and type is {type(value)}.")
                    value_new = self._check_speedup_config_str_value(key, value)
                    self.set_param(set_func, value_new)
        except (TypeError, ValueError) as exo:
            raise ValueError(str(exo) + "\nFor 'context.set_context', "
                                        "open or load the 'speedup_config_path' file {} "
                                        "failed, please check whether 'speedup_config_path' is json file and correct, "
                                        "or may not have permission to read it.".format(speedup_config_real_path)) \
                                        from exo


def _context():
    """
    Get the global _context, if context is not created, create a new one.

    Returns:
        _Context, the global context in PyNative mode.
    """
    global K_CONTEXT
    if K_CONTEXT is None:
        default_backend = 'debug'
        try:
            from mindspore import default_config
            default_backend = default_config.__backend__
        except ImportError:
            logger.error("import default config fail")
        K_CONTEXT = _Context()
        K_CONTEXT.enable_debug_runtime = False
        if default_backend == 'debug':
            K_CONTEXT.enable_debug_runtime = True
            default_backend = 'vm'
            K_CONTEXT.set_backend_policy(default_backend)
    return K_CONTEXT


@args_type_check(device_num=int, global_rank=int, gradients_mean=bool, gradient_fp32_sync=bool, parallel_mode=str,
                 auto_parallel_search_mode=str, search_mode=str, parameter_broadcast=bool, strategy_ckpt_load_file=str,
                 strategy_ckpt_save_file=str, full_batch=bool, enable_parallel_optimizer=bool, enable_alltoall=bool,
                 all_reduce_fusion_config=list, pipeline_stages=int, pipeline_segments=int,
                 pipeline_result_broadcast=bool, parallel_optimizer_config=dict,
                 pipeline_config=dict,
                 comm_fusion=dict, strategy_ckpt_config=dict, force_fp32_communication=bool)
def set_auto_parallel_context(**kwargs):
    r"""
    Set auto parallel context, this api will be deprecated and removed in future versions, please use the api
    :class:`mindspore.parallel.auto_parallel.AutoParallel` instead.

    Note:
        CPU only support data parallel.

    Some configurations are parallel mode specific, see the below table for details:

    ===========================  ===========================
    Common                       AUTO_PARALLEL
    ===========================  ===========================
    device_num                   gradient_fp32_sync
    global_rank                  loss_repeated_mean
    gradients_mean               search_mode
    parallel_mode                parameter_broadcast
    all_reduce_fusion_config     strategy_ckpt_load_file
    enable_parallel_optimizer    strategy_ckpt_save_file
    parallel_optimizer_config    dataset_strategy
    enable_alltoall              pipeline_stages
    pipeline_config              auto_parallel_search_mode
    force_fp32_communication     pipeline_result_broadcast
               \                 comm_fusion
               \                 strategy_ckpt_config
               \                 group_ckpt_save_file
               \                 auto_pipeline
               \                 dump_local_norm
               \                 dump_local_norm_path
               \                 dump_device_local_norm
    ===========================  ===========================

    Args:
        device_num (int): Available device number, the value must be in [1, 4096]. Default: ``1`` .
        global_rank (int): Global rank id, the value must be in [0, 4095]. Default: ``0`` .
        gradients_mean (bool): Whether to perform mean operator after allreduce of gradients.
                     "stand_alone" do not support gradients_mean. Default: ``False`` .
        gradient_fp32_sync (bool): Run allreduce of gradients in fp32. "stand_alone", "data_parallel"
                     and "hybrid_parallel" do not support gradient_fp32_sync. Default: ``True`` .
        loss_repeated_mean (bool) - Indicates whether the mean operator is executed backwards when the
                     calculation is repeated. Default: ``True`` .
        parallel_mode (str): There are five kinds of parallel modes, ``"stand_alone"`` , ``"data_parallel"`` ,
                     ``"hybrid_parallel"`` , ``"semi_auto_parallel"`` and ``"auto_parallel"`` . Note the pynative mode
                     only supports the ``"stand_alone"`` and ``"data_parallel"`` mode. Default: ``"stand_alone"`` .

                     - stand_alone: Only one processor is working.

                     - data_parallel: Distributes the data across different processors.

                     - hybrid_parallel: Achieves data parallelism and model parallelism manually.

                     - semi_auto_parallel: Achieves data and model parallelism by setting parallel strategies.

                     - auto_parallel: Achieving parallelism automatically.
        search_mode (str): There are three kinds of shard strategy search modes: ``"recursive_programming"`` ,
                     ``"sharding_propagation"`` and ``"dynamic_programming"`` (Not recommended).
                     Only works in ``"auto_parallel"`` mode.
                     Default: ``"recursive_programming"`` .

                     - recursive_programming: Recursive programming search mode. In order to obtain optimal performance,
                       it is recommended that users set the batch size to be greater than or equal to the product of
                       the number of devices and the number of multi-copy parallelism.

                     - sharding_propagation: Propagate shardings from configured ops to non-configured ops. Dynamic
                       shapes are not supported currently.

                     - dynamic_programming: Dynamic programming search mode.
        auto_parallel_search_mode (str): This is the old version of 'search_mode'. Here, remaining this attribute is
                     for forward compatibility, and this attribute will be deleted in a future MindSpore version.
        parameter_broadcast (bool): Whether to broadcast parameters before training. Before training, in order to have
                     the same network initialization parameter values for all devices, broadcast the parameters
                     on device 0 to other devices. Parameter broadcasting in different parallel modes is different,
                     ``data_parallel`` mode, all parameters are broadcast except for the parameter whose attribute
                     layerwise_parallel is ``True`` . ``Hybrid_parallel`` , ``semi_auto_parallel``  and
                     ``auto_parallel mode`` , the segmented parameters do not participate in broadcasting.
                     Default: ``False`` .
        strategy_ckpt_load_file (str): The path to load parallel strategy checkpoint. The parameter is not to be
                       recommended currently, it is better using 'strategy_ckpt_config' to replace it. Default: ``''``
        strategy_ckpt_save_file (str): The path to save parallel strategy checkpoint. The parameter is not to be
                       recommended currently, it is better using 'strategy_ckpt_config' to replace it. Default: ``''``
        full_batch (bool): If you load whole batch datasets in ``auto_parallel`` mode, this parameter
                       should be set as ``True`` . Default: ``False`` . The interface is not to be recommended
                       currently, it is better using 'dataset_strategy' to replace it.
        dataset_strategy (Union[str, tuple]): Dataset sharding strategy. Default: ``"data_parallel"`` .
                       dataset_strategy="data_parallel" is equal to full_batch=False, dataset_strategy="full_batch" is
                       equal to full_batch=True. For execution mode is 'GRAPH_MODE' and dataset load into net by model
                       parallel strategy likes ds_stra ((1, 8), (1, 8)), it requires using
                       set_auto_parallel_context(dataset_strategy=ds_stra). The dataset sharding strategy is not
                       affected by the currently configured parallel mode. parallel strategy also supports tuple of
                       Layout.
        enable_parallel_optimizer (bool): This is a developing feature, which shards the weight update computation for
                       data parallel training in the benefit of time and memory saving. Currently, auto and semi auto
                       parallel mode support all optimizers in both Ascend and GPU. Data parallel mode only supports
                       `Lamb` and `AdamWeightDecay` in Ascend . Default: ``False`` .
        force_fp32_communication (bool): A switch that determines whether reduce operators (AllReduce, ReduceScatter)
                        are forced to use the fp32 data type for communication during communication. True is the enable
                        switch. Default: ``False`` .
        enable_alltoall (bool): A switch that allows AllToAll operators to be generated during communication. If its
                        value is ``False`` , there will be a combination of operators such as AllGather, Split and
                        Concat instead of AllToAll. Default: ``False`` .
        all_reduce_fusion_config (list): Set allreduce fusion strategy by parameters indices. Only support ReduceOp.SUM
                       and HCCL_WORLD_GROUP/NCCL_WORLD_GROUP. No Default, if it is not set, the fusion is closed.
        pipeline_stages (int): Set the stage information for pipeline parallel. This indicates how the devices are
                        distributed alone in the pipeline. The total devices will be divided into 'pipeline_stages'
                        stages.
                        Default: ``1`` .
        pipeline_result_broadcast (bool): A switch that broadcast the last stage result to all other stage in pipeline
                        parallel inference. Default: ``False`` .
        pipeline_config (dict): A dict contains the keys and values for setting the pipeline parallelism configuration.
                        It supports the following keys:

                        - pipeline_interleave(bool): Indicates whether to enable the interleaved execution mode.
                        - pipeline_scheduler(str): Indicates the scheduling mode for pipeline parallelism. Only support
                          ``gpipe/1f1b/seqpipe/seqvpp/seqsmartvpp/zero_bubble_v``. When applying seqsmartvpp,
                          the pipeline parallel must be an even number.
        parallel_optimizer_config (dict): A dict contains the keys and values for setting the parallel optimizer
                        configure. The configure provides more detailed behavior control about parallel training
                        when parallel optimizer is enabled. The configure will be effective when we use
                        mindspore.set_auto_parallel_context(enable_parallel_optimizer=True).
                        It supports the following keys.

                        - gradient_accumulation_shard(bool): Please using optimizer_level: ``level2`` to replace
                          this config.
                          If ``true`` , the accumulation gradient parameters will be
                          sharded across the data parallel devices. This will
                          introduce additional communication(ReduceScatter) at
                          each step when accumulate the gradients, but saves a
                          lot of device memories, thus can make model be trained
                          with larger batch size. This configure is effective only
                          when the model runs on pipeline training or gradient
                          accumulation with data parallel. Default ``False`` .

                        - parallel_optimizer_threshold(int): Set the threshold of parallel optimizer. When parallel
                          optimizer is enabled, parameters with size smaller than this threshold will not be sharded
                          across the devices. Parameter size is calculated as:
                          shape[0] \* ... \* shape[n] \* size(dtype). Non-negative.
                          Unit: KB. Default: ``64`` .

                        - optimizer_weight_shard_size(int): Set the optimizer weight shard group size, if you want to
                          specific the maximum group size across devices when the parallel optimizer is enabled.
                          The numerical range can be (0, device_num]. If pipeline parallel is enabled, the numerical
                          range is (0, device_num/stage]. If the size of data parallel communication domain
                          of the parameter cannot be divided by `optimizer_weight_shard_size`, then the specified
                          communication group size will not take effect. Default value is ``-1`` , which means the
                          optimizer weight shard group size will be the size of data parallel group of each parameter.

                        - optimizer_level(str, optional): optimizer_level configuration is used to specify
                          the splitting level for optimizer sharding. It is important to note that the implementation
                          of optimizer sharding in static graph is inconsistent with dynamic graph like megatron,
                          but the memory optimization effect is the same. When optimizer_level= ``level1`` ,
                          splitting is performed on weights and optimizer state. When optimizer_level= ``level2`` ,
                          splitting is performed on weights, optimizer state, and gradients.
                          When optimizer_level= ``level3`` , splitting is performed on weights, optimizer state,
                          gradients, additionally, before the backward pass, the weights are further applied with
                          allgather communication to release the memory used by the forward pass allgather.
                          It must be one of [``level1``, ``level2``, ``level3``]. Default: ``level1``.

        comm_fusion (dict): A dict contains the types and configurations for setting the communication fusion. each
                        communication fusion config has two keys: "mode" and "config".
                        It supports following communication fusion types and configurations:

                        - openstate: Whether turn on the communication fusion or not. If `openstate` is ``True`` ,
                          turn on the communication fusion, otherwise, turn off the communication fusion.
                          Default: ``True`` .

                        - allreduce: If communication fusion type is `allreduce`. The `mode` contains: `auto`, `size`
                          and `index`. In `auto` mode, AllReduce fusion is configured by gradients size and the default
                          fusion threshold is `64` MB. In 'size' mode, AllReduce fusion is configured by gradients size
                          manually, and the fusion threshold must be larger than `0` MB. In `index` mode, it is same as
                          `all_reduce_fusion_config`.

                        - allgather: If communication fusion type is `allgather`. The `mode` contains: `auto`, `size`.
                          In `auto` mode, AllGather fusion is configured by gradients size, and the default fusion
                          threshold is `64` MB. In 'size' mode, AllGather fusion is configured by gradients size
                          manually, and the fusion threshold must be larger than `0` MB.

                        - reducescatter: If communication fusion type is `reducescatter`. The `mode` contains: `auto`
                          and `size`. Config is same as `allgather`.

        strategy_ckpt_config (dict): A dict contains the configurations for setting the parallel strategy file. This
                        interface contains the functions of parameter `strategy_ckpt_load_file` and
                        `strategy_ckpt_save_file`, it is recommonded to use this parameter to replace those two
                        parameters.
                        It contains following configurations:

                        - load_file (str): The path to load parallel strategy checkpoint. If the file name extension is
                          `.json`, the file is loaded in JSON format. Otherwise, the file is loaded in ProtoBuf
                          format.
                          Default: ``''``

                        - save_file (str): The path to save parallel strategy checkpoint. If the file name extension is
                          `.json`, the file is saved in JSON format. Otherwise, the file is saved in ProtoBuf format.
                          Default: ``''``

                        - only_trainable_params (bool): Only save/load the strategy information for trainable parameter.
                          Default: ``True`` .
        group_ckpt_save_file (str): The path to save parallel group checkpoint.
        auto_pipeline (bool): Set the pipeline stage number to automatic. Its value will be selected between 1 and the
                        parameter `pipeline_stages`. This option requires the `parallel_mode` to be ``auto_parallel``
                        and the `search_mode` to be ``recursive_programming``. Default: ``False`` .
        dump_local_norm (bool): Whether to dump local_norm value, when the `parallel_mode` is set to
                        ``semi_auto_parallel`` or ``auto_parallel``.
                        Default: ``False`` .
        dump_local_norm_path (str): The path to save dump files of local_norm value.
                        Default: ``''`` .
        dump_device_local_norm (bool): Whether to dump device_local_norm value, when the `parallel_mode` is set to
                        ``semi_auto_parallel`` or ``auto_parallel``.
                        Default: ``False`` .

    Raises:
        ValueError: If input key is not attribute in auto parallel context.

    Examples:
        >>> import mindspore as ms
        >>> ms.set_auto_parallel_context(device_num=8)
        >>> ms.set_auto_parallel_context(global_rank=0)
        >>> ms.set_auto_parallel_context(gradients_mean=True)
        >>> ms.set_auto_parallel_context(gradient_fp32_sync=False)
        >>> ms.set_auto_parallel_context(parallel_mode="auto_parallel")
        >>> ms.set_auto_parallel_context(search_mode="recursive_programming")
        >>> ms.set_auto_parallel_context(auto_parallel_search_mode="recursive_programming")
        >>> ms.set_auto_parallel_context(parameter_broadcast=False)
        >>> ms.set_auto_parallel_context(strategy_ckpt_load_file="./strategy_stage1.ckpt")
        >>> ms.set_auto_parallel_context(strategy_ckpt_save_file="./strategy_stage1.ckpt")
        >>> ms.set_auto_parallel_context(dataset_strategy=((1, 8), (1, 8)))
        >>> ms.set_auto_parallel_context(enable_parallel_optimizer=False)
        >>> ms.set_auto_parallel_context(enable_alltoall=False)
        >>> ms.set_auto_parallel_context(all_reduce_fusion_config=[8, 160])
        >>> ms.set_auto_parallel_context(pipeline_stages=2)
        >>> ms.set_auto_parallel_context(pipeline_stages=2, pipeline_result_broadcast=True)
        >>> parallel_config = {"gradient_accumulation_shard": True, "parallel_optimizer_threshold": 24,
        ...                    "optimizer_weight_shard_size": 2, "optimizer_level": "level3"}
        >>> ms.set_auto_parallel_context(parallel_optimizer_config=parallel_config, enable_parallel_optimizer=True)
        >>> config = {"allreduce": {"mode": "size", "config": 32}, "allgather": {"mode": "size", "config": 32}}
        >>> ms.set_auto_parallel_context(comm_fusion=config)
        >>> stra_ckpt_dict = {"load_file": "./stra0.ckpt", "save_file": "./stra1.ckpt", "only_trainable_params": False}
        >>> ms.set_auto_parallel_context(strategy_ckpt_config=stra_ckpt_dict)
    """
    _set_auto_parallel_context(**kwargs)


def get_auto_parallel_context(attr_key):
    """
    Get auto parallel context attribute value according to the key, this api will be deprecated and removed in future
    versions.

    Args:
        attr_key (str): The key of the attribute.

    Returns:
        Returns attribute value according to the key.

    Raises:
        ValueError: If input key is not attribute in auto parallel context.

    Examples:
        >>> import mindspore as ms
        >>> parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        >>> dataset_strategy = ms.get_auto_parallel_context("dataset_strategy")
    """
    return _get_auto_parallel_context(attr_key)


def reset_auto_parallel_context():
    """
    Reset auto parallel context attributes to the default values, this api will be deprecated and removed in future
    versions, please use the api :class:`mindspore.parallel.auto_parallel.AutoParallel` instead.

    - device_num: 1.
    - global_rank: 0.
    - gradients_mean: False.
    - gradient_fp32_sync: True.
    - parallel_mode: 'stand_alone'.
    - search_mode: 'recursive_programming'.
    - auto_parallel_search_mode: 'recursive_programming'.
    - parameter_broadcast: False.
    - strategy_ckpt_load_file: ''.
    - strategy_ckpt_save_file: ''.
    - full_batch: False.
    - enable_parallel_optimizer: False.
    - force_fp32_communication: False.
    - enable_alltoall: False.
    - pipeline_stages: 1.
    - pipeline_result_broadcast: False.
    - fusion_threshold: 64.
    - auto_pipeline: False.
    - dump_local_norm: False.
    - dump_local_norm_path: ''.
    - dump_device_local_norm: False.

    Examples:
        >>> import mindspore as ms
        >>> ms.reset_auto_parallel_context()
    """
    _reset_auto_parallel_context()
    api.ms_compile_cache.clear()


def _check_target_specific_cfgs(device, arg_key):
    """Checking whether a config is suitable for a specified device"""
    device_cfgs = {
        'enable_reduce_precision': ['Ascend'],
        'print_file_path': ['Ascend'],
        'variable_memory_max_size': ['Ascend'],
        'max_device_memory': ['Ascend', 'GPU'],
        'mempool_block_size': ['GPU', 'Ascend'],
        'disable_format_transform': ['GPU'],
        'ascend_config': ['Ascend'],
        'gpu_config': ['GPU'],
    }
    # configs not in map device_cfgs are supposed to be suitable for all devices
    if arg_key not in device_cfgs:
        return True
    supported_devices = device_cfgs[arg_key]
    if device in supported_devices:
        return True
    logger.warning(f"For 'context.set_context', when set the argument '{arg_key}', "
                   f"the argument 'device_target' only supports devices in '{supported_devices}', "
                   f"but got '{device}', ignore it.")
    return False


def _check_ascend_device_context_initialized(device_target, settings):
    if device_target == 'Ascend' and is_initialized(device_target):
        for key, _ in settings.items():
            if key in ('ascend_config', 'deterministic', 'jit_compile', 'device_id'):
                logger.warning("For 'context.set_context' in Ascend backend, the backend is already initialized, "
                               "please set it before the definition of any Tensor and Parameter, and the "
                               "instantiation and execution of any operation and net, otherwise the settings may not "
                               "take effect. ")
                break


def _check_key(key):
    if key in ('precision_mode', 'jit_compile', 'atomic_clean_policy', 'matmul_allow_hf32', 'conv_allow_hf32',
               'op_precision_mode', 'host_scheduling_max_threshold', 'ge_options', 'op_debug_option'):
        raise ValueError(f"Please set '{key}' through parameter ascend_config")


def _check_context_deprecated(key):
    """Checking whether a context key will be deprecated."""
    deprecated_context_dict = {'save_graphs': 'env MS_DEV_SAVE_GRAPHS',
                               'save_graphs_path': 'env MS_DEV_SAVE_GRAPHS_PATH',
                               'precompile_only': 'env MS_DEV_PRECOMPILE_ONLY',
                               'check_bprop': '',
                               'max_call_depth': 'api mindspore.set_recursion_limit()',
                               'grad_for_scalar': 'tensor derivative',
                               'enable_compile_cache': 'env MS_COMPILER_CACHE_ENABLE',
                               'enable_cache_path': 'env MS_COMPILER_CACHE_PATH',
                               'debug_level': '',
                               'device_target': 'api mindspore.set_device()',
                               'device_id': 'api mindspore.set_device()',
                               'deterministic': 'api mindspore.set_deterministic()',
                               'inter_op_parallel_num': 'api mindspore.runtime.dispatch_threads_num()',
                               'pynative_synchronize': 'api mindspore.runtime.launch_blocking()',
                               'max_device_memory': 'api mindspore.runtime.set_memory()',
                               'variable_memory_max_size': 'api mindspore.runtime.set_memory()',
                               'mempool_block_size': 'api mindspore.runtime.set_memory()',
                               'memory_optimize_level': 'api mindspore.runtime.set_memory()',
                               'ascend_config': '''api mindspore.device_context.ascend.op_precision.precision_mode(),
                                                       mindspore.device_context.ascend.op_precision.op_precision_mode(),
                                                       mindspore.device_context.ascend.op_precision.matmul_allow_hf32(),
                                                       mindspore.device_context.ascend.op_precision.conv_allow_hf32(),
                                                       mindspore.device_context.ascend.op_tuning.op_compile()''',
                               'aoe_tune_mode': 'api mindspore.device_context.ascend.op_tuning.aoe_tune_mode()',
                               'aoe_config': 'api mindspore.device_context.ascend.op_tuning.aoe_job_type()',
                               'op_timeout': 'api mindspore.device_context.ascend.op_debug.execute_timeout()',
                               'op_debug_option': 'api mindspore.device_context.ascend.op_debug.debug_option()',
                               'gpu_config': '''api mindspore.device_context.gpu.op_precision.conv_allow_tf32(),
                                                     mindspore.device_context.gpu.op_precision.matmul_allow_tf32(),
                                                     mindspore.device_context.gpu.op_precision.conv_fprop_algo(),
                                                     mindspore.device_context.gpu.op_precision.conv_wgrad_algo(),
                                                     mindspore.device_context.gpu.op_precision.conv_dgrad_algo()''',
                               'runtime_num_threads': 'api mindspore.device_context.cpu.op_tuning.threads_num()'}
    invalid_context_dict = {
        'exception_dump': {'version': '2.6', 'interface': 'device_context.ascend.op_debug.aclinit_config()'}
    }
    if key in deprecated_context_dict:
        log = f"For 'context.set_context', the parameter '{key}' will be deprecated and removed in a future version."
        if deprecated_context_dict.get(key) != '':
            log += f" Please use the {deprecated_context_dict.get(key)} instead."
        logger.warning(log)
    if key in invalid_context_dict:
        info = invalid_context_dict.get(key)
        deprecated_version, new_interface = info.get('version'), info.get('interface')
        log = (
            f"For 'context.set_context', the parameter '{key}' has been removed"
            f" since MindSpore {deprecated_version} version."
        )
        if new_interface:
            log += f" Please use the {new_interface} instead."
        raise ValueError(log)

@args_type_check(mode=int, precompile_only=bool, device_target=str, device_id=int, save_graphs=(bool, int),
                 save_graphs_path=str, aoe_tune_mode=str, aoe_config=dict,
                 enable_reduce_precision=bool, variable_memory_max_size=str,
                 enable_auto_mixed_precision=bool, inter_op_parallel_num=int,
                 enable_graph_kernel=bool, reserve_class_name_in_scope=bool, check_bprop=bool,
                 max_device_memory=str, print_file_path=str, max_call_depth=int, env_config_path=str,
                 graph_kernel_flags=str, save_compile_cache=bool, runtime_num_threads=int, load_compile_cache=bool,
                 grad_for_scalar=bool, pynative_synchronize=bool, mempool_block_size=str, disable_format_transform=bool,
                 op_timeout=int, deterministic=str, ascend_config=dict, jit_syntax_level=int, debug_level=int,
                 jit_enable_inplace_ops=bool, gpu_config=dict, jit_config=dict, enable_compile_cache=bool)
def set_context(**kwargs):
    r"""
    Set context for running environment, this interface will be deprecated in future versions, and its
    parameter-related functionalities will be provided through new APIs.

    Args:
        mode (int): GRAPH_MODE(0) or PYNATIVE_MODE(1). Default ``PYNATIVE_MODE`` .
        device_id (int): ID of the target device. Default ``0`` . This parameter will be deprecated
            and removed in future versions. Please use the api :func:`mindspore.set_device` instead.
        device_target (str): The target device to run, support ``"Ascend"``, ``"GPU"``, and ``"CPU"``. This parameter
            will be deprecated and removed in future versions. Please use the api :func:`mindspore.set_device` instead.
        deterministic (str): Deterministic computation of operators. Default ``"OFF"`` .
            This parameter will be deprecated and removed in future versions. Please use the api
            :func:`mindspore.set_deterministic` instead.
        max_call_depth (int): The maximum depth of function call. Default ``1000`` .
            This parameter will be deprecated and removed in a future version. Please use the api
            :func:`mindspore.set_recursion_limit` instead.
        variable_memory_max_size (str): This parameter will be deprecated and removed in future versions.
            Please use the api :func:`mindspore.runtime.set_memory` instead.
        mempool_block_size (str): Set the size of the memory pool block for devices. Default ``"1GB"`` .
            This parameter will be deprecated and removed in future versions. Please use
            the api :func:`mindspore.runtime.set_memory` instead.
        memory_optimize_level (str): The memory optimize level. Default ``"O0"``.
            This parameter will be deprecated and removed in future versions. Please use
            the api :func:`mindspore.runtime.set_memory` instead.
        max_device_memory (str): Set the maximum memory available for devices.
            Default ``"1024GB"`` . This parameter will be deprecated and removed in future versions. Please use
            the api :func:`mindspore.runtime.set_memory` instead.
        pynative_synchronize (bool): Whether to enable synchronous execution of the device in PyNative mode.
            Default ``False`` . This parameter will be deprecated and removed in future versions.Please use
            the api :func:`mindspore.runtime.launch_blocking` instead.
        compile_cache_path (str): Path to save the compile cache. Default ``"."``.
            This parameter will be deprecated and removed in a future version. Please use the environment variable
            `MS_COMPILER_CACHE_PATH` instead.
        inter_op_parallel_num(int): The thread number of op parallel at the same time.
            Default ``0`` . This parameter will be deprecated and removed in future versions.
            Please use the api :func:`mindspore.runtime.dispatch_threads_num` instead.
        disable_format_transform (bool): Whether to disable the automatic format transform function from NCHW
            to NHWC. Default ``False`` . This parameter will be deprecated and removed in future versions. Please
            use the related parameter of :func:`mindspore.jit` instead.
        jit_syntax_level (int): Set JIT syntax support level. Default ``LAX`` . This parameter is deprecated
            and removed in future versions. Please use the `fullgraph` parameter of :func:`mindspore.jit` instead.
            Setting the `fullgraph` parameter to True is equivalent to setting the `jit_syntax_level` parameter to
            ``STRICT``, and setting the `fullgraph` parameter to False is equivalent to setting the `jit_syntax_level`
            parameter to ``LAX``.
        jit_config (dict): Set the global jit config for compile. This parameter is deprecated
            and removed in future versions. Please use the related parameter of :func:`mindspore.jit` instead.
        exec_order (str): The sorting method for operator execution. This parameter is deprecated
            and removed in future versions. Please use the related parameter of :func:`mindspore.jit` instead.
        op_timeout (int): Set the maximum duration of executing an operator in seconds. Default ``900`` .
            This parameter will be deprecated and removed in future versions. Please use the
            api :func:`mindspore.device_context.ascend.op_debug.execute_timeout` instead.
        aoe_tune_mode (str): AOE tuning mode.
            This parameter will be deprecated and removed in future versions. Please use the
            api :func:`mindspore.device_context.ascend.op_tuning.aoe_tune_mode` instead.
        aoe_config (dict): AOE-specific parameters. This parameter will be deprecated and removed in future
            versions. Please use the api :func:`mindspore.device_context.ascend.op_tuning.aoe_job_type` instead.
        runtime_num_threads(int): The thread pool number of cpu kernel used in runtime. Default ``30`` .
            This parameter will be deprecated and removed in future versions. Please use the
            api :func:`mindspore.device_context.cpu.op_tuning.threads_num` instead.
        save_graphs (bool or int): Whether to save intermediate compilation graphs. Default ``0`` .
            This parameter will be deprecated and removed in a future version. Please use the environment variable
            `MS_DEV_SAVE_GRAPHS` instead.
        save_graphs_path (str): Path to save graphs. Default ``"."``.
            This parameter will be deprecated and removed in a future version. Please use the environment variable
            `MS_DEV_SAVE_GRAPHS_PATH` instead.
        precompile_only (bool): Whether to only precompile the network. Default ``False`` .
            This parameter will be deprecated and removed in a future version. Please use the environment variable
            `MS_DEV_PRECOMPILE_ONLY` instead.
        enable_compile_cache (bool): Whether to save or load the compiled cache of the graph.
            Default ``False`` . This is an experimental prototype that is subject to change and/or deletion.
            This parameter will be deprecated and removed in a future version. Please use the environment variable
            `MS_COMPILER_CACHE_ENABLE` instead.
        ascend_config (dict): Set the parameters specific to Ascend hardware platform.

            - precision_mode (str): Mixed precision mode setting. Default ``"force_fp16"`` .
              This parameter will be deprecated and removed in future versions. Please use the
              api :func:`mindspore.device_context.ascend.op_precision.precision_mode` instead.
            - jit_compile (bool): Whether to select online compilation. This parameter will be deprecated and removed
              in future versions. Please use the api :func:`mindspore.device_context.ascend.op_tuning.op_compile`
              instead.
            - matmul_allow_hf32 (bool): Whether to convert FP32 to HF32 for Matmul operators. Default ``False``.
              This parameter will be deprecated and removed in future versions. Please use the
              api :func:`mindspore.device_context.ascend.op_precision.matmul_allow_hf32` instead.
            - conv_allow_hf32 (bool): Whether to convert FP32 to HF32 for Conv operators. Default ``True``.
              This parameter will be deprecated and removed in future versions. Please use the
              api :func:`mindspore.device_context.ascend.op_precision.conv_allow_hf32` instead.
            - op_precision_mode (str): Path to config file of op precision mode.
              This parameter will be deprecated and removed in future versions. Please use the
              api :func:`mindspore.device_context.ascend.op_precision.op_precision_mode` instead.
            - op_debug_option (str): Enable debugging options for Ascend operators.
              This parameter will be deprecated and removed in future versions. Please use the
              api :func:`mindspore.device_context.ascend.op_debug.debug_option` instead.
            - ge_options (dict): Set options for CANN. This parameter will be deprecated and removed in future versions.
              Please use the related parameter of :func:`mindspore.jit` instead.
            - atomic_clean_policy (int): The policy for cleaning memory occupied by atomic operators in the network.
              Default ``1`` represents that memory is not cleaned centrally, ``0`` represents that memory is cleaned
              centrally. This parameter will be deprecated and removed in future versions. Please
              use the related parameter of :func:`mindspore.jit` instead.
            - exception_dump (str): Enable Ascend operator exception dump. Default ``"2"`` . This parameter has been
              deprecated and removed. Please use the api
              :func:`mindspore.device_context.ascend.op_debug.aclinit_config` instead.
            - host_scheduling_max_threshold(int): The max threshold to control whether the dynamic shape process is
              used when run the static graph. Default ``0`` . This parameter will be deprecated and removed in future
              versions. Please use the related parameter of :func:`mindspore.jit` instead.
            - parallel_speed_up_json_path(Union[str, None]): The path to the parallel speed up json file.
              This parameter will be deprecated and removed in future versions. Please use the
              api :func:`mindspore.parallel.auto_parallel.AutoParallel.transformer_opt` instead.
            - hccl_watchdog (bool): Enable a thread to monitor the failure of collective communication.
              Default ``True`` . This parameter will be deprecated and removed in future versions. Please use the
              environment variable `MS_ENABLE_THM="{HCCL_WATCHDOG:1}"` instead.

        gpu_config (dict): Set the parameters specific to gpu hardware platform. It is not set by default.

            - conv_fprop_algo (str): Specifies convolution forward algorithm. Default ``"normal"`` .
              This parameter will be deprecated and removed in future versions. Please use the
              api :func:`mindspore.device_context.gpu.op_tuning.conv_fprop_algo` instead.
            - conv_dgrad_algo (str): Specifies convolution data grad algorithm. Default ``"normal"`` .
              This parameter will be deprecated and removed in future versions. Please use the
              api :func:`mindspore.device_context.gpu.op_tuning.conv_dgrad_algo` instead.
            - conv_wgrad_algo (str): Specifies convolution filter grad algorithm. Default ``"normal"`` .
              This parameter will be deprecated and removed in future versions. Please use the
              api :func:`mindspore.device_context.gpu.op_tuning.conv_wgrad_algo` instead.
            - conv_allow_tf32 (bool): Controls to allow Tensor core TF32 computation on CUDNN.
              Default ``True``.
              This parameter will be deprecated and removed in future versions. Please use the
              api :func:`mindspore.device_context.gpu.op_precision.conv_allow_tf32` instead.
            - matmul_allow_tf32 (bool): Controls to allow Tensor core TF32 computation on CUBLAS.
              Default ``False``.
              This parameter will be deprecated and removed in future versions. Please use the
              api :func:`mindspore.device_context.gpu.op_precision.matmul_allow_tf32` instead.
        print_file_path (str): This parameter will be deprecated and removed in future versions.
        env_config_path (str): This parameter will be deprecated and removed in future versions.
        debug_level (int): This parameter will be deprecated and removed in future versions.
        reserve_class_name_in_scope (bool): This parameter will be deprecated and removed in future versions.
        check_bprop (bool): This parameter will be deprecated and removed in future versions.
        enable_reduce_precision (bool): This parameter will be deprecated and removed in a future versions.
        grad_for_scalar (bool): This parameter will be deprecated and removed in future versions.
        support_binary (bool): Whether to support run .pyc or .so in graph mode. This parameter will be deprecated and
            removed in a future version. Please use the environment variable `MS_SUPPORT_BINARY` instead.

    Examples:
        >>> import mindspore as ms
        >>> ms.set_context(mode=ms.PYNATIVE_MODE)
        >>> ms.set_context(precompile_only=True)
        >>> ms.set_context(device_target="Ascend")
        >>> ms.set_context(device_id=0)
        >>> ms.set_context(save_graphs=True, save_graphs_path="./model.ms")
        >>> ms.set_context(enable_reduce_precision=True)
        >>> ms.set_context(reserve_class_name_in_scope=True)
        >>> ms.set_context(variable_memory_max_size="6GB")
        >>> ms.set_context(aoe_tune_mode="online")
        >>> ms.set_context(aoe_config={"job_type": "2"})
        >>> ms.set_context(check_bprop=True)
        >>> ms.set_context(max_device_memory="3.5GB")
        >>> ms.set_context(mempool_block_size="1GB")
        >>> ms.set_context(print_file_path="print.pb")
        >>> ms.set_context(max_call_depth=80)
        >>> ms.set_context(env_config_path="./env_config.json")
        >>> ms.set_context(grad_for_scalar=True)
        >>> ms.set_context(enable_compile_cache=True, compile_cache_path="./cache.ms")
        >>> ms.set_context(pynative_synchronize=True)
        >>> ms.set_context(runtime_num_threads=10)
        >>> ms.set_context(inter_op_parallel_num=4)
        >>> ms.set_context(disable_format_transform=True)
        >>> ms.set_context(memory_optimize_level='O0')
        >>> ms.set_context(deterministic='ON')
        >>> ms.set_context(ascend_config={"precision_mode": "force_fp16", "jit_compile": True,
        ...                "atomic_clean_policy": 1, "op_precision_mode": "./op_precision_config_file",
        ...                "op_debug_option": "oom",
        ...                "ge_options": {"global": {"ge.opSelectImplmode": "high_precision"},
        ...                               "session": {"ge.exec.atomicCleanPolicy": "0"}}})
        >>> ms.set_context(jit_syntax_level=ms.STRICT)
        >>> ms.set_context(debug_level=ms.context.DEBUG)
        >>> ms.set_context(gpu_config={"conv_fprop_algo": "performance", "conv_allow_tf32": True,
        ...                "matmul_allow_tf32": True})
        >>> ms.set_context(jit_config={"jit_level": "O0"})
        >>> ms.set_context(exec_order="bfs")
    """
    ctx = _context()
    # set device target first
    if 'device_target' in kwargs:
        ctx.set_device_target(kwargs['device_target'])
    device = ctx.get_param(ms_ctx_param.device_target)
    _check_ascend_device_context_initialized(device, kwargs)

    for key, value in kwargs.items():
        _check_context_deprecated(key)
        if key in ('enable_sparse', 'auto_tune_mode'):
            logger.warning(f"For 'context.set_context', '{key}' parameter is deprecated, "
                           "and will be removed in the next version.")
            continue
        if key in ('enable_auto_mixed_precision',):
            logger.warning(f"For 'context.set_context', '{key}' parameter is deprecated. "
                           "For details, please see the interface parameter API comments")
            continue
        if key == "print_file_path":
            logger.warning(f"For 'context.set_context', '{key}' parameter is deprecated due to changes in the behavior"
                           f" of the print operator. Recommend not using this parameter and"
                           f" directly viewing the screen output.")
        if key in ('reserve_class_name_in_scope', 'env_config_path'):
            logger.warning(f"For 'context.set_context', '{key}' parameter is deprecated, "
                           "and will be removed in the next version.")
        _check_key(key)
        if key == 'save_graphs':
            if value is True:
                value = 2
            if value is False:
                value = 0
            if value > 3:
                raise ValueError(f"value for save_graphs should be 0-3 but got '{value}'")
        if key == 'jit_syntax_level' and value not in (STRICT, COMPATIBLE, LAX):
            raise ValueError(f"For 'jit_syntax_level', the value should be context.STRICT"
                             f" or context.LAX, but got {value}.")
        if key == 'debug_level' and value not in (RELEASE, DEBUG):
            raise ValueError(f"For 'debug_level', the value should be context.DEBUG"
                             f" or context.RELEASE, but got {value}.")
        if key == 'enable_compile_cache':
            setattr(ctx, key, value)
            ctx.set_param(ms_ctx_param.__members__[key], int(value))
            continue
        if key == 'enable_graph_kernel':
            logger.warning(f"For 'context.set_context', '{key}' parameter is deprecated, "
                           "and will be removed in the next version. "
                           "Please use jit_config={'jit_level': 'O1'} instead.")
        if key == 'graph_kernel_flags':
            logger.warning(f"For 'context.set_context', '{key}' parameter is deprecated, "
                           "and will be removed in the next version. "
                           "Please use environ variable 'MS_DEV_GRAPH_KERNEL_FLAGS' instead.")
        if not _check_target_specific_cfgs(device, key):
            continue
        if key in ctx.setters:
            ctx.setters[key](ctx, value)
            continue
        if hasattr(ctx, key):
            setattr(ctx, key, value)
            continue
        # enum variables beginning with '_' are for internal use
        if key in ms_ctx_param.__members__ and key[0] != '_':
            ctx.set_param(ms_ctx_param.__members__[key], value)
            continue
        raise ValueError(f"For 'context.set_context', the keyword argument {key} is not recognized! For detailed "
                         f"usage of 'set_context', please refer to the Mindspore official website.")


def get_context(attr_key):

    """
    Get context attribute value according to the input key, this api will be deprecated and removed in future versions,
    please use :func:`mindspore.get_current_device` instead.

    If some attributes are not set, they will be automatically obtained.

    Args:
        attr_key (str): The key of the attribute.

    Returns:
        Object, The value of given attribute key.

    Raises:
        ValueError: If input key is not an attribute in context.
    Examples:
        >>> import mindspore as ms
        >>> ms.get_context("device_target")
        >>> ms.get_context("device_id")
    """
    ctx = _context()
    device = ctx.get_param(ms_ctx_param.device_target)
    _ = _check_target_specific_cfgs(device, attr_key)
    if hasattr(ctx, attr_key):
        return getattr(ctx, attr_key)
    # enum variables beginning with '_' are for internal use
    if attr_key in ms_ctx_param.__members__ and attr_key[0] != '_':
        return ctx.get_param(ms_ctx_param.__members__[attr_key])
    raise ValueError(f"For 'context.get_context', the argument {attr_key} is not recognized! For detailed "
                     f"usage of 'get_context', please refer to the Mindspore official website.")


def _get_mode():
    """
    Get execution mode. Only for internal using.

    Returns:
        Object: The Value of execution mode.
    """
    ctx = _context()
    return ctx.get_mode()


def get_jit_config():
    """
    Get global jit config.

    Returns:
        Object: The Value of jit config.
    """
    ctx = _context()
    return ctx.get_jit_config()


class ParallelMode:
    """
    Parallel mode options.

    There are five kinds of parallel modes, ``STAND_ALONE``, ``DATA_PARALLEL``,
    ``HYBRID_PARALLEL``, ``SEMI_AUTO_PARALLEL`` and ``AUTO_PARALLEL``. Default: ``STAND_ALONE``.

    - ``STAND_ALONE``: Only one processor is working.
    - ``DATA_PARALLEL``: Distributes the data across different processors.
    - ``HYBRID_PARALLEL``: Achieves data parallelism and model parallelism manually.
    - ``SEMI_AUTO_PARALLEL``: Achieves data parallelism and model parallelism by setting parallel strategies.
    - ``AUTO_PARALLEL``: Achieves parallelism automatically.

    ``MODE_LIST``: The list of all supported parallel modes.
    """

    STAND_ALONE = "stand_alone"
    DATA_PARALLEL = "data_parallel"
    HYBRID_PARALLEL = "hybrid_parallel"
    SEMI_AUTO_PARALLEL = "semi_auto_parallel"
    AUTO_PARALLEL = "auto_parallel"
    MODE_LIST = [STAND_ALONE, DATA_PARALLEL, HYBRID_PARALLEL, SEMI_AUTO_PARALLEL, AUTO_PARALLEL]


@args_type_check(enable_ps=bool)
def set_ps_context(**kwargs):
    """
    Set parameter server training mode context, this api will be deprecated and removed in future versions.

    Note:
        Parameter server mode is only supported in graph mode.
        Some other environment variables should also be set for parameter server training mode.
        These environment variables are listed below:

        - MS_SERVER_NUM: Server number
        - MS_WORKER_NUM: Worker number
        - MS_SCHED_HOST: Scheduler IP address
        - MS_SCHED_PORT: Scheduler port
        - MS_ROLE: The role of this process:

          - MS_SCHED: represents the scheduler,
          - MS_WORKER: represents the worker,
          - MS_PSERVER/MS_SERVER: represents the Server

    Args:
        enable_ps (bool): Whether to enable parameter server training mode.
                          Only after `enable_ps` is set ``True``, the environment variables will be effective.
                          Default: ``False`` .
        config_file_path (str): Configuration file path used by recovery,
                                   parameter server training mode only
                                   supports Server disaster recovery currently. Default: ``''`` .
        enable_ssl (bool): Set PS SSL mode enabled or disabled. Default: ``False``.
                           When set to False, users need to review and confirm the security of network environment
                           where the distributed job is located.
        client_password (str): Password to decrypt the secret key stored in the client certificate. Default: ``''`` .
        server_password (str): Password to decrypt the secret key stored in the server certificate. Default: ``''`` .

    Raises:
        ValueError: If input key is not the attribute in parameter server training mode context.

    Examples:
        >>> import mindspore as ms
        >>> ms.set_ps_context(enable_ps=True, enable_ssl=True, client_password='', server_password='')
    """
    _set_ps_context(**kwargs)


def get_ps_context(attr_key):
    """
    Get parameter server training mode context attribute value according to the key, this api will be deprecated and
    removed in future versions.

    Args:
        attr_key (str): The key of the attribute:

            - enable_ps (bool, optional): Whether to enable parameter server training mode. Default: ``False`` .
            - config_file_path (str, optional): Configuration file path used by recovery,
              parameter server training mode only
              supports Server disaster recovery currently. Default: ``''`` .
            - enable_ssl (bool, optional): Set PS SSL mode enabled or disabled. Default: ``False`` .
              When set to False, users need to review and confirm the security of network environment
              where the distributed job is located.

    Returns:
        Returns attribute value according to the key.

    Raises:
        ValueError: If input key is not attribute in auto parallel context.

    Examples:
        >>> import mindspore as ms
        >>> ms.get_ps_context("enable_ps")
    """
    return _get_ps_context(attr_key)


def reset_ps_context():
    """
    Reset parameter server training mode context attributes to the default values, this api will be deprecated and
    removed in future versions.

    Meaning of each field and its default value refer to :func:`mindspore.set_ps_context`.

    Examples:
        >>> import mindspore as ms
        >>> ms.reset_ps_context()
    """
    _reset_ps_context()


_hccl_connect_timeout = '600'


def _init_parallel_env():
    """Set hccl connect timeout."""
    if 'HCCL_CONNECT_TIMEOUT' not in os.environ:
        os.environ['HCCL_CONNECT_TIMEOUT'] = _hccl_connect_timeout


_init_parallel_env()
