# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
from mindspore._c_expression import MSContext, ms_ctx_param
from mindspore import _checkparam as Validator
from mindspore._checkparam import args_type_check
from mindspore.parallel._auto_parallel_context import _set_auto_parallel_context, _get_auto_parallel_context, \
    _reset_auto_parallel_context
from mindspore.parallel._ps_context import _set_ps_context, _get_ps_context, _reset_ps_context, \
    _need_reset_device_target_for_ps
from mindspore.parallel._offload_context import _set_offload_context, _get_offload_context

__all__ = ['GRAPH_MODE', 'PYNATIVE_MODE', 'STRICT', 'COMPATIBLE', 'LAX', 'set_context', 'get_context',
           'set_auto_parallel_context', 'get_auto_parallel_context', 'reset_auto_parallel_context', 'ParallelMode',
           'set_ps_context', 'get_ps_context', 'reset_ps_context', 'set_offload_context', 'get_offload_context']

GRAPH_MODE = 0
PYNATIVE_MODE = 1
_DEVICE_APP_MEMORY_SIZE = 31  # The max memory size of graph plus variable.
_RE_PATTERN = r'[1-9][0-9]*(\.)?[0-9]*GB|0\.[0-9]*GB'
K_CONTEXT = None

# Enumerate for the property 'jit_syntax_level'.
STRICT = 0
COMPATIBLE = 1
LAX = 2


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
            os.makedirs(path)
        except FileExistsError:
            logger.debug("The directory(%s) already exist.", path)
        except PermissionError as e:
            logger.critical(f"No write permission on the directory '{path}'', error = {e}")
            raise ValueError(e.__str__() + f"\nNo write permission on the directory '{path}'.")
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
        super(_ThreadLocalInfo, self).__init__()
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
        super(_ContextSwitchInfo, self).__init__()
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
            cls._instance_lock.acquire()
            cls._instance = object.__new__(cls)
            cls._instance_lock.release()
        return cls._instance

    def __init__(self):
        self._thread_local_info = _ThreadLocalInfo()
        self._context_switches = _ContextSwitchInfo(False)
        self._context_handle = MSContext.get_instance()
        self._support_binary = False

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
        return self.get_param(ms_ctx_param.mode)

    def set_mode(self, mode):
        """
        Switch between Graph mode and PyNative mode.

        Args:
            mode (int): GRAPH_MODE or PYNATIVE_MODE.
        """
        if mode == PYNATIVE_MODE:
            if self.enable_debug_runtime:
                self.set_backend_policy("vm")
            parallel_mode = _get_auto_parallel_context("parallel_mode")
            if parallel_mode not in (ParallelMode.DATA_PARALLEL, ParallelMode.STAND_ALONE, ParallelMode.AUTO_PARALLEL):
                raise ValueError(f"Got {parallel_mode}, when the user enabled SEMI_AUTO_PARALELL, "
                                 f"pynative mode dose not support, you should set either "
                                 f"context.set_auto_parallel_context(parallel_mode='data_parallel'), "
                                 f"context.set_auto_parallel_context(parallel_mode='stand_alone') "
                                 f"or context.set_auto_parallel_context(parallel_mode='auto_parallel').")
            self._context_switches.push(True, None)
        elif mode == GRAPH_MODE:
            if self.enable_debug_runtime:
                self.set_backend_policy("ge")
            self._context_switches.push(False, None)
        else:
            raise ValueError(f"For 'context.set_context', the argument 'mode' should be context.GRAPH_MODE (0) "
                             f"or context.PYNATIVE_MODE (1), but got {mode}.")
        self.set_param(ms_ctx_param.mode, mode)

    def set_jit_syntax_level(self, level):
        """"Set the JIT syntax level for graph compiling"""
        if level != STRICT and level != COMPATIBLE and level != LAX:
            raise ValueError(f"For 'context.set_jit_syntax_level', the argument 'level' should be context.STRICT "
                             f"or context.LAX, but got {level}.")
        self.set_param(ms_ctx_param.jit_syntax_level, level)

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

    def set_memory_offload(self, memory_offload):
        """
        Enable memory offload or not, support "ON", "OFF".

        Args:
            memory_offload (str): "ON", "OFF"
        """
        memory_offload_options = ["ON", "OFF"]
        if memory_offload not in memory_offload_options:
            raise ValueError(f"For 'context.set_context', the argument 'memory_offload' must be one of "
                             f"{memory_offload_options}, but got {memory_offload}.")
        if memory_offload == "ON":
            self.set_param(ms_ctx_param.memory_offload, True)
        else:
            self.set_param(ms_ctx_param.memory_offload, False)

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
        self.set_param(ms_ctx_param.deterministic, deterministic)

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
                - op_precision_mode (str): config file path.
                - parallel_speed_up_json_path(Union[str, None]): The path to the parallel speed up json file.
                  If its value is None or '', it does not take effect. Default None.
        """
        ascend_cfg_modes = {
            'precision_mode': ["force_fp16", "allow_fp32_to_fp16", "allow_mix_precision", "must_keep_origin_dtype",
                               "force_fp32", "allow_fp32_to_bf16", "allow_mix_precision_fp16",
                               "allow_mix_precision_bf16"],
            'jit_compile': [True, False],
            'atomic_clean_policy': [0, 1],
            'matmul_allow_hf32': [True, False],
            'conv_allow_hf32': [True, False],
            'op_precision_mode': (str,),
            'parallel_speed_up_json_path': (str, None)
        }
        ascend_cfg_setters = {
            'precision_mode': self._get_ascend_config_setter('precision_mode'),
            'jit_compile': self._get_ascend_config_setter('jit_compile', lambda v: "1" if v else "0"),
            'atomic_clean_policy': self._get_ascend_config_setter('atomic_clean_policy', str),
            'matmul_allow_hf32': self._get_ascend_config_setter('matmul_allow_hf32', lambda v: "1" if v else "0"),
            'conv_allow_hf32': self._get_ascend_config_setter('conv_allow_hf32', lambda v: "1" if v else "0"),
            'op_precision_mode': self._set_op_precision_mode,
            'parallel_speed_up_json_path': self._set_speedup_config_path
        }
        ascend_cfg_set = tuple(ascend_cfg_modes.keys())
        for ascend_key, ascend_value in ascend_config.items():
            if ascend_key not in ascend_cfg_set:
                raise ValueError(f"For 'context.set_context', the key of argument 'ascend_config' must be one of "
                                 f"{ascend_cfg_set}, but got {ascend_key}.")
            supported_modes = ascend_cfg_modes.get(ascend_key)
            if isinstance(supported_modes, list) and ascend_value not in supported_modes:
                raise ValueError(f"For 'ascend_config', the value of argument {ascend_key} must be one of "
                                 f"{supported_modes}, but got {ascend_value}.")
            if isinstance(supported_modes, tuple) and not isinstance(ascend_value, supported_modes):
                raise TypeError(f"For 'ascend_config', the type of argument {ascend_key} must be one of "
                                f"{supported_modes}, but got {type(ascend_value)}.")
            cfg_setter = ascend_cfg_setters.get(ascend_key)
            cfg_setter(ascend_value)

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
        if _get_mode() == GRAPH_MODE:
            logger.warning("Graph mode doesn't support to set parameter 'mempool_block_size' of context currently, "
                           "you can use context.set_context to set pynative mode.")
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
            with open(env_config_path, 'r') as f:
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
        'op_timeout': set_op_timeout,
        'memory_offload': set_memory_offload,
        'deterministic': set_deterministic,
        'ascend_config': set_ascend_config,
        'jit_syntax_level': set_jit_syntax_level,
        'gpu_config': set_gpu_config,
        'aoe_config': set_aoe_config,
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
            trans_fn = lambda x: x
        return _config_setter

    def _set_op_precision_mode(self, ascend_value):
        op_precision_path = ascend_value
        real_path = os.path.realpath(op_precision_path)
        if not os.path.exists(real_path):
            raise ValueError(f"For 'ascend_config', the 'op_precision_mode' is invalid path, "
                             f"got '{op_precision_path}'.")
        self.set_param(ms_ctx_param.op_precision_mode, ascend_value)

    def _set_speedup_config_path(self, speedup_config_path):
        """"Check and set speedup config for auto parallel."""
        if speedup_config_path is None or speedup_config_path == "":
            return
        speedup_config_real_path = os.path.abspath(speedup_config_path)
        if not os.path.exists(speedup_config_real_path):
            raise ValueError(f"For 'ascend_config', the path to parallel_speed_up_json: "
                             f"{speedup_config_real_path} does not exist, please check whether the "
                             f"'parallel_speed_up_json_path' is correct.")
        try:
            valid_option = {"recompute_comm_overlap": ms_ctx_param.recompute_comm_overlap,
                            "matmul_grad_comm_overlap": ms_ctx_param.matmul_grad_comm_overlap,
                            "enable_task_opt": ms_ctx_param.enable_task_opt,
                            "enable_grad_comm_opt": ms_ctx_param.enable_grad_comm_opt,
                            "interleaved_matmul_comm": ms_ctx_param.interleaved_matmul_comm,
                            "interleaved_layernorm_comm": ms_ctx_param.interleaved_layernorm_comm}
            with open(speedup_config_real_path, 'r') as f:
                speedup_config = json.load(f)
                for k, v in speedup_config.items():
                    if not isinstance(k, str):
                        raise TypeError("key {} is not a str".format(k))
                    if k not in valid_option:
                        raise ValueError("key {} should be one of {}.".format(k, valid_option.keys()))
                    if not isinstance(v, bool):
                        raise TypeError("value {} is not a bool".format(v))
                    self.set_param(valid_option.get(k), v)
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
                 parallel_optimizer_config=dict,
                 comm_fusion=dict, strategy_ckpt_config=dict)
def set_auto_parallel_context(**kwargs):
    r"""
    Set auto parallel context, only data parallel supported on CPU.

    Note:
        Attribute name is required for setting attributes.
        If a program has tasks on different parallel modes, before setting a new parallel mode for the
        next task, interface :func:`mindspore.reset_auto_parallel_context` should be called to reset
        the configuration.
        Setting or changing parallel modes must be called before creating any Initializer, otherwise,
        it may have RuntimeError when compiling the network.

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
               \                 auto_parallel_search_mode
               \                 comm_fusion
               \                 strategy_ckpt_config
    ===========================  ===========================

    Args:
        device_num (int): Available device number, the value must be in [1, 4096]. Default: ``1`` .
        global_rank (int): Global rank id, the value must be in [0, 4095]. Default: ``0`` .
        gradients_mean (bool): Whether to perform mean operator after allreduce of gradients.
                     "stand_alone" do not support gradients_mean. Default: ``False`` .
        gradient_fp32_sync (bool): Run allreduce of gradients in fp32. "stand_alone", "data_parallel"
                     and "hybrid_parallel" do not support gradient_fp32_sync. Default: ``True`` .
        parallel_mode (str): There are five kinds of parallel modes, ``"stand_alone"`` , ``"data_parallel"`` ,
                     ``"hybrid_parallel"`` , ``"semi_auto_parallel"`` and ``"auto_parallel"`` . Note the pynative mode
                     only supports the ``"stand_alone"`` and ``"data_parallel"`` mode. Default: ``"stand_alone"`` .

                     - stand_alone: Only one processor is working.

                     - data_parallel: Distributes the data across different processors.

                     - hybrid_parallel: Achieves data parallelism and model parallelism manually.

                     - semi_auto_parallel: Achieves data and model parallelism by setting parallel strategies.

                     - auto_parallel: Achieving parallelism automatically.
        search_mode (str): There are three kinds of shard strategy search modes: ``"recursive_programming"`` ,
                     ``"dynamic_programming"`` and ``"sharding_propagation"`` . Default: ``"recursive_programming"`` .

                     - recursive_programming: Recursive programming search mode. In order to obtain optimal performance,
                       it is recommended that users set the batch size to be greater than or equal to the product of
                       the number of devices and the number of multi-copy parallelism.

                     - dynamic_programming: Dynamic programming search mode.

                     - sharding_propagation: Propagate shardings from configured ops to non-configured ops.
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
                       set_auto_parallel_context(dataset_strategy=ds_stra).
        enable_parallel_optimizer (bool): This is a developing feature, which shards the weight update computation for
                       data parallel training in the benefit of time and memory saving. Currently, auto and semi auto
                       parallel mode support all optimizers in both Ascend and GPU. Data parallel mode only supports
                       `Lamb` and `AdamWeightDecay` in Ascend . Default: ``False`` .
        enable_alltoall (bool): A switch that allows AllToAll operators to be generated during communication. If its
                        value is ``False`` , there will be a combination of operators such as AllGather, Split and
                        Concat instead of AllToAll. Default: ``False`` .
        all_reduce_fusion_config (list): Set allreduce fusion strategy by parameters indices. Only support ReduceOp.SUM
                       and HCCL_WORLD_GROUP/NCCL_WORLD_GROUP. No Default, if it is not set, the fusion is closed.
        pipeline_stages (int): Set the stage information for pipeline parallel. This indicates how the devices are
                        distributed alone in the pipeline. The total devices will be divided into 'pipeline_stags'
                        stages.
                        Default: ``1`` .
        parallel_optimizer_config (dict): A dict contains the keys and values for setting the parallel optimizer
                        configure. The configure provides more detailed behavior control about parallel training
                        when parallel optimizer is enabled. The configure will be effective when we use
                        mindspore.set_auto_parallel_context(enable_parallel_optimizer=True).
                        It supports the following keys.

                        - gradient_accumulation_shard(bool): If ``true`` , the accumulation gradient parameters will be
                          sharded across the data parallel devices. This will
                          introduce additional communication(ReduceScatter) at
                          each step when accumulate the gradients, but saves a
                          lot of device memories, thus can make model be trained
                          with larger batch size. This configure is effective only
                          when the model runs on pipeline training or gradient
                          accumulation with data parallel. Default ``False`` .

                        - parallel_optimizer_threshold(int): Set the threshold of parallel optimizer. When parallel
                          optimizer is enabled, parameters with size smaller than this threshold will not be sharded
                          across the devices. Parameter size = shape[0] \* ... \* shape[n] \* size(dtype). Non-negative.
                          Unit: KB. Default: ``64`` .

                        - optimizer_weight_shard_size(int): Set the optimizer weight shard group size, if you want to
                          specific the maximum group size across devices when the parallel optimizer is enabled.
                          The numerical range can be (0, device_num]. If pipeline parallel is enabled, the numerical
                          range is (0, device_num/stage]. If the size of data parallel communication domain
                          of the parameter cannot be divided by `optimizer_weight_shard_size`, then the specified
                          communication group size will not take effect. Default value is ``-1`` , which means the
                          optimizer weight shard group size will be the size of data parallel group of each parameter.

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
                          Default: ''

                        - save_file (str): The path to save parallel strategy checkpoint. If the file name extension is
                          `.json`, the file is saved in JSON format. Otherwise, the file is saved in ProtoBuf format.
                          Default: ''

                        - only_trainable_params (bool): Only save/load the strategy information for trainable parameter.
                          Default: ``True`` .

    Raises:
        ValueError: If input key is not attribute in auto parallel context.

    Examples:
        >>> import mindspore as ms
        >>> ms.set_auto_parallel_context(device_num=8)
        >>> ms.set_auto_parallel_context(global_rank=0)
        >>> ms.set_auto_parallel_context(gradients_mean=True)
        >>> ms.set_auto_parallel_context(gradient_fp32_sync=False)
        >>> ms.set_auto_parallel_context(parallel_mode="auto_parallel")
        >>> ms.set_auto_parallel_context(search_mode="dynamic_programming")
        >>> ms.set_auto_parallel_context(auto_parallel_search_mode="dynamic_programming")
        >>> ms.set_auto_parallel_context(parameter_broadcast=False)
        >>> ms.set_auto_parallel_context(strategy_ckpt_load_file="./strategy_stage1.ckpt")
        >>> ms.set_auto_parallel_context(strategy_ckpt_save_file="./strategy_stage1.ckpt")
        >>> ms.set_auto_parallel_context(dataset_strategy=((1, 8), (1, 8)))
        >>> ms.set_auto_parallel_context(enable_parallel_optimizer=False)
        >>> ms.set_auto_parallel_context(enable_alltoall=False)
        >>> ms.set_auto_parallel_context(all_reduce_fusion_config=[8, 160])
        >>> ms.set_auto_parallel_context(pipeline_stages=2)
        >>> parallel_config = {"gradient_accumulation_shard": True, "parallel_optimizer_threshold": 24,
        ...                    "optimizer_weight_shard_size": 2}
        >>> ms.set_auto_parallel_context(parallel_optimizer_config=parallel_config, enable_parallel_optimizer=True)
        >>> config = {"allreduce": {"mode": "size", "config": 32}, "allgather": {"mode": "size", "config": 32}}
        >>> ms.set_auto_parallel_context(comm_fusion=config)
        >>> stra_ckpt_dict = {"load_file": "./stra0.ckpt", "save_file": "./stra1.ckpt", "only_trainable_params": False}
        >>> ms.set_auto_parallel_context(strategy_ckpt_config=stra_ckpt_dict)
    """
    _set_auto_parallel_context(**kwargs)


def get_auto_parallel_context(attr_key):
    """
    Get auto parallel context attribute value according to the key.

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
    Reset auto parallel context attributes to the default values.

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
    - enable_alltoall: False.
    - pipeline_stages: 1.
    - fusion_threshold: 64.

    Examples:
        >>> import mindspore as ms
        >>> ms.reset_auto_parallel_context()
    """
    _reset_auto_parallel_context()


@args_type_check(offload_config=dict)
def set_offload_context(offload_config):
    r"""
    Configure heterogeneous training detailed parameters to adjust the offload strategy.

    Note:
        The offload configuration is only used if the memory offload feature is enabled
        via mindspore.set_context(memory_offload="ON").

    Args:
        offload_config (dict): A dict contains the keys and values for setting the offload context
            configure.It supports the following keys.

            - offload_path (str):  The path of offload, relative paths are supported. Default: ``"./offload"``.
            - offload_cpu_size (str):  The cpu memory size for offload. The format is "xxGB".
            - offload_disk_size (str): The disk size for offload. The format is "xxGB"
            - hbm_ratio (float): The ratio that can be used based on the maximum device memory.
              The range is (0,1], Default: ``1.0``.
            - cpu_ratio (float): The ratio that can be used based on the maximum host memory.
              The range is (0,1], Default: ``1.0``.
            - enable_pinned_mem (bool): The flag of whether enabling Pinned Memory. Default: ``True``.
            - enable_aio (bool): The flag of whether enabling aio. Default: ``True``.
            - aio_block_size (str): The size of aio block. The format is "xxGB".
            - aio_queue_depth (int): The depth of aio queue.
            - offload_param (str):  The param for offload destination, cpu or disk, Default: ``""``.
            - offload_checkpoint (str):  The checkpoint for offload destination, only valid if recompute is turned on,
              cpu or disk, Default: ``""``.
            - auto_offload (bool): The flag of whether auto offload. Default: ``True``.
            - host_mem_block_size (str): The memory block size of host memory pool. The format is "xxGB"

    Raises:
        ValueError: If input key is not attribute in auto parallel context.

    Examples:
        >>> from mindspore import context
        >>> context.set_offload_context(offload_config={"offload_param":"cpu"})
    """
    _set_offload_context(offload_config)


def get_offload_context():
    """
    Gets the offload configuration parameters. Configure through interface mindspore.set_offload_context().
    If the user is not set, the default configuration is obtained.

    Returns:
        Dict, heterogeneous training offload detailed configuration parameters.

    Examples:
        >>> from mindspore import context
        >>> offload_config = context.get_offload_context()
    """
    return _get_offload_context()


def _check_target_specific_cfgs(device, arg_key):
    """Checking whether a config is suitable for a specified device"""
    device_cfgs = {
        'enable_graph_kernel': ['Ascend', 'GPU', 'CPU'],
        'graph_kernel_flags': ['Ascend', 'GPU', 'CPU'],
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


@args_type_check(mode=int, precompile_only=bool, device_target=str, device_id=int, save_graphs=(bool, int),
                 save_graphs_path=str, enable_dump=bool, aoe_tune_mode=str, aoe_config=dict,
                 save_dump_path=str, enable_reduce_precision=bool, variable_memory_max_size=str,
                 enable_auto_mixed_precision=bool, inter_op_parallel_num=int,
                 enable_graph_kernel=bool, reserve_class_name_in_scope=bool, check_bprop=bool,
                 max_device_memory=str, print_file_path=str, max_call_depth=int, env_config_path=str,
                 graph_kernel_flags=str, save_compile_cache=bool, runtime_num_threads=int, load_compile_cache=bool,
                 grad_for_scalar=bool, pynative_synchronize=bool, mempool_block_size=str, disable_format_transform=bool,
                 op_timeout=int, deterministic=str, ascend_config=dict, jit_syntax_level=int,
                 jit_enable_inplace_ops=bool, gpu_config=dict)
def set_context(**kwargs):
    """
    Set context for running environment.

    Context should be configured before running your program. If there is no configuration,
    it will be automatically set according to the device target by default.

    Note:
        Attribute name is required for setting attributes.
        The mode is not recommended to be changed after net was initialized because the implementations of some
        operations are different in graph mode and pynative mode. Default: ``PYNATIVE_MODE`` .

    Some configurations are device specific, see the below table for details:

    +-------------------------+------------------------------+----------------------------+
    | Function Classification |   Configuration Parameters   |   Hardware Platform Support|
    +=========================+==============================+============================+
    | System Configuration    |   device_id                  |   CPU/GPU/Ascend           |
    |                         +------------------------------+----------------------------+
    |                         |   device_target              |   CPU/GPU/Ascend           |
    |                         +------------------------------+----------------------------+
    |                         |  max_device_memory           |  GPU/Ascend                |
    |                         +------------------------------+----------------------------+
    |                         |  variable_memory_max_size    |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  mempool_block_size          |  GPU/Ascend                |
    |                         +------------------------------+----------------------------+
    |                         |  op_timeout                  |  Ascend                    |
    +-------------------------+------------------------------+----------------------------+
    | Debug Configuration     |  save_graphs                 |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  save_graphs_path            |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  enable_dump                 |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  save_dump_path              |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  deterministic               |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  print_file_path             |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  env_config_path             |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  precompile_only             |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  reserve_class_name_in_scope |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  pynative_synchronize        |  CPU/GPU/Ascend            |
    +-------------------------+------------------------------+----------------------------+
    | Executive Control       |   mode                       |   CPU/GPU/Ascend           |
    |                         +------------------------------+----------------------------+
    |                         |  enable_graph_kernel         |  Ascend/GPU                |
    |                         +------------------------------+----------------------------+
    |                         |  graph_kernel_flags          |  Ascend/GPU                |
    |                         +------------------------------+----------------------------+
    |                         |  enable_reduce_precision     |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  aoe_tune_mode               |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  aoe_config                  |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  check_bprop                 |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  max_call_depth              |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  grad_for_scalar             |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  enable_compile_cache        |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  inter_op_parallel_num       |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  runtime_num_threads         |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  compile_cache_path          |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  disable_format_transform    |  GPU                       |
    |                         +------------------------------+----------------------------+
    |                         |  support_binary              |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  memory_optimize_level       |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  memory_offload              |  GPU/Ascend                |
    |                         +------------------------------+----------------------------+
    |                         |  ascend_config               |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  jit_syntax_level            |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  gpu_config                  |  GPU                       |
    +-------------------------+------------------------------+----------------------------+

    Args:
        device_id (int): ID of the target device, the value must be in [0, device_num_per_host-1],
            while device_num_per_host should be no more than 4096. Default: ``0`` .
        device_target (str): The target device to run, support "Ascend", "GPU", and "CPU".
            If device target is not set, the version of MindSpore package is used.
        max_device_memory (str): Set the maximum memory available for devices. The format is "xxGB".
            Default: ``" 1024GB"`` . The actual used memory size is the minimum of the available memory of the device
            and max_device_memory. 'max_device_memory' should be set before the program runs.
        variable_memory_max_size (str): This parameter is deprecated, and will be removed in a future version.
            Please use parameter 'max_device_memory' instead.
        mempool_block_size (str): Set the size of the memory pool block in PyNative mode for devices.
            The format is "xxGB". Default: ``"1GB"`` . Minimum size is "1G". The actual used memory block size is the
            minimum of the available memory of the device and mempool_block_size.
        op_timeout (int): Set the maximum duration of executing an operator in seconds.
            If the execution time exceeds this value, system will terminate the task. 0 means endless wait.
            Default: ``1900`` .
        save_graphs (bool or int): Whether to save intermediate compilation graphs. Default: ``0`` .
            Available values are:

            - False or 0: disable saving of intermediate compilation graphs.
            - 1: some intermediate files will be generated during graph compilation.
            - True or 2: Generate more ir files related to backend process.
            - 3: Generate visualization computing graphs and detailed frontend ir graphs.

            When the `save_graphs` attribute is set as ``True`` , ``1`` , ``2`` or ``3`` , attribute of
            `save_graphs_path` is used to set the intermediate compilation graph storage path. By default, the graphs
            are saved in the current directory.
        save_graphs_path (str): Path to save graphs. Default: ".".
            If the specified directory does not exist, the system will automatically create the directory.
            During distributed training, graphs will be saved to the directory of
            `save_graphs_path/rank_${rank_id}/`. `rank_id` is the ID of the current device in the cluster.
        deterministic (str): Whether to enable op run in deterministic mode. The value must be in the
            range of ['ON', 'OFF'], and the default value is ``'OFF'`` .

            - "ON": Enable operator deterministic running mode.
            - "OFF": Disable operator deterministic running mode.

            When deterministic mode is on, model ops will be deterministic in Ascend. This means that if op run
            multiple times with the same inputs on the same hardware, it will have the exact same outputs each time.
            This is useful for debugging models.
        enable_dump (bool): This parameters is deprecated, and will be deleted in the next version.
        save_dump_path (str): This parameters is deprecated, and will be deleted in the next version.
        print_file_path (str): The path of saving print data. If this parameter is set, print data is saved to
            a file by default, and print_file_path is not set, the screen will be displayed.
            If the saved file already exists, the timestamp suffix will be added to the file. Saving data to a file
            solves the problem of data loss in screen printing when a large amount of data is generated.
            If it is not set, an error will be reported: prompt to set the upper absolute path.
        env_config_path (str): Config path for DFX.
            Through mindspore.set_context(env_config_path="./mindspore_config.json")

            configure RDR:

            - enable: controls whether the RDR is enabled to collect the key data during training and
              save key data in the fault scenario. When set to ``true`` , the RDR will be turned on.
              When set to ``false`` , the RDR will be turned off.
            - mode: sets the mode of RDR on exporting data. When set to ``1`` , the RDR only exports data
              in the fault scenario. When set to ``2`` , the RDR exports data in the fault scenario and the
              normal end scenario. Default: ``1`` .
            - path: sets the path where RDR saves data. The current path must be absolute.

            Memory reuse:

            - mem_Reuse: controls whether the memory reuse function is turned on. When set to ``True`` ,
              the memory reuse function is turned on. When set to ``False`` , the memory reuse function is turned off.

        precompile_only (bool): Whether to only precompile the network. Default: ``False`` .
            If set to ``True`` , the network will only be compiled, not executed.
        reserve_class_name_in_scope (bool) : Whether to save the network class name in the scope. Default: ``True`` .
            Each node has a scope. A scope of a subnode is the name of its parent node. If reserve_class_name_in_scope
            is set to ``True`` , the class name will be saved after keyword 'net-' in the scope.
            For example:

            Default/net-Net1/net-Net2 (reserve_class_name_in_scope=True)

            Default/net/net (reserve_class_name_in_scope=False)

        pynative_synchronize (bool): Whether to enable synchronous execution of the device in PyNative mode.
            Default: ``False`` . When the value is set to ``False`` , the operator is executed asynchronously on the
            device. When an error occurs in the execution of the operator, the specific error script code location
            cannot be located, when the value is set to ``True`` , the operator is executed synchronously on the
            device. It will reduce the execution performance of the program. At this time, when an error occurs in the
            execution of the operator, the location of the error script code can be located according to the call stack
            of the error.
        mode (int): Running in GRAPH_MODE(0) or PYNATIVE_MODE(1).
            Both modes support all backends. Default: ``PYNATIVE_MODE`` .
        enable_graph_kernel (bool): Whether to enable graph kernel fusion to optimize network execution performance.
            Default: ``False`` .
            Indicates whether to enable image-computing convergence to optimize network execution performance.
            If enable_graph_kernel is set to ``True`` , acceleration can be enabled.
            For details of graph kernel fusion, please check
            `Enabling Graph Kernel Fusion
            <https://www.mindspore.cn/tutorials/experts/en/master/optimize/graph_fusion_engine.html>`_.
        graph_kernel_flags (str):
            Optimization options of graph kernel fusion, and the priority is higher when it conflicts
            with enable_graph_kernel. Only for experienced users.
            For example,

            .. code-block::

                mindspore.set_context(graph_kernel_flags="--opt_level=2 --dump_as_text")

            Some general options:

            - opt_level: Set the optimization level.
              Default: ``2`` . Graph kernel fusion can be enabled equivalently by setting opt_level greater than 0.
              Available values are:

              - 0: disables graph kernel fusion;
              - 1: enables the basic fusion of operators;
              - 2: includes all optimizations of level 1,
                and turns on more optimizations such as CSE, arithmetic simplification and so on;
              - 3: includes all optimizations of level 2, and turns on more optimizations such as SitchingFusion,
                ParallelFusion and so on. Optimizations of this level are radical and unstable in some scenarios.
                Be caution when using this level.

            - dump_as_text: dumps detail info as text files. Default: ``False`` .

        enable_reduce_precision (bool): Whether to enable precision reduction.
            If the operator does not support the user-specified precision, the precision will
            be changed automatically. Default: ``True`` .
        aoe_tune_mode (str): AOE tuning mode setting, which is not set by default.
            When set to ``"online"`` , the tuning in online function is turned on.
            When set to ``"offline"`` , ge graph will be save for offline tuning.
        aoe_config (dict): Set the parameters specific to Ascend Optimization Engine. It is not set by default.

            - job_type (str): Mode type setting, default value is ``"2"``.

              - ``"1"``: subgraph tuning;
              - ``"2"``: operator tuning.

        check_bprop (bool): Whether to check back propagation nodes. The checking ensures that the shape and dtype
            of back propagation node outputs is the same as input parameters. Default: ``False`` .
        max_call_depth (int): Specify the maximum depth of function call. Must be positive integer. Default: ``1000`` .
            The max_call_depth parameter needs to be set when the nested call is too deep or the number
            of subgraphs is too large. If max_call_depth is set larger than before, the system max stack depth should be
            set larger too, otherwise a `core dumped` exception may be raised because of system stack overflow.
        grad_for_scalar (bool):  Whether to get gradient for scalar. Default: ``False`` .
            When grad_for_scalar is set to ``True`` , the function's scalar input can be derived.
            The default value is ``False`` . Because the back-end does not support scaling operations currently,
            this interface only supports simple operations that can be deduced by the front-end.
        enable_compile_cache (bool): Whether to save or load the cache of the graph compiled by front-end.
            After enable_compile_cache is set to ``True`` , during the first execution, a hardware-independent
            compilation cache is generated and exported to a MINDIR file. When the network is executed again,
            if enable_compile_cache is still set to ``True`` and the network scripts are not changed,
            the compile cache is loaded. Note that only limited automatic detection for the changes of
            python scripts is supported by now, which means that there is a correctness risk. Default: ``False`` .
            This is an experimental prototype that is subject to change and/or deletion.
        compile_cache_path (str): Path to save the compile cache. Default: ".".
            If the specified directory does not exist, the system will automatically create the directory.
            The cache will be saved to the directory of `compile_cache_path/rank_${rank_id}/`. The `rank_id` is
            the ID of the current device in the cluster.
        inter_op_parallel_num(int): The thread number of op parallel at the same time. Default value is ``0`` ,
            which means use the default num.
        runtime_num_threads(int): The thread pool number of cpu kernel used in runtime,
            which must bigger than or equal to 0. Default value is ``30`` , if you run many processes at
            the same time, you should set the value smaller to avoid thread contention.
        disable_format_transform (bool): Whether to disable the automatic format transform function from NCHW to NHWC.
            When the network training performance of fp16 is worse than fp32, `disable_format_transform` can be set to
            ``True`` to try to improve training performance. Default: ``False`` .
        support_binary (bool): Whether to support run .pyc or .so in graph mode. If want to support run .so or .pyc
            in graph mode, coulde set 'support_binary' to be ``True`` , and run once .py file. It would save the source
            of the interfaces would be compiled by MindSpore to the interfaces definition .py file that should be
            guaranteed to be writable. Then compile the .py file to the .pyc or .so file, and could run in Graph mode.
        memory_optimize_level (str): The memory optimize level.
            Default: O0. The value must be in ['O0', 'O1'].

            - O0: priority performance option, disable SOMAS (Safe Optimized Memory Allocation Solver).
            - O1: priority memory option, enable SOMAS.
        memory_offload (str): Whether to enable the memory offload function. When it is enabled, the idle data will be
            temporarily copied to the host side in the case of insufficient device memory. The value must be in the
            range of ['ON', 'OFF'], and the default value is ``'OFF'`` .

            - ON: Enable the memory Offload function. On Ascend hardware platform, this parameter does not take effect
              when the environment variable "GRAPH_OP_RUN=1" is not set; This parameter does not take effect when
              memory_optimize_level is set 'O1'.
            - OFF: Turn off the memory Offload function.
        ascend_config (dict): Set the parameters specific to Ascend hardware platform. It is not set by default.
            The default value of `precision_mode`, `jit_compile` and
            `atomic_clean_policy` are experimental parameters, may change in the future.

            - precision_mode (str): Mixed precision mode setting, and the default value of inference network
              is ``force_fp16`` . The value range is as follows:

              - force_fp16: When the operator supports both float16 and float32, select float16 directly.
              - allow_fp32_to_fp16: When the operator does not support the float32 data type, directly reduce
                the precision of float16.
              - allow_mix_precision: Automatic mixing precision, facing the whole network operator, according
                to the built-in optimization strategy, automatically reduces the precision of some operators
                to float16 or bfloat16.
              - must_keep_origin_dtype: Keep the accuracy of the original drawing.
              - force_fp32: When the input of the matrix calculation operator is float16 and the output supports
                float16 and float32, output is forced to float32.
              - allow_fp32_to_bf16: When the operator does not support the float32 data type, directly reduce
                the precision of bfloat16.
              - allow_mix_precision_fp16: Automatic mixing precision, facing the whole network operator, automatically
                reduces the precision of some operators to float16 according to the built-in optimization strategy.
              - allow_mix_precision_bf16: Automatic mixing precision, facing the whole network operator, according to
                the built-in optimization strategy, automatically reduces the precision of some operators to bfloat16.

            - jit_compile (bool): Whether to select online compilation. the default value is based on CANN.
            - atomic_clean_policy (int): The policy for cleaning memory occupied by atomic operators in the network.
              Default: ``1`` .

              - 0: The memory occupied by all atomic operators in the network is cleaned centrally.
              - 1: Memory is not cleaned centrally and each atomic operator in the network is cleaned separately.
                When the memory of the network exceeds the limit, you may try this cleaning policy, but it may cause
                performance loss.
            - matmul_allow_hf32 (bool): Whether to convert FP32 to HF32 for Matmul operators. Default value: ``False``.
              This is an experimental prototype that is subject to change and/or deletion.
              For detailed information, please refer to `Ascend community <https://www.hiascend.com/>`_ .
            - conv_allow_hf32 (bool): Whether to convert FP32 to HF32 for Conv operators. Default value: ``True``.
              This is an experimental prototype that is subject to change and/or deletion.
              For detailed information, please refer to `Ascend community <https://www.hiascend.com/>`_ .
            - op_precision_mode (str): Path to config file of op precision mode. For detailed information, please refer
              to `Ascend community <https://www.hiascend.com/>`_ .
            - parallel_speed_up_json_path(Union[str, None]): The path to the parallel speed up json file, configuration
              can refer to `parallel_speed_up.json
              <https://gitee.com/mindspore/mindspore/blob/master/config/parallel_speed_up.json>`_ .
              If its value is None or '', it does not take effect. Default None.

              - recompute_comm_overlap (bool): Enable overlap between recompute ops and communication ops if True.
                Default: False.
              - matmul_grad_comm_overlap (bool): Enable overlap between grad ops and communication ops if True.
                Default: False.
              - enable_task_opt (bool): Enable the optimization of the number of tasks for each communication if True.
                Default: False.
              - interleaved_matmul_comm (bool): Enable interleaved optimization of Matmul-Comm if True. Default: False.
              - interleaved_layernorm_comm (bool): Enable interleaved optimization of LayerNorm-Comm if True.
                Default: False.

        jit_syntax_level (int): Set JIT syntax level for graph compiling, triggered by GRAPH_MODE and @jit decorator.
            The value must be ``STRICT`` or ``LAX`` . Default: ``LAX`` . All levels support all backends.

            - ``STRICT`` : Only basic syntax is supported, and execution performance is optimal. Can be used for MindIR
              load and export.
            - ``LAX`` : Compatible with all Python syntax as much as possible. However, execution performance may be
              affected and not optimal. Cannot be used for MindIR load and export due to some syntax that may not be
              able to be exported.

        gpu_config (dict): Set the parameters specific to gpu hardware platform. It is not set by default.
            Currently, only setting `conv_fprop_algo` and `conv_dgrad_algo` and `conv_wgrad_algo` and `conv_allow_tf32`
            and `matmul_allow_tf32` are supported on GPU hardware platform.

            - conv_fprop_algo (str): Specifies convolution forward algorithm and the default value is 'normal',
              The value range is as follows:

              - normal: Use the heuristic search algorithm.
              - performance: Use the trial search algorithm.
              - implicit_gemm: This algorithm expresses the convolution as a matrix product without actually explicitly
                forming the matrix that holds the input tensor data.
              - implicit_precomp_gemm: This algorithm expresses convolution as a matrix product without actually
                explicitly forming the matrix that holds the input tensor data, but still needs some memory workspace to
                precompute some indices in order to facilitate the implicit construction of the matrix that holds the
                input tensor data.
              - gemm: This algorithm expresses the convolution as an explicit matrix product. A significant memory
                workspace is needed to store the matrix that holds the input tensor data.
              - direct: This algorithm expresses the convolution as a direct convolution (for example, without
                implicitly or explicitly doing a matrix multiplication).
              - fft: This algorithm uses the Fast-Fourier Transform approach to compute the convolution. A significant
                memory workspace is needed to store intermediate results.
              - fft_tiling: This algorithm uses the Fast-Fourier Transform approach but splits the inputs into tiles.
                A significant memory workspace is needed to store intermediate results but less than fft algorithm for
                large size images.
              - winograd: This algorithm uses the Winograd Transform approach to compute the convolution. A reasonably
                sized workspace is needed to store intermediate results.
              - winograd_nonfused: This algorithm uses the Winograd Transform approach to compute the convolution. A
                significant workspace may be needed to store intermediate results.
            - conv_dgrad_algo (str): Specifies convolution data grad algorithm and the default value is 'normal',
              The value range is as follows:

              - normal: Use the heuristic search algorithm.
              - performance: Use the trial search algorithm.
              - algo_0: This algorithm expresses the convolution as a sum of matrix products without actually explicitly
                forming the matrix that holds the input tensor data. The sum is done using the atomic add operation,
                thus the results are non-deterministic.
              - algo_1: This algorithm expresses the convolution as a matrix product without actually explicitly forming
                the matrix that holds the input tensor data. The results are deterministic.
              - fft: This algorithm uses a Fast-Fourier Transform approach to compute the convolution. A significant
                memory workspace is needed to store intermediate results. The results are deterministic.
              - fft_tiling: This algorithm uses the Fast-Fourier Transform approach but splits the inputs into tiles.
                A significant memory workspace is needed to store intermediate results but less than fft for large size
                images. The results are deterministic.
              - winograd: This algorithm uses the Winograd Transform approach to compute the convolution. A reasonably
                sized workspace is needed to store intermediate results. The results are deterministic.
              - winograd_nonfused: This algorithm uses the Winograd Transform approach to compute the convolution.
                A significant workspace may be needed to store intermediate results. The results are deterministic.
            - conv_wgrad_algo (str): Specifies convolution filter grad algorithm and the default value is 'normal',
              The value range is as follows:

              - normal: Use the heuristic search algorithm.
              - performance: Use the trial search algorithm.
              - algo_0: This algorithm expresses the convolution as a sum of matrix products without actually explicitly
                forming the matrix that holds the input tensor data. The sum is done using the atomic add operation,
                thus the results are non-deterministic.
              - algo_1: This algorithm expresses the convolution as a matrix product without actually explicitly forming
                the matrix that holds the input tensor data. The results are deterministic.
              - fft: This algorithm uses a Fast-Fourier Transform approach to compute the convolution. A significant
                memory workspace is needed to store intermediate results. The results are deterministic.
              - algo_3: This algorithm is similar to algo_0 but uses some small workspace to precompute some indices.
                The results are also non-deterministic.
              - winograd_nonfused: This algorithm uses the Winograd Transform approach to compute the convolution.
                A significant workspace may be needed to store intermediate results. The results are deterministic.
              - fft_tiling: This algorithm uses the Fast-Fourier Transform approach but splits the inputs into tiles.
                A significant memory workspace is needed to store intermediate results but less than fft for large size
                images. The results are deterministic.
            - conv_allow_tf32 (bool): The flag below controls to allow Tensor core TF32 computation on CUDNN and the
              default value is ``True``.
            - matmul_allow_tf32 (bool): The flag below controls to allow Tensor core TF32 computation on CUBLAS and the
              default value is ``False``.

    Raises:
        ValueError: If input key is not an attribute in context.

    Examples:
        >>> import mindspore as ms
        >>> ms.set_context(mode=ms.PYNATIVE_MODE)
        >>> ms.set_context(precompile_only=True)
        >>> ms.set_context(device_target="Ascend")
        >>> ms.set_context(device_id=0)
        >>> ms.set_context(save_graphs=True, save_graphs_path="./model.ms")
        >>> ms.set_context(enable_reduce_precision=True)
        >>> ms.set_context(enable_graph_kernel=True)
        >>> ms.set_context(graph_kernel_flags="--opt_level=2 --dump_as_text")
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
        >>> ms.set_context(memory_offload='ON')
        >>> ms.set_context(deterministic='ON')
        >>> ms.set_context(ascend_config={"precision_mode": "force_fp16", "jit_compile": True,
        ...                "atomic_clean_policy": 1, "op_precision_mode": "./op_precision_config_file"})
        >>> ms.set_context(jit_syntax_level=ms.STRICT)
        >>> ms.set_context(gpu_config={"conv_fprop_algo": "performance", "conv_allow_tf32": True,
        ...                "matmul_allow_tf32": True})
    """
    ctx = _context()
    # set device target first
    if 'device_target' in kwargs:
        ctx.set_device_target(kwargs['device_target'])
    device = ctx.get_param(ms_ctx_param.device_target)
    for key, value in kwargs.items():
        if key in ('enable_sparse', 'auto_tune_mode'):
            logger.warning(f"For 'context.set_context', '{key}' parameter is deprecated, "
                           "and will be removed in the next version.")
            continue
        if key in ('enable_auto_mixed_precision', 'enable_dump', 'save_dump_path'):
            logger.warning(f"For 'context.set_context', '{key}' parameter is deprecated. "
                           "For details, please see the interface parameter API comments")
            continue
        if key in ('precision_mode', 'jit_compile', 'atomic_clean_policy', 'matmul_allow_hf32', 'conv_allow_hf32',
                   'op_precision_mode'):
            raise ValueError(f"Please set '{key}' through parameter ascend_config")
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
        if not _check_target_specific_cfgs(device, key):
            continue
        if hasattr(ctx, key):
            setattr(ctx, key, value)
            continue
        if key in ctx.setters:
            ctx.setters[key](ctx, value)
            continue
        # enum variables beginning with '_' are for internal use
        if key in ms_ctx_param.__members__ and key[0] != '_':
            ctx.set_param(ms_ctx_param.__members__[key], value)
            continue
        raise ValueError(f"For 'context.set_context', the keyword argument {key} is not recognized! For detailed "
                         f"usage of 'set_context', please refer to the Mindspore official website.")


def get_context(attr_key):
    """
    Get context attribute value according to the input key.
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
    Set parameter server training mode context.

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
                          Only after enable_ps is set True, the environment variables will be effective.
                          Default: ``False`` .
        config_file_path (string): Configuration file path used by recovery, parameter server training mode only
                                   supports Server disaster recovery currently. Default: ``''`` .
        scheduler_manage_port (int): Scheduler manage port used to scale out/in. Default: ``11202`` .
        enable_ssl (bool): Set PS SSL mode enabled or disabled. Default: ``False`` .
        client_password (str): Password to decrypt the secret key stored in the client certificate. Default: ``''`` .
        server_password (str): Password to decrypt the secret key stored in the server certificate. Default: ``''`` .

    Raises:
        ValueError: If input key is not the attribute in parameter server training mode context.

    Examples:
        >>> import mindspore as ms
        >>> ms.set_ps_context(enable_ps=True, enable_ssl=True, client_password='123456', server_password='123456')
    """
    _set_ps_context(**kwargs)


def get_ps_context(attr_key):
    """
    Get parameter server training mode context attribute value according to the key.

    Args:
        attr_key (str): The key of the attribute:

            - enable_ps (bool): Whether to enable parameter server training mode. Default: ``False`` .
            - config_file_path (string): Configuration file path used by recovery, parameter server training mode only
              supports Server disaster recovery currently. Default: ``''`` .
            - scheduler_manage_port (int): Scheduler manage port used to scale out/in. Default: ``11202`` .
            - enable_ssl (bool): Set PS SSL mode enabled or disabled. Default: ``False`` .
            - client_password (str): Password to decrypt the secret key stored in the client certificate.
              Default: ``''`` .
            - server_password (str): Password to decrypt the secret key stored in the server certificate.
              Default: ``''`` .

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
    Reset parameter server training mode context attributes to the default values:

    - enable_ps: False.

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
