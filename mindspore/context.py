# Copyright 2020 Huawei Technologies Co., Ltd
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
including execution mode, execution backend and other feature switchs.
"""
import threading
from collections import namedtuple
from types import FunctionType
from mindspore import log as logger
from mindspore._c_expression import MSContext
from mindspore._extends.pynative_helper import args_type_check
from mindspore.parallel._auto_parallel_context import _set_auto_parallel_context, _get_auto_parallel_context, \
    _reset_auto_parallel_context


__all__ = ['GRAPH_MODE', 'PYNATIVE_MODE', 'set_context', 'get_context', 'set_auto_parallel_context',
           'get_auto_parallel_context', 'reset_auto_parallel_context']

GRAPH_MODE = 0
PYNATIVE_MODE = 1


class _ThreadLocalInfo(threading.local):
    """
    Thread local Info used for store thread local attributes.
    """
    def __init__(self):
        super(_ThreadLocalInfo, self).__init__()
        self._reserve_class_name_in_scope = True

    @property
    def reserve_class_name_in_scope(self):
        """Gets whether to save the network class name in the scope."""
        return self._reserve_class_name_in_scope

    @reserve_class_name_in_scope.setter
    def reserve_class_name_in_scope(self, reserve_class_name_in_scope):
        """Sets whether to save the network class name in the scope."""
        if not isinstance(reserve_class_name_in_scope, bool):
            raise ValueError("Set reserve_class_name_in_scope value must be bool!")
        self._reserve_class_name_in_scope = reserve_class_name_in_scope


_ContextRecord = namedtuple("_ContextRecord", ["is_pynative_mode", "switch_context_fn"])


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
        self.context_stack.append(_ContextRecord(is_pynative, switch_context_fn))

    def pop(self):
        self.context_stack.pop()


class _Context:
    """
    _Context is the environment in which operations are executed

    Note:
        Create a context through instantiating Context object is not recommended.
        should use context() to get the context since Context is singleton.
    """
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._thread_local_info = _ThreadLocalInfo()
        self._context_switches = _ContextSwitchInfo(True)
        self._context_handle = MSContext.get_instance()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance_lock.acquire()
            cls._instance = object.__new__(cls)
            cls._instance_lock.release()
        return cls._instance

    def __getattribute__(self, attr):
        value = object.__getattribute__(self, attr)
        if attr == "_context_handle" and value is None:
            raise ValueError("Context handle is none in context!!!")
        return value

    # For Ascend task sink mode execution
    @property
    def enable_task_sink(self):
        return self._context_handle.get_task_sink_flag()

    @enable_task_sink.setter
    def enable_task_sink(self, task_sink):
        self._context_handle.set_task_sink_flag(task_sink)

    @property
    def mode(self):
        return self._context_handle.get_execution_mode()

    @mode.setter
    def mode(self, mode):
        """
        Switch between Graph mode and PyNative mode.

        Args:
            mode (int): GRAPH_MODE or PYNATIVE_MODE.
        """
        self._context_handle.set_execution_mode(mode)
        if mode == PYNATIVE_MODE:
            if self.enable_debug_runtime:
                self.set_backend_policy("vm")
            self._context_switches.push(True, None)
        else:
            if self.enable_debug_runtime:
                self.set_backend_policy("ge")
            self._context_switches.push(False, None)

    def set_backend_policy(self, policy):
        success = self._context_handle.set_backend_policy(policy)
        if not success:
            raise RuntimeError("Backend policy must be one of ge, vm, ms.")

    @property
    def precompile_only(self):
        return self._context_handle.get_precompile_only()

    @precompile_only.setter
    def precompile_only(self, precompile_only):
        self._context_handle.set_precompile_only(precompile_only)

    @property
    def save_graphs(self):
        return self._context_handle.get_save_graphs_flag()

    @save_graphs.setter
    def save_graphs(self, save_graphs_flag):
        self._context_handle.set_save_graphs_flag(save_graphs_flag)

    @property
    def save_graphs_path(self):
        return self._context_handle.get_save_graphs_path()

    @save_graphs_path.setter
    def save_graphs_path(self, save_graphs_path):
        self._context_handle.set_save_graphs_path(save_graphs_path)

    @property
    def device_target(self):
        return self._context_handle.get_device_target()

    @device_target.setter
    def device_target(self, target):
        success = self._context_handle.set_device_target(target)
        if not success:
            raise ValueError("target device name is invalid!!!")

    @property
    def device_id(self):
        return self._context_handle.get_device_id()

    @device_id.setter
    def device_id(self, device_id):
        if device_id < 0 or device_id > 4095:
            raise ValueError("Device id must be in [0, 4095], but got {}".format(device_id))
        success = self._context_handle.set_device_id(device_id)
        if not success:
            raise RuntimeError("Device id set failed!!!")

    @property
    def enable_hccl(self):
        return self._context_handle.get_hccl_flag()

    @enable_hccl.setter
    def enable_hccl(self, hccl):
        self._context_handle.set_hccl_flag(hccl)

    @property
    def enable_ir_fusion(self):
        return self._context_handle.get_ir_fusion_flag()

    @enable_ir_fusion.setter
    def enable_ir_fusion(self, enable_ir_fusion):
        self._context_handle.set_ir_fusion_flag(enable_ir_fusion)

    @property
    def enable_loop_sink(self):
        return self._context_handle.get_loop_sink_flag()

    @enable_loop_sink.setter
    def enable_loop_sink(self, enable_loop_sink):
        self._context_handle.set_loop_sink_flag(enable_loop_sink)

    @property
    def enable_mem_reuse(self):
        return self._context_handle.get_enable_mem_reuse()

    @enable_mem_reuse.setter
    def enable_mem_reuse(self, enable_mem_reuse):
        self._context_handle.set_enable_mem_reuse(enable_mem_reuse)

    @property
    def save_ms_model(self):
        return self._context_handle.get_save_ms_model_flag()

    @save_ms_model.setter
    def save_ms_model(self, save_ms_model_flag):
        self._context_handle.set_save_ms_model_flag(save_ms_model_flag)

    @property
    def save_ms_model_path(self):
        return self._context_handle.get_save_ms_model_path()

    @save_ms_model_path.setter
    def save_ms_model_path(self, save_ms_model_path):
        self._context_handle.set_save_ms_model_path(save_ms_model_path)

    @property
    def enable_gpu_summary(self):
        return self._context_handle.get_enable_gpu_summary()

    @enable_gpu_summary.setter
    def enable_gpu_summary(self, enable_gpu_summary):
        self._context_handle.set_enable_gpu_summary(enable_gpu_summary)

    @property
    def enable_auto_mixed_precision(self):
        return self._context_handle.get_auto_mixed_precision_flag()

    @enable_auto_mixed_precision.setter
    def enable_auto_mixed_precision(self, enable_auto_mixed_precision):
        self._context_handle.set_auto_mixed_precision_flag(enable_auto_mixed_precision)

    @property
    def enable_reduce_precision(self):
        return self._context_handle.get_enable_reduce_precision_flag()

    @enable_reduce_precision.setter
    def enable_reduce_precision(self, enable_reduce_precision):
        self._context_handle.set_enable_reduce_precision_flag(enable_reduce_precision)

    @property
    def enable_dump(self):
        return self._context_handle.get_enable_dump()

    @enable_dump.setter
    def enable_dump(self, enable_dump):
        self._context_handle.set_enable_dump(enable_dump)

    @property
    def save_dump_path(self):
        return self._context_handle.get_save_dump_path()

    @save_dump_path.setter
    def save_dump_path(self, save_dump_path):
        self._context_handle.set_save_dump_path(save_dump_path)

    @property
    def reserve_class_name_in_scope(self):
        """Gets whether to save the network class name in the scope."""
        return self._thread_local_info.reserve_class_name_in_scope

    @reserve_class_name_in_scope.setter
    def reserve_class_name_in_scope(self, reserve_class_name_in_scope):
        """Sets whether to save the network class name in the scope."""
        self._thread_local_info.reserve_class_name_in_scope = reserve_class_name_in_scope

    @property
    def enable_dynamic_memory(self):
        return self._context_handle.get_enable_dynamic_mem_pool()

    @enable_dynamic_memory.setter
    def enable_dynamic_memory(self, enable_dynamic_memory):
        self._context_handle.set_enable_dynamic_mem_pool(enable_dynamic_memory)

    @property
    def graph_memory_max_size(self):
        return None

    @graph_memory_max_size.setter
    def graph_memory_max_size(self, graph_memory_max_size):
        if check_input_fotmat(graph_memory_max_size):
            graph_memory_max_size_ = graph_memory_max_size[:-2] + " * 1024 * 1024 * 1024"
            self._context_handle.set_graph_memory_max_size(graph_memory_max_size_)
        else:
            raise ValueError("Context param graph_memory_max_size should be in correct format! Such as \"26GB\"")

    @property
    def variable_memory_max_size(self):
        return None

    @variable_memory_max_size.setter
    def variable_memory_max_size(self, variable_memory_max_size):
        if check_input_fotmat(variable_memory_max_size):
            variable_memory_max_size_ = variable_memory_max_size[:-2] + " * 1024 * 1024 * 1024"
            self._context_handle.set_variable_memory_max_size(variable_memory_max_size_)
        else:
            raise ValueError("Context param variable_memory_max_size should be in correct format! Such as \"5GB\"")

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


def check_input_fotmat(x):
    import re
    pattern = r'[1-9][0-9]*(\.)?[0-9]*GB|0\.[0-9]*GB'
    result = re.match(pattern, x)
    return result is not None

_k_context = None


def _context():
    """
    Get the global _context, if context is not created, create a new one.

    Returns:
        _Context, the global context in PyNative mode.
    """
    global _k_context
    if _k_context is None:
        default_backend = 'debug'
        try:
            from mindspore import default_config
            default_backend = default_config.__backend__
        except ImportError:
            logger.error("import default config fail")
        _k_context = _Context()
        _k_context.enable_debug_runtime = False
        if default_backend == 'debug':
            _k_context.enable_debug_runtime = True
            default_backend = 'vm'
        _k_context.set_backend_policy(default_backend)
    return _k_context


@args_type_check(device_num=int, global_rank=int, mirror_mean=bool, cast_before_mirror=bool, parallel_mode=str,
                 parameter_broadcast=bool)
def set_auto_parallel_context(**kwargs):
    """
    Set auto parallel context.

    Note:
        Attribute name is required for setting attributes.

    Args:
        device_num (int): Available device number, the value must be in [1, 4096]. Default: 1.
        global_rank (int): Global rank id, the value must be in [0, 4095]. Default: 0.
        mirror_mean (bool): Whether to perform mean operator after all-reduce of mirror. Default: False.
        cast_before_mirror (bool): Insert Mirror Op after the cast if this flag is True. Default: True.
        parallel_mode (str): There are five kinds of parallel modes, "stand_alone", "data_parallel",
                     "hybrid_parallel", "semi_auto_parallel" and "auto_parallel". Default: "stand_alone".

                     - stand_alone: Only one processor working.

                     - data_parallel: Distributing the data across different processors.

                     - hybrid_parallel: Achieving data parallelism and model parallelism manually.

                     - semi_auto_parallel: Achieving data parallelism and model parallelism by
                       setting parallel strategies.

                     - auto_parallel: Achieving parallelism automatically.
        parameter_broadcast (bool): Indicating whether to broadcast parameters before training.
                       "stand_alone", "semi_auto_parallel" and "auto_parallel" do not support parameter
                       broadcast. Default: False.

    Raises:
        ValueError: If input key is not attribute in auto parallel context.

    Examples:
        >>> context.set_auto_parallel_context(device_num=8)
        >>> context.set_auto_parallel_context(global_rank=0)
        >>> context.set_auto_parallel_context(mirror_mean=True)
        >>> context.set_auto_parallel_context(cast_before_mirror=False)
        >>> context.set_auto_parallel_context(parallel_mode="auto_parallel")
        >>> context.set_auto_parallel_context(parameter_broadcast=False)
    """
    _set_auto_parallel_context(**kwargs)


def get_auto_parallel_context(attr_key):
    """
    Gets auto parallel context attribute value according to the key.

    Args:
        attr_key (str): The key of the attribute.

    Returns:
        Returns attribute value according to the key.

    Raises:
        ValueError: If input key is not attribute in auto parallel context.
    """
    return _get_auto_parallel_context(attr_key)


def reset_auto_parallel_context():
    """
    Reset auto parallel context attributes to the default values:

    - device_num: 1.
    - global_rank: 0.
    - mirror_mean: False.
    - cast_before_mirror: True.
    - parallel_mode: "stand_alone".
    - parameter_broadcast: False.
    """
    _reset_auto_parallel_context()


@args_type_check(mode=int, precompile_only=bool, device_target=str,
                 device_id=int, enable_ir_fusion=bool, save_graphs=bool, enable_hccl=bool,
                 enable_task_sink=bool, save_graphs_path=str, enable_loop_sink=bool,
                 enable_mem_reuse=bool, save_ms_model=bool, save_ms_model_path=str, enable_gpu_summary=bool,
                 enable_auto_mixed_precision=bool, enable_dump=bool, save_dump_path=str,
                 enable_reduce_precision=bool, enable_dynamic_memory=bool, graph_memory_max_size=str,
                 variable_memory_max_size=str)
def set_context(**kwargs):
    """
    Set context for running environment.

    Context should be configured before running your program. If there is no configuration,
    the "Ascend" device target will be used by default. GRAPH_MODE or
    PYNATIVE_MODE can be set by `mode` attribute and both modes support all backends, default
    mode is PYNATIVE_MODE.

    When the `save_graphs` attribute is set to True, attribute of `save_graphs_path` is used to set the
    intermediate compilation graph storage path. By default, the graphs are saved in the current directory.
    As for other configurations and arguments, please refer to the corresponding module
    description, the configuration is optional and can be enabled when needed.

    Note:
        Attribute name is required for setting attributes.
        If need to config graph max memory size and variable max memory size, one must make sure:
            The sum of graph_memory_max_size and variable_memory_max_size should be less than total memory size of
            a device, while the total memory is supposed to be no more than 256GB.

    Args:
        mode (int): Running in GRAPH_MODE(0) or PYNATIVE_MODE(1). Default: PYNATIVE_MODE.
        device_target (str): The target device to run, support "Ascend", "GPU", "CPU". Default: "Ascend".
        device_id (int): Id of target device, the value must be in [0, device_num_per_host-1],
                    while device_num_per_host should no more than 4096. Default: 0.
        enable_ir_fusion (bool): Whether to enable ir fusion. Default: True.
        save_graphs (bool): Whether to save graphs. Default: False.
        enable_hccl (bool): Whether to enable hccl. Default: False.
        enable_loop_sink (bool): Whether to enable loop sink. Default: False.
        enable_task_sink (bool): Whether to enable task sink. Default: True.
        enable_mem_reuse (bool): Whether to enable memory reuse. Default: True.
        save_ms_model (bool): Whether to save model converted by graph. Default: False.
        save_ms_model_path (str): Path to save converted model. Default: "."
        enable_gpu_summary (bool): Whether to enable gpu summary. Default: True.
        save_graphs_path (str): Path to save graphs. Default: "."
        enable_auto_mixed_precision (bool): Whether to enable auto mixed precision. Default: True.
        reserve_class_name_in_scope (bool) : Whether to save the network class name in the scope. Default: True.
        enable_reduce_precision (bool): Whether to enable precision reduction. Default: True.
        enable_dump (bool): Whether to enable dump. Default: False.
        save_dump_path (str): Set path to dump data. Default: ".".
        enable_dynamic_memory (bool): Whether to enable dynamic memory. Default: False.
        graph_memory_max_size (str): Set graph memory max size. Default: "26GB".
        variable_memory_max_size (str): Set variable memory max size. Default: "5GB".

    Raises:
        ValueError: If input key is not an attribute in context.

    Examples:
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> context.set_context(mode=context.PYNATIVE_MODE)
        >>> context.set_context(device_target="Ascend")
        >>> context.set_context(device_id=0)
        >>> context.set_context(save_graphs=True, save_graphs_path="./model.ms")
        >>> context.set_context(enable_task_sink=True)
        >>> context.set_context(enable_mem_reuse=True)
        >>> context.set_context(enable_reduce_precision=True)
        >>> context.set_context(save_ms_model=True, save_ms_model_path=".")
        >>> context.set_context(enable_gpu_summary=False)
        >>> context.set_context(enable_dump=False, save_dump_path=".")
        >>> context.set_context(reserve_class_name_in_scope=True)
        >>> context.set_context(enable_dynamic_memory=True)
        >>> context.set_context(graph_memory_max_size="25GB")
        >>> context.set_context(variable_memory_max_size="6GB")
        >>> context.set_context(mode=context.GRAPH_MODE,
        >>>                     device_target="Ascend",device_id=0, save_graphs=True,
        >>>                     save_graphs_path="/mindspore")
    """
    for key, value in kwargs.items():
        if not hasattr(_context(), key):
            raise ValueError("Set context keyword %s is not recognized!" % key)
        setattr(_context(), key, value)


def get_context(attr_key):
    """
    Gets context attribute value according to the input key.

    Args:
        attr_key (str): The key of the attribute.

    Returns:
        Object, The value of given attribute key.

    Raises:
        ValueError: If input key is not an attribute in context.
    """
    if not hasattr(_context(), attr_key):
        raise ValueError("Get context keyword %s is not recognized!" % attr_key)
    return getattr(_context(), attr_key)
