# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import json
import os
import time
import threading
from collections import namedtuple
from types import FunctionType

from mindspore import log as logger
from mindspore._c_expression import MSContext, ms_ctx_param
from mindspore._checkparam import args_type_check, Validator, args_unreset_check
from mindspore.parallel._auto_parallel_context import _set_auto_parallel_context, _get_auto_parallel_context, \
    _reset_auto_parallel_context
from mindspore.parallel._ps_context import _set_ps_context, _get_ps_context, _reset_ps_context
from .default_config import __device_target__, __package_name__

__all__ = ['GRAPH_MODE', 'PYNATIVE_MODE', 'set_context', 'get_context', 'set_auto_parallel_context',
           'get_auto_parallel_context', 'reset_auto_parallel_context', 'ParallelMode', 'set_ps_context',
           'get_ps_context', 'reset_ps_context', 'set_fl_context', 'get_fl_context']

GRAPH_MODE = 0
PYNATIVE_MODE = 1
_DEVICE_APP_MEMORY_SIZE = 31  # The max memory size of graph plus variable.
_re_pattern = r'[1-9][0-9]*(\.)?[0-9]*GB|0\.[0-9]*GB'
_k_context = None


def _make_directory(path):
    """Make directory."""
    real_path = None
    if path is None or not isinstance(path, str) or path.strip() == "":
        raise ValueError(f"Input path `{path}` is invalid type")

    # convert the relative paths
    path = os.path.realpath(path)
    logger.debug("The absolute path is %r", path)

    # check whether the path is already existed and has written permissions
    if os.path.exists(path):
        real_path = path
    else:
        # All exceptions need to be caught because create directory maybe have some limit(permissions)
        logger.debug("The directory(%s) doesn't exist, will create it", path)
        try:
            os.makedirs(path)
            real_path = path
        except PermissionError as e:
            logger.error(f"No write permission on the directory `{path}, error = {e}")
            raise ValueError(f"No write permission on the directory `{path}`.")
    return real_path


def _get_print_file_name(file_name):
    """Add timestamp suffix to file name. Rename the file name:  file_name + "." + time(seconds)."""
    time_second = str(int(time.time()))
    file_name = file_name + "." + time_second
    if os.path.exists(file_name):
        ValueError("This file {} already exists.".format(file_name))
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
        if not isinstance(reserve_class_name_in_scope, bool):
            raise ValueError(
                "Set reserve_class_name_in_scope value must be bool!")
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
        should use context() to get the context since Context is singleton.
    """
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._thread_local_info = _ThreadLocalInfo()
        self._context_switches = _ContextSwitchInfo(False)
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

    def get_param(self, param):
        return self._context_handle.get_param(param)

    def set_param(self, param, value):
        self._context_handle.set_param(param, value)

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
            raise ValueError(f'The execution mode {mode} is invalid!')
        self.set_param(ms_ctx_param.mode, mode)

    def set_backend_policy(self, policy):
        success = self._context_handle.set_backend_policy(policy)
        if not success:
            raise RuntimeError("Backend policy must be one of ge, vm, ms.")

    def set_save_graphs_path(self, save_graphs_path):
        self.set_param(ms_ctx_param.save_graphs_path, _make_directory(save_graphs_path))

    def set_device_target(self, target):
        valid_targets = ["CPU", "GPU", "Ascend", "Davinci"]
        if not target in valid_targets:
            raise ValueError(f"Target device name {target} is invalid! It must be one of {valid_targets}")
        if target == "Davinci":
            target = "Ascend"
        self.set_param(ms_ctx_param.device_target, target)
        if self.enable_debug_runtime and target == "CPU":
            self.set_backend_policy("vm")

    def set_auto_tune_mode(self, tune_mode):
        candidate = ["NO_TUNE", "RL", "GA", "RL,GA", "GA,RL"]
        if tune_mode in candidate:
            self.set_param(ms_ctx_param.tune_mode, tune_mode)
        else:
            raise ValueError(f"Tune mode must be in ['NO_TUNE', 'RL', 'GA', 'RL,GA', 'GA,RL'], but got {tune_mode}")

    def set_device_id(self, device_id):
        if device_id < 0 or device_id > 4095:
            raise ValueError(f"Device id must be in [0, 4095], but got {device_id}")
        self.set_param(ms_ctx_param.device_id, device_id)

    def set_max_call_depth(self, max_call_depth):
        if max_call_depth <= 0:
            raise ValueError(f"Max call depth must be greater than 0, but got {max_call_depth}")
        self.set_param(ms_ctx_param.max_call_depth, max_call_depth)

    def set_profiling_options(self, option):
        if not isinstance(option, str):
            raise TypeError("The parameter option must be str.")
        self.set_param(ms_ctx_param.profiling_options, option)

    def set_variable_memory_max_size(self, variable_memory_max_size):
        """set values of variable_memory_max_size and graph_memory_max_size"""
        if not Validator.check_str_by_regular(variable_memory_max_size, _re_pattern):
            raise ValueError("Context param variable_memory_max_size should be in correct format! Such as \"5GB\"")
        if int(variable_memory_max_size[:-2]) > _DEVICE_APP_MEMORY_SIZE:
            raise ValueError("Context param variable_memory_max_size should be not greater than 31GB.")
        variable_memory_max_size_ = variable_memory_max_size[:-2] + " * 1024 * 1024 * 1024"
        graph_memory_max_size = _DEVICE_APP_MEMORY_SIZE - int(variable_memory_max_size[:-2])
        graph_memory_max_size_ = str(graph_memory_max_size) + " * 1024 * 1024 * 1024"
        self.set_param(ms_ctx_param.variable_memory_max_size, variable_memory_max_size_)
        self.set_param(ms_ctx_param._graph_memory_max_size, graph_memory_max_size_)

    def set_max_device_memory(self, max_device_memory):
        if not Validator.check_str_by_regular(max_device_memory, _re_pattern):
            raise ValueError("Context param max_device_memory should be in correct format! Such as \"3.5GB\"")
        max_device_memory_value = float(max_device_memory[:-2])
        if max_device_memory_value == 0:
            raise ValueError("Context param max_device_memory should be in correct format! Such as \"3.5GB\"")
        self.set_param(ms_ctx_param.max_device_memory, max_device_memory_value)

    def set_print_file_path(self, file_path):
        """Add timestamp suffix to file name. Sets print file path."""
        print_file_path = os.path.realpath(file_path)
        if os.path.isdir(print_file_path):
            raise IOError("Print_file_path should be file path, but got {}.".format(file_path))

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
            raise ValueError("The 'env_config_path' is not supported, please enable ENABLE_DUMP_IR "
                             "with '-D on' and recompile source.")
        env_config_path = os.path.realpath(env_config_path)
        if not os.path.isfile(env_config_path):
            raise ValueError("The %r set by 'env_config_path' should be an existing json file." % env_config_path)
        try:
            with open(env_config_path, 'r') as f:
                json.load(f)
        except (TypeError, ValueError) as exo:
            raise ValueError("The %r set by 'env_config_path' should be a json file. "
                             "Detail: %s." % (env_config_path, str(exo)))
        self.set_param(ms_ctx_param.env_config_path, env_config_path)

    setters = {
        'mode': set_mode,
        'save_graphs_path': set_save_graphs_path,
        'device_target': set_device_target,
        'device_id': set_device_id,
        'auto_tune_mode': set_auto_tune_mode,
        'max_call_depth': set_max_call_depth,
        'profiling_options': set_profiling_options,
        'variable_memory_max_size': set_variable_memory_max_size,
        'max_device_memory': set_max_device_memory,
        'print_file_path': set_print_file_path,
        'env_config_path': set_env_config_path
    }

    @property
    def reserve_class_name_in_scope(self):
        """Get whether to save the network class name in the scope."""
        return self._thread_local_info.reserve_class_name_in_scope

    @reserve_class_name_in_scope.setter
    def reserve_class_name_in_scope(self, reserve_class_name_in_scope):
        """Set whether to save the network class name in the scope."""
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


@args_type_check(device_num=int, global_rank=int, gradients_mean=bool, gradient_fp32_sync=bool, parallel_mode=str,
                 auto_parallel_search_mode=str, parameter_broadcast=bool, strategy_ckpt_load_file=str,
                 strategy_ckpt_save_file=str, full_batch=bool, enable_parallel_optimizer=bool,
                 all_reduce_fusion_config=list, pipeline_stages=int, grad_accumulation_step=int)
def set_auto_parallel_context(**kwargs):
    r"""
    Set auto parallel context, which is valid only for Ascend and GPU target.

    Auto parallel context should be configured before the initialization of your network.

    Note:
        Attribute name is required for setting attributes.
        If a program has tasks on different parallel modes, before setting a new parallel mode for the
        next task, interface mindspore.context.reset_auto_parallel_context() should be called to reset
        the configuration.
        Setting or changing parallel modes must be called before creating any Initializer, otherwise,
        it may have RuntimeError when compiling the network.

    Some configurations are parallel mode specific, see the below table for details:

    ===========================  ===========================
    Common                       AUTO_PARALLEL
    ===========================  ===========================
    device_num                   gradient_fp32_sync
    global_rank                  loss_repeated_mean
    gradients_mean               auto_parallel_search_mode
    parallel_mode                strategy_ckpt_load_file
    all_reduce_fusion_config     strategy_ckpt_save_file
    enable_parallel_optimizer    dataset_strategy
               \                 pipeline_stages
               \                 grad_accumulation_step
    ===========================  ===========================

    Args:
        device_num (int): Available device number, the value must be in [1, 4096]. Default: 1.
        global_rank (int): Global rank id, the value must be in [0, 4095]. Default: 0.
        gradients_mean (bool): Whether to perform mean operator after allreduce of gradients.
                     "stand_alone" do not support gradients_mean. Default: False.
        gradient_fp32_sync (bool): Run allreduce of gradients in fp32. "stand_alone", "data_parallel"
                     and "hybrid_parallel" do not support gradient_fp32_sync. Default: True.
        parallel_mode (str): There are five kinds of parallel modes, "stand_alone", "data_parallel",
                     "hybrid_parallel", "semi_auto_parallel" and "auto_parallel". Default: "stand_alone".

                     - stand_alone: Only one processor is working.

                     - data_parallel: Distributes the data across different processors.

                     - hybrid_parallel: Achieves data parallelism and model parallelism manually.

                     - semi_auto_parallel: Achieves data and model parallelism by setting parallel strategies.

                     - auto_parallel: Achieving parallelism automatically.
        auto_parallel_search_mode (str): There are two kinds of shard strategy search modes, "recursive_programming"
                     and "dynamic_programming". Default: "dynamic_programming".

                     - recursive_programming: Recursive programming search mode.

                     - dynamic_programming: Dynamic programming search mode.
        parameter_broadcast (bool): Whether to broadcast parameters before training. Before training, in order to have
                     the same network initialization parameter values for all devices, broadcast the parameters
                     on device 0 to other devices. Parameter broadcasting in different parallel modes is different,
                     data_parallel mode, all parameters are broadcast except for the parameter whose attribute
                     layerwise_parallel is True. Hybrid_parallel, semi_auto_parallel and auto_parallel mode, the
                     segmented parameters do not participate in broadcasting. Default: False.
        strategy_ckpt_load_file (str): The path to load parallel strategy checkpoint. Default: ''
        strategy_ckpt_save_file (str): The path to save parallel strategy checkpoint. Default: ''
        full_batch (bool): If you load whole batch datasets in auto_parallel mode, this parameter
                       should be set as True. Default: False. The interface is not be recommended currently,
                       it is better using 'dataset_strategy' to replace it.
        dataset_strategy (Union[str, tuple]): Dataset sharding strategy. Default: "data_parallel".
                       dataset_strategy="data_parallel" is equal to full_batch=False, dataset_strategy="full_batch" is
                       equal to full_batch=True. For dataset load into net by model parallel strategy likes
                       ds_stra ((1, 8), (1, 8)), it requires using set_auto_parallel_context(dataset_strategy=ds_stra).
        enable_parallel_optimizer (bool): This is a developing feature, which shards the weight update computation for
                       data parallel training in the benefit of time and memory saving. Currently, auto and semi auto
                       parallel mode support all optimizers in both Ascend and GPU. Data parallel mode only supports
                       `Lamb` and `AdamWeightDecay` in Ascend . Default: False.
        all_reduce_fusion_config (list): Set allreduce fusion strategy by parameters indices. Only support ReduceOp.SUM
                       and HCCL_WORLD_GROUP/NCCL_WORLD_GROUP. No Default, if it is not set, the fusion is closed.
        pipeline_stages (int): Set the stage information for pipeline parallel. This indicates how the devices are
                        distributed alone the pipeline. The total devices will be divided into 'pipeline_stags' stages.
                        Currently this could only be used when parallel mode semi_auto_parallel is enabled. Default: 1.
        grad_accumulation_step (int): Set the accumulation steps of gradients in auto and semi auto parallel mode.
                        This should be a positive int. Default: 1.

    Raises:
        ValueError: If input key is not attribute in auto parallel context.

    Examples:
        >>> context.set_auto_parallel_context(device_num=8)
        >>> context.set_auto_parallel_context(global_rank=0)
        >>> context.set_auto_parallel_context(gradients_mean=True)
        >>> context.set_auto_parallel_context(gradient_fp32_sync=False)
        >>> context.set_auto_parallel_context(parallel_mode="auto_parallel")
        >>> context.set_auto_parallel_context(auto_parallel_search_mode="dynamic_programming")
        >>> context.set_auto_parallel_context(parameter_broadcast=False)
        >>> context.set_auto_parallel_context(strategy_ckpt_load_file="./strategy_stage1.ckpt")
        >>> context.set_auto_parallel_context(strategy_ckpt_save_file="./strategy_stage1.ckpt")
        >>> context.set_auto_parallel_context(dataset_strategy=((1, 8), (1, 8)))
        >>> context.set_auto_parallel_context(enable_parallel_optimizer=False)
        >>> context.set_auto_parallel_context(all_reduce_fusion_config=[8, 160])
        >>> context.set_auto_parallel_context(pipeline_stages=2)
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
    """
    return _get_auto_parallel_context(attr_key)


def reset_auto_parallel_context():
    """
    Reset auto parallel context attributes to the default values:

    - device_num: 1.
    - global_rank: 0.
    - gradients_mean: False.
    - gradient_fp32_sync: True.
    - parallel_mode: 'stand_alone'.
    - auto_parallel_search_mode: 'dynamic_programming'.
    - parameter_broadcast: False.
    - strategy_ckpt_load_file: ''.
    - strategy_ckpt_save_file: ''.
    - full_batch: False.
    - enable_parallel_optimizer: False.
    - pipeline_stages: 1.
    """
    _reset_auto_parallel_context()


def _check_target_specific_cfgs(device, arg_key):
    """Checking whether a config is suitable for a specified device"""
    device_cfgs = {
        'enable_dump': ['Ascend'],
        'save_dump_path': ['Ascend'],
        'enable_graph_kernel': ['Ascend', 'GPU'],
        'graph_kernel_flags': ['Ascend', 'GPU'],
        'enable_reduce_precision': ['Ascend'],
        'enable_profiling': ['Ascend'],
        'profiling_options': ['Ascend'],
        'print_file_path': ['Ascend'],
        'variable_memory_max_size': ['Ascend'],
        'auto_tune_mode': ['Ascend'],
        'max_device_memory': ['GPU']
    }
    # configs not in map device_cfgs are supposed to be suitable for all devices
    if not arg_key in device_cfgs:
        return True
    supported_devices = device_cfgs[arg_key]
    if device in supported_devices:
        return True
    logger.warning(f"Config '{arg_key}' only supports devices in {supported_devices}, current device is '{device}'"
                   ", ignore it.")
    return False


@args_unreset_check(device_id=int, variable_memory_max_size=str, max_device_memory=str)
@args_type_check(mode=int, precompile_only=bool, device_target=str, device_id=int, save_graphs=bool,
                 save_graphs_path=str, enable_dump=bool, auto_tune_mode=str,
                 save_dump_path=str, enable_reduce_precision=bool, variable_memory_max_size=str,
                 enable_profiling=bool, profiling_options=str, enable_auto_mixed_precision=bool,
                 enable_graph_kernel=bool, reserve_class_name_in_scope=bool, check_bprop=bool,
                 max_device_memory=str, print_file_path=str, enable_sparse=bool, max_call_depth=int,
                 env_config_path=str, graph_kernel_flags=str, save_compile_cache=bool,
                 load_compile_cache=bool, grad_for_scalar=bool, pynative_synchronize=bool)
def set_context(**kwargs):
    """
    Set context for running environment.

    Context should be configured before running your program. If there is no configuration,
    it will be automatically set according to the device target by default.

    Note:
        Attribute name is required for setting attributes.
        The mode is not recommended to be changed after net was initialized because the implementations of some
        operations are different in graph mode and pynative mode. Default: GRAPH_MODE.

    Some configurations are device specific, see the below table for details:

    +-------------------------+------------------------------+----------------------------+
    | Function Classification |   Configuration Parameters   |   Hardware Platform Support|
    +=========================+==============================+============================+
    | System Configuration    |   device_id                  |   CPU/GPU/Ascend           |
    |                         +------------------------------+----------------------------+
    |                         |   device_target              |   CPU/GPU/Ascend           |
    |                         +------------------------------+----------------------------+
    |                         |  max_device_memory           |  GPU                       |
    |                         +------------------------------+----------------------------+
    |                         |  variable_memory_max_size    |  Ascend                    |
    +-------------------------+------------------------------+----------------------------+
    | Debug Configuration     |  save_graphs                 |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  save_graphs_path            |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  enable_dump                 |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  save_dump_path              |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  enable_profiling            |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  profiling_options           |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  print_file_path             |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  env_config_path             |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  precompile_only             |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  reserve_class_name_in_scope |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  pynative_synchronize        |  GPU/Ascend                |
    +-------------------------+------------------------------+----------------------------+
    | Executive Control       |   mode                       |   CPU/GPU/Ascend           |
    |                         +------------------------------+----------------------------+
    |                         |  enable_graph_kernel         |  Ascend/GPU                |
    |                         +------------------------------+----------------------------+
    |                         |  graph_kernel_flags          |  Ascend/GPU                |
    |                         +------------------------------+----------------------------+
    |                         |  enable_reduce_precision     |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  auto_tune_mode              |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  check_bprop                 |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  max_call_depth              |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  enable_sparse               |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  grad_for_scalar             |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  save_compile_cache          |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  load_compile_cache          |  CPU/GPU/Ascend            |
    +-------------------------+------------------------------+----------------------------+

    Args:
        device_id (int): ID of the target device, the value must be in [0, device_num_per_host-1],
            while device_num_per_host should be no more than 4096. Default: 0.
        device_target (str): The target device to run, support "Ascend", "GPU", and "CPU".
            If device target is not set, the version of MindSpore package is used.
        max_device_memory (str): Set the maximum memory available for devices.
            Currently, it is only supported on GPU. The format is "xxGB". Default: "1024GB".
            The actual used memory size is the minimum of the available memory of the device and max_device_memory.
        variable_memory_max_size (str): Set the maximum size of the variable memory max size. Default: "30GB".
            After this parameter is set, the maximum memory used by the framework is restricted to the configured value.
        save_graphs (bool): Whether to save graphs. Default: False.
            When the `save_graphs` attribute is set as True, attribute of `save_graphs_path` is used to set the
            intermediate compilation graph storage path. By default, the graphs are saved in the current directory.
        save_graphs_path (str): Path to save graphs. Default: ".".
            If the specified directory does not exist, the system will automatically create the directory.
            During distributed training, graphs will be saved to the directory of
            `save_graphs_path/rank_${rank_id}/`. `rank_id` is the ID of the current device in the cluster.
        enable_dump (bool): Whether to enable dump on Ascend. Default: False.
        save_dump_path (str): When the program is executed on Ascend, operators can dump data in this path.
            The root dump path is configured in /home/HwHiAiUser/ide_daemon/ide_daemon.cfg.
            So the real dump path is "{configured root dump path}/{`save_dump_path`}". Default: ".".
        enable_profiling (bool): This parameters is deprecated, and will be deleted in the next version.
            Please use mindspore.profiler.Profiler api instead.
        profiling_options (str): This parameters is deprecated, and will be deleted in the next version.
            Please use mindspore.profiler.Profiler api instead.
        print_file_path (str): The path of saving print data. If this parameter is set, print data is saved to
            a file by default, and print_file_path is not set, the screen will be displayed.
            If the saved file already exists, the timestamp suffix will be added to the file. Saving data to a file
            solves the problem of data loss in screen printing when a large amount of data is generated.
            If it is not set, an error will be reported: prompt to set the upper absolute path.
        env_config_path (str): Config path for DFX.
            Through context.set_context(env_config_path="./mindspore_config.json")

            configure RDR:

            - enable: controls whether the RDR is enabled to collect the key data during training and
              save key data in the fault scenario. When set to true, the RDR will be turned on.
              When set to false, the RDR will be turned off.
            - path: sets the path where RDR saves data. The current path must be absolute.

            Memory reuse:

            - mem_Reuse: controls whether the memory reuse function is turned on. When set to True,
            - the memory reuse function is turned on. When set to False, the memory reuse function is turned off.

        precompile_only (bool): Whether to only precompile the network. Default: False.
            If set to True, the network will only be compiled, not executed.
        reserve_class_name_in_scope (bool) : Whether to save the network class name in the scope. Default: True.
            Each node has a scope. A scope of a subnode is the name of its parent node. If reserve_class_name_in_scope
            is set, the class name will be saved after keyword 'net-' in the scope.
            For example:

            Default/net-Net1/net-Net2 (reserve_class_name_in_scope=True)

            Default/net/net (reserve_class_name_in_scope=False)

        pynative_synchronize (bool): Whether to enable synchronous execution of the device in PyNative mode.
            Default: False. When the value is set to False, the operator is executed asynchronously on the device.
            When an error occurs in the execution of the operator, the specific error script code location cannot
            be located, when the value is set to True, the operator is executed synchronously on the device. It will
            reduce the execution performance of the program. At this time, when an error occurs in the execution of
            the operator, the location of the error script code can be located according to the call stack of the error.
        mode (int): Running in GRAPH_MODE(0) or PYNATIVE_MODE(1). Default: GRAPH_MODE(0).
            GRAPH_MODE or PYNATIVE_MODE can be set by `mode` attribute and both modes support all backends, default
            mode is GRAPH_MODE.
        enable_graph_kernel (bool): Whether to enable graph kernel fusion to optimize network execution performance.
            Default: False.
            Indicates whether to enable image-computing convergence to optimize network execution performance.
            If enable_graph_kernel is set to True, acceleration can be enabled.
            For details of sparsity and sparse tensor, please check
            `Enabling Graph-Accounting Convergence <https://www.mindspore.cn/docs/programming_guide
            /en/master/enable_graph_kernel_fusion.html>`_.
        graph_kernel_flags (str) –
            Optimization options of graph kernel fusion, and the priority is higher when it conflicts
            with enable_graph_kernel. Experienced user only.
            For example, context.set_context(graph_kernel_flags="–opt_level=2 –dump_as_text"). Some general options:

            - opt_level: Set the optimization level.
              Default: 2. Graph kernel fusion can be enabled equivalently by setting opt_level greater than 0.
              Available values are:

              - 0: Disable graph kernel fusion;
              - 1: enable the basic fusion of operators;
              - 2: includes all optimizations of level 1,
                and turns on more optimizations such as CSE, arithmetic simplication and so on;
              - 3: includes all optimizations of level 2, and turns on more optimizations such as SitchingFusion,
                ParallelFusion and so on. Optimizations of this level are radical and unstable in some scenarios.
                Be caution when using this level.

            - dump_as_text: dump detail info as text files. Default: false.

            More options can refer to the implementation code. These options can also be set by environment
            variable MS_GRAPH_KERNEL_FLAGS, without modifying network source code.
            For example, export MS_GRAPH_KERNEL_FLAGS="–opt_level=2 –dump_as_text".
        enable_reduce_precision (bool): Whether to enable precision reduction. Default: True.
        auto_tune_mode (str): The mode of auto tune when op building, get the best tiling performance.
            Default: NO_TUNE. The value must be in ['RL', 'GA', 'RL,GA'].

            - RL: Reinforcement Learning tune.
            - GA: Genetic Algorithm tune.
            - RL,GA: When both RL and GA optimization are enabled, the tool automatically selects RL or GA based on
              different types of operators in the network model. The sequence of RL and GA is not differentiated.
              (Automatic selection).

            For more information about the enable operator tuning tool settings, please check
            `Enable the operator optimization tool <https://www.mindspore.cn/docs/programming_guide/en
            /master/enable_auto_tune.html>`_.
        check_bprop (bool): Whether to check back propagation nodes. The checking ensures that the shape and dtype
            of back propagation node outputs is the same as input parameters. Default: False.
        max_call_depth (int): Specify the maximum depth of function call. Must be positive integer. Default: 1000.
            The max_call_depth parameter needs to be set when the nested call is too deep or the number
            of subgraphs is too large. If max_call_depth is set larger than before, the system max stack depth should be
            set larger too, otherwise a `core dumped` exception may be raised because of system stack overflow.
        enable_sparse (bool): Whether to enable sparsity feature. Default: False.
            For details of sparsity and sparse tensor, please check
            `sparse tensor <https://www.mindspore.cn/docs/programming_guide/en/master/tensor.html#sparse-tensor>`_.
        grad_for_scalar (bool):  Whether to get gradient for scalar. Default: False.
            When grad_for_scalar is set to True, the function's scalar input can be derived.
            The default value is False. Because the back-end does not support scaling operations currently,
            this interface only supports simple operations that can be deduced by the front-end.
        save_compile_cache (bool): Whether to cache the graph compiled by front-end. Default: False.
            After save_compile_cache is set to True, a hardware-independent compilation cache is
            generated and exported to a MINDIR file, This is an experimental prototype that is
            subject to change and/or deletion.
        load_compile_cache (bool): Whether to use the cache of the graph compiled by front-end.
            This parameter must be used together with save_compile_cache. After save_compile_cache is set to True,
            a hardware-independent compilation cache is generated and exported to a MINDIR file.
            When the network is executed again, if load_compile_cache is set to True, the compile cache is loaded.
            By now, we do not support automatic checking for changes.
            Default: False.
            This is an experimental prototype that is subject to change and/or deletion.
    Raises:
        ValueError: If input key is not an attribute in context.

    Examples:
        >>> context.set_context(mode=context.PYNATIVE_MODE)
        >>> context.set_context(precompile_only=True)
        >>> context.set_context(device_target="Ascend")
        >>> context.set_context(device_id=0)
        >>> context.set_context(save_graphs=True, save_graphs_path="./model.ms")
        >>> context.set_context(enable_reduce_precision=True)
        >>> context.set_context(enable_dump=True, save_dump_path=".")
        >>> context.set_context(enable_graph_kernel=True)
        >>> context.set_context(graph_kernel_flags="--opt_level=2 --dump_as_text")
        >>> context.set_context(reserve_class_name_in_scope=True)
        >>> context.set_context(variable_memory_max_size="6GB")
        >>> context.set_context(enable_profiling=True,
        ...                     profiling_options='{"output":"/home/data/output","training_trace":"on"}')
        >>> context.set_context(check_bprop=True)
        >>> context.set_context(max_device_memory="3.5GB")
        >>> context.set_context(print_file_path="print.pb")
        >>> context.set_context(enable_sparse=True)
        >>> context.set_context(max_call_depth=80)
        >>> context.set_context(env_config_path="./env_config.json")
        >>> context.set_context(auto_tune_mode="GA,RL")
        >>> context.set_context(grad_for_scalar=True)
        >>> context.set_context(save_compile_cache=True)
        >>> context.set_context(load_compile_cache=True)
        >>> context.set_context(pynative_synchronize=True)
    """
    ctx = _context()
    # set device target first
    if 'device_target' in kwargs:
        ctx.set_device_target(kwargs['device_target'])
        device = ctx.get_param(ms_ctx_param.device_target)
        if not device.lower() in __device_target__:
            raise ValueError(f"Error, package type {__package_name__} support device type {__device_target__}, "
                             f"but got device target {device}")
    device = ctx.get_param(ms_ctx_param.device_target)
    for key, value in kwargs.items():
        if key == "enable_auto_mixed_precision":
            logger.warning(f" '{key}' mixing accuracy is controlled by amp, '{key}' will be deleted later.")
            continue
        if key in ('enable_profiling', 'profiling_options'):
            logger.warning(f" '{key}' is deprecated. Please use Profiler instead. The parameter will"
                           "be deleted in the next version.")
            continue
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
        raise ValueError("Set context keyword %s is not recognized!" % key)


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
    """
    ctx = _context()
    device = ctx.get_param(ms_ctx_param.device_target)
    _ = _check_target_specific_cfgs(device, attr_key)
    if hasattr(ctx, attr_key):
        return getattr(ctx, attr_key)
    # enum variables beginning with '_' are for internal use
    if attr_key in ms_ctx_param.__members__ and attr_key[0] != '_':
        return ctx.get_param(ms_ctx_param.__members__[attr_key])
    raise ValueError("Get context keyword %s is not recognized!" % attr_key)


class ParallelMode:
    """
    Parallel mode options.

    There are five kinds of parallel modes, "STAND_ALONE", "DATA_PARALLEL",
    "HYBRID_PARALLEL", "SEMI_AUTO_PARALLEL" and "AUTO_PARALLEL". Default: "STAND_ALONE".

    - STAND_ALONE: Only one processor is working.
    - DATA_PARALLEL: Distributes the data across different processors.
    - HYBRID_PARALLEL: Achieves data parallelism and model parallelism manually.
    - SEMI_AUTO_PARALLEL: Achieves data parallelism and model parallelism by setting parallel strategies.
    - AUTO_PARALLEL: Achieves parallelism automatically.

    MODE_LIST: The list of all supported parallel modes.
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
        Some other environment variables should also be set for parameter server training mode.
        These environment variables are listed below:

    - MS_SERVER_NUM: Server number
    - MS_WORKER_NUM: Worker number
    - MS_SCHED_HOST: Scheduler IP address
    - MS_SCHED_PORT: Scheduler port
    - MS_ROLE: The role of this process:
    - MS_SCHED: represents the scheduler,
    - MS_WORKER: represents the worker,
    - MS_PSERVER: represents the Server

    Args:
        enable_ps (bool): Whether to enable parameter server training mode.
                          Only after enable_ps is set True, the environment variables will be effective.
                          Default: False.

    Raises:
        ValueError: If input key is not the attribute in parameter server training mode context.

    Examples:
        >>> context.set_ps_context(enable_ps=True)
    """
    _set_ps_context(**kwargs)


def get_ps_context(attr_key):
    """
    Get parameter server training mode context attribute value according to the key.

    Args:
        attr_key (str): The key of the attribute:
            - enable_ps (bool): Whether to enable parameter server training mode.

    Returns:
        Returns attribute value according to the key.

    Raises:
        ValueError: If input key is not attribute in auto parallel context.
    """
    return _get_ps_context(attr_key)


def reset_ps_context():
    """
    Reset parameter server training mode context attributes to the default values:

    - enable_ps: False.
    """
    _reset_ps_context()


def set_fl_context(**kwargs):
    """
    Set federated learning training mode context.

    Args:
        enable_fl (bool): Whether to enable federated learning training mode.
                          Default: False.
        server_mode (str): Describe the server mode, which must one of 'FEDERATED_LEARNING' and 'HYBRID_TRAINING'.
                              Default: 'FEDERATED_LEARNING'.
        ms_role (str): The process's role in the federated learning mode,
                          which must be one of 'MS_SERVER', 'MS_WORKER' and 'MS_SCHED'.
                          Default: 'MS_SERVER'.
        worker_num (int): The number of workers. For current version, this must be set to 1 or 0.
        server_num (int): The number of federated learning servers. Default: 0.
        scheduler_ip (str): The scheduler IP. Default: '0.0.0.0'.
        scheduler_port (int): The scheduler port. Default: 6667.
        fl_server_port (int): The http port of the federated learning server.
                              Normally for each server this should be set to the same value. Default: 6668.
        enable_fl_client (bool): Whether this process is federated learning client. Default: False.
        start_fl_job_threshold (int): The threshold count of startFLJob. Default: 1.
        start_fl_job_time_window (int): The time window duration for startFLJob in millisecond. Default: 3000.
        share_secrets_ratio (float): The ratio for computing the threshold count of share secrets. Default: 1.0.
        update_model_ratio (float): The ratio for computing the threshold count of updateModel. Default: 1.0.
        cipher_time_window (int): The time window duration for each cipher round in millisecond. Default: 300000.
        reconstruct_secrets_threshold (int): The threshold count of reconstruct threshold. Default: 0.
        update_model_time_window (int): The time window duration for updateModel in millisecond. Default: 3000.
        fl_name (string): The federated learning job name. Default: ''.
        fl_iteration_num (int): Iteration number of federated learning,
                                which is the number of interactions between client and server. Default: 20.
        client_epoch_num (int): Client training epoch number. Default: 25.
        client_batch_size (int): Client training data batch size. Default: 32.
        client_learning_rate (float): Client training learning rate. Default: 0.001.
        worker_step_num_per_iteration (int): The worker's standalone training step number before communicating with
                                             server. Default: 65.
        dp_eps (float): Epsilon budget of differential privacy mechanism. The smaller the dp_eps, the better the
            privacy protection effect. Default: 50.0.
        dp_delta (float): Delta budget of differential privacy mechanism, which is usually equals the reciprocal of
            client number. The smaller the dp_delta, the better the privacy protection effect. Default: 0.01.
        dp_norm_clip (float): A factor used for clipping model's weights for differential mechanism. Its value is
            suggested to be 0.5~2. Default: 1.0.
        encrypt_type (string): Secure schema for federated learning, which can be 'NOT_ENCRYPT', 'DP_ENCRYPT' or
            'PW_ENCRYPT'. If 'DP_ENCRYPT', differential privacy schema would be applied for clients and the privacy
            protection effect would be determined by dp_eps, dp_delta and dp_norm_clip as described above. If
            'PW_ENCRYPT', pairwise secure aggregation would be applied to protect clients' model from stealing.
            Default: 'NOT_ENCRYPT'.
        config_file_path (string): Configuration file path used by recovery. Default: ''.
        scheduler_manage_port (int): scheduler manage port used to scale out/in. Default: 11202.
        enable_ssl (bool): Set PS SSL mode enabled or disabled. Default: true.
        client_password (str): Password to decrypt the secret key stored in the client certificate.
        server_password (str): Password to decrypt the secret key stored in the server certificate.

    Raises:
        ValueError: If input key is not the attribute in federated learning mode context.

    Examples:
        >>> context.set_fl_context(enable_fl=True, server_mode='FEDERATED_LEARNING')
    """
    _set_ps_context(**kwargs)


def get_fl_context(attr_key):
    """
    Get federated learning mode context attribute value according to the key.

    Args:
        attr_key (str): The key of the attribute.
                        Please refer to `set_fl_context`'s parameters to decide what key should be passed.

    Returns:
        Returns attribute value according to the key.

    Raises:
        ValueError: If input key is not attribute in federated learning mode context.

    Examples:
        >>> context.get_fl_context("server_mode")
    """
    return _get_ps_context(attr_key)
