# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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
"""Providing interface methods."""
import types
import sys
import os
import time
import ast
import importlib
from collections import OrderedDict
from functools import wraps

from mindspore import context
from mindspore import log as logger
from mindspore._extends.remote import kernel_build_server
from .tensor import Tensor as MsTensor
from .tensor import CSRTensor as MsCSRTensor
from .._c_expression import GraphExecutor_, Tensor, MetaTensor, CSRTensor, PynativeExecutor_, _ms_memory_recycle
from .._c_expression import verify_inputs_signature, init_exec_dataset, _set_dataset_mode_config, init_pipeline
from ..parallel._ps_context import _is_role_pserver, _is_role_sched
from ..parallel._utils import _get_device_num, _get_global_rank, _need_to_full, _check_full_batch, _to_full_tensor, \
    _get_parameter_broadcast, _get_pipeline_stages
from .._checkparam import Validator

# store ms_function class compiled pipeline cache
ms_compile_cache = set()
# store cell compiled pipeline cache,
# {cell1:set_cache1, cell2:set_cache2, ...}
cells_compile_cache = {}

BROADCAST_PHASE = "_broadcast_"


def _wrap_func(fn):
    """
    Wrapper function, convert return data to tensor or tuple of tensor.

    Args:
        fn (Function): The function need be wrapped.

    Returns:
        Function, a new function with return suitable format data.
    """

    @wraps(fn)
    def wrapper(*arg, **kwargs):
        results = fn(*arg, **kwargs)

        def _convert_data(data):
            if isinstance(data, Tensor) and not isinstance(data, MsTensor):
                return MsTensor(data)
            if isinstance(data, CSRTensor) and not isinstance(data, MsCSRTensor):
                return MsCSRTensor(csr_tensor=data)
            if isinstance(data, tuple):
                return tuple(_convert_data(x) for x in data)
            if isinstance(data, list):
                return list(_convert_data(x) for x in data)
            return data

        return _convert_data(results)

    return wrapper


def _exec_init_graph(obj, init_phase):
    """Execute the parameter initializer graph."""
    inst_executor = GraphExecutor_.get_instance()
    param_dict = OrderedDict()
    for name, param in obj.parameters_dict().items():
        if not param.is_init:
            param_dict[name] = param
            param.is_init = True
            param.data.init_flag = True

    if param_dict:
        inst_executor.run_init_graph(param_dict, init_phase)


def _check_all_tensor(sequence):
    for element in sequence:
        if not isinstance(element, Tensor) and not (isinstance(element, tuple) and _check_all_tensor(element)):
            return False
    return True


sys_path = list(sys.path)
# Get the entry script path.
if sys.argv and sys.argv[0] != '':
    entry_script_path = os.path.realpath(sys.argv[0])
    entry_script_path_dir = os.path.split(entry_script_path)[0]
    if entry_script_path_dir in sys_path:
        sys_path.remove(entry_script_path_dir)


def _in_sys_path(file_path):
    for path in sys_path:
        if file_path.startswith(path):
            return True
    return False


def __get_compile_cache_dep_files(file_path, compile_cache_dep_files, pkg):
    """Get the dependency files of the network"""
    with open(file_path) as fh:
        root = ast.parse(fh.read(), file_path)
    for node in ast.iter_child_nodes(root):
        module_name = ""
        if isinstance(node, ast.ImportFrom):
            module_name = node.module
            if node.level == 1:
                module_name = "." + module_name
        elif not isinstance(node, ast.Import):
            continue
        # Do not care the files in mindspore package
        if module_name.startswith("mindspore"):
            continue

        for n in node.names:
            if n.name.startswith("mindspore"):
                continue
            if module_name == "":
                whole_module = n.name
            else:
                whole_module = module_name
                if not n.name is None:
                    whole_module += "." + n.name
            try:
                module_spec = importlib.util.find_spec(whole_module, pkg)
            except (ModuleNotFoundError, ValueError):
                whole_module = whole_module[0:whole_module.rfind('.')]
                module_spec = importlib.util.find_spec(whole_module, pkg)
            if module_spec is None:
                continue
            module = importlib.util.module_from_spec(module_spec)
            if hasattr(module, '__file__'):
                dep_file_path = module.__file__
            else:
                continue
            # Exclude the installed modules.
            if not _in_sys_path(dep_file_path) and not dep_file_path in compile_cache_dep_files:
                logger.debug(f"dependent file path: {dep_file_path}")
                compile_cache_dep_files.append(dep_file_path)
                __get_compile_cache_dep_files(dep_file_path, compile_cache_dep_files, module.__package__)


def _get_compile_cache_dep_files():
    """Get the dependency files of the network"""
    if entry_script_path is None:
        logger.warning("Can not get the entry script file path.")
        return []
    compile_cache_dep_files = []
    logger.debug(f"entry script file path: {entry_script_path}")
    compile_cache_dep_files.append(entry_script_path)
    __get_compile_cache_dep_files(entry_script_path, compile_cache_dep_files, None)
    return compile_cache_dep_files


class _MindsporeFunctionExecutor:
    """
    Represents a function compiled by graph compiler.

    _MindsporeFunctionExecutor will compile the original function for every combination
    of argument types and shapes it is given (as well as their values, optionally).

    Args:
        fn (Function): The root function to compile.
        input_signature (Function): User defines signature to verify input.
        ms_create_time(TimeStamp): The time ms_function created
        obj (Object): If function is a method, obj is the owner of function,
             else, obj is none.

    Returns:
        The result of pipeline running in graph mode.
    """

    def __init__(self, fn, ms_create_time, input_signature=None, obj=None):
        init_pipeline()
        if not isinstance(fn, (types.FunctionType, types.MethodType)):
            raise RuntimeError('fn {} is not function or method'.format(fn))

        self.fn = fn
        self.input_signature = input_signature
        self.obj = None
        if obj and hasattr(obj, fn.__name__):
            self.obj = obj
        self._graph_executor = GraphExecutor_.get_instance()
        self._create_time = ms_create_time

    def build_data_init_graph(self, graph_name):
        """Build GE data graph and init graph for the given graph name."""
        if self.obj is None:
            logger.warning("Make sure parameter should not be used in function")
            para_dict = OrderedDict()
            self._graph_executor.build_data_graph(para_dict, graph_name)
            return
        self._graph_executor.build_data_graph(self.obj.parameters_dict(), graph_name,
                                              self.obj.parameters_broadcast_dict())
        init_phase = "init_subgraph" + graph_name[graph_name.find("."):]
        _exec_init_graph(self.obj, init_phase)

    def compile(self, args_list, method_name):
        """Returns pipeline for the given args."""
        # Verify the signature for both function and method
        if self.input_signature is not None:
            signatures = []
            for sig_spec in self.input_signature:
                if not isinstance(sig_spec, MetaTensor):
                    raise TypeError("Input_signature is not MetaTensor")
                signatures.append(sig_spec)
            is_valid_input = verify_inputs_signature(signatures, args_list)
            if not is_valid_input:
                raise ValueError("Inputs is incompatible with input signature!")

        generate_name = self.fn.__module__ + "." + self.fn.__name__ + "." + self.fn.__code__.co_filename + "." + \
                        str(self.fn.__code__.co_firstlineno) + '.' + str(id(self.fn))
        if _pynative_executor.grad_flag():
            generate_name = generate_name + ".grad"
        self.fn.__parse_method__ = method_name

        # Add key with obj
        if self.obj is not None:
            if self.obj.__module__ != self.fn.__module__:
                logger.error(f'`obj` module not equal to `fn` module: {self.obj.__module__}, {self.fn.__module__}')
            self.obj.__parse_method__ = method_name
            generate_name = generate_name + '.' + str(self.obj.create_time) + '.' + str(id(self.obj))
        else:
            # Different instance of same class may use same memory(means same obj_id) at diff times.
            # To avoid unexpected phase matched, add create_time to generate_name.
            generate_name = generate_name + '.' + str(self._create_time)

        if hasattr(self.obj, "enable_tuple_broaden"):
            self.enable_tuple_broaden = self.obj.enable_tuple_broaden
        else:
            self.enable_tuple_broaden = False
        self._graph_executor.set_enable_tuple_broaden(self.enable_tuple_broaden)
        key = self._graph_executor.generate_arguments_key(args_list, self.enable_tuple_broaden)
        phase = generate_name + '.' + str(key)
        if phase in ms_compile_cache:
            return phase

        if self.obj is None:
            is_compile = self._graph_executor.compile(self.fn, args_list, phase, True)
        else:
            self._graph_executor.set_weights_values(self.obj.parameters_dict())
            is_compile = self._graph_executor.compile(self.obj, args_list, phase, True)
        if not is_compile:
            raise RuntimeError("Executor compile failed.")
        if context.get_context("enable_ge"):
            self.build_data_init_graph(phase)
        ms_compile_cache.add(phase)
        return phase

    @_wrap_func
    def __call__(self, *args):
        args_list = args
        if self.obj is not None:
            args_list = args_list[1:]

        phase = self.compile(args_list, self.fn.__name__)

        if context.get_context("precompile_only"):
            return None
        new_inputs = []
        for i in args_list:
            if isinstance(i, (Tensor, CSRTensor)):
                new_inputs.append(i)
            elif context.get_context("grad_for_scalar") and isinstance(i, (int, float)):
                new_inputs.append(i)
            elif self.enable_tuple_broaden and isinstance(i, tuple) and _check_all_tensor(i):
                new_inputs.append(i)
        output = self._graph_executor(tuple(new_inputs), phase)
        if context.get_context("mode") == context.PYNATIVE_MODE:
            _pynative_executor.set_graph_phase(phase)
            output = _pynative_executor.grad_ms_function(output, *new_inputs)

        return output


def ms_function(fn=None, obj=None, input_signature=None):
    """
    Create a callable MindSpore graph from a Python function.

    This allows the MindSpore runtime to apply optimizations based on graph.

    Args:
        fn (Function): The Python function that will be run as a graph. Default: None.
        obj (Object): The Python object is used to distinguish the compiled function. Default: None.
        input_signature (Tensor): The Tensor which describes the input arguments. The shape and dtype of the Tensor
            will be supplied to this function. If input_signature is specified, each input to `fn` must be a `Tensor`.
            And the input parameters of `fn` cannot accept `**kwargs`. The shape and dtype of actual inputs should
            keep the same as input_signature. Otherwise, TypeError will be raised. Default: None.

    Returns:
        Function, if `fn` is not None, returns a callable function that will execute the compiled function; If `fn` is
        None, returns a decorator and when this decorator invokes with a single `fn` argument, the callable function is
        equal to the case when `fn` is not None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import ms_function
        ...
        >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        ...
        >>> # create a callable MindSpore graph by calling ms_function
        >>> def tensor_add(x, y):
        ...     z = x + y
        ...     return z
        ...
        >>> tensor_add_graph = ms_function(fn=tensor_add)
        >>> out = tensor_add_graph(x, y)
        ...
        >>> # create a callable MindSpore graph through decorator @ms_function
        >>> @ms_function
        ... def tensor_add_with_dec(x, y):
        ...     z = x + y
        ...     return z
        ...
        >>> out = tensor_add_with_dec(x, y)
        ...
        >>> # create a callable MindSpore graph through decorator @ms_function with input_signature parameter
        >>> @ms_function(input_signature=(Tensor(np.ones([1, 1, 3, 3]).astype(np.float32)),
        ...                               Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))))
        ... def tensor_add_with_sig(x, y):
        ...     z = x + y
        ...     return z
        ...
        >>> out = tensor_add_with_sig(x, y)
    """

    def wrap_mindspore(func):
        ms_create_time = int(time.time() * 1e9)

        @wraps(func)
        def staging_specialize(*args):
            if obj is not None:
                logger.warning("Obj is no longer in use, and the function's own object has been used to \
                                distinguish whether it has been compiled.")
            process_obj = None
            if args and not isinstance(args[0], MsTensor) and hasattr(args[0], func.__name__):
                process_obj = args[0]
            out = _MindsporeFunctionExecutor(func, ms_create_time, input_signature, process_obj)(*args)
            return out

        return staging_specialize

    if fn is not None:
        return wrap_mindspore(fn)
    return wrap_mindspore


def _get_auto_split_param_names(parameter_layout_dict):
    auto_split_param_names = []
    for key, value in parameter_layout_dict.items():
        for dim in value[1]:
            if dim != -1:
                auto_split_param_names.append(key)
                break
    return auto_split_param_names


def _build_broadcast_graph(broadcast_params_dict, broadcast_phase):
    """Build broadcast graph."""
    from mindspore.nn.wrap.cell_wrapper import _BroadCastCell
    if not broadcast_params_dict:
        broadcast_params_dict = {}
    broadcast_params = []
    for param in broadcast_params_dict.values():
        broadcast_params.append(Tensor(param.asnumpy()))
    _broadcast_net = _BroadCastCell(broadcast_params)
    _broadcast_net.phase = broadcast_phase
    broadcasted_params = _broadcast_net()
    for param_name, param in zip(broadcast_params_dict.keys(), broadcasted_params):
        broadcast_params_dict[param_name].set_data(param)


def _parameter_broadcast(obj, auto_parallel_mode):
    """Parameter broadcast."""
    auto_split_param_names = []
    if auto_parallel_mode:
        auto_split_param_names = _get_auto_split_param_names(obj.parameter_layout_dict)

    broadcast_params_dict = obj.parameters_broadcast_dict()
    if auto_split_param_names and broadcast_params_dict:
        broadcast_params_dict = OrderedDict()
        for param_name, param in obj.parameters_broadcast_dict().items():
            if param_name not in auto_split_param_names:
                broadcast_params_dict[param_name] = param
    broadcast_phase = "_broadcast_subgraph"
    _build_broadcast_graph(broadcast_params_dict, broadcast_phase)


class _PynativeExecutor:
    """
    A pynative executor used to compile/manage/run single op.

    The main functions include: construct op graph, compile op graph, auto grad and run op graph.

    Args:
        obj (Object): The python network that will be run in pynative mode.
        args (Tuple(Tensor...)): The inputs of network in tuple form.

    Returns:
        gradients (Tuple(Tensor...)): The gradients of network parameters and inputs.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self):
        self._executor = PynativeExecutor_.get_instance()
        self._executor.set_py_exe_path(sys.executable)
        self._executor.set_kernel_build_server_dir(os.path.split(kernel_build_server.__file__)[0] + os.sep)

    def new_graph(self, obj, *args, **kwargs):
        self._executor.new_graph(obj, *args, *(kwargs.values()))

    def end_graph(self, obj, output, *args, **kwargs):
        self._executor.end_graph(obj, output, *args, *(kwargs.values()))

    def check_graph(self, obj, *args, **kwargs):
        return self._executor.check_graph(obj, *args, *(kwargs.values()))

    def check_run(self, grad, obj, *args, **kwargs):
        return self._executor.check_run(grad, obj, *args, *(kwargs.values()))

    def set_grad_position(self, grad, grad_position):
        return self._executor.set_grad_position(grad, grad_position)

    def grad(self, grad, obj, weights, grad_position, *args, **kwargs):
        self._executor.grad_net(grad, obj, weights, grad_position, *args, *(kwargs.values()))

    def del_cell(self, cell_id=""):
        self._executor.clear_cell(cell_id)

    def clear_res(self):
        return self._executor.clear_res()

    def clear_grad(self, obj, *args, **kwargs):
        self._executor.clear_grad(obj, *args, *(kwargs.values()))

    def sync(self):
        self._executor.sync()

    def set_lazy_build(self, enable):
        self._executor.set_lazy_build(enable)

    def execute_all_task(self):
        self._executor.execute_all_task()

    def grad_ms_function(self, output, *args):
        return self._executor.grad_ms_function(output, *args)

    def set_graph_phase(self, phase):
        self._executor.set_graph_phase(phase)

    def grad_flag(self):
        return self._executor.grad_flag()

    def set_grad_flag(self, flag):
        self._executor.set_grad_flag(flag)

    def parameter_broadcast(self, obj, phase, auto_parallel_mode):
        if BROADCAST_PHASE not in phase and _get_parameter_broadcast():
            _parameter_broadcast(obj, auto_parallel_mode)

    def enter_cell(self):
        self._executor.enter_cell()

    def exit_cell(self):
        self._executor.exit_cell()

    def is_top_cell(self):
        return self._executor.is_top_cell()

    def __call__(self, obj, *args, **kwargs):
        args = args + tuple(kwargs.values())
        return self._executor(obj, args)


class _CellGraphExecutor:
    """
    An executor used to compile/manage/run graph for a Cell.

    Including data_graph, train_graph, eval_graph and predict graph.

    Args:
        obj (Function/Cell): The function or cell instance need compile.
        args (tuple): Function or cell input arguments.

    Returns:
        Graph, return the result of pipeline running.
    """

    VALID_JIT_CONFIG_PARAM = ["jit_level"]
    VALID_JIT_CONFIG_PARAM_VALUE = {
        "jit_level": ["o0", "o1"]
    }

    def __init__(self):
        # create needed graph by lazy mode
        self.is_init = False
        self._graph_executor = GraphExecutor_.get_instance()
        self._graph_executor.set_py_exe_path(sys.executable)
        self._graph_executor.set_kernel_build_server_dir(os.path.split(kernel_build_server.__file__)[0] + os.sep)

    def init_dataset(self, queue_name, dataset_size, batch_size, dataset_types, dataset_shapes,
                     input_indexs, phase='dataset'):
        """
        Initialization interface for calling data subgraph.

        Args:
            queue_name (str): The name of tdt queue on the device.
            dataset_size (int): The size of dataset.
            batch_size (int): The size of batch.
            dataset_types (list): The output types of element in dataset.
            dataset_shapes (list): The output shapes of element in dataset.
            input_indexs (list): The index of data with net.
            phase (str): The name of phase, e.g., train_dataset/eval_dataset. Default: 'dataset'.

        Returns:
            bool, specifies whether the data subgraph was initialized successfully.
        """
        if not init_exec_dataset(queue_name=queue_name,
                                 size=dataset_size,
                                 batch_size=batch_size,
                                 types=dataset_types,
                                 shapes=dataset_shapes,
                                 input_indexs=input_indexs,
                                 phase=phase):
            raise RuntimeError("Failure to init and dataset subgraph!")
        self._graph_executor.set_queue_name(queue_name)
        return True

    def _build_data_graph(self, obj, phase):
        self._graph_executor.build_data_graph(obj.parameters_dict(), phase, obj.parameters_broadcast_dict())

    def set_queue_name(self, queue_name):
        """
        while a mode use shared dataset with others, need set queue_name which saved in data_set
        :param queue_name:
        :return:
        """
        self._graph_executor.set_queue_name(queue_name)

    def _set_dataset_mode(self, args_list):
        """set dataset mode."""
        # decide whether to sink based on whether the inputs is virtual or args_list is ()
        if (args_list and isinstance(args_list[0], Tensor) and args_list[0].virtual_flag) or \
                (args_list is not None and args_list == ()):
            _set_dataset_mode_config('sink')
        else:
            _set_dataset_mode_config('normal')

    @staticmethod
    def _use_vm_mode():
        enable_ge = context.get_context("enable_ge")
        enable_debug_runtime = context.get_context("enable_debug_runtime")
        exe_mode = context.get_context("mode") == context.PYNATIVE_MODE
        return not enable_ge or (enable_debug_runtime and exe_mode)

    def _set_compile_cache_dep_files(self, phase):
        # If enable compile cache, get the dependency files list
        enable_compile_cache = context.get_context("enable_compile_cache")
        if enable_compile_cache is None:
            enable_compile_cache = os.getenv('MS_COMPILER_CACHE_ENABLE')
        if "train" in phase and (enable_compile_cache is True or enable_compile_cache == "1"):
            self._graph_executor.set_compile_cache_dep_files(_get_compile_cache_dep_files())

    def compile(self, obj, *args, phase='predict', do_convert=True, auto_parallel_mode=False):
        """
        Compiles graph.

        Args:
            obj (Function/Cell): The function or cell instance need compile.
            args (tuple): Function or cell input arguments.
            phase (str): The name of compile phase. Default: 'predict'.
            do_convert (bool): When set to True, convert ME graph to GE graph after compiling graph.
            auto_parallel_mode: When set to True, use auto parallel mode to compile graph.

        Return:
            Str, the full phase of the cell.
            Bool, if the graph has been compiled before, return False, else return True.
        """
        obj.__parse_method__ = 'construct'
        if not hasattr(obj, obj.__parse_method__):
            raise AttributeError(
                'The class {} dose not have method {}'.format(obj.__class__.__name__, obj.__parse_method__))
        args_list = args
        if hasattr(obj, "enable_tuple_broaden"):
            self.enable_tuple_broaden = obj.enable_tuple_broaden
        else:
            self.enable_tuple_broaden = False
        self._graph_executor.set_enable_tuple_broaden(self.enable_tuple_broaden)
        key = self._graph_executor.generate_arguments_key(args_list, self.enable_tuple_broaden)
        obj.arguments_key = str(key)
        phase = phase + '.' + str(obj.create_time) + '.' + str(id(obj)) + '.' + obj.arguments_key

        if phase in obj.compile_cache and self.has_compiled(phase):
            logger.debug("%r graph has existed.", phase)
            return phase, False

        obj.check_names()
        _check_full_batch()
        self._set_dataset_mode(args_list)
        self._set_compile_cache_dep_files(phase)

        is_sink_mode = args and isinstance(args[0], Tensor) and args[0].virtual_flag
        if auto_parallel_mode and _need_to_full() and not is_sink_mode and obj.auto_parallel_compile_and_run():
            args_list = _to_full_tensor(args, _get_device_num(), _get_global_rank())

        enable_ge = context.get_context("enable_ge")
        self._graph_executor.set_weights_values(obj.parameters_dict())
        result = self._graph_executor.compile(obj, args_list, phase, self._use_vm_mode())
        obj.compile_cache.add(phase)
        if not result:
            raise RuntimeError("Executor compile failed.")
        graph = self._graph_executor.get_func_graph(phase)

        if graph is None:
            raise RuntimeError("Compile graph failed for phase {}.".format(phase))

        self._auto_parallel_process(obj, phase, is_sink_mode, auto_parallel_mode, *args)

        if not do_convert:
            return phase, True

        # the following GE init process is not needed when use vm or ms backend
        if enable_ge:
            self._build_data_graph(obj, phase)
            if "export" not in phase:
                init_phase = "init_subgraph." + str(obj.create_time) + "." + str(id(obj))
                _exec_init_graph(obj, init_phase)
        elif "export" in phase:
            self._build_data_graph(obj, phase)
        elif BROADCAST_PHASE not in phase and _get_parameter_broadcast():
            _parameter_broadcast(obj, auto_parallel_mode)

        return phase, True

    def _auto_parallel_process(self, obj, phase, is_sink_mode, auto_parallel_mode, *args):
        """compile graph in auto parallel mode."""
        if not auto_parallel_mode:
            replace = obj.init_parameters_data(auto_parallel_mode=auto_parallel_mode)
            self._update_param_node_default_input(phase, replace)
            return

        obj.parameter_layout_dict = self._graph_executor.get_parameter_layout(phase)
        obj.parallel_parameter_name_list = self._graph_executor.get_parallel_parameter_name_list(phase)
        replace = obj.init_parameters_data(auto_parallel_mode=True)
        if _get_pipeline_stages() > 1 and (not hasattr(obj, "is_first_iteration") or not obj.is_first_iteration):
            obj.remove_redundant_parameters()
        if not context.get_context("enable_debug_runtime") or context.get_context("enable_ge"):
            obj.load_parameter_slice(None)

        self._update_param_node_default_input(phase, replace)

        # set parallel inputs in sink mode
        if is_sink_mode:
            obj.set_parallel_input_with_inputs(*args)

    def _update_param_node_default_input(self, phase, replace):
        new_param = {x.name: replace[x] for x in replace if id(x) != id(replace[x])}
        return self._graph_executor.updata_param_node_default_input(phase, new_param)

    def _get_shard_strategy(self, obj):
        real_phase = obj.phase + '.' + str(obj.create_time) + '.' + str(id(obj)) + '.' + obj.arguments_key
        return self._graph_executor.get_strategy(real_phase)

    def _get_num_parallel_ops(self, obj):
        real_phase = obj.phase + '.' + str(obj.create_time) + '.' + str(id(obj)) + '.' + obj.arguments_key
        return self._graph_executor.get_num_parallel_ops(real_phase)

    def _get_allreduce_fusion(self, obj):
        real_phase = obj.phase + '.' + str(obj.create_time) + '.' + str(id(obj)) + '.' + obj.arguments_key
        return self._graph_executor.get_allreduce_fusion(real_phase)

    def has_compiled(self, phase='predict'):
        """
        Specify whether have been compiled.

        Args:
            phase (str): The phase name. Default: 'predict'.

        Returns:
            bool, specifies whether the specific graph has been compiled.
        """
        return self._graph_executor.has_compiled(phase)

    def __call__(self, obj, *args, phase='predict'):
        if context.get_context("precompile_only") or _is_role_pserver() or _is_role_sched():
            return None
        return self.run(obj, *args, phase=phase)

    @_wrap_func
    def _exec_pip(self, obj, *args, phase=''):
        """Execute the generated pipeline."""
        fn = obj.construct
        obj.__parse_method__ = fn.__name__
        return self._graph_executor(args, phase)

    def run(self, obj, *args, phase='predict'):
        """
        Run the specific graph.

        Args:
            phase (str): The phase name. Default: 'predict'.

        Returns:
            Tensor/Tuple, return execute result.
        """
        if phase == 'save':
            return self._graph_executor((), phase + '.' + str(obj.create_time) + '.' + str(id(obj)))

        phase_real = phase + '.' + str(obj.create_time) + '.' + str(id(obj)) + '.' + obj.arguments_key
        if self.has_compiled(phase_real):
            return self._exec_pip(obj, *args, phase=phase_real)
        raise KeyError('{} graph is not exist.'.format(phase_real))

    def del_net_res(self, net_id):
        self._graph_executor.del_net_res(net_id)

    def _get_func_graph_proto(self, obj, exec_id, ir_type="onnx_ir", use_prefix=False):
        """Get graph proto from pipeline."""
        if use_prefix:
            exec_id = exec_id + '.' + obj.arguments_key
        if self._graph_executor.has_compiled(exec_id) is False:
            return None
        return self._graph_executor.get_func_graph_proto(exec_id, ir_type)

    def get_optimize_graph_proto(self, obj):
        """Return optimize graph binary proto."""
        exec_id = obj.phase + "." + str(obj.create_time) + '.' + str(id(obj)) + '.' + obj.arguments_key
        if self._graph_executor.has_compiled(exec_id) is False:
            return None
        graph_proto = self._graph_executor.get_optimize_graph_proto(exec_id)
        if isinstance(graph_proto, str) and graph_proto == "":
            logger.warning("Can not get optimize graph proto. Instead, try to find function graph.")
            graph_proto = obj.get_func_graph_proto()
        return graph_proto

    def export(self, file_name, graph_id):
        """
        Export graph.

        Args:
            file_name (str): File name of model to export
            graph_id (str): id of graph to be exported
        """
        from .._c_expression import export_graph
        export_graph(file_name, 'AIR', graph_id)

    def fetch_info_for_quant_export(self, exec_id):
        """Get graph proto from pipeline."""
        if self._graph_executor.has_compiled(exec_id) is False:
            return None
        return self._graph_executor.fetch_info_for_quant_export(exec_id)

    def set_jit_config(self, jit_config):
        """Set jit config."""
        self._check_jit_config(jit_config)
        self._graph_executor.set_jit_config(jit_config)

    def _check_jit_config(self, jit_config):
        """Check the value of jit config."""
        if not isinstance(jit_config, dict):
            raise ValueError("The jit_config should be a string.")
        for param_name, param_value in jit_config.items():
            Validator.check_string(param_name, self.VALID_JIT_CONFIG_PARAM, "jit_config")
            Validator.check_string(param_value, self.VALID_JIT_CONFIG_PARAM_VALUE.get(param_name), param_name,
                                   "jit_config")


def ms_memory_recycle():
    """
    Recycle memory used by MindSpore.
    When train multi Neural network models in one process, memory used by mindspore is very large,
    this is because mindspore cached runtime memory for every model.
    To recycle these cached memory, users can call this function after training of one model.
    """
    if ms_compile_cache:
        _cell_graph_executor.del_net_res(ms_compile_cache)
        ms_compile_cache.clear()
    for cell_cache in cells_compile_cache.values():
        if cell_cache:
            _cell_graph_executor.del_net_res(cell_cache)
            cell_cache.clear()
    _ms_memory_recycle()


_cell_graph_executor = _CellGraphExecutor()
_pynative_executor = _PynativeExecutor()

__all__ = ['ms_function', 'ms_memory_recycle']
