# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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
"""Providing interface methods."""
import types
import sys
from collections import OrderedDict
from functools import wraps

from mindspore import context
from mindspore import log as logger
from .tensor import Tensor as MsTensor
from .._c_expression import generate_key, Executor_, Tensor, MetaTensor, PynativeExecutor_
from .._c_expression import verify_inputs_signature, init_exec_dataset, _set_dataset_mode_config, init_pipeline
from ..parallel._ps_context import _is_role_pserver
from ..parallel._utils import _get_device_num, _get_global_rank, _need_to_full, _check_full_batch, _to_full_tensor, \
    _get_parameter_broadcast, _get_pipeline_stages

# store ms_function class compiled pipeline cache
ms_compile_cache = {}

BROADCAST_PHASE = "_broadcast_"


def _convert_function_arguments(fn, *args):
    """
    Process the fn default parameters.

    Args:
        fn (Function): The function to be parsed.
        args (tuple): The parameters of the function.
    """
    arguments_dict = OrderedDict()
    parse_method = None
    if isinstance(fn, (types.FunctionType, types.MethodType)):
        parse_method = fn.__name__
        index = 0
        for value in args:
            arguments_dict[f'arg{index}'] = value
            index = index + 1
        logger.debug("fn(%r) full parameters dict is: %r", fn, arguments_dict)
        converted = True
    else:
        logger.warning("Find error: fn isn't function or method")
        converted = False
    return converted, arguments_dict, parse_method


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
            if isinstance(data, tuple):
                return tuple(_convert_data(x) for x in data)
            if isinstance(data, list):
                return list(_convert_data(x) for x in data)
            return data

        return _convert_data(results)

    return wrapper


def _exec_init_graph(obj, init_phase):
    """Execute the parameter initializer graph."""
    inst_executor = Executor_.get_instance()
    param_dict = OrderedDict()
    for name, param in obj.parameters_dict().items():
        if not param.is_init:
            param_dict[name] = param
            param.is_init = True
            param.data.init_flag = True

    if param_dict:
        inst_executor.run_init_graph(param_dict, init_phase)


class _MindSporeFunction:
    """
    Represents a function compiled by mind expression.

    _MindSporeFunction will compile the original function for every combination
    of argument types and shapes it is given (as well as their values, optionally).

    Args:
        fn (Function): The root function to compile.
        input_signature (Function): User defines signature to verify input.
        obj (Object): If function is a method, obj is the owner of function,
             else, obj is none.
    """

    def __init__(self, fn, input_signature=None, obj=None):
        self.fn = fn
        self.save_graphs = context.get_context("save_graphs")
        self.save_graphs_path = context.get_context("save_graphs_path")
        self.input_signature = input_signature
        self.obj = None
        self.identify_obj = None
        if hasattr(obj, fn.__name__):
            self.obj = obj
        elif obj is not None:
            self.identify_obj = obj
        self._executor = Executor_.get_instance()

    def build_data_init_graph(self, graph_name):
        """Build GE data graph and init graph for the given graph name."""
        if self.obj is None:
            logger.warning("Make sure parameter should not be used in function")
            para_dict = OrderedDict()
            self._executor.build_data_graph(para_dict, graph_name)
            return
        self._executor.build_data_graph(self.obj.parameters_dict(), graph_name, self.obj.parameters_broadcast_dict())
        init_phase = "init_subgraph" + graph_name[graph_name.find("."):]
        _exec_init_graph(self.obj, init_phase)

    def compile(self, arguments_dict, method_name):
        """Returns pipeline for the given args."""
        args_list = tuple(arguments_dict.values())
        arg_names = tuple(arguments_dict.keys())

        # remove first self parameter when fn is a method
        if self.obj is not None:
            args_list = args_list[1:]
            arg_names = arg_names[1:]

        # verify the signature for both function and method
        if self.input_signature is not None:
            signatures = []
            for sig_spec in self.input_signature:
                if not isinstance(sig_spec, MetaTensor):
                    raise TypeError("Input_signature is not MetaTensor")
                signatures.append(sig_spec)
            is_valid_input = verify_inputs_signature(signatures, args_list)
            if not is_valid_input:
                raise ValueError("Inputs is incompatible with input signature!")

        dic = dict(zip(arg_names, args_list))
        generate_name = self.fn.__module__ + "." + self.fn.__name__
        self.fn.__parse_method__ = method_name

        # replace key with obj info and object ext info when fn is a method
        if self.obj is not None:
            self.obj.__parse_method__ = method_name
            generate_name = self.obj.__module__ + "."
            if self.obj.__class__.__name__ != "ClipByNorm":
                generate_name = generate_name + str(self.obj.create_time)
        if self.identify_obj is not None:
            generate_name = generate_name + str(id(self.identify_obj))

        key = generate_key(generate_name, dic)
        phase = str(key[1]) + generate_name
        if key not in ms_compile_cache.keys():
            is_compile = False
            if self.obj is None:
                is_compile = self._executor.compile(self.fn, args_list, phase, True)
            else:
                is_compile = self._executor.compile(self.obj, args_list, phase, True)
            if not is_compile:
                raise RuntimeError("Executor compile failed.")
            if context.get_context("enable_ge"):
                self.build_data_init_graph(phase)
            # since function can be redefined, we only cache class method pipeline
            if self.obj is not None or self.identify_obj is not None:
                ms_compile_cache[key] = phase
            return phase

        return ms_compile_cache[key]

    @_wrap_func
    def __call__(self, *args):
        init_pipeline()
        converted, arguments_dict, parse_method = _convert_function_arguments(self.fn, *args)
        if not converted:
            raise RuntimeError('Process function parameter is failure')

        args_list = tuple(arguments_dict.values())
        if self.obj is not None:
            args_list = args_list[1:]

        phase = self.compile(arguments_dict, parse_method)

        if context.get_context("precompile_only"):
            return None
        new_inputs = []
        for i in args_list:
            if isinstance(i, Tensor):
                new_inputs.append(i)
            elif context.get_context("grad_for_scalar") and isinstance(i, (int, float)):
                new_inputs.append(i)
        return self._executor(tuple(new_inputs), phase)


def ms_function(fn=None, obj=None, input_signature=None):
    """
    Create a callable MindSpore graph from a python function.

    This allows the MindSpore runtime to apply optimizations based on graph.

    Args:
        fn (Function): The Python function that will be run as a graph. Default: None.
        obj (Object): The Python Object that provides the information for identifying the compiled function.Default:
            None.
        input_signature (Tensor): The Tensor which describes the input arguments. The shape and dtype of the Tensor
            will be supplied to this function. If input_signature is specified, each input to `fn` must be a `Tensor`.
            And the input parameters of `fn` cannot accept `**kwargs`. The shape and dtype of actual inputs should
            keep the same as input_signature. Otherwise, TypeError will be raised. Default: None.

    Returns:
        Function, if `fn` is not None, returns a callable function that will execute the compiled function; If `fn` is
        None, returns a decorator and when this decorator invokes with a single `fn` argument, the callable function is
        equal to the case when `fn` is not None.

    Examples:
        >>> from mindspore.ops import functional as F
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
        @wraps(func)
        def staging_specialize(*args):
            process_obj = obj
            if args and not isinstance(args[0], MsTensor) and hasattr(args[0], func.__name__):
                process_obj = args[0]
            return _MindSporeFunction(func, input_signature, process_obj)(*args)

        return staging_specialize

    if fn is not None:
        return wrap_mindspore(fn)
    return wrap_mindspore


def _generate_pip_args(obj, *args, method="construct"):
    """Generate arguments for pipeline."""
    if hasattr(obj, method):
        fn = getattr(obj, method)
    else:
        raise AttributeError('The process method is not exist')
    converted, arguments_dict, parse_method = _convert_function_arguments(fn, *args)
    if not converted:
        raise RuntimeError('Process method parameter is failure')
    args_list = tuple(arguments_dict.values())
    args_names = tuple(arguments_dict.keys())
    obj.__parse_method__ = parse_method
    return args_names, args_list


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
    An pynative executor used to compile/manage/run graph.

    Returns:
        Graph, return the result of pipeline running.
    """

    def __init__(self):
        self._executor = PynativeExecutor_.get_instance()

    def new_graph(self, obj, *args, **kwargs):
        self._executor.new_graph(obj, *args, *(kwargs.values()))

    def end_graph(self, obj, output, *args, **kwargs):
        self._executor.end_graph(obj, output, *args, *(kwargs.values()))

    def check_graph(self, obj, *args, **kwargs):
        return self._executor.check_graph(obj, *args, *(kwargs.values()))

    def check_run(self, obj, *args, **kwargs):
        return self._executor.check_run(obj, *args, *(kwargs.values()))

    def grad(self, grad, obj, weights, *args, **kwargs):
        self._executor.grad_net(grad, obj, weights, *args, *(kwargs.values()))

    def del_cell(self, cell_id=""):
        self._executor.clear_cell(cell_id)

    def clear_grad(self, obj, *args, **kwargs):
        self._executor.clear_grad(obj, *args, *(kwargs.values()))

    def sync(self):
        self._executor.sync()

    def set_grad_flag(self, flag):
        self._executor.set_grad_flag(flag)

    def enter_construct(self, cell):
        self._executor.enter_construct(cell)

    def leave_construct(self, cell):
        self._executor.leave_construct(cell)

    def parameter_broadcast(self, obj, phase, auto_parallel_mode):
        if BROADCAST_PHASE not in phase and _get_parameter_broadcast():
            _parameter_broadcast(obj, auto_parallel_mode)

    def __call__(self, obj, *args, **kwargs):
        args = args + tuple(kwargs.values())
        return self._executor(obj, args, "")


class _Executor:
    """
    An executor used to compile/manage/run graph.

    Including data_graph, train_graph, eval_graph and predict graph.

    Returns:
        Graph, return the result of pipeline running.
    """

    def __init__(self):
        # create needed graph by lazy mode
        self.is_init = False
        self._executor = Executor_.get_instance()
        self.compile_cache = {}
        self._executor.set_py_exe_path(sys.executable)

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
        return True

    def _build_data_graph(self, obj, phase):
        self._executor.build_data_graph(obj.parameters_dict(), phase, obj.parameters_broadcast_dict())

    def _set_dataset_mode(self, args_list):
        """set dataset mode."""
        # decide whether to sink based on whether the inputs is virtual or args_list is ()
        if (args_list and isinstance(args_list[0], Tensor) and args_list[0].virtual_flag) or \
                (args_list is not None and args_list == ()):
            _set_dataset_mode_config('sink')
        else:
            _set_dataset_mode_config('normal')

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

        args_names, args_list = _generate_pip_args(obj, *args)
        dic = dict(zip(args_names, args_list))
        key = generate_key(phase, dic)
        obj.phase_prefix = str(key[1])
        if 'export' in phase:
            phase = phase + '.' + obj.phase_prefix + '.' + str(obj.create_time)
        else:
            phase = obj.phase_prefix + phase + '.' + str(obj.create_time)

        if phase in self.compile_cache.keys():
            logger.debug("%r graph has existed.", phase)
            return phase, False

        obj.check_names()
        _check_full_batch()
        self._set_dataset_mode(args_list)

        is_sink_mode = args and isinstance(args[0], Tensor) and args[0].virtual_flag
        if auto_parallel_mode and _need_to_full() and not is_sink_mode and obj.auto_parallel_compile_and_run():
            args_full = _to_full_tensor(args, _get_device_num(), _get_global_rank())
            _, args_list = _generate_pip_args(obj, *args_full)

        enable_debug_runtime = context.get_context("enable_debug_runtime")
        enable_ge = context.get_context("enable_ge")
        use_vm = not enable_ge or (enable_debug_runtime and context.get_context("mode") == context.PYNATIVE_MODE)
        result = self._executor.compile(obj, args_list, phase, use_vm)
        self.compile_cache[phase] = phase
        if not result:
            raise RuntimeError("Executor compile failed.")
        graph = self._executor.get_func_graph(phase)

        if graph is None:
            logger.error("%r graph compile failed.", phase)

        self._auto_parallel_process(obj, phase, is_sink_mode, auto_parallel_mode, *args)

        if not do_convert:
            return phase, True

        # the following GE init process is not needed when use vm or ms backend
        if enable_ge:
            self._build_data_graph(obj, phase)

            if "export" not in phase:
                init_phase = "init_subgraph" + "." + str(obj.create_time)
                _exec_init_graph(obj, init_phase)
        elif not enable_ge and "export" in phase:
            self._build_data_graph(obj, phase)
        elif BROADCAST_PHASE not in phase and _get_parameter_broadcast():
            _parameter_broadcast(obj, auto_parallel_mode)

        return phase, True

    def _auto_parallel_process(self, obj, phase, is_sink_mode, auto_parallel_mode, *args):
        """compile graph in auto parallel mode."""
        if not auto_parallel_mode:
            replace = obj.init_parameters_data(auto_parallel_mode=auto_parallel_mode)
            self._updata_param_node_default_input(phase, replace)
            return

        obj.parameter_layout_dict = self._executor.get_parameter_layout(phase)
        if _get_pipeline_stages() > 1:
            obj.parallel_parameter_name_list = self._executor.get_parallel_parameter_name_list(phase)
            obj.remove_redundant_parameters()
        replace = obj.init_parameters_data(auto_parallel_mode=True)
        if not context.get_context("enable_debug_runtime") or context.get_context("enable_ge"):
            obj.load_parameter_slice(None)

        self._updata_param_node_default_input(phase, replace)

        # set parallel inputs in sink mode
        if is_sink_mode:
            obj.set_parallel_input_with_inputs(*args)

    def _updata_param_node_default_input(self, phase, replace):
        new_param = {x.name: replace[x] for x in replace if id(x) != id(replace[x])}
        return self._executor.updata_param_node_default_input(phase, new_param)

    def _get_shard_strategy(self, obj):
        real_phase = obj.phase_prefix + obj.phase + '.' + str(obj.create_time)
        return self._executor.get_strategy(real_phase)

    def _get_num_parallel_ops(self, obj):
        real_phase = obj.phase_prefix + obj.phase + '.' + str(obj.create_time)
        return self._executor.get_num_parallel_ops(real_phase)

    def _get_allreduce_fusion(self, obj):
        real_phase = obj.phase_prefix + obj.phase + '.' + str(obj.create_time)
        return self._executor.get_allreduce_fusion(real_phase)

    def has_compiled(self, phase='predict'):
        """
        Specify whether have been compiled.

        Args:
            phase (str): The phase name. Default: 'predict'.

        Returns:
            bool, specifies whether the specific graph has been compiled.
        """
        return self._executor.has_compiled(phase)

    def __call__(self, obj, *args, phase='predict'):
        if context.get_context("precompile_only") or _is_role_pserver():
            return None
        return self.run(obj, *args, phase=phase)

    @_wrap_func
    def _exec_pip(self, obj, *args, phase=''):
        """Execute the generated pipeline."""
        fn = obj.construct
        converted, arguments_dict, parse_method = _convert_function_arguments(fn, *args)
        if not converted:
            raise RuntimeError('Process method parameter is failure')
        args_list = tuple(arguments_dict.values())
        obj.__parse_method__ = parse_method
        return self._executor(args_list, phase)

    def run(self, obj, *args, phase='predict'):
        """
        Run the specific graph.

        Args:
            phase (str): The phase name. Default: 'predict'.

        Returns:
            Tensor/Tuple, return execute result.
        """
        if phase == 'save':
            return self._executor((), phase + '.' + str(obj.create_time))

        phase_real = obj.phase_prefix + phase + '.' + str(obj.create_time)
        if self.has_compiled(phase_real):
            return self._exec_pip(obj, *args, phase=phase_real)
        raise KeyError('{} graph is not exist.'.format(phase_real))

    def del_net_res(self, net_id):
        self._executor.del_net_res(net_id)

    def _get_func_graph_proto(self, obj, exec_id, ir_type="onnx_ir", use_prefix=False):
        """Get graph proto from pipeline."""
        if use_prefix:
            exec_id = obj.phase_prefix + exec_id
        if self._executor.has_compiled(exec_id) is False:
            return None
        return self._executor.get_func_graph_proto(exec_id, ir_type)

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
        if self._executor.has_compiled(exec_id) is False:
            return None
        return self._executor.fetch_info_for_quant_export(exec_id)


_executor = _Executor()
_pynative_exec = _PynativeExecutor()

__all__ = ['ms_function']
