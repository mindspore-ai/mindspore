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
from collections import OrderedDict
from functools import wraps
from mindspore import context
from mindspore import log as logger
from mindspore.parallel._utils import _get_parallel_mode
from .._c_expression import generate_key, Executor_, Tensor, MetaTensor
from .._c_expression import verify_inputs_signature, init_exec_dataset, _set_dataset_mode_config, init_backend
from .tensor import Tensor as MsTensor

# store ms_function class compiled pipeline cache
ms_compile_cache = {}


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
            return data

        if isinstance(results, tuple):
            return tuple(_convert_data(x) for x in results)
        if isinstance(results, list):
            return list(_convert_data(x) for x in results)
        return _convert_data(results)

    return wrapper


def _exec_init_graph(obj, init_phase):
    """Execute the parameter initializer graph."""
    inst_executor = Executor_.get_instance()
    exec_init_graph = False
    for param in obj.get_parameters():
        if not param.is_init:
            param.is_init = True
            exec_init_graph = True

    if exec_init_graph:
        inst_executor.run_init_graph(obj.parameters_dict(), init_phase)


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
        """Returns pipline for the given args."""
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
            generate_name = self.obj.__module__ + "." + str(self.obj.create_time)
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
        init_backend()
        converted, arguments_dict, parse_method = _convert_function_arguments(self.fn, *args)
        if not converted:
            raise RuntimeError('Process function parameter is failure')

        args_list = tuple(arguments_dict.values())
        if self.obj is not None:
            args_list = args_list[1:]

        phase = self.compile(arguments_dict, parse_method)

        if context.get_context("precompile_only"):
            return None
        return self._executor(args_list, phase)


def ms_function(fn=None, obj=None, input_signature=None):
    """
    Creates a callable MindSpore graph from a python function.

    This allows the MindSpore runtime to apply optimizations based on graph.

    Args:
        fn (Function): The Python function that will be run as a graph. Default: None.
        obj (Object): The Python Object that provide information for identify compiled function. Default: None.
        input_signature (MetaTensor): The MetaTensor to describe the input arguments. The MetaTensor specifies
            the shape and dtype of the Tensor and they will be supplied to this function. If input_signature
            is specified, every input to `fn` must be a `Tensor`. And the input parameters of `fn` cannot accept
            `**kwargs`. The shape and dtype of actual inputs should keep same with input_signature, or TypeError
            will be raised. Default: None.

    Returns:
        Function, if `fn` is not None, returns a callable that will execute the compiled function; If `fn` is None,
        returns a decorator and when this decorator invokes with a single `fn` argument, the callable is equal to the
        case when `fn` is not None.

    Examples:
        >>> def tensor_add(x, y):
        >>>     z = F.tensor_add(x, y)
        >>>     return z
        >>>
        >>> @ms_function
        >>> def tensor_add_with_dec(x, y):
        >>>     z = F.tensor_add(x, y)
        >>>     return z
        >>>
        >>> @ms_function(input_signature=(MetaTensor(mindspore.float32, (1, 1, 3, 3)),
        >>>                               MetaTensor(mindspore.float32, (1, 1, 3, 3))))
        >>> def tensor_add_with_sig(x, y):
        >>>     z = F.tensor_add(x, y)
        >>>     return z
        >>>
        >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>>
        >>> tensor_add_graph = ms_function(fn=tensor_add)
        >>> out = tensor_add_graph(x, y)
        >>> out = tensor_add_with_dec(x, y)
        >>> out = tensor_add_with_sig(x, y)
    """
    def wrap_mindspore(func):
        @wraps(func)
        def staging_specialize(*args):
            process_obj = obj
            if args and not isinstance(args[0], MsTensor) and hasattr(args[0], func.__name__):
                process_obj = args[0]
            args = (x.default_input if hasattr(x, 'default_input') else x for x in args)
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
        self.phase_prefix = ""

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

    def _build_data_graph(self, obj, params, phase):
        if params is None:
            self._executor.build_data_graph(obj.parameters_dict(), phase, obj.parameters_broadcast_dict())
        elif isinstance(params, OrderedDict):
            self._executor.build_data_graph(params, phase)
        else:
            raise TypeError('Parameters need OrderedDict type, but got {}'.
                            format(type(params)))

    def compile(self, obj, *args, phase='predict', params=None):
        """
        Compiles graph.

        Args:
            obj (Function/Cell): The function or cell instance need compile.
            args (tuple): Function or cell input arguments.
            phase (str): The name of compile phase. Default: 'predict'.
            params (OrderedDict): The parameters dictionary used for init data graph. Default: None.

        Return:
            Str, the full phase of the cell.
            Bool, if the graph has been compiled before, return False, else return True.
        """
        obj.check_names()
        args_names, args_list = _generate_pip_args(obj, *args)
        dic = dict(zip(args_names, args_list))
        key = generate_key(phase, dic)
        self.phase_prefix = str(key[1])
        if phase == 'export':
            phase = phase + '.' + str(obj.create_time)
        else:
            phase = self.phase_prefix + phase + '.' + str(obj.create_time)
        enable_debug_runtime = context.get_context("enable_debug_runtime")
        enable_ge = context.get_context("enable_ge")

        use_vm = not enable_ge or (enable_debug_runtime and context.get_context("mode") == context.PYNATIVE_MODE)

        if phase in self.compile_cache.keys():
            logger.debug("%r graph has existed.", phase)
            return phase, False

        result = self._executor.compile(obj, args_list, phase, use_vm)
        self.compile_cache[phase] = phase
        if not result:
            raise RuntimeError("Executor compile failed.")
        graph = self._executor.get_func_graph(phase)

        if graph is None:
            logger.error("%r graph compile failed.", phase)

        if not enable_debug_runtime or enable_ge:
            if _get_parallel_mode() in ["auto_parallel", "semi_auto_parallel"]:
                obj.parameter_layout_dict = self._executor.get_parameter_layout(phase)
                obj.load_parameter_slice(params)

        # the following GE init process is not needed when use vm or ms backend
        if enable_ge:
            # decide whether to sink based on whether the inputs is virtual or not
            if args_list and isinstance(args_list[0], Tensor) and args_list[0].virtual_flag:
                _set_dataset_mode_config('sink')
            else:
                _set_dataset_mode_config('normal')

            self._build_data_graph(obj, params, phase)

            if "export" not in phase:
                init_phase = "init_subgraph" + "." + str(obj.create_time)
                _exec_init_graph(obj, init_phase)
        elif not enable_ge and "export" in phase:
            self._build_data_graph(obj, params, phase)

        return phase, True

    def _get_strategy(self, obj):
        real_phase = self.phase_prefix + obj.phase + '.' + str(obj.create_time)
        return self._executor.get_strategy(real_phase)

    def _get_allreduce_fusion(self, obj):
        real_phase = self.phase_prefix + obj.phase + '.' + str(obj.create_time)
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
        if context.get_context("precompile_only"):
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

        phase_real = self.phase_prefix + phase + '.' + str(obj.create_time)
        if self.has_compiled(phase_real):
            return self._exec_pip(obj, *args, phase=phase_real)
        raise KeyError('{} graph is not exist.'.format(phase_real))

    def del_net_res(self, net_id):
        self._executor.del_net_res(net_id)

    def _get_func_graph_proto(self, exec_id, ir_type="onnx_ir", use_prefix=False):
        """Get graph proto from pipeline."""
        if use_prefix:
            exec_id = self.phase_prefix + exec_id
        if self._executor.has_compiled(exec_id) is False:
            return None
        return self._executor.get_func_graph_proto(exec_id, ir_type)

    def export(self, net, file_name, file_format='GEIR'):
        """
        Export graph.

        Args:
            net (Cell): MindSpore network
            file_name (str): File name of model to export
            file_format (str): MindSpore currently support 'GEIR' and 'ONNX' format for exported model
        """
        from .._c_expression import export_graph
        phase = 'export' + '.' + str(net.create_time)
        export_graph(file_name, file_format, phase)


_executor = _Executor()

__all__ = ['ms_function']
