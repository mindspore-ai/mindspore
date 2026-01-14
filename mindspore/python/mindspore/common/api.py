# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2020-2025 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import

__all__ = ['ms_memory_recycle', 'jit', 'jit_class', 'flops_collection']

import gc
import types
import sys
import os
import time
import ast
import inspect
import importlib
import contextlib
import json
from collections import OrderedDict, namedtuple
from functools import wraps
from typing import Optional, Callable
import mindspore as ms
from mindspore import context
from mindspore import log as logger
from mindspore._extends.remote import kernel_build_server
from mindspore.common.jit_config import JitConfig
from mindspore.common.tensor import Tensor as PythonTensor
from mindspore.common.sparse_tensor import CSRTensor as PythonCSRTensor
from mindspore.common.sparse_tensor import COOTensor as PythonCOOTensor
from mindspore.common.sparse_tensor import RowTensor as PythonRowTensor
from mindspore._c_expression.amp import get_curr_amp_strategy
from mindspore._c_expression import GraphExecutor_, JitExecutor_, CSRTensor, RowTensor, COOTensor, \
    PyNativeExecutor_, verify_inputs_signature, init_exec_dataset, _set_dataset_mode_config, init_pipeline, \
    _run_jit_pipeline, _ms_memory_recycle, _bind_device_ctx, TensorPy as Tensor, dump_func_graph, _GraphFragment_
from mindspore.parallel._ps_context import _is_role_sched
from mindspore.parallel._utils import _check_full_batch, _get_parameter_broadcast, _is_in_auto_parallel_mode, \
    _is_parallel_mode
from mindspore import _checkparam as Validator
from mindspore._checkparam import is_stub_tensor
from mindspore.common._utils import is_shape_unknown, get_func
from mindspore.common.mutable import mutable, _check_element_type
from mindspore.common.dynamic_shape.auto_dynamic_shape import get_auto_dynamic_shape_args, \
    update_auto_dynamic_shape_phase
from mindspore.common.dynamic_shape.enable_dynamic import generate_dynamic_tensor_args, ENABLE_DYNAMIC
from mindspore.common._pijit_context import PIJitCaptureContext
from mindspore.common.parameter import Parameter
from mindspore.common.hook_handle import _hook_version
from mindspore.common.jit_context import jit_context
from mindspore.common.jit_trace import _jit_trace
from mindspore.parallel._utils import _init_auto_parallel_context, _clear_auto_parallel_context
from mindspore._check_jit_forbidden_api import jit_forbidden_register
from mindspore.runtime.event import Event

# Store jit class compiled pipeline cache.
ms_compile_cache = set()
# Store cell compiled pipeline cache.
cells_compile_cache = {}
# Store function compiled times information.
function_phases = {}

BROADCAST_PHASE = "_broadcast_"
_PYNATIVE_PARALLEL_FUNC_NAME = "after_shard"

ARG_SPECIFIED = "arg_specified_infos"
TOTAL_ARG_LEN = "total_arg_length"


def _real_phase(phase, obj):
    real_phase = phase + '.' + str(obj.create_time) + '.' + str(id(obj)) + '.' + obj.arguments_key
    return real_phase


def _check_recompile_args(compile_args, kwargs):
    """Check recompile of graph"""

    def _check_constant_tensor_arg(arg):
        if hasattr(arg, "__ms_mutable__"):
            return False
        if isinstance(arg, (list, tuple)):
            return any(_check_constant_tensor_arg(x) for x in arg)
        return isinstance(arg, Tensor)

    for v in kwargs.values():
        compile_args += (v,)
    for arg in compile_args:
        if not isinstance(arg, tuple) and not isinstance(arg, list):
            continue
        if _check_constant_tensor_arg(arg):
            logger.warning(f"Constant value tensor are detected in tuple or list, which might cause recompiling "
                           f"when tensor value changes. You can use mutable(Tensor) or mutable(tuple(Tensor)) "
                           f"to set tensor's value as variable to to avoid recompiling. The tuple or list arg "
                           f"is: {arg} .")
            return


def _check_recompile(obj, compile_args, kwargs, full_function_name, create_time, echo_function_name):
    """Warning when the function has been compiled."""
    # pylint: disable=C3002
    ignore_dirs = ["mindspore/ops", "mindspore/nn"]
    if any((lambda x: x in full_function_name)(x) for x in ignore_dirs):
        return

    if full_function_name in function_phases:
        warning_times = 1
        if len(function_phases[full_function_name]) >= warning_times \
                and create_time not in function_phases[full_function_name]:
            if isinstance(obj, ms.nn.Cell):
                tips = f"Please try to create {echo_function_name} instance only once to avoid recompiling. "
                logger.info(f"The {echo_function_name} has been compiled again. "
                            f"{tips} ")
            else:
                tips = "Try to reuse the function object decorated by @jit to reduce the compile time. " \
                       "For more details, get instructions about `jit` at " \
                       "https://www.mindspore.cn/search?inputValue=jit."
                logger.warning(f"The {echo_function_name} has been compiled again. "
                               f"{tips} ")
        else:
            _check_recompile_args(compile_args, kwargs)
    else:
        function_phases[full_function_name] = set()
    function_phases[full_function_name].add(create_time)


def _convert_python_data(data):
    """
    Convert C++ data to python.

    Args:
        data : The data need be convert.

    Returns:
        data, a data convert C++ to python
    """
    # pylint: disable=C0200
    if isinstance(data, PythonTensor):
        return data
    if data.__class__ is tuple:
        # Handle namedtuple since its type is tuple.
        if hasattr(data, "_fields"):
            type_name = data.__class__.__name__
            data_dict = data._asdict()
            fields = data_dict.keys()
            return namedtuple(type_name, fields)(**_convert_python_data(data_dict))
        return tuple(_convert_python_data(x) for x in data)
    if isinstance(data, CSRTensor) and not isinstance(data, PythonCSRTensor):
        return PythonCSRTensor(csr_tensor=data)
    if isinstance(data, COOTensor) and not isinstance(data, PythonCOOTensor):
        return PythonCOOTensor(coo_tensor=data)
    if isinstance(data, RowTensor) and not isinstance(data, PythonRowTensor):
        return PythonRowTensor(row_tensor=data)
    if data.__class__ is list:
        # Keep list object not change for inplace operation.
        for i,_ in enumerate(data):
            data[i] = _convert_python_data(data[i])
        return data
    if data.__class__ is dict:
        # Keep the dict object not change.
        keys = tuple(data.keys())
        for key in keys:
            data[_convert_python_data(key)] = _convert_python_data(data.pop(key))
        return data
    return data


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
        return _convert_python_data(results)

    return wrapper


def _check_all_tensor(sequence):
    for element in sequence:
        if not isinstance(element, Tensor) and not is_stub_tensor(element) and not (isinstance(element, tuple)
                                                                                    and _check_all_tensor(element)):
            return False
    return True


def _handle_func_args(func, *args, **kwargs):
    """Handle the *args and **kwargs inputs of the function."""
    if not isinstance(func, (types.FunctionType, types.MethodType)):
        raise RuntimeError('fn {} is not function or method'.format(func))
    if kwargs:
        bound_arguments = inspect.signature(func).bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        args = bound_arguments.args
        kwargs = bound_arguments.kwargs

    return args, kwargs


def _check_func_args(func, *args):
    """Check the *args inputs of the function"""
    positional_args = 0
    default_args = 0
    has_var = False
    for value in inspect.signature(func).parameters.values():
        if value.kind is inspect.Parameter.VAR_POSITIONAL or value.kind is inspect.Parameter.VAR_KEYWORD:
            has_var = True
        if value.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if value.default is inspect.Parameter.empty:
                positional_args += 1
            else:
                default_args += 1

    if has_var:
        return

    if len(args) < positional_args:
        raise TypeError(f"Function {func.__name__} needs {positional_args} positional argument, but got {len(args)}.")
    if len(args) > positional_args + default_args:
        raise TypeError(f"Function {func.__name__} needs {positional_args} positional argument and {default_args} "
                        f"default argument, total {positional_args + default_args}, but got {len(args)}.")


sys_path = list(sys.path)
# Get the entry script path.
entry_script_path = None
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
    # pylint: disable=W1514
    with open(file_path) as fh:
        root = ast.parse(fh.read(), file_path)
    for node in ast.iter_child_nodes(root):
        module_name = ""
        if isinstance(node, ast.ImportFrom):
            if node.module is not None:
                module_name = node.module
            module_name = "." * node.level + module_name
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
                if n.name is not None:
                    if not whole_module.endswith("."):
                        whole_module += "."
                    whole_module += n.name
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
                # Exclude the installed modules.
                if not _in_sys_path(dep_file_path) and dep_file_path not in compile_cache_dep_files:
                    logger.debug(f"dependent file path: {dep_file_path}")
                    compile_cache_dep_files.append(dep_file_path)
                    __get_compile_cache_dep_files(dep_file_path, compile_cache_dep_files, module.__package__)
            else:
                continue


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


def _contains_auto_grad_tensor(obj):
    """Check object is or contains auto grad tensor element"""
    if isinstance(obj, PythonTensor):
        return obj._has_auto_grad()
    if isinstance(obj, (tuple, list)):
        for element in obj:
            if _contains_auto_grad_tensor(element):
                return True
    if isinstance(obj, dict):
        for key in obj:
            if _contains_auto_grad_tensor(obj[key]):
                return True
    return False


def _add_mutable_attr(args_list, compile_args, is_grad):
    """Restore the mutable attr for every arg."""
    new_compile_args = ()
    for idx, arg in enumerate(args_list):
        if hasattr(arg, "__ms_mutable__") and getattr(arg, "__ms_mutable__") and \
                not (hasattr(arg, "const_arg") and getattr(arg, "const_arg")):
            if hasattr(arg, "__ms_dynamic_len__"):
                new_compile_args += (mutable(compile_args[idx], getattr(arg, "__ms_dynamic_len__")),)
            else:
                new_compile_args += (mutable(compile_args[idx], False),)
        else:
            if is_grad and _contains_auto_grad_tensor(arg):
                if not _check_element_type(arg):
                    raise RuntimeError("Input \"%s\" contains tensor with gradient but can not mutable." % (str(arg)))
                new_compile_args += (mutable(compile_args[idx], False),)
            else:
                new_compile_args += (compile_args[idx],)
    return new_compile_args


def _get_parameter_layout():
    graph_executor = GraphExecutor_.get_instance()
    layout = {}
    for phase in ms_compile_cache:
        layout.update(graph_executor.get_parameter_layout(phase))
    return layout


def _handle_arg(obj, arg, has_mutable_arg, is_predict):
    """Handle arg for runtime .If need handle the arg, return True"""
    from mindspore._extends.parse import compile_config
    if isinstance(arg, PythonTensor):
        if arg.has_init:
            arg.init_data()
        if not arg.const_arg:
            return arg
    elif isinstance(arg, (Tensor, CSRTensor, COOTensor)):
        return arg
    elif has_mutable_arg:
        # mutable([]) will be eliminated by FuncGraphSpecializer, and empty list is not supported by backend.
        if isinstance(arg, list) and not arg:
            return None
        return arg
    elif not is_predict and (context.get_context("grad_for_scalar") or str(compile_config.GRAD_FOR_SCALAR) == '1') and \
            isinstance(arg, (int, float)):
        return arg
    elif hasattr(obj, "enable_tuple_broaden") and obj.enable_tuple_broaden and isinstance(arg, tuple) and \
            _check_all_tensor(arg):
        return arg
    elif isinstance(arg, Event):
        return arg
    return None


def _handle_arg_predict(obj, arg, has_mutable_arg):
    """Handle arg for runtime .If need handle the arg, return True"""
    if arg is None:
        return None

    if isinstance(arg, (int, float)):
        return None

    if isinstance(arg, (list, tuple)):
        if has_mutable_arg:
            # mutable([]) will be eliminated by FuncGraphSpecializer, and empty list is not supported by backend.
            if isinstance(arg, list) and not arg:
                return None
            return arg
        if hasattr(obj, "enable_tuple_broaden") and obj.enable_tuple_broaden and isinstance(arg, tuple) and \
                _check_all_tensor(arg):
            return arg
        return None
    return arg


def _get_args_for_run(obj, args, kwargs, has_mutable_args_list, sequence_modified, is_predict=False,
                      is_in_phase_cache=False):
    """Get the actual input args and kwargs for runtime."""
    new_args = []
    sequence_index = 0
    for arg, has_mutable_arg in zip(args, has_mutable_args_list):
        new_arg = _handle_arg(obj, arg, has_mutable_arg, is_predict)
        if new_arg is not None:
            new_args.append(new_arg)
        elif not is_in_phase_cache and sequence_modified and isinstance(arg, (list, tuple)):
            if sequence_index < len(sequence_modified) and sequence_modified[
                sequence_index] is True:
                logger.debug(f'The list or tuple need append: `{arg}')
                new_args.append(arg)
            sequence_index = sequence_index + 1

    for _, value in kwargs.items():
        new_value = _handle_arg(obj, value, None, is_predict)
        if new_value is not None:
            new_args.append(new_value)

    return new_args


def _get_mutable_flags(compile_args):
    """Get a list of booleans indicating whether each argument is marked as mutable"""
    new_args = []
    for compile_arg in compile_args:
        has_mutable_arg = compile_arg is not None and hasattr(compile_arg, "__ms_mutable__") and \
                          getattr(compile_arg, "__ms_mutable__")
        new_args.append(has_mutable_arg)
    return new_args


def _is_args_fullmode(args, is_init=True):
    """Check whether the arguments is for incremental-mode.

    Args:
        args (Union[list, tuple, dict, Tensor]): Given arguments.
        is_init (bool): Is check in argument initialization phase.

    Raises:
        RuntimeError: loss necessary keys and values for incremental-mode.

    Returns:
        bool: Fullmode or not.
    """
    if not isinstance(args, dict):
        return True
    if not is_init and (args.get(ARG_SPECIFIED, None) is None or args.get(TOTAL_ARG_LEN, None) is None):
        raise RuntimeError(
            "The incremental inputs should be processed(with \"%s\" and \"%s\"), but got %s." %
            (ARG_SPECIFIED, TOTAL_ARG_LEN, str(args)))
    return False


def _process_dyn_args(fn, dyn_args):
    """Process the dynamic arguments, return the necessary data for latter processing.

    Args:
        fn (Function): The root function to compile.
        dyn_args (Union[dict, list, tuple, None]): Given arguments for dynamic compilation.
            None for nothing, list or tuple for fullmode setting, dict for incremental configuration.

    Returns:
        A dict which contains args for dynamic compilation. None for nothing dynamic.
    """
    if dyn_args is None:
        # nothing should be done for None.
        return dyn_args

    if isinstance(dyn_args, dict) and ARG_SPECIFIED in dyn_args:
        return dyn_args

    args_sig = inspect.signature(fn)
    if _is_args_fullmode(dyn_args):
        if not isinstance(dyn_args, (list, tuple)):
            temp_dyn_args = (dyn_args,)
        else:
            temp_dyn_args = dyn_args

        # If dyn_args is fullmode, it should be apply directly.
        args_sig_parameters = list(args_sig.parameters.values())
        if not args_sig_parameters:
            return ()

        # fn may be Cell's construct while the first input is 'self'.
        if args_sig_parameters[0].name == "self" and (len(temp_dyn_args) + 1) == len(args_sig_parameters):
            bound_args = args_sig.bind(None, *temp_dyn_args)
            bound_args.apply_defaults()
            return bound_args.args[1:]

        bound_args = args_sig.bind(*temp_dyn_args)
        bound_args.apply_defaults()
        return bound_args.args

    # The dyn_args is not fullmode, a real compilation arguments should be assembled by latter procession...
    arg_names = []
    args_sig_parameters = list(args_sig.parameters.values())
    for arg_p in args_sig_parameters:
        if arg_p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            arg_names.append(arg_p.name)
        else:
            raise TypeError("Dynamic arguments is not accepted for VAR_POSITIONAL or VAR_KEYWORD parameters!")

    offset = -1 if fn.__name__ == 'construct' and args_sig_parameters[0].name == "self" else 0
    meet_index = set()

    def _check_index_valid(index):
        if index >= len(arg_names):
            raise ValueError("For dict mode, valid index is \"0\"-\"%d\", but got %s!" % (len(arg_names) - 1, index))
        if index in meet_index:
            raise ValueError("For dict mode, there are more than one same specified key for real index: %d!" % index)
        meet_index.add(index)

    arg_handler_infos = []
    for k, v in dyn_args.items():
        if not isinstance(k, str):
            raise TypeError("For dict mode, only string key is accepted, but got %s!" % k)
        if k in arg_names:
            cur_id = arg_names.index(k)
            _check_index_valid(cur_id)
            arg_handler_infos.append([cur_id + offset, v])
        else:
            raise ValueError("For dict mode, valid key is %s, but got %s!" % (arg_names, k))
    return {ARG_SPECIFIED: arg_handler_infos, TOTAL_ARG_LEN: len(args_sig_parameters)}


def _generate_dyn_compile_args(compile_args, dyn_args):
    """Generate the dynamic compile arguments."""
    if not dyn_args:
        return compile_args
    if _is_args_fullmode(dyn_args, False):
        if not isinstance(dyn_args, (list, tuple)):
            return (dyn_args,)
        return dyn_args
    arg_specified_infos = dyn_args.get(ARG_SPECIFIED, None)
    if arg_specified_infos is None:
        raise RuntimeError("For dict mode, a key with \"%s\" should exist, but got %s!" %
                           (ARG_SPECIFIED, str(dyn_args)))
    new_compile_args = list(compile_args)
    for index, arg in arg_specified_infos:
        new_compile_args[index] = arg
    return tuple(new_compile_args)


def _get_parameter_ids(args, kwargs):
    """Get the ids of parameters."""
    parameter_ids = ""
    for arg in args:
        if isinstance(arg, Parameter):
            parameter_ids += str(id(arg))
    for _, value in kwargs.items():
        # The type of key is usually String type.
        if isinstance(value, Parameter):
            parameter_ids += str(id(value))
    return parameter_ids


def _get_tensor_hook_key(tensor):
    """Get the hook key of Tensor/Parameter"""
    return ".".join(map(str, map(id, tensor.hooks())))


def _get_hook_key(*args, **kwargs):
    """Get the hook key of Tensors/Parameters"""
    hook_key = ""
    for idx, arg in enumerate(args):
        if idx != 0:
            hook_key += "."
        # Only arg of the type Tensor or Parameter is supported now
        if isinstance(arg, (Tensor, Parameter)):
            hook_key += _get_tensor_hook_key(arg)

    for idx, value in enumerate(kwargs.values()):
        if idx != 0:
            hook_key += "."
        # Only kwarg of the type Tensor or Parameter is supported now
        if isinstance(value, (Tensor, Parameter)):
            hook_key += _get_tensor_hook_key(value)

    return hook_key


class _JitExecutor:
    """
    Represents a function compiled by graph compiler.

    _JitExecutor will compile the original function for every combination
    of argument types and shapes it is given (as well as their values, optionally).

    Args:
        fn (Function): The root function to compile.
        input_signature (Function): User defines signature to verify input.
        ms_create_time(TimeStamp): Time the function was created
        obj (Object): If function is a method, obj is the owner of function,
             else, obj is none.

    Returns:
        The result of pipeline running in graph mode.
    """

    def __init__(self, fn, ms_create_time, input_signature=None, obj=None, jit_config=None, dynamic=0,
                 cell_cache_key_extend=''):
        init_pipeline()
        if not isinstance(fn, (types.FunctionType, types.MethodType)):
            raise RuntimeError('fn {} is not function or method'.format(fn))

        self.fn = fn
        self.input_signature = input_signature
        self.dynamic_args_shapes = getattr(get_func(fn), ENABLE_DYNAMIC, None)
        self.enable_jit_dynamic = self.dynamic_args_shapes is not None
        self.obj = None
        if obj and hasattr(obj, fn.__name__):
            self.obj = obj
        self.shard_parent_obj = obj
        self.enable_tuple_broaden = False
        if _run_jit_pipeline():
            self._graph_executor = JitExecutor_.get_instance()
        else:
            self._graph_executor = GraphExecutor_.get_instance()
        self._create_time = ms_create_time
        self._mutable_flags = None
        self.sequence_modified = []
        self._enable_auto_dynamic = dynamic == 1
        self.jit_config_dict = jit_config.jit_config_dict if jit_config else None
        self._cell_cache_key_extend = cell_cache_key_extend

    def _predict(self, *args, **kwargs):
        """Dedicated routine for predict."""
        # check infer_boost instead check model phase to run infer mode
        # the phaes check condition will delete in the next version
        is_infer = self.jit_config_dict["infer_boost"] == "on"

        if not is_infer and not hasattr(self.obj, "phase"):
            return False, None

        predict_vailid_phase = {"prefill", 'increment'}
        predict_phase = self.obj.phase
        if not is_infer and predict_phase not in predict_vailid_phase:
            return False, None

        args_list = args
        if self.obj is not None:
            args_list = args_list[1:]

        if predict_phase not in self.obj.phase_cache:
            try:
                predict_phase = self.compile(self.fn.__name__, *args_list, **kwargs)
            except Exception as err:
                _pynative_executor.clear_res()
                raise err

        new_inputs = self._generate_run_args(args_list, kwargs, self.sequence_modified, is_predict=True)
        if self.jit_config_dict:
            jit_config_dict = self.jit_config_dict
        else:
            jit_config_dict = JitConfig().jit_config_dict
        self._graph_executor.set_jit_config(jit_config_dict)
        output = self._graph_executor(
            tuple(new_inputs),
            self.obj.phase_cache[self.obj.phase]
        )
        res = _convert_python_data(output)
        return True, res

    def compile_frontend(self, *args, **kwargs):
        """Only compile to the frontend graph."""
        args_list = args
        if self.obj is not None:
            args_list = args_list[1:]
        os.environ['MS_DEV_PRECOMPILE_ONLY'] = '1'
        phase = ""
        _pynative_executor.set_jit_compile_phase(phase)
        phase = self.compile(self.fn.__name__, *args_list, **kwargs)
        _pynative_executor.set_jit_compile_phase(phase)
        os.unsetenv('MS_DEV_PRECOMPILE_ONLY')
        return self._graph_executor.get_func_graph(phase), self._mutable_flags, phase, self.enable_tuple_broaden

    @_wrap_func
    def __call__(self, *args, **kwargs):
        predict, res = self._predict(*args, **kwargs)
        if predict:
            return res
        _check_func_args(self.fn, *args)
        if jit_context() and jit_context().is_nested():
            return jit_context().run_graph("", None, *())
        args_list = args
        if self.obj is not None:
            args_list = args_list[1:]
        phase = ""
        try:
            _pynative_executor.set_jit_compile_phase(phase)
            phase = self.compile(self.fn.__name__, *args_list, **kwargs)
            _pynative_executor.set_jit_compile_phase(phase)
        except Exception as err:
            _pynative_executor.clear_res()
            raise err

        if context.get_context("precompile_only") or os.getenv('MS_DEV_PRECOMPILE_ONLY') == '1':
            return None

        new_inputs = self._generate_run_args(args_list, kwargs, self.sequence_modified)
        if self.jit_config_dict:
            jit_config_dict = self.jit_config_dict
        else:
            jit_config_dict = JitConfig().jit_config_dict
        self._graph_executor.set_jit_config(jit_config_dict)
        output = _pynative_executor.grad_jit(*new_inputs)
        if jit_context():
            if is_stub_tensor(output):
                output = output.stub_sync()
            return jit_context().run_graph(phase, output, *tuple(new_inputs))
        return output

    def compile(self, method_name, *args, phase="", **kwargs):
        """Returns pipeline for the given args."""
        # Chose dynamic shape tensors or actual input tensors as compile args.
        self._graph_executor.set_real_args(args, kwargs)
        compile_args = self._generate_compile_args(args)
        key_id = self._get_key_id()
        if self.input_signature is None:
            compile_args = get_auto_dynamic_shape_args(
                compile_args, key_id, self._enable_auto_dynamic, self.enable_jit_dynamic
            )

        # Add mutable for compile_args for two scene:
        # 1) Origin args is mutable.
        # 2) Args contains sequence with gradient tensor.
        compile_args = _add_mutable_attr(args, compile_args, _pynative_executor.requires_grad())
        mutable_flags = _get_mutable_flags(compile_args)
        self._mutable_flags = mutable_flags
        # Store the _mutable_flags in the cell obj for incremental inference.
        if self.obj is not None:
            self.obj._mutable_flags = mutable_flags
        generate_name, echo_function_name = self._get_generate_name()
        # The full Function name
        full_function_name = generate_name
        create_time = ''

        # Add key with obj
        if self.obj is not None:
            if self.obj.__module__ != self.fn.__module__:
                logger.info(
                    f'The module of `self.obj`: `{self.obj.__module__}` is not same with the module of `self.fn`: '
                    f'`{self.fn.__module__}`')
            self.obj.__parse_method__ = method_name
            if isinstance(self.obj, ms.nn.Cell):
                generate_name = generate_name + '.' + str(self.obj.create_time) + self.obj.phase
                create_time = str(self.obj.create_time)
            else:
                generate_name = generate_name + '.' + str(self._create_time)
                create_time = str(self._create_time)

            generate_name = generate_name + '.' + str(id(self.obj))
            full_function_name = generate_name
        else:
            # Different instance of same class may use same memory(means same obj_id) at diff times.
            # To avoid unexpected phase matched, add create_time to generate_name.
            generate_name = generate_name + '.' + str(self._create_time)
            create_time = str(self._create_time)

        self.enable_tuple_broaden = False
        if hasattr(self.obj, "enable_tuple_broaden"):
            self.enable_tuple_broaden = self.obj.enable_tuple_broaden

        self._graph_executor.set_enable_tuple_broaden(self.enable_tuple_broaden)
        key = self._graph_executor.generate_arguments_key(self.fn, compile_args, kwargs, self.enable_tuple_broaden)
        key = str(key)

        parameter_ids = _get_parameter_ids(args, kwargs)
        if parameter_ids != "":
            key += '.' + parameter_ids

        key += "." + _get_hook_key(*args, **kwargs)
        key += "." + str(_hook_version())

        phase = phase + generate_name + '.' + key

        if self.input_signature is None:
            update_auto_dynamic_shape_phase(compile_args, key_id, phase)

        phase = phase + self._cell_cache_key_extend

        if phase in ms_compile_cache and self._graph_executor.has_compiled(phase):
            # Release resource should be released when CompileInner won't be executed, such as cur_convert_input_
            # generated in generate_arguments_key.
            self._graph_executor.clear_compile_arguments_resource()
            return phase

        _check_recompile(self.obj, compile_args, kwargs, full_function_name, create_time, echo_function_name)

        # If enable compile cache, get the dependency files list and set to graph executor.
        self._set_compile_cache_dep_files()
        if self.jit_config_dict:
            jit_config_dict = self.jit_config_dict
        else:
            jit_config_dict = JitConfig().jit_config_dict

        if self.obj is None:
            # Set an attribute to fn as an identifier.
            setattr(get_func(self.fn), "__jit_function__", True)
            is_compile = self._graph_executor.compile(self.fn, compile_args, kwargs, phase, jit_config_dict)
            delattr(get_func(self.fn), "__jit_function__")
        else:
            if isinstance(self.obj, ms.nn.Cell):
                self._graph_executor.set_weights_values(self.obj.parameters_dict())
            is_compile = self._graph_executor.compile(
                self.obj, compile_args, kwargs, phase, jit_config_dict)

        if not is_compile:
            raise RuntimeError("Executor compile failed.")
        ms_compile_cache.add(phase)
        if hasattr(self.obj, "phase"):
            self.obj.phase_cache[self.obj.phase] = phase

        # If a sequence is modified in-place, it must be included as an input to the top-level graph.
        self.sequence_modified = self._graph_executor.check_func_graph_sequence_parameter(phase)
        return phase

    @staticmethod
    def _optimizer_state_init(opt_states):
        """set data for all optimizer states in case it is executed in graph mode"""
        prefix_list = ["moments", "accum", "moment1", "moment2", "lamb_m", "lamb_v", "mean_grad",
                       "mean_square", "prev"]
        for opt_param in opt_states:
            prefix = opt_param.name[:opt_param.name.find(".")]
            if opt_param.has_init and (prefix in prefix_list or opt_param.name == "global_step"):
                opt_param.init_data()

    def _get_key_id(self):
        """get key id."""
        if isinstance(self.obj, ms.nn.Cell):
            key_id = str(id(self.obj)) + str(self.obj.create_time)
        else:
            key_id = str(id(self.obj)) + str(self._create_time)

        if _pynative_executor.requires_grad():
            key_id = key_id + ".grad"
        return key_id

    def _get_generate_name(self):
        """get generate name."""
        generate_name = self.fn.__module__ + "." + self.fn.__name__ + "." + self.fn.__code__.co_filename + "." + str(
            self.fn.__code__.co_firstlineno)
        echo_function_name = "function \"" + self.fn.__name__ + "\" at the file \"" + self.fn.__code__.co_filename \
                             + "\", line " + str(self.fn.__code__.co_firstlineno)
        if _pynative_executor.requires_grad():
            generate_name = generate_name + ".grad"
        if self.fn.__name__ == _PYNATIVE_PARALLEL_FUNC_NAME:
            generate_name = generate_name[:generate_name.rfind(str(id(self.fn)))] + str(id(self.shard_parent_obj))
        return generate_name, echo_function_name

    def _set_compile_cache_dep_files(self):
        # If enable compile cache, get the dependency files list
        enable_compile_cache = context.get_context("enable_compile_cache")
        if enable_compile_cache is None:
            enable_compile_cache = os.getenv('MS_COMPILER_CACHE_ENABLE')
        if enable_compile_cache is True or enable_compile_cache == "1":
            self._graph_executor.set_compile_cache_dep_files(_get_compile_cache_dep_files())

    def _generate_compile_args_by_enable_dynamic(self, args_list):
        """Generate compile args by enable_dynamic."""
        compile_args = generate_dynamic_tensor_args(args_list, self.dynamic_args_shapes)
        compile_args = _add_mutable_attr(args_list, compile_args, _pynative_executor.requires_grad())
        if self.obj is not None:
            _pynative_executor.set_dynamic_input(self.obj, *compile_args)
        else:
            _pynative_executor.set_dynamic_input(self.fn, *compile_args)
        logger.info(f"dynamic shape compile_args: {compile_args}")
        Validator.check_symbolic_shape(compile_args, args_list)
        return compile_args

    def _generate_compile_args_by_set_inputs(self, args_list):
        """Generate compile args by set_inputs."""
        compile_args = _generate_dyn_compile_args(args_list, self.obj.get_inputs())
        if len(compile_args) != len(args_list):
            raise ValueError(f"The number of actual input tensors: {len(args_list)} is not equal to the number of "
                             f"dynamic shape tensors: {len(compile_args)}.")
        self._graph_executor.check_argument_consistency(compile_args, args_list, "set_inputs")
        Validator.check_symbolic_shape(compile_args, args_list)
        return compile_args

    def _generate_compile_args_by_input_signature(self, args_list):
        """Generate compile args by input_signature."""
        # pylint: disable=R1729
        compile_args = list(_generate_dyn_compile_args(args_list, self.input_signature))
        dyn_shape = any([is_shape_unknown(elem.shape) for elem in compile_args if isinstance(elem, PythonTensor)])
        Validator.check_symbolic_shape(self.input_signature, args_list)
        if dyn_shape:
            # Checkout whether the `sens` has been added to args_list.
            if len(compile_args) == len(args_list) - 1:
                logger.warning(f"The number of actual input args '{len(args_list)}' is one more than the number "
                               f"of input_signature args '{len(compile_args)}'. The last actual args may "
                               f"be 'sens' and added it to compile args.")
                compile_args.append(args_list[-1])
            compile_args = tuple(compile_args)
            self._graph_executor.check_argument_consistency(compile_args, args_list, "input_signature")
            if self.obj is not None:
                _pynative_executor.set_dynamic_input(self.obj, *compile_args)
            else:
                _pynative_executor.set_dynamic_input(self.fn, *compile_args)
        else:
            if not verify_inputs_signature(compile_args, args_list):
                raise ValueError("The input args is incompatible with the args in `input_signature`!")
        return compile_args

    def _check_set_inputs(self):
        """Check if the `set_inputs()` of Cell object has been set."""
        return self.fn.__name__ == 'construct' and isinstance(self.obj, ms.nn.Cell) and self.obj.get_inputs()

    def _generate_compile_args(self, args_list):
        """Chose dynamic shape tensors or actual input tensors as compile args."""
        # Case: The `enable_dynamic` is provided and `set_inputs()` of Cell object has been set.
        if self.enable_jit_dynamic and self._check_set_inputs():
            raise ValueError("When `enable_dynamic` is provided, the `set_inputs()` cannot be set!")
        # Case: The `enable_dynamic` is provided.
        if self.enable_jit_dynamic:
            return self._generate_compile_args_by_enable_dynamic(args_list)
        # Case: The `set_inputs()` of Cell object has been set, using these dynamic shape args as compile args.
        if self._check_set_inputs():
            return self._generate_compile_args_by_set_inputs(args_list)
        # Case: If dynamic shape tensors have been assigned to `input_signature`, they are preferred as compile args.
        if self.input_signature is not None:
            return self._generate_compile_args_by_input_signature(args_list)
        # Case: If the shape of input args is dynamic, get dynamic shape tensor from context and use it to compile.
        return _pynative_executor.get_dynamic_input(args_list)

    def _generate_run_args(self, args_list, kwargs, sequence_modified, is_predict=False):
        """
        Generate input args, which are required for running.

        Args:
            args_list (Tuple): Actual input args.
            kwargs (Dict): Actual input kwargs.
            sequence_modified (List) : The flags of sequence parameter which requires append.

        Returns:
            new_inputs, new input args, which are required for running.
        """
        if self.obj is not None and hasattr(self.obj, '_mutable_flags'):
            mutable_flags = self.obj._mutable_flags
        else:
            mutable_flags = self._mutable_flags
        return _get_args_for_run(self, args_list, kwargs, mutable_flags, sequence_modified, is_predict)

    def _get_func_graph_proto(self, obj, exec_id, ir_type="onnx_ir", use_prefix=False, incremental=False):
        """Get graph proto from pipeline."""
        if use_prefix:
            exec_id = exec_id + '.' + obj.arguments_key
        if self._graph_executor.has_compiled(exec_id) is False:
            return None
        return self._graph_executor.get_func_graph_proto(exec_id, ir_type, incremental)


# The attributes used to identify a given object.
attr_op = {"__str__": lambda x: x.__str__(),
           "__hash__": lambda x: str(x.__hash__()),
           "__module__": lambda x: x.__module__,
           "__name__": lambda x: x.__name__,
           "__qualname__": lambda x: x.__qualname__,
           "__len__": lambda x: str(x.__len__()),
           "__code__": lambda x: x.__code__.co_filename + str(x.__code__.co_firstlineno)
           }


def _is_inner_func(func):
    """Check whether the func is an inner func which needs hash_args parameter."""
    # This is a workaround for inner api, should fix it later.
    inner_func = ["after_shard", "_wrap_container"]
    return func.__name__ in inner_func


def _get_obj_id(input_obj):
    """Get hash id of single object."""
    obj_id = ".".join(
        (map(lambda x: attr_op.get(x)(input_obj) if hasattr(input_obj, x) and getattr(input_obj, x) else "", attr_op)))
    return obj_id + str(id(input_obj))


def _get_jit_hash(hash_input):
    """Get hash value of single object or list of objects."""
    if isinstance(list, tuple):
        return ".".join(map(_get_obj_id, hash_input))
    return _get_obj_id(hash_input)


def _get_hash_obj(options):
    hash_obj = None
    if "hash_args" in options:
        hash_obj = _get_jit_hash(options["hash_args"])
        del options["hash_args"]
    return hash_obj


def _check_option_device(option, device):
    """Check jit options wiwh device"""
    option_device_cfgs = {
        'disable_format_transform': ['GPU'],
        'exec_order': ['Ascend'],
        'ge_options': ['Ascend'],
        'infer_boost': ['Ascend'],
    }
    if option in option_device_cfgs and device not in option_device_cfgs[option]:
        logger.warning(f"For 'jit(options)', the option '{option}' is only support device in "
                       f"'{option_device_cfgs[option]}', but got '{device}', ignore it.")


def _check_option_backend(option, backend):
    """Check jit options wiwh backend"""
    option_backend_cfgs = {
        'disable_format_transform': ['ms_backend'],
        'exec_order': ['ms_backend'],
        'ge_options': ['GE'],
        'infer_boost': ['ms_backend'],
    }
    if option in option_backend_cfgs and backend != '' and backend not in option_backend_cfgs[option]:
        logger.warning(f"For 'jit(options)', the option '{option}' is only support backend in "
                       f"'{option_backend_cfgs[option]}', but got '{backend}', ignore it.")


def _check_disable_format_transform_value(option, disable_format_transform):
    """check disable_format_transform option value"""
    if not isinstance(disable_format_transform, bool):
        raise TypeError(f"For 'jit(options)', the type of '{option}' must be bool, "
                        f"but got {type(disable_format_transform)}.")


def _check_exec_order_value(option, exec_order):
    """check exec_order option value"""
    if not isinstance(exec_order, str):
        raise TypeError(f"For 'jit(options)', the type of '{option}' must be str, but got {type(exec_order)}.")

    if exec_order not in ['bfs', 'dfs']:
        raise ValueError(f"For '{option}', the value of '{option}' must be one of "
                         f"['bfs', 'dfs'], but got '{exec_order}'.")


def _check_ge_options_value(option, ge_options):
    """check ge_options option value"""
    if not isinstance(ge_options, dict):
        raise TypeError(f"For 'jit(options)', the type of '{option}' must be dict, but got {type(ge_options)}.")

    for level, options in ge_options.items():
        if level not in ['global', 'session']:
            raise ValueError(f"For '{option}', the key of '{option}' must be one of "
                             f"['global', 'session'], but got '{level}'.")

        if not isinstance(options, dict):
            raise TypeError(f"For '{option}', the type of {level} options must be dict, "
                            f"but got {type(options)}. The error options: {options}.")

        for key, value in options.items():
            if not isinstance(key, str):
                raise TypeError(f"For '{option}', the type of key and value must be str, "
                                f"but got {type(key)}. The error key is {key}.")
            if not isinstance(value, str):
                raise TypeError(f"For '{option}', the type of key and value must be str, "
                                f"but got {type(value)}. The error value is {value}")


def _check_infer_boost_value(option, value):
    """check infer_boost option value"""
    if not isinstance(value, str):
        raise TypeError(f"For 'jit(options)', the type of '{option}' must be str, but got {type(value)}.")

    if value not in ['on', 'off']:
        raise ValueError(f"For '{option}', the value of '{option}' must be one of ['on', 'off'], but got '{value}'.")


def _check_auto_offload_value(option, value):
    """check auto_offload option value"""
    if not isinstance(value, str):
        raise TypeError(f"For 'jit(options)', the type of '{option}' must be str, but got {type(value)}.")

    if value not in ['weight', 'activaction', 'all']:
        raise ValueError(f"For '{option}', the value of '{option}' must be one of ['weight', 'activaction', 'all'], "
                         f"but got '{value}'.")


def _check_option_value(option, value):
    """check jit options wiwh value"""
    option_valuecheck_funcs = {
        'disable_format_transform': _check_disable_format_transform_value,
        'exec_order': _check_exec_order_value,
        'ge_options': _check_ge_options_value,
        'infer_boost': _check_infer_boost_value,
        'auto_offload': _check_auto_offload_value,
    }
    if option in option_valuecheck_funcs:
        option_valuecheck_funcs[option](option, value)
    else:
        logger.warning(f"For 'jit(options)', the option argument '{option}' is not recognized, please check!"
                       f"For detailed usage of 'jit(options)', please refer to the Mindspore official website.")


def _check_options(options, backend):
    """Check jit options"""
    # check whether there are deprecated parameters in the dict `options`.
    deprecated_args = {'mode': 'capture_mode', 'input_signature': 'dynamic', 'hash_args: ': '',
                       'jit_config': 'jit_level, fullgraph or options', 'compile_once': ''}
    for key, value in deprecated_args.items():
        if key in options:
            log = f"For 'jit', the parameter '{key}' has been deprecated."
            if value != '':
                log += f" Please use the parameter '{value}' instead. For more details, please refer to " \
                       f"https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.jit.html."
            logger.warning(log)
            del options[key]

    # check options' device, backend and value
    for option, value in options.items():
        _check_option_backend(option, backend)
        _check_option_value(option, value)


def _jit_ast(hash_obj, dynamic, jit_config, jit_graph_name):
    """Return the wrapped function for ast mode jit."""

    def wrap_func(func):
        nonlocal hash_obj
        if hasattr(func, "construct"):
            if isinstance(func, ms.nn.Cell):
                # Bound the cell object to get the self arg.
                return types.MethodType(_jit_ast(
                    hash_obj, dynamic, jit_config, func._jit_graph_name)(func.construct.__func__), func)
            if isinstance(func, type) and issubclass(func, ms.nn.Cell):
                func.construct = _jit_ast(
                    hash_obj, dynamic, jit_config, '')(func.construct)
            return func

        if isinstance(func, types.MethodType):
            return types.MethodType(_jit_ast(hash_obj, dynamic, jit_config, '')(func.__func__), func.__self__)

        if not isinstance(func, types.FunctionType):
            logger.warning(f"The func should be function, method or cell instance/class, but got {func}")
            return func

        if hasattr(func, "__wrapped_by_jit__"):
            logger.warning(f"The func {func} should be wrapped by jit only once.")

        if hash_obj is None or not _is_inner_func(func):
            hash_obj = int(time.time() * 1e9)

        @wraps(func)
        def staging_specialize(*args, **kwargs):
            if os.getenv("MS_JIT") == '0':
                return func(*args, **kwargs)

            args, kwargs = _handle_func_args(func, *args, **kwargs)
            process_obj = None
            if args and not isinstance(args[0], PythonTensor) and hasattr(args[0], func.__name__):
                process_obj = args[0]
            # Handle auto mixed precision strategy.
            if not hasattr(func, "amp_strategy"):
                setattr(get_func(func), "amp_strategy", get_curr_amp_strategy())

            jit_graph_name = ''
            if hasattr(staging_specialize, "__jit_graph_name__"):
                jit_graph_name = staging_specialize.__jit_graph_name__
            jit_executor = _JitExecutor(
                func, hash_obj, None, process_obj, jit_config, dynamic, jit_graph_name)
            out = jit_executor(*args, **kwargs)
            if isinstance(process_obj, ms.nn.Cell):
                _clear_auto_parallel_context(process_obj)
            return out

        # `inspect.getfullargspec(func)` will get the specification of the decorated function by default. By set
        # `__signature__` for the decorated function, `inspect.getfullargspec(func)` will get the specification of
        # original `func`.
        staging_specialize.__signature__ = inspect.signature(func)
        setattr(staging_specialize, "__wrapped_by_jit__", True)
        setattr(staging_specialize, "__jit_graph_name__", jit_graph_name)
        return staging_specialize

    return wrap_func


def jit(
        function: Optional[Callable] = None,
        *,
        capture_mode: str = "ast",
        jit_level: str = "O0",
        dynamic: int = 0,
        fullgraph: bool = False,
        backend: str = "",
        **options):
    """
    Create a callable MindSpore graph from a Python function.

    This allows the MindSpore runtime to apply optimizations based on graph.

    Note:
        - It is not supported to run a function with decoration @jit(capture_mode=“bytecode”)
          in static graph mode, in which case the decoration @jit(capture_mode=“bytecode”) is considered invalid.
        - Calls to functions with decorated @jit(capture_mode=“bytecode”) inside functions
          decorated with @jit(capture_mode=“ast”) are not supported,
          and the decoration @jit(capture_mode=“bytecode”) is considered invalid.

    Args:
        function (Callable, optional): The Python function or Cell that will be run as a graph. Default: ``None``.

    Keyword Args:
        capture_mode (str, optional): The method to create a callable MindSpore graph. The value of capture_mode
            should be ``"ast"`` , ``"bytecode"`` or ``"trace"`` . Default: ``"ast"`` .

            - ast: Parse Python ast to build graph.
            - bytecode: Parse Python bytecode to build graph at runtime. This is an experimental prototype
              that is subject to change and/or deletion. Python 3.12 and higher versions are currently
              not supported.
            - trace: Trace the execution of Python code to build graph. This is an experimental prototype
              that is subject to change and/or deletion.

        jit_level (str, optional): Used to control the compilation optimization level. Currently is only effective
            with ms_backend. The value of jit_level should be ``"O0"`` or ``"O1"`` . Default: ``"O0"`` .

            - O0: Except for optimizations that may affect functionality, all other optimizations are turned off.
            - O1: Using commonly used optimizations and automatic operator fusion optimizations. This optimization
              level is experimental and is being improved.

        dynamic (int, optional): Whether dynamic shape compilation should be performed. Default: ``0``. The value range
            is as follows:

            - 0: Do not perform dynamic shape compilation.
            - 1: Enable dynamic shape compilation and automatically detect shape changes.

        fullgraph (bool, optional): Whether to capture the entire function into graph. If False, jit attempts to
            be compatible with all Python syntax in the function as much as possible. If True, we require that the
            entire function can be captured into graph. If this is not possible (that is, if there is Python syntax
            not supported), then it will raise an exception. This currently only applies when capture_mode is ``ast``
            or ``bytecode``. Default: ``False``.
        backend (str, optional): The compilation backend to be used. If this parameter is not set, the framework will
            use ``"GE"`` backend for Atlas training series products and ``"ms_backend"`` backend for others including
            Atlas A2 training series products by default.

            - ms_backend: Utilizes the built-in backend engine of MindSpore for hardware-related compilation
              optimization and execution, supporting multiple hardware forms such as Ascend, GPU, and CPU.
            - GE: Utilizes the GraphEngine, a graph compilation and execution engine within CANN,
              for Ascend model compilation and execution. Note: This backend only supports GRAPH Mode in Ascend,
              only supports whole graph sinking or sub graph sinking in pipeline parallel, and does not support
              dynamic shape scenes. In addition, this backend incurs additional compilation costs and is difficult to
              debug and tune.

        **options (dict): A dictionary of options to pass to the compilation backend.

            Some options are device specific, see the below table for details:

            +---------------------------+---------------------------+-------------------------+
            |  Option Parameters        | Hardware Platform Support |  Backend Support        |
            +===========================+===========================+=========================+
            | disable_format_transform  |  GPU                      |  ms_backend             |
            +---------------------------+---------------------------+-------------------------+
            | exec_order                |  Ascend                   |  ms_backend             |
            +---------------------------+---------------------------+-------------------------+
            | ge_options                |  Ascend                   |  GE                     |
            +---------------------------+---------------------------+-------------------------+
            | infer_boost               |  Ascend                   |  ms_backend             |
            +---------------------------+---------------------------+-------------------------+
            | auto_offload              |  Ascend                   |  ms_backend             |
            +---------------------------+---------------------------+-------------------------+

            - disable_format_transform (bool, optional): Whether to disable the automatic format transform function
              from NCHW to NHWC. When the network training performance of fp16 is worse than fp32,
              `disable_format_transform` can be set to ``True`` to try to improve training performance.
              Default: ``False`` .
            - exec_order (str, optional): Set the sorting method for operator execution, currently only two sorting
              methods are supported: ``"bfs"`` and ``"dfs"`` . Default: ``"bfs"`` .

              - bfs: The default sorting method, breadth priority, good communication masking, relatively good
                performance.
              - dfs: An optional sorting method, depth-first sorting. The performance is relatively worse than that
                of bfs execution order, but it occupies less memory. It is recommended to try dfs in scenarios where
                other execution orders run out of memory (OOM).

            - ge_options (dict): Set options for ge backend. The options are divided into two categories: global,
              and session. This is an experimental prototype that is subject to change and/or deletion.
              For detailed information, please refer to `Ascend community <https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/apiref/ascendgraphapi/atlasgeapi_07_0146.html>`_ .

              - global (dict): Set global options.
              - session (dict): Set session options.

            - infer_boost (str, optional): Used to control the inference mode. Default: ``"off"``, which means
              the inference mode is disabled. The range is as follows:

              - on: Enable inference mode, get better infer performance.
              - off: Disable inference mode, use forward for inference. The performance is poor.

            - auto_offload (str, optional): Used to control whether automatic node offloading is performed. This
              interface is experimental. Default: ``""``, not offload.

              - weight: Automatic offload weight which is initialized with device 'Remote'.
              - activaction: Automatic offload activation tensor.
              - all: Automatic both weight and activation.

    Returns:
        Function, if `function` is not ``None``, returns a callable function
        that will execute the compiled function; If `function` is
        ``None``, returns a decorator and when this decorator invokes
        with a single `function` argument, the callable function is
        equal to the case when `function` is not ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> from mindspore import jit
        ...
        >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        ...
        >>> # Create a callable MindSpore graph by calling jit.
        >>> def tensor_add(x, y):
        ...     z = x + y
        ...     return z
        ...
        >>> tensor_add_graph = jit(function=tensor_add)
        >>> out = tensor_add_graph(x, y)
        >>> print(out)
        Tensor(shape=[1, 1, 3, 3], dtype=Float32, value=
        [[[[ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00],
           [ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00],
           [ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00]]]])
        ...
        >>> # Create a callable MindSpore graph through decorator @jit.
        >>> @jit
        ... def tensor_add_with_dec(x, y):
        ...     z = x + y
        ...     return z
        ...
        >>> out = tensor_add_with_dec(x, y)
        >>> print(out)
        Tensor(shape=[1, 1, 3, 3], dtype=Float32, value=
        [[[[ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00],
           [ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00],
           [ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00]]]])
        ...
        >>> # Create a callable MindSpore graph and capture the entire function into the graph.
        >>> @jit(fullgraph=True)
        ... def tensor_add_fullgraph(x, y):
        ...     z = x + y
        ...     return z
        ...
        >>> out = tensor_add_fullgraph(x, y)
        >>> print(out)
        Tensor(shape=[1, 1, 3, 3], dtype=Float32, value=
        [[[[ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00],
           [ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00],
           [ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00]]]])
        ...
        >>> # Create a callable MindSpore graph by trace mode.
        >>> @jit(capture_mode="trace")
        ... def tensor_add_by_trace(x, y):
        ...     z = x + y
        ...     return z
        ...
        >>> out = tensor_add_by_trace(x, y)
        >>> print(out)
        Tensor(shape=[1, 1, 3, 3], dtype=Float32, value=
        [[[[ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00],
           [ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00],
           [ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00]]]])
        ...
        >>> # Create a callable MindSpore graph with ms_backend and jit_level="O1".
        >>> @jit(backend="ms_backend", jit_level="O1")
        ... def tensor_add_by_trace(x, y):
        ...     z = x + y
        ...     return z
        ...
        >>> out = tensor_add_by_trace(x, y)
        >>> print(out)
        Tensor(shape=[1, 1, 3, 3], dtype=Float32, value=
        [[[[ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00],
           [ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00],
           [ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00]]]])
        ...
        >>> # Create a callable MindSpore graph with GE backend and some ge options on Ascend.
        >>> @jit(backend="GE", ge_options={"global": {"ge.opSelectImplmode": "high_precision"}})
        ... def tensor_add_by_trace(x, y):
        ...     z = x + y
        ...     return z
        ...
        >>> out = tensor_add_by_trace(x, y)
        >>> print(out)
        Tensor(shape=[1, 1, 3, 3], dtype=Float32, value=
        [[[[ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00],
           [ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00],
           [ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00]]]])
        ...
    """

    capture_mode = Validator.check_string(capture_mode, ["ast", "bytecode", "trace"], "capture_mode", "jit")
    jit_level = Validator.check_string(jit_level, ["O0", "O1"], "jit_level", "jit")
    dynamic = Validator.check_int_range(dynamic, 0, 1, Validator.INC_BOTH, "dynamic", "jit")
    fullgraph = Validator.check_bool(fullgraph, "fullgraph", "jit")
    jit_syntax_level = "LAX" if fullgraph is False else "STRICT"
    hash_obj = _get_hash_obj(options)
    _check_options(options, backend)
    options_str = json.dumps(options)
    infer_boost = options['infer_boost'] if 'infer_boost' in options else "off"
    exc_mode = options['exc_mode'] if 'exc_mode' in options else "auto"
    jit_config = JitConfig(jit_level=jit_level, exc_mode=exc_mode, jit_syntax_level=jit_syntax_level,
                           infer_boost=infer_boost, backend=backend, options=options_str)

    if capture_mode == "ast":
        wrap_func = _jit_ast(hash_obj, dynamic, jit_config, '')
    elif capture_mode == "bytecode":
        wrap_func = PIJitCaptureContext(fullgraph=fullgraph, jit_config=jit_config)
    else:
        wrap_func = _jit_trace(jit_config)

    if function is not None:
        return wrap_func(function)
    return wrap_func


def _core(fn=None, **flags):
    """
    A decorator that adds a flag to the function.

    By default, the function is marked as True, enabling to use this decorator to
    set flag to a graph.

    Args:
        fn (Function): Function to add flag. Default: ``None``.
        flags (dict): The following flags can be set core, which indicates that this is a core function or
                      other flag. Default: ``None``.

    Returns:
        Function, the function with core flag.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    # need set the attr and access on c++
    def deco(fn):
        fn._func_graph_flags = {
            'core': True,
            **flags,
        }
        return fn

    if fn is not None:
        ret = deco(fn)
    else:
        ret = deco
    return ret


def _add_flags(fn=None, **flags):
    """
    A decorator that adds a flag to the function.

    Note:
        Only supports bool value.

    Args:
        fn (Function): Function or cell to add flag. Default: ``None``.
        flags (dict): Flags use kwargs. Default: ``None``.

    Returns:
        Function, the function with added flags.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def deco(fn):
        # need set the attr and access on c++
        if not hasattr(fn, "_func_graph_flags"):
            fn._func_graph_flags = {}

        fn._func_graph_flags.update({**flags})
        return fn

    ret = deco
    if fn is not None:
        ret = deco(fn)
    return ret


def _no_recursive(callable_obj):
    """
    Method or function decorator for ignoring recursive check.

    This allows MindSpore to skip the procedure of checking function or method recursive.

    Args:
        callable_obj (Union(method, function)): The function or method to call.

    Returns:
        Function or method with no_recursive flag.

    Raises:
        TypeError: If ms_class is used for non-class types or nn.Cell.
        AttributeError: If the private attributes or magic methods of the class decorated by ms_class is called.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    is_cell_subclass = inspect.isclass(callable_obj) and issubclass(callable_obj, ms.nn.Cell)
    if not is_cell_subclass and not inspect.ismethod(callable_obj) and not inspect.isfunction(callable_obj):
        raise TypeError(f"Decorator no_recursive is used for callable object, but got {callable_obj}.")
    _add_flags(callable_obj, no_recursive=True)
    return callable_obj


def jit_class(cls):
    """
    Class decorator for user-defined classes.

    This allows MindSpore to identify user-defined classes and thus obtain their attributes and methods.

    Args:
        cls (Class): User-defined class.

    Returns:
        Class.

    Raises:
        TypeError: If `jit_class` is used for non-class types or nn.Cell.
        AttributeError: If the private attributes or magic methods of the class decorated with `jit_class` is called.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> from mindspore import jit_class
        ...
        >>> @jit_class
        ... class UserDefinedNet:
        ...     def __init__(self):
        ...         self.value = 10
        ...
        ...     def func(self, x):
        ...         return 2 * x
        ...
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.net = UserDefinedNet()
        ...
        ...     def construct(self, x):
        ...         out = self.net.value + self.net.func(x)
        ...         return out
        ...
        >>> net = Net()
        >>> out = net(5)
        >>> print(out)
        20
    """
    from mindspore import nn
    # Check if cls is of type class.
    if not inspect.isclass(cls):
        raise TypeError(f'Decorator jit_class can only be used for class type, but got {cls}.')
    # Check if cls is nn.Cell.
    if issubclass(cls, nn.cell.Cell):
        raise TypeError(f"Decorator jit_class is used for user-defined classes and cannot be used for nn.Cell: {cls}.")
    setattr(cls, '__ms_class__', True)
    return cls


def _function_forbid_reuse(func):
    if not inspect.isfunction(func):
        raise TypeError(f'Decorator _function_forbid_reuse can only be used for function type, but got {func}.')
    setattr(func, '__function_forbid_reuse__', True)
    return func


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
        broadcast_params_dict.get(param_name).set_data(param)


def _get_auto_split_param_names(parameter_layout_dict):
    auto_split_param_names = []
    for key, value in parameter_layout_dict.items():
        for dim in value[1]:
            if dim != -1:
                auto_split_param_names.append(key)
                break
    return auto_split_param_names


def _parameter_broadcast(obj):
    """
    Parameter broadcast.
    When the parallel mode is 'semi_auto_parallel' or 'auto_parallel', it will broadcast the parameters that have not
    split.
    """
    auto_split_param_names = []
    if _is_in_auto_parallel_mode():
        auto_split_param_names = _get_auto_split_param_names(obj.parameter_layout_dict)

    broadcast_params_dict = obj.parameters_broadcast_dict()
    if auto_split_param_names and broadcast_params_dict:
        broadcast_params_dict = OrderedDict()
        for param_name, param in obj.parameters_broadcast_dict().items():
            if param_name not in auto_split_param_names:
                broadcast_params_dict[param_name] = param
    broadcast_phase = "_broadcast_subgraph"
    _build_broadcast_graph(broadcast_params_dict, broadcast_phase)


def _run_in_jit():
    """In jit, this function always returns true. Otherwise, returns false."""

    def _temp_func():
        return 0

    from mindspore.ops.primitive import constexpr

    @constexpr(check=False)
    def _check_func(func):
        return func is None

    return _check_func(_temp_func)


class _no_grad(contextlib.ContextDecorator):
    """
    Context Manager to disable gradient calculation. When enter this context, we will disable calculate
    gradient. When exit this context, we will resume its prev state.
    Currently, it can only use in Pynative mode. It also can be used as decorator.
    """

    def __init__(self):
        self.prev_state = False

    def __enter__(self):
        self.prev_state = _pynative_executor.enable_grad()
        _pynative_executor.set_enable_grad(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pynative_executor.set_enable_grad(self.prev_state)
        return False


class saved_tensors_hooks:
    """
    A context manager used to customize how saved tensors are packed and unpacked.

    Certain tensors from the forward pass are stored for use in the backward process.
    By using this context, users can specify:

    - How these tensors are packed before saving (pack stage) .
    - How they are restored when accessed during gradient computation (unpack stage) .

    The hooks should have the following signatures:

    - pack_hook(tensor: Tensor) -> Any:
      Accepts a tensor and returns an arbitrary object that represents the stored form of the tensor.

    - unpack_hook(packed: Any) -> Tensor:
      Accepts the object returned by `pack_hook` and restores the corresponding tensor.

    .. note::
        This context manager is currently not supported in Graph and Jit mode.

    .. warning ::
        - To prevent undefined behavior, in-place modification of the original tensor passed to the `pack_hook`
          will throw an exception.
        - To prevent reference cycles, the object returned by `pack_hook` cannot hold a direct reference
          to the original tensor.


    Args:
        pack_hook (Callable): A function that defines how to process a tensor before it is saved during the forward
                              pass.
        unpack_hook (Callable): A function that defines how to recover the tensor when it is needed during the
                                backward computation.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import ops
        >>> def pack_hook(x):
        ...     print("packing ", x)
        ...     return x + 1
        >>>
        >>> def unpack_hook(x):
        ...     print("unpacking ", x)
        ...     return x
        >>>
        >>> def forward_fn(x, y):
        ...     with ms.saved_tensors_hooks(pack_hook, unpack_hook):
        ...         out = x * y
        ...     print("forward end")
        ...     return out
        >>> x = ops.ones(2, dtype=ms.float32)
        >>> y = ops.ones(2, dtype=ms.float32)
        >>> ms.value_and_grad(forward_fn, grad_position=(0,1))(x, y)
        packing [1. 1.]
        packing [1. 1.]
        forward end
        unpacking [2. 2.]
        unpacking [2. 2.]
    """

    @jit_forbidden_register
    def __init__(self, pack_hook, unpack_hook):
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook
        self.pre_disable_async = False
        self.pushed = False

    def __enter__(self):
        self.pre_disable_async = _pynative_executor.disable_frontend_and_bprop_pipeline()
        _pynative_executor.push_saved_tensor_hook(self.pack_hook, self.unpack_hook)
        self.pushed = True

    def __exit__(self, *args):
        if self.pushed:
            _pynative_executor.pop_saved_tensor_hook()
        if not self.pre_disable_async:
            _pynative_executor.enable_frontend_and_bprop_pipeline()


@jit_forbidden_register
@contextlib.contextmanager
def _disable_saved_tensors_hooks(error_msg: str, *, is_error_on_outer_hook=True):
    pre_error_msg = _pynative_executor.disable_saved_tensor_hook(error_msg, is_error_on_outer_hook)
    yield
    _pynative_executor.set_saved_tensor_hook_disable_error_message(pre_error_msg)


class _PyNativeExecutor:
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
        self._executor = PyNativeExecutor_.get_instance()
        self._executor.set_py_exe_path(sys.executable)
        self._executor.set_kernel_build_server_dir(os.path.split(kernel_build_server.__file__)[0] + os.sep)

    @staticmethod
    def parameter_broadcast(obj, phase):
        """
        Run broadcast for parameter.

        Args:
            obj (Cell): The cell instance.
            phase (str): The phase of cell instance.

        Return:
            None.
        """
        if BROADCAST_PHASE not in phase and _get_parameter_broadcast():
            _parameter_broadcast(obj)

    def real_run_op(self, *args):
        """
        Run single op.

        Args:
            args (tuple): Op prim and input arguments.

        Return:
            Tensor, result of run op.
        """
        return self._executor.real_run_op(*args)

    def run_op_async(self, *args):
        """
        Run single op async.

        Args:
            args (tuple): Op prim and input arguments.

        Return:
            StubNode, result of run op.
        """
        return self._executor.run_op_async(*args)

    def new_graph(self, obj, *args, **kwargs):
        """
        Initialize resources for building forward and backward graph.

        Args:
            obj (Function/Cell): The function or cell instance.
            args (tuple): Function or cell input arguments.
            kwargs (dict): keyword arguments.

        Return:
            None.
        """
        self._executor.new_graph(obj, *args, *(kwargs.values()))

    def end_graph(self, obj, output, *args, **kwargs):
        """
        Clean resources after building forward and backward graph.

        Args:
            obj (Function/Cell): The function or cell instance.
            output (Tensor/tuple/list): Function or cell output object.
            args (tuple): Function or cell input arguments.
            kwargs (dict): keyword arguments.

        Return:
            None.
        """
        self._executor.end_graph(obj, output, *args, *(kwargs.values()))

    def check_run(self, grad, obj, weights, grad_hash_id, *args, **kwargs):
        """
        Whether the forward graph need to construct.

        Args:
            grad (GradOperation): The gradoperation object.
            obj (Function/Cell): The function or cell instance.
            grad_hash_id (tuple): The id of objects, which contributes to cache of compiled graph in pynative mode.
            args (tuple): Function or cell input arguments.

        Return:
            bool, specifies whether the forward graph needs to construct.
        """
        return self._executor.check_run(grad, obj, weights, grad_hash_id, *args, **kwargs)

    def grad(self, obj, grad, weights, grad_position, *args):
        """
        Get grad graph.

        Args:
            obj (Function/Cell): The function or cell instance.
            grad (GradOperation): The gradoperation object.
            weights (ParameterTuple): The weights of cell instance.
            grad_position (Union(int, tuple[int])): If int, get the gradient with respect to single input.
              If tuple, get the gradients with respect to selected inputs. 'grad_position' begins with 0. Default: 0.
            args (tuple): Function or cell input arguments.

        Return:
            None.
        """
        return self._executor.grad(grad, obj, weights, grad_position, False, *args)

    def grad_aux(self, obj, grad, weights, grad_position, *args):
        """
        Run grad graph with aux

        Args:
            obj (Function/Cell): The function or cell instance.
            grad (GradOperation): The gradoperation object.
            weights (ParameterTuple): The weights of cell instance.
            grad_position (Union(int, tuple[int])): If int, get the gradient with respect to single input.
              If tuple, get the gradients with respect to selected inputs. 'grad_position' begins with 0. Default: 0.
            args (tuple): Function or cell input arguments.

        Return:
            None.
        """
        return self._executor.grad(grad, obj, weights, grad_position, True, *args)

    def clear_res(self):
        """
        Clean resource for _PyNativeExecutor.

        Return:
            None.
        """
        return self._executor.clear_res()

    def sync(self):
        """
        SyncStream.

        Return:
            None.
        """
        self._executor.sync()

    def grad_jit(self, *args):
        """
        Building grad graph decorated by jit.

        Args:
            args (tuple): Function or cell decorated by jit input arguments.

        Return:
            output: The output object of function or cell decorated by jit.
        """
        output = self._executor.grad_jit(*args)
        return output

    def call_custom_bprop(self, obj, output, *args, **kwargs):
        """
        Call custom bprop to build variable for cell bprop.
        Args:
            obj (Cell): The function or cell instance.
            output (Tensor/tuple/list): Function or cell output object.
            args (tuple): Function or cell input arguments.
            kwargs (dict): keyword arguments.

        Return:
            None.
        """
        return self._executor.call_custom_bprop(obj, output, *args, *(kwargs.values()))

    def grad_flag(self):
        """
        The flag of whether the net building grad graph.

        Return:
            bool, whether building grad graph.
        """
        return self._executor.grad_flag()

    def set_grad_flag(self, flag):
        """
        Set the flag of building grad graph.

        Args:
            flag (bool): Specifying whether building grad graph.

        Return:
            None.
        """
        self._executor.set_grad_flag(flag)

    def set_async_for_graph(self, flag):
        """
        Set the flag for graph async run.

        Args:
            flag (bool): Specifying whether enable graph async run.

        Return:
            None.
        """
        self._executor.set_async_for_graph(flag)

    def enable_grad(self):
        """
        The global flag that whether need to calculate gradient use in no_grad.

        Return:
            bool, whether needing to calculate gradient.
        """
        return self._executor.enable_grad()

    def set_enable_grad(self, flag):
        """
        Set the flag of calculating gradient.

        Args:
            flag (bool): Specifying whether calculating gradient.

        Return:
            None.
        """
        self._executor.set_enable_grad(flag)

    def requires_grad(self):
        """
        When both enable_grad is true and grad_flag is true, that the flag requires_grad will be true.

        Args:
            flag (bool): Specifying whether calculating gradient.

        Return:
            None.
        """
        return self._executor.requires_grad()

    def set_jit_compile_phase(self, phase):
        """
        Set jit phase

        Args:
            phase (str): The phase of cell/function instance.
        Return:
            None.
        """
        self._executor.set_jit_compile_phase(phase)

    def set_is_run_recompute(self, status):
        """
        Set recompute grad is compiling

        Args:
            status(bool): grad is in recompute status
        Return:
            None.
        """
        self._executor.set_is_run_recompute(status)

    def high_order(self):
        """
        Is high order of current scene, this is a inner interface.

        Return:
            Bool.
        """
        return self._executor.high_order()

    def set_cell_use_dynamic_shape_process(self, flag):
        """
        Set the dynamic shape flag of eval process.

        Args:
            flag (bool): Specifying whether using a dynamic process.

        Return:
            None.
        """
        self._executor.set_cell_use_dynamic_shape_process(flag)

    def set_dynamic_input(self, obj, *args):
        """
        Set dynamic shape tensor of input arguments.

        Args:
            obj (Function/Cell): The function or cell instance.
            args (tuple): Function or cell dynamic input arguments.

        Return:
            None.
        """
        self._executor.set_dynamic_input(obj, *args)

    def get_dynamic_input(self, *actual_args):
        """
        Get dynamic shape arguments according to actual input arguments.

        Args:
            actual_args(tuple): Actual input arguments of Function or Cell.

        Return:
            dynamic_shape_args(tuple): Dynamic shape arguments of Function or Cell.
        """
        return self._executor.get_dynamic_input(*actual_args)

    def set_mixed_precision_type(self, mixed_precision_type, is_push=True):
        """
        The value of mixed precision type.

        Args:
            type(MixedPrecisionType): Mix precision type.
            is_push(bool): If called by __enter__, is push will be True

        Return:
            None.
        """

        return self._executor.set_mixed_precision_type(mixed_precision_type, is_push)

    def constant_folding(self, *args):
        """
        Get value by infer value.

        Args:
            args (tuple): Op prim and input arguments.

        Return:
            Tensor, the value get by op infer.
        """
        return self._executor.constant_folding(*args)

    def set_creation_type(self, tensor, creation_type):
        """
        Set tensor's view creation type

        Args:
            tensor (Tensor): input tensor.
            creation_type (CreationType): The type of view tensor when it is created.

        Return:
            None.
        """
        return self._executor.set_creation_type(tensor, creation_type)

    def queue_backward_final_callback(self, callback):
        """
        add backward final callback

        Args:
            callback(Function): callback function.

        Return:
            None.
        """
        return self._executor.queue_backward_final_callback(callback)

    def push_saved_tensor_hook(self, pack_hook, unpack_hook):
        """
        Push default saved tensor hook

        Args:
            pack_hook (Callable): pack hook
            unpack_hook (Callable): unpack hook

        Return:
            None.
        """
        return self._executor.push_saved_tensor_hook(pack_hook, unpack_hook)

    def pop_saved_tensor_hook(self):
        """
        Pop default saved tensor hook

        Return:
            None.
        """
        return self._executor.pop_saved_tensor_hook()

    def disable_saved_tensor_hook(self, error_msg, is_error_on_outer_hook):
        """
        Disable default saved tensor hook

        Args:
            error_msg (str): error message to raise when disabling conflicts.
            is_error_on_outer_hook (bool): Whether to raise an error if called inside
            an active saved tensor hook scope.

        Return:
            None.
        """
        return self._executor.disable_saved_tensor_hook(error_msg, is_error_on_outer_hook)

    def set_saved_tensor_hook_disable_error_message(self, error_msg):
        """
        Set saved tensor hook disable error message

        Args:
            error_msg (str): current disable message

        Return:
            None.
        """
        self._executor.set_saved_tensor_hook_disable_error_message(error_msg)

    def disable_frontend_and_bprop_pipeline(self):
        """
        Disable frontend and bprop pipeline.

        Return:
            pre_is_disable(bool), indicates whether the pipeline was already disabled before this call.
        """
        return self._executor.disable_frontend_and_bprop_pipeline()

    def enable_frontend_and_bprop_pipeline(self):
        """
        Enable frontend and bprop pipeline.

        Return:
            None.
        """
        self._executor.enable_frontend_and_bprop_pipeline()

    def is_saved_tensor_hook_active(self):
        """
        Is current saved tensor hook active

        Return:
            bool.
        """
        return self._executor.is_saved_tensor_hook_active()

    def get_current_autodiff_engine_id(self):
        """
        get current autodiff engine id, if not in autodiff engine, return -1

        Return:
            autodiff engine id.
        """
        return self._executor.get_current_autodiff_engine_id()


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

    def __init__(self):
        # create needed graph by lazy mode
        self.is_init = False
        self.enable_tuple_broaden = False
        self._graph_executor = GraphExecutor_.get_instance()
        self._graph_executor.set_py_exe_path(sys.executable)
        self._graph_executor.set_kernel_build_server_dir(os.path.split(kernel_build_server.__file__)[0] + os.sep)

    def init_dataset(self, queue_name, dataset_size, batch_size, dataset_types, dataset_shapes,
                     input_indexs, phase='dataset', need_run=True):
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
                                 phase=phase,
                                 need_run=need_run):
            raise RuntimeError("Failure to init and dataset subgraph!")
        self._graph_executor.set_queue_name(queue_name)
        return True

    def set_queue_name(self, queue_name):
        """
        while a mode use shared dataset with others, need set queue_name which saved in data_set
        :param queue_name:
        :return:
        """
        self._graph_executor.set_queue_name(queue_name)

    def get_queue_name(self, dataset_phase):
        """
        Get cached queue name for the graph loaded from compile cache.
        :return: cached queue name
        """
        return self._graph_executor.get_queue_name(dataset_phase)

    @staticmethod
    def _set_dataset_mode(obj):
        """set dataset mode."""
        # decide whether to sink based on the sink_mode flag which is set in connect_network_with_dataset
        if 'sink_mode' in obj.get_flags().keys() and obj.get_flags()['sink_mode'] is True:
            _set_dataset_mode_config('sink')
        else:
            _set_dataset_mode_config('normal')

    def _build_data_graph(self, obj, phase):
        self._graph_executor.build_data_graph(obj.parameters_dict(), phase)

    def _set_compile_cache_dep_files(self, phase):
        # If enable compile cache, get the dependency files list
        enable_compile_cache = context.get_context("enable_compile_cache")
        if enable_compile_cache is None:
            enable_compile_cache = os.getenv('MS_COMPILER_CACHE_ENABLE')
        if enable_compile_cache is True or enable_compile_cache == "1":
            self._graph_executor.set_compile_cache_dep_files(_get_compile_cache_dep_files())

    def compile(self, obj, *args, phase='predict', do_convert=True, jit_config_dict=None, **kwargs):
        """
        Compiles graph.

        Args:
            obj (Function/Cell): The function or cell instance need compile.
            phase (str): The name of compile phase. Default: 'predict'.
            do_convert (bool): When set to True, convert ME graph to GE graph after compiling graph.
            jit_config_dict (dict): Jit config for compile. Default: ``None``.
            args (tuple): Args of the Cell object.
            kwargs (dict): Kwargs of the Cell object.

        Return:
            Str, the full phase of the cell.
            Bool, if the graph has been compiled before, return False, else return True.
        """
        _init_auto_parallel_context(obj)
        obj.__parse_method__ = 'construct'
        if not hasattr(obj, obj.__parse_method__):
            raise AttributeError(
                'The class {} does not have method {}'.format(obj.__class__.__name__, obj.__parse_method__))
        inner_func = inspect.unwrap(obj.construct)
        if hasattr(get_func(inner_func), ENABLE_DYNAMIC):
            raise ValueError(
                "When using set_context(mode=GRAPH_MODE) together with nn.Cell, the 'enable_dynamic' cannot be set!"
            )
        key_id = str(id(obj)) + str(obj.create_time)
        args = get_auto_dynamic_shape_args(args, key_id)

        self.enable_tuple_broaden = False
        if hasattr(obj, "enable_tuple_broaden"):
            self.enable_tuple_broaden = obj.enable_tuple_broaden
        logger.debug(f"Convert the network: {do_convert}.")
        self._graph_executor.set_enable_tuple_broaden(self.enable_tuple_broaden)

        key = self._graph_executor.generate_arguments_key(obj, args, kwargs, self.enable_tuple_broaden)
        key = str(key)

        # When exist parameter in the top graph inputs, need check if the parameter object has changed.
        parameter_ids = _get_parameter_ids(args, kwargs)
        if parameter_ids != "":
            key += '.' + parameter_ids

        key += "." + _get_hook_key(*args, **kwargs)
        key += "." + str(_hook_version())

        obj.arguments_key = key

        raw_phase = phase

        phase = _real_phase(phase, obj)
        obj.phase_cache[raw_phase] = phase
        update_auto_dynamic_shape_phase(args, key_id, phase)
        obj.current_phase = phase
        obj._add_attr("compile_phase", phase)
        obj.compile_phase = phase
        if phase in obj.compile_cache and self.has_compiled(phase):
            logger.debug("%r graph has existed.", phase)
            # Release resource should be released when CompileInner won't be executed, such as cur_convert_input_
            # generated in generate_arguments_key.
            self._graph_executor.clear_compile_arguments_resource()
            _clear_auto_parallel_context(obj)
            return phase, False

        full_function_name = obj.__class__.__name__ + '.' + str(obj.total_instance_count) + '.' + str(id(type(obj)))
        echo_function_name = obj.__class__.__name__
        _check_recompile(obj, args, kwargs, full_function_name, obj.create_time, echo_function_name)

        obj.check_names()
        _check_full_batch()
        self._set_dataset_mode(obj)
        self._set_compile_cache_dep_files(phase)

        self._graph_executor.set_weights_values(obj.parameters_dict())
        if not jit_config_dict:
            jit_config_dict = JitConfig().jit_config_dict
        gc.collect()
        result = self._graph_executor.compile(
            obj, args, kwargs, phase, jit_config_dict)
        obj.compile_cache.add(phase)
        if not result:
            raise RuntimeError("Executor compile failed.")
        graph = self._graph_executor.get_func_graph(phase)

        if graph is None:
            raise RuntimeError("Compile graph failed for phase {}.".format(phase))

        auto_parallel_mode = _is_in_auto_parallel_mode() or _is_parallel_mode()
        if not auto_parallel_mode:
            replace = obj.init_parameters_data(auto_parallel_mode=auto_parallel_mode)
            self._update_param_node_default_input(phase, replace)
        elif 'skip_auto_parallel_compile' not in obj.get_flags().keys():
            obj.parameter_layout_dict = self._graph_executor.get_parameter_layout(phase)
            obj.parallel_parameter_name_list = self._graph_executor.get_parallel_parameter_name_list(phase)
        if "export.air" in phase:
            self._build_data_graph(obj, phase)
        elif BROADCAST_PHASE not in phase and _get_parameter_broadcast():
            _parameter_broadcast(obj)
        _clear_auto_parallel_context(obj)
        return phase, True

    def _update_param_node_default_input(self, phase, replace):
        new_param = {x.name: replace[x] for x in replace if id(x) != id(replace[x])}
        return self._graph_executor.updata_param_node_default_input(phase, new_param)

    def set_real_args(self, args, kwargs):
        """Set real arguments to graph executor."""
        self._graph_executor.set_real_args(args, kwargs)

    def _get_shard_strategy(self, obj):
        real_phase = _real_phase(obj.phase, obj)
        return self._graph_executor.get_strategy(real_phase)

    def _get_num_parallel_ops(self, obj):
        real_phase = _real_phase(obj.phase, obj)
        return self._graph_executor.get_num_parallel_ops(real_phase)

    def _get_allreduce_fusion(self, obj):
        real_phase = _real_phase(obj.phase, obj)
        return self._graph_executor.get_allreduce_fusion(real_phase)

    def __call__(self, obj, *args, phase='predict'):
        if context.get_context("precompile_only") or os.getenv('MS_DEV_PRECOMPILE_ONLY') == '1' or _is_role_sched():
            return None
        return self.run(obj, *args, phase=phase)

    def has_compiled(self, phase='predict'):
        """
        Specify whether have been compiled.

        Args:
            phase (str): The phase name. Default: 'predict'.

        Returns:
            bool, specifies whether the specific graph has been compiled.
        """
        return self._graph_executor.has_compiled(phase)

    def flops_collection(self, phase='train'):
        """
        Specify whether have been compiled.

        Args:
            phase (str): The phase name. Default: 'predict'.

        Returns:
            bool, specifies whether the specific graph has been compiled.
        """
        return self._graph_executor.flops_collection(phase)

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
            obj (Cell): The cell object.
            args (tuple): Args of the Cell object.
            phase (str): The phase name. Default: 'predict'.

        Returns:
            Tensor/Tuple, return execute result.
        """
        if phase == 'save':
            exe_phase = _real_phase(phase, obj)
            return self._graph_executor((), exe_phase)

        phase_real = _real_phase(phase, obj)
        if self.has_compiled(phase_real):
            return self._exec_pip(obj, *args, phase=phase_real)
        raise KeyError('{} graph is not exist.'.format(phase_real))

    def del_net_res(self, obj, net_id):
        """Clear the memory resource of a network."""
        self._graph_executor.del_net_res(obj, net_id)

    def _get_func_graph(self, obj, exec_id, use_prefix=False):
        """Get func graph from pipeline."""
        if use_prefix:
            exec_id = exec_id + '.' + obj.arguments_key
        if self._graph_executor.has_compiled(exec_id) is False:
            return None
        return self._graph_executor.get_func_graph(exec_id)

    def _get_func_graph_proto(self, obj, exec_id, ir_type="onnx_ir", use_prefix=False, incremental=False):
        """Get graph proto from pipeline."""
        if use_prefix:
            exec_id = exec_id + '.' + obj.arguments_key
        if self._graph_executor.has_compiled(exec_id) is False:
            return None
        return self._graph_executor.get_func_graph_proto(exec_id, ir_type, incremental)

    def _get_onnx_func_graph_proto(self, obj, exec_id, use_prefix=False, input_names=None, output_names=None,
                                   opset_version=11, export_params=True, keep_initializers_as_inputs=False,
                                   dynamic_axes=None, extra_save_params=False, save_file_dir=None):
        """Get graph proto from pipeline."""
        if use_prefix:
            exec_id = exec_id + '.' + obj.arguments_key
        if self._graph_executor.has_compiled(exec_id) is False:
            return None

        return self._graph_executor.get_onnx_func_graph_proto(exec_id, input_names, output_names, opset_version,
                                                              export_params, keep_initializers_as_inputs, dynamic_axes,
                                                              extra_save_params, save_file_dir)

    def get_optimize_graph_proto(self, obj):
        """Return optimize graph binary proto."""
        exec_id = _real_phase(obj.phase, obj)
        if self._graph_executor.has_compiled(exec_id) is False:
            return None
        graph_proto = self._graph_executor.get_optimize_graph_proto(exec_id)
        if isinstance(graph_proto, str) and graph_proto == "":
            logger.warning("Can not get optimize graph proto. Instead, try to find function graph.")
            graph_proto = obj.get_func_graph_proto()
        return graph_proto

    def export(self, file_name, graph_id, enc_key=None, encrypt_func=None):
        """
        Export graph.

        Args:
            file_name (str): File name of model to export
            graph_id (str): id of graph to be exported
        """
        self._graph_executor.export_graph(file_name, graph_id, encrypt_func, enc_key)


def ms_memory_recycle():
    """
    Recycle memory used by MindSpore.
    When train multi Neural network models in one process, memory used by MindSpore is very large,
    this is because MindSpore cached runtime memory for every model.
    To recycle these cached memory, users can call this function after training of one model.

    Examples:
        >>> import mindspore as ms
        >>> ms.ms_memory_recycle()
    """
    if ms_compile_cache:
        _cell_graph_executor.del_net_res(None, ms_compile_cache)
        if os.getenv('MS_DEV_JIT_PIPELINE') != '0':
            JitExecutor_.get_instance().del_net_res(None, ms_compile_cache)
        ms_compile_cache.clear()
    for cell_cache in cells_compile_cache.values():
        if cell_cache:
            _cell_graph_executor.del_net_res(None, cell_cache)
            cell_cache.clear()
    _ms_memory_recycle()


def set_recursion_limit(recursion_limit=1000):
    """
    Specify the recursion depth limit of function call before compiling graph.
    It needs to be call when the nested function call is too deep or the number of sub graphs is too large.
    If `recursion_limit` is set larger than before, the system max stack depth should be set larger too,
    otherwise a `core dumped` exception may be raised because of system stack overflow.

    Args:
        recursion_limit (int, optional): The recursion depth limit. Must be a positive integer. Default: ``1000`` .

    Examples:
        >>> import mindspore as ms
        >>> ms.set_recursion_limit(10000)
    """
    recursion_limit = Validator.check_positive_int(recursion_limit)
    GraphExecutor_.get_instance().set_max_call_depth(recursion_limit)


def _bind_device_context():
    """Bind device context to current thread"""
    _bind_device_ctx()


def flops_collection(phase='train'):
    """
    Recycle memory used by MindSpore.
    When train multi Neural network models in one process, memory used by MindSpore is very large,
    this is because MindSpore cached runtime memory for every model.
    To recycle these cached memory, users can call this function after training of one model.

    Examples:
        >>> import mindspore as ms
        >>> ms.ms_memory_recycle()
    """
    return _cell_graph_executor.flops_collection(phase)


class _ScriptGraph:
    """Store the graph compiled by the frontend compiler."""

    def __init__(self, func_graph, func, origin_cell, mutable_flags, phase, enable_tuple_broaden):
        self.func_graph = func_graph
        self.func = func
        self.origin_cell = origin_cell
        self.mutable_flags = mutable_flags
        self.phase = phase
        self.enable_tuple_broaden = enable_tuple_broaden

    def print(self):
        """Print the MindIR of the frontend graph."""
        graph_str = dump_func_graph(self.func_graph)
        print(graph_str, flush=True)


def _frontend_compile_ast(dynamic, jit_config, jit_graph_name=''):
    """Return the wrapped function for ast mode jit."""

    def wrap_func(func):
        if hasattr(func, "construct") and isinstance(func, ms.nn.Cell):
            # Bound the cell object to get the self arg.
            return types.MethodType(_frontend_compile_ast(dynamic, jit_config,
                                                          func._jit_graph_name)(func.construct.__func__), func)

        if isinstance(func, types.MethodType):
            return types.MethodType(_frontend_compile_ast(dynamic, jit_config)(func.__func__), func.__self__)

        if not isinstance(func, types.FunctionType):
            logger.warning(f"The func should be function, method or cell instance/class, but got {func}")
            return func

        hash_obj = int(time.time() * 1e9)

        @wraps(func)
        def staging_specialize(*args, **kwargs):
            if os.getenv("MS_JIT") == '0':
                return func(*args, **kwargs)

            args, kwargs = _handle_func_args(func, *args, **kwargs)
            process_obj = None
            if args and not isinstance(args[0], PythonTensor) and hasattr(args[0], func.__name__):
                process_obj = args[0]
            # Handle auto mixed precision strategy.
            if not hasattr(func, "amp_strategy"):
                setattr(get_func(func), "amp_strategy", get_curr_amp_strategy())

            jit_graph_name = ''
            if hasattr(staging_specialize, "__jit_graph_name__"):
                jit_graph_name = staging_specialize.__jit_graph_name__
            jit_executor = _JitExecutor(func, hash_obj, None, process_obj, jit_config, dynamic, jit_graph_name)
            func_graph, mutable_flags, phase, enable_tuple_broaden = jit_executor.compile_frontend(*args, **kwargs)
            return _ScriptGraph(func_graph, func, process_obj, mutable_flags, phase, enable_tuple_broaden)

        # `inspect.getfullargspec(func)` will get the specification of the decorated function by default. By set
        # `__signature__` for the decorated function, `inspect.getfullargspec(func)` will get the specification of
        # original `func`.
        staging_specialize.__signature__ = inspect.signature(func)
        setattr(staging_specialize, "__jit_graph_name__", jit_graph_name)
        return staging_specialize

    return wrap_func


def _frontend_compile(function: Callable,
                      *,
                      dynamic: int = 0,
                      fullgraph: bool = False):
    """
    Create a frontend MindSpore graph from a Python function by the ast capture mode.

    Args:
        function (Callable, optional): The Python function or Cell instance that will be compiled as a frontend graph.
            Default: ``None``.

    Keyword Args:
        dynamic (int, optional): Whether dynamic shape compilation should be performed. Default: ``0``. The value range
            is as follows:

            - `0`: Do not perform dynamic shape compilation.
            - `1`: Enable dynamic shape compilation and automatically detect shape changes.

        fullgraph (bool, optional): Whether to capture the entire function into graph. If False, jit attempts to
            be compatible with all Python syntax in the function as much as possible. If True, we require that the
            entire function can be captured into graph. If this is not possible (that is, if there is Python syntax
            not supported), then it will raise an exception. This currently only applies when capture_mode is ``ast``
            or ``bytecode``. Default: ``False``.

    Returns:
        a :class:`_ScriptGraph` object.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> from mindspore.common.api import _frontend_compile
        ...
        >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        ...
        >>> def tensor_add(x, y):
        ...     z = x + y
        ...     return z
        ...
        >>> tensor_add_graph = _frontend_compile(tensor_add)(x, y)
        >>> tensor_add_graph.print()
        ...
    """

    dynamic = Validator.check_int_range(dynamic, 0, 1, Validator.INC_BOTH, "dynamic", "jit")
    fullgraph = Validator.check_bool(fullgraph, "fullgraph", "jit")
    jit_syntax_level = "LAX" if fullgraph is False else "STRICT"
    jit_config = JitConfig(jit_syntax_level=jit_syntax_level)
    return _frontend_compile_ast(dynamic, jit_config)(function)


class _GraphFragment(_GraphFragment_):
    """
    Represents the output by backend graph split.
    """

    def __init__(self, frag):
        if frag is None or not isinstance(frag, _GraphFragment_):
            raise TypeError(f"Expect input `frag` to be a _GraphFragment_, but got {type(frag)}")
        _GraphFragment_.__init__(self, frag)

    def __call__(self, *args):
        return super().__call__(args)

    def __repr__(self):
        return self.__str__()

    def id(self):
        return self.id_()

    def is_graph(self):
        return self.is_graph_()

    def py_key(self):
        return self.py_key_()

    def args_list(self):
        return self.args_list_()


def _graph_split(script_graph):
    """
    Split the script_graph into several fragments according to the nodes with the split op attribute.

    Args:
        a :class:`_ScriptGraph` object.

    Returns:
        several :class:`_GraphFragment` object.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> from mindspore.common.api import _frontend_compile, _graph_split
        ...
        >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> add = ops.Add().add_prim_attr("split_op", True).add_prim_attr("func_id", "add_func")
        ...
        >>> def tensor_add(x, y):
        ...     z1 = x + y
        ...     z2 = add(z1, x)
        ...     return z2
        ...
        >>> tensor_add_graph = _frontend_compile(tensor_add)(x, y)
        >>> frags = _graph_split(tensor_add_graph)
        >>> print(frags)
        ...
    """
    outputs = JitExecutor_.get_instance().split_graph(script_graph.func_graph)
    fragments = []
    for arg in outputs:
        fragments.append(_GraphFragment(arg))
    return fragments


def register_saved_tensors_hooks(pack_hook, unpack_hook):
    """
    A decorator used in graph mode to customize the packing and unpacking of saved tensors.

    It is functionally equivalent to using `with mindspore.saved_tensors_hooks(pack_hook, unpack_hook)` in
    PyNative mode. 

    For more details, please refer to :class:`mindspore.saved_tensors_hooks`.

    .. note::
        - This decorator only supports graph mode.
        - `pack_hook` and `unpack_hook` must satisfy the syntax constraints of the graph mode.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import register_saved_tensors_hooks
        >>> from mindspore import ops
        >>>
        >>> def pack_hook(x):
        ...     print("packing ", x)
        ...     return x + 1
        >>>
        >>> def unpack_hook(x):
        ...     print("unpacking ", x)
        ...     return x
        >>>
        >>> @register_saved_tensors_hooks(pack_hook, unpack_hook)
        ... def forward_fn(x, y):
        ...     out = x * y
        ...     return out
        >>>
        >>> x = ops.ones(2, dtype=ms.float32)
        >>> y = ops.ones(2, dtype=ms.float32)
        >>> ms.jit(ms.value_and_grad(forward_fn, grad_position=(0,1)))(x, y)
        packing
        Tensor(shape=[2], dtype=Float32, value=[ 1.00000000e+00  1.00000000e+00])
        packing
        Tensor(shape=[2], dtype=Float32, value=[ 1.00000000e+00  1.00000000e+00])
        unpacking
        Tensor(shape=[2], dtype=Float32, value=[ 2.00000000e+00  2.00000000e+00])
        unpacking
        Tensor(shape=[2], dtype=Float32, value=[ 2.00000000e+00  2.00000000e+00])
    """

    def decorator(func):
        # If the function is wrapped (e.g., by functools.wraps), apply hooks to the original function
        # to ensure they are accessible even when multiple decorators are stacked.
        wrapped_func = inspect.unwrap(func)
        setattr(wrapped_func, "_saved_tensors_pack_hook", pack_hook)
        setattr(wrapped_func, "_saved_tensors_unpack_hook", unpack_hook)
        return func
    return decorator


_cell_graph_executor = _CellGraphExecutor()
_pynative_executor = _PyNativeExecutor()
