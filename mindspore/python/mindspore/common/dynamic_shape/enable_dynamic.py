# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Define enable_dynamic decorator."""
import types
import inspect
from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore.common._utils import get_func, is_dim_unknown
from mindspore.common.dynamic_shape.auto_dynamic_shape import SHAPE_DIM_ANY


ENABLE_DYNAMIC = "__enable_dynamic__"


def _check_element_valid(item, shape, name):
    """Check elements in shape."""
    if item is not SHAPE_DIM_ANY and (isinstance(item, int) and item <= 0):
        raise TypeError(f"The argument '{name}' has invalid shape '{shape}', only supports None " \
                        f"or a tuple/list of positive integers and None.")
    return True


def _check_arg_shape_valid(arg, name):
    """Check if the shape of arg is valid"""
    #if the shape of arg is None
    if isinstance(arg, Tensor) and is_dim_unknown(arg.shape):
        return True
    if isinstance(arg, Tensor) and \
        SHAPE_DIM_ANY in arg.shape and \
        all(_check_element_valid(item, arg.shape, name) for item in arg.shape):
        return True
    if isinstance(arg, (tuple, list)) and any(_check_arg_shape_valid(item, name) for item in arg):
        return True
    return False


def _check_arg_type_valid(arg, name):
    """Check if the type of arg is valid."""
    if isinstance(arg, Tensor):
        return
    if isinstance(arg, (tuple, list)):
        for item in arg:
            _check_arg_type_valid(item, name)
    else:
        raise TypeError(f"The decorator enable_dynamic only supports Tensor " \
                        f"or a tuple/list of Tensor, but the argument : {name} is type of:{type(arg)}.")


def _check_input_valid(arg):
    """Check if real argument is valid."""
    if isinstance(arg, Tensor):
        if not all(isinstance(item, int) and item > 0 for item in arg.shape):
            raise ValueError("When using decorator enable_dynamic, the corresponding shape of inputs should be " \
                             "a tuple/list of positive integers")
    elif isinstance(arg, (tuple, list)):
        for item in arg:
            _check_input_valid(item)
    else:
        raise TypeError("When using decorator enable_dynamic, the corresponding inputs only supports Tensor " \
                        "or a tuple/list of Tensor.")


def _check_arg_type_shape(arg, dyn_arg, name):
    """Check the type, shape and dtype of real argument."""
    if isinstance(arg, Tensor) and isinstance(dyn_arg, Tensor):
        if arg.dtype != dyn_arg.dtype:
            raise TypeError(f"When using decorator enable_dynamic, input tensor dtype = {arg.dtype}, " \
                            f"dynamic tensor dtype = {dyn_arg.dtype}, tensor dtypes are not the same.")
        if is_dim_unknown(dyn_arg.shape):
            return
        if len(arg.shape) != len(dyn_arg.shape) or \
            any(y is not SHAPE_DIM_ANY and x != y for x, y in zip(arg.shape, dyn_arg.shape)):
            raise ValueError(f"When using decorator enable_dynamic, input tensor shape = {arg.shape}, " \
                             f"dynamic tensor shape = {dyn_arg.shape}, tensor shapes are not the same.")
    elif isinstance(arg, (tuple, list)) and isinstance(dyn_arg, (tuple, list)):
        if len(arg) != len(dyn_arg):
            raise ValueError("Input sequences must have the same structure and length.")
        for x, y in zip(arg, dyn_arg):
            _check_arg_type_shape(x, y, name)
    else:
        raise TypeError(f"When using decorator enable_dynamic, the type between argument '{name}' " \
                        f"and corresponding input are not the same.")


def generate_dynamic_sequence_args(args_list, dyn_args_list):
    """Generate dynamic shapes for input sequence"""
    if isinstance(args_list, Tensor):
        return dyn_args_list if args_list.shape != dyn_args_list.shape else args_list
    result = []
    for x, y in zip(args_list, dyn_args_list):
        result.append(generate_dynamic_sequence_args(x, y))
    return type(args_list)(result)


def generate_dynamic_tensor_args(args_list, dynamic_shapes):
    """Generate compile args with dynamic_shapes"""
    new_compile_args = list(args_list)
    for index, arg in enumerate(args_list):
        if isinstance(arg, (tuple, list)) and not hasattr(arg, "__ms_mutable__"):
            raise ValueError("When using decorator enable_dynamic, the corresponding attribute of input should be " \
                             "mutable(tuple/list)")
        if index not in dynamic_shapes:
            continue
        _check_input_valid(arg)
        name, dyn_arg = dynamic_shapes[index]
        _check_arg_type_shape(arg, dyn_arg, name)
        new_compile_args[index] = generate_dynamic_sequence_args(arg, dyn_arg)
    return new_compile_args


def enable_dynamic(**kwargs):
    r"""
    Use to specify whether the shape of the parameter is dynamic shape or dynamic rank.

    Note:
        - It needs to be used in conjunction with the JIT interface. Without using the JIT decorator,
          the dynamic shape and dynamic rank functions will not be enabled.
        - In the scenario where both set_context(mode=GRAPH_MODE) and nn.Cell are set simultaneously,
          use enabled_dynamic to report an error.

    Args:
        \*\*kwargs (dict[str, Union[Tensor, tuple[Tensor], list[Tensor]]]): The input types are Tensor,
            tuple[Tensor] and list[Tensor]. If one or
            more dimensions in the shape of the parameter need to be specified as dynamic shapes,
            the corresponding dimensions in the shape can be set to None. If the shape that needs
            to generate specified parameters is dynamic rank, the shape can be set to None.

    Returns:
        Function, decorator function which is used to specify the dynamic shape information of the parameters for
        the decorated function.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore import enable_dynamic
        >>> from mindspore import jit
        ...
        >>> x = Tensor(np.random.randn(2, 3), ms.float32)
        >>> y = Tensor(np.random.randn(2, 3), ms.float32)
        ...
        >>> # Specify parameter y as dynamic shape
        >>> @enable_dynamic(y=Tensor(shape=None, dtype=ms.float32))
        >>> @jit
        >>> def func(x, y):
        ...     return x + 1, y + 1
        ...
        >>> out = func(x, y)
    """
    # Check inputs at first.
    if not kwargs:
        raise ValueError("When using decorator enable_dynamic, the input cannot be empty!")
    for name, arg in kwargs.items():
        _check_arg_type_valid(arg, name)
        if not _check_arg_shape_valid(arg, name):
            raise TypeError(f"When using decorator enable_dynamic, the shape of argument '{name}' " \
                            f"at least have one None.")

    def decorator(func):
        if not isinstance(func, (types.FunctionType, types.MethodType)):
            raise ValueError(f"Decorator enable_dynamic can only be used for function or method " \
                             f"decrocated by ms.jit, but got {func}.")
        signature = inspect.signature(func)
        sigs_name = [sig_name for sig_name in signature.parameters if sig_name != "self"]
        if len(kwargs) > len(sigs_name):
            raise ValueError(f"When using decorator enable_dynamic, the number of arguments {len(kwargs)} " \
                             f"exceeds the number of function arguments {len(sigs_name)}.")
        # Generate dynamic args.
        dynamic_args = {}
        for key, value in kwargs.items():
            index = sigs_name.index(key)
            if index in dynamic_args:
                raise ValueError(f"keyword argument repeated: {key}")
            dynamic_args[index] = (key, value)
        # Set dynamic_tensor_shape to func.
        inner_func = inspect.unwrap(func, stop=lambda f: not hasattr(f, '__wrapped__'))
        setattr(get_func(inner_func), ENABLE_DYNAMIC, dynamic_args)
        logger.info(f"Set enable dynamic: {dynamic_args} to {inner_func}")
        return func
    return decorator
