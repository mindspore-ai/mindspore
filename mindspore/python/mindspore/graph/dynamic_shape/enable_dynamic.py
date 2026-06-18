# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
from mindspore.graph.dynamic_shape.auto_dynamic_shape import SHAPE_DIM_ANY


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


def enable_dynamic(**kwargs):
    r"""
    Use to specify whether the shape of the parameter is dynamic shape or dynamic rank.

    Note:
        - It needs to be used in conjunction with the JIT interface. Without using the JIT decorator,
          the dynamic shape and dynamic rank functions will not be enabled.
        - In the scenario where both set_context(mode=GRAPH_MODE) and nn.Cell are set simultaneously,
          use enable_dynamic to report an error.

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
