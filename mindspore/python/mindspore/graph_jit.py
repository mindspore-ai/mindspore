# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Providing interface method."""
from copy import deepcopy
import inspect
import types
import warnings
from mindspore._c_expression import jit_mode_pi_enable, jit_mode_pi_compile, TestGraphIRCodeGen

GraphJit_config_Default = {
    # replace call nn.Cell with call nn.Cell.construct
    # inline call nn.Cell if True
    "replace_nncell_by_construct": False,
    # print the bytecode or information of each graph after all passes(e.g. DFG, CodeGen)
    "print_after_all": False,
    # print the traceback for a @Graphjit decorated function
    "print_tb": False,
    "print_last_frame_if_break_graph": False,
    "print_bb": False,
    "print_cfg": False,
    # interpret new bytecode if True, not compile
    "interpret_captured_code": False,
    # not captured and rewrite code, pass function directly to backend compiler
    "compile_without_capture": False,
    # not inline any function if True
    "not_inline_any_function": False,
    # capture mindspore support operaions to build new function to compile
    # reshape bytecodes if True
    "graph_break_at_unsupported_operations": True,
    # guard control option
    "enable_guard": True,
    "specialize_int_float": True,
    "specialize_tensor": False,
    "guard_subroutine": False,
    "print_guard": False,
    "auto_clean_cache": False,
    "prune_case": True,
    "loop_unrolling": True,
    "infer_primitive": True,
    # inilne whitelist, only inline the function of the list,
    # default 'mindspore', the top module of @GraphJit decorated function
    "allowed_inline_modules": ["mindspore"],
    "MAX_INLINE_DEPTH": 10,
    "MAX_PRUNE_CASE": -1,
    "MAX_LOOP_UNROLLING": 30,
    "INFER_PRIMITIVE_MASK": 7,
    "INFER_PRIMITIVE_MAX": 0
}

UNSUPPORTED_CODE_TYPE = (inspect.CO_GENERATOR | inspect.CO_COROUTINE |
                         inspect.CO_ASYNC_GENERATOR | inspect.CO_ITERABLE_COROUTINE)

jit_mode_pi_enable()


def GraphJit(fn=None, **kwvargs):
    """
    Create a callable MindSpore graph from a Python function.

    This allows the MindSpore runtime to apply optimizations based on graph.

    Note:
        The input arguments for `fn` will not accept `**kwargs`.

    Args:
        fn (Function): The Python function that will be run as a graph. Default: ``None`` .
        jit_config (JitConfig): Jit config for compile. Default: ``None`` .

    Returns:
        Function, if `fn` is not None, returns a callable function that will execute the compiled function; If `fn` is
        None, returns a decorator and when this decorator invokes with a single `fn` argument, the callable function is
        equal to the case when `fn` is not None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import GraphJit
        ...
        >>> x = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> y = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        ...
        >>> # create a callable MindSpore graph by calling decorator @GraphJit
        >>> def tensor_add(x, y):
        ...     z = x + y
        ...     return z
        ...
        >>> tensor_add_graph = GraphJit(fn=tensor_add)
        >>> out = tensor_add_graph(x, y)
        ...
        >>> # create a callable MindSpore graph through decorator @GraphJit
        >>> @GraphJit
        ... def tensor_add_with_dec(x, y):
        ...     z = x + y
        ...     return z
        ...
        >>> out = tensor_add_with_dec(x, y)
    """
    tag = None
    for k, v in kwvargs.items():
        if k == "config":
            tag = v

    def tag_func(fn):
        # check options
        config = deepcopy(GraphJit_config_Default)
        if isinstance(tag, dict):
            if tag.get("test_graph_code_gen"):
                return TestGraphIRCodeGen(fn)
            for k, v in tag.items():
                if GraphJit_config_Default.get(k) is None:
                    raise Exception("unkonw options {}".format(k))
                if k == "allowed_inline_modules":
                    v.extend(GraphJit_config_Default.get(k))
            config.update(tag)

        func_obj = None
        if isinstance(fn, (types.FunctionType, types.MethodType)):
            func_obj = fn
        elif isinstance(fn, type):
            func_obj = fn.__call__
        else:
            warnings.warn('unkonw function {}'.format(fn))
            return fn

        top_module = fn.__module__.split('.')[0]
        config.get("allowed_inline_modules").append(top_module)

        # cpython use this flag mark the optimization that replace
        # variable name dictionary with fast local (PyFrameObject->f_localsplus)
        # GraphJit depends on fast local
        assert func_obj.__code__.co_flags & inspect.CO_OPTIMIZED

        # generator, coroutine, awaitable and a function that return them is unsupported
        if func_obj.__code__.co_flags & UNSUPPORTED_CODE_TYPE:
            raise RuntimeError(
                '{}. generator, coroutine, awaitable function is unsupported'.format(fn))

        if jit_mode_pi_compile(func_obj, config) is False:
            warnings.warn('add fn {} to compile failed '.format(fn))

        return fn

    if fn is not None:
        assert callable(fn)
        return tag_func(fn)
    return tag_func


__all__ = [
    "GraphJit"
]
