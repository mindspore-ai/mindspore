# Copyright 2022 Huawei Technologies Co., Ltd
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
"""sparsify utils"""
import ast
import builtins
import inspect
import types
from enum import Enum, auto
from typing import NamedTuple, Any, Optional, Callable, Union

import mindspore
from mindspore import ops, Tensor, CSRTensor, COOTensor
from mindspore.rewrite.namespace import get_functional


class ArgType(Enum):
    """
    Argument types for sparsify.

    - CSR represents a CSRTensor.
    - COO represents a COOTensor.
    - NONSPARSE represents a non-sparse value.

    .. warning::
        This is a set of experimental APIs that is subject to change or deletion.
    """
    NONSPARSE = auto()
    CSR = auto()
    COO = auto()
    ANY = auto()


class SparseFunc(NamedTuple):
    """
    Represents a sparse function in sparsify.

    Note:
        If `fn` is a function with type hints, `inputs` and/or `outputs`, when provided, override function type hints.

    .. warning::
        This is a set of experimental APIs that is subject to change or deletion.

    Args:
        fn (Union[str, Callable]): a sparse function. If `fn` is a string, the function represents a mindspore
            functional op; or `fn` can be any function object.
        inputs (Any, optional): input types for the function. If `inputs` is None, use the input types in function
            type hints. Default: ``None`` .
        outputs (Any, optional): output types for the function. If `outputs` is None, use the output types in function
            type hints. Default: ``None`` .
    """
    fn: Union[str, Callable]
    inputs: Optional[Any] = None
    outputs: Optional[Any] = None


# maps function to a list of strings or SparseFunc, each representing the name of a sparse_func
sparse_rules = {
    ops.reduce_sum: ["csr_reduce_sum"],
    ops.mul: ["csr_mul"],
    ops.matmul: ["csr_mv"],
    "+": [],
    "-": [],
    "*": ["csr_mul"],
    "/": ["csr_div"]
}


builtin_ops = {i for i, v in vars(builtins).items() if isinstance(v, types.BuiltinFunctionType)}
tensor_to_arg_type_map = {Tensor: ArgType.NONSPARSE, CSRTensor: ArgType.CSR, COOTensor: ArgType.COO}
arg_type_to_tensor_map = {ArgType.CSR: CSRTensor, ArgType.COO: COOTensor}
arg_type_to_prefix_map = {ArgType.CSR: "csr", ArgType.COO: "coo"}


def get_arg_type(annotation):
    """Returns arg_type based on typing annotation."""
    if isinstance(annotation, str):
        annotation = getattr(mindspore, annotation, None)
    arg_type = tensor_to_arg_type_map.get(annotation, None)
    if arg_type is None:
        if annotation in (int, float, bool, str):
            return ArgType.NONSPARSE
        raise ValueError(f"Type {annotation} cannot be mapped to ArgType!")
    return arg_type


def get_tuple(x):
    """get tuple"""
    if not isinstance(x, (tuple, list)):
        return (x,)
    return tuple(x)


def get_inputs_outputs(fn):
    """Returns input and output types for function based on typing."""
    sig = inspect.signature(fn)
    inputs = []
    for i in sig.parameters.values():
        if i.annotation == inspect.Parameter.empty:
            inputs = None
            break
        input_type = get_arg_type(i.annotation)
        inputs.append(input_type)
    if sig.return_annotation == inspect.Parameter.empty:
        outputs = None
    else:
        outputs = get_tuple(get_arg_type(sig.return_annotation))
    return inputs, outputs


def get_sparse_method_outputs(method_name, sparse_type):
    """Returns output types for sparse tensor method."""
    tensor = arg_type_to_tensor_map.get(sparse_type, None)
    if tensor is None:
        raise ValueError(f"Unrecognized sparse type {sparse_type}!")
    method = getattr(tensor, method_name, None)
    if method is None:
        raise ValueError(f"{tensor} does not have attr {method_name}!")
    _, outputs = get_inputs_outputs(method)
    return outputs


def get_sparse_func(rule):
    """
    Returns SparseFunc with string for `fn`, `inputs` and `outputs` extracted from
    function annotation.
    """
    if isinstance(rule, str):
        # only mindspore functional ops can be passed as strings
        sparse_func = get_functional(rule)
        if not sparse_func:
            raise ValueError(f"{rule} not a valid name for mindspore functional op!")
        inputs, outputs = get_inputs_outputs(sparse_func)
        return SparseFunc(rule, inputs, outputs)
    if isinstance(rule, SparseFunc):
        if isinstance(rule.fn, str):
            return get_sparse_func(rule.fn)
        if callable(rule.fn):
            inputs, outputs = get_inputs_outputs(rule.fn)
            if rule.inputs:
                inputs = get_tuple(rule.inputs)
            elif inputs is None:
                raise ValueError(f"Input types not provided for {rule}!")
            if rule.outputs:
                outputs = get_tuple(rule.outputs)
            elif outputs is None:
                raise ValueError(f"Output types not provided for {rule}!")
            return SparseFunc(rule.fn.__name__, inputs, outputs)
        raise ValueError(f"`fn` {rule.fn} for SparseFunc should be either a string or a function!")
    if callable(rule):
        inputs, outputs = get_inputs_outputs(rule)
        if inputs is None or outputs is None:
            raise ValueError(f"Both input types and output types should be provided for {rule}!")
        return SparseFunc(rule.__name__, inputs, outputs)
    raise ValueError(f"Sparse rule {rule} should be either a string or a SparseFunc!")


def get_binop_name(binop):
    """Maps ast.BinOp operator to string."""
    if binop == ast.Add():
        return "+"
    if binop == ast.Sub():
        return "-"
    if binop == ast.Mult():
        return "*"
    if binop == ast.Div():
        return "/"
    return ""
