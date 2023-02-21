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
"""Utility functions to help distribution class."""
import numpy as np
from mindspore import context
from mindspore._checkparam import Validator as validator
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops.primitive import constexpr, _primexpr, PrimitiveWithInfer, prim_attr_register
import mindspore.ops as ops
import mindspore.nn as nn


def cast_to_tensor(t, hint_type=mstype.float32):
    """
    Cast an user input value into a Tensor of dtype.
    If the input t is of type Parameter, t is directly returned as a Parameter.

    Args:
        t (int, float, list, numpy.ndarray, Tensor, Parameter): object to be cast to Tensor.
        dtype (mindspore.dtype): dtype of the Tensor. Default: mstype.float32.

    Raises:
        RuntimeError: if t cannot be cast to Tensor.

    Returns:
        Tensor.
    """
    if t is None:
        raise ValueError(f'Input cannot be None in cast_to_tensor')
    if isinstance(t, Parameter):
        return t
    if isinstance(t, bool):
        raise TypeError(f'Input cannot be Type Bool')
    if isinstance(t, (Tensor, np.ndarray, list, int, float)):
        return Tensor(t, dtype=hint_type)
    invalid_type = type(t)
    raise TypeError(
        f"Unable to convert input of type {invalid_type} to a Tensor of type {hint_type}")


def cast_type_for_device(dtype):
    """
    use the alternative dtype supported by the device.
    Args:
        dtype (mindspore.dtype): input dtype.
    Returns:
        mindspore.dtype.
    """
    if context.get_context("device_target") == "GPU":
        if dtype in mstype.uint_type or dtype == mstype.int8:
            return mstype.int16
        if dtype == mstype.int64:
            return mstype.int32
        if dtype == mstype.float64:
            return mstype.float32
    return dtype


def check_greater_equal_zero(value, name):
    """
    Check if the given Tensor is greater zero.

    Args:
        value (Tensor, Parameter): value to be checked.
        name (str) : name of the value.

    Raises:
        ValueError: if the input value is less than zero.

    """
    if isinstance(value, Parameter):
        if not isinstance(value.data, Tensor):
            return
        value = value.data
    comp = np.less(value.asnumpy(), np.zeros(value.shape))
    if comp.any():
        raise ValueError(f'{name} must be greater than or equal to zero.')


def check_greater_zero(value, name):
    """
    Check if the given Tensor is strictly greater than zero.

    Args:
        value (Tensor, Parameter): value to be checked.
        name (str) : name of the value.

    Raises:
        ValueError: if the input value is less than or equal to zero.

    """
    if value is None:
        raise ValueError(f'input value cannot be None in check_greater_zero')
    if isinstance(value, Parameter):
        if not isinstance(value.data, Tensor):
            return
        value = value.data
    comp = np.less(np.zeros(value.shape), value.asnumpy())
    if not comp.all():
        raise ValueError(f'{name} must be greater than zero.')


def check_greater(a, b, name_a, name_b):
    """
    Check if Tensor b is strictly greater than Tensor a.

    Args:
        a (Tensor, Parameter): input tensor a.
        b (Tensor, Parameter): input tensor b.
        name_a (str): name of Tensor_a.
        name_b (str): name of Tensor_b.

    Raises:
        ValueError: if b is less than or equal to a
    """
    if a is None or b is None:
        raise ValueError(f'input value cannot be None in check_greater')
    if isinstance(a, Parameter) or isinstance(b, Parameter):
        return
    comp = np.less(a.asnumpy(), b.asnumpy())
    if not comp.all():
        raise ValueError(f'{name_a} must be less than {name_b}')


def check_prob(p):
    """
    Check if p is a proper probability, i.e. 0 < p <1.

    Args:
        p (Tensor, Parameter): value to be checked.

    Raises:
        ValueError: if p is not a proper probability.
    """
    if p is None:
        raise ValueError(f'input value cannot be None in check_greater_zero')
    if isinstance(p, Parameter):
        if not isinstance(p.data, Tensor):
            return
        p = p.data
    comp = np.less(np.zeros(p.shape), p.asnumpy())
    if not comp.all():
        raise ValueError('Probabilities must be greater than zero')
    comp = np.greater(np.ones(p.shape), p.asnumpy())
    if not comp.all():
        raise ValueError('Probabilities must be less than one')


def check_sum_equal_one(probs):
    """
    Used in categorical distribution. check if probabilities of each category sum to 1.
    """
    if probs is None:
        raise ValueError(f'input value cannot be None in check_sum_equal_one')
    if isinstance(probs, Parameter):
        if not isinstance(probs.data, Tensor):
            return
        probs = probs.data
    if isinstance(probs, Tensor):
        probs = probs.asnumpy()
    prob_sum = np.sum(probs, axis=-1)
    # add a small tolerance here to increase numerical stability
    comp = np.allclose(prob_sum, np.ones(prob_sum.shape),
                       rtol=np.finfo(prob_sum.dtype).eps * 10, atol=np.finfo(prob_sum.dtype).eps)
    if not comp:
        raise ValueError(
            'Probabilities for each category should sum to one for Categorical distribution.')


def check_rank(probs):
    """
    Used in categorical distribution. check Rank >=1.
    """
    if probs is None:
        raise ValueError(f'input value cannot be None in check_rank')
    if isinstance(probs, Parameter):
        if not isinstance(probs.data, Tensor):
            return
        probs = probs.data
    if probs.asnumpy().ndim == 0:
        raise ValueError(
            'probs for Categorical distribution must have rank >= 1.')


def logits_to_probs(logits, is_binary=False):
    """
    converts logits into probabilities.
    Args:
        logits (Tensor)
        is_binary (bool)
    """
    if is_binary:
        return nn.Sigmoid()(logits)
    return nn.Softmax(axis=-1)(logits)


def clamp_probs(probs):
    """
    clamp probs boundary
    """
    eps = P.Eps()(probs)
    return ops.clip_by_value(probs, eps, 1-eps)


def probs_to_logits(probs, is_binary=False):
    """
    converts probabilities into logits.
        Args:
        probs (Tensor)
        is_binary (bool)
    """
    ps_clamped = clamp_probs(probs)
    if is_binary:
        return P.Log()(ps_clamped) - P.Log()(1-ps_clamped)
    return P.Log()(ps_clamped)


@constexpr
def raise_none_error(name):
    raise TypeError(f"the type {name} must be subclass of Tensor."
                    f" It can not be None since it is not specified during initialization.")


@_primexpr
def raise_broadcast_error(shape_a, shape_b):
    raise ValueError(f"Shape {shape_a} and {shape_b} is not broadcastable.")


@constexpr
def raise_not_impl_error(name):
    raise ValueError(
        f"{name} function must be implemented for non-linear transformation")


@constexpr
def raise_not_implemented_util(func_name, obj, *args, **kwargs):
    raise NotImplementedError(
        f"{func_name} is not implemented for {obj} distribution.")


@constexpr
def raise_type_error(name, cur_type, required_type):
    raise TypeError(
        f"For {name} , the type must be or be subclass of {required_type}, but got {cur_type}")


@constexpr
def raise_not_defined(func_name, obj, *args, **kwargs):
    raise ValueError(
        f"{func_name} is undefined for {obj} distribution.")


@constexpr
def check_distribution_name(name, expected_name):
    if name is None:
        raise ValueError(
            f"Input dist must be a constant which is not None.")
    if name != expected_name:
        raise ValueError(
            f"Expected dist input is {expected_name}, but got {name}.")


class CheckTuple(PrimitiveWithInfer):
    """
    Check if input is a tuple.
    """
    @prim_attr_register
    def __init__(self):
        super(CheckTuple, self).__init__("CheckTuple")
        self.init_prim_io_names(inputs=['x', 'name'], outputs=['dummy_output'])

    def __infer__(self, x, name):
        if not isinstance(x['dtype'], tuple):
            raise TypeError(
                f"For {name['value']}, Input type must b a tuple.")

        out = {'shape': None,
               'dtype': None,
               'value': x["value"]}
        return out

    def __call__(self, x, name):
        # The op is not used in a cell
        if isinstance(x, tuple):
            return x
        if context.get_context("mode") == 0:
            return x["value"]
        raise TypeError(f"For {name}, input type must be a tuple.")


class CheckTensor(PrimitiveWithInfer):
    """
    Check if input is a Tensor.
    """
    @prim_attr_register
    def __init__(self):
        super(CheckTensor, self).__init__("CheckTensor")
        self.init_prim_io_names(inputs=['x', 'name'], outputs=['dummy_output'])

    def __infer__(self, x, name):
        src_type = x['dtype']
        validator.check_subclass(
            "input", src_type, [mstype.tensor], name["value"])

        out = {'shape': None,
               'dtype': None,
               'value': None}
        return out

    def __call__(self, x, name):
        # we skip this check in graph mode as it is checked in the infer stage
        # and in the graph mode x is None if x is not const in the graph
        if x is None or isinstance(x, Tensor):
            return x
        raise TypeError(
            f"For {name}, input type must be a Tensor or Parameter.")


def set_param_type(args, hint_type):
    """
    Find the common type among arguments.

    Args:
        args (dict): dictionary of arguments, {'name':value}.
        hint_type (mindspore.dtype): hint type to return.

    Raises:
        TypeError: if tensors in args are not the same dtype.
    """
    int_type = mstype.int_type + mstype.uint_type
    if hint_type in int_type or hint_type is None:
        hint_type = mstype.float32
    common_dtype = None
    for name, arg in args.items():
        if hasattr(arg, 'dtype'):
            if isinstance(arg, np.ndarray):
                cur_dtype = mstype.pytype_to_dtype(arg.dtype)
            else:
                cur_dtype = arg.dtype
            if common_dtype is None:
                common_dtype = cur_dtype
            elif cur_dtype != common_dtype:
                raise TypeError(
                    f"{name} should have the same dtype as other arguments.")
    if common_dtype in int_type or common_dtype == mstype.float64:
        return mstype.float32
    return hint_type if common_dtype is None else common_dtype
