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
"""Utitly functions to help distribution class."""
import numpy as np
from mindspore.ops import _utils as utils
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import composite as C
import mindspore.nn as nn
import mindspore.nn.probability as msp

def cast_to_tensor(t, hint_dtype=mstype.float32):
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
    if isinstance(t, Parameter):
        return t
    if isinstance(t, Tensor):
        if t.dtype != hint_dtype:
            raise TypeError(f"Input tensor should be type {hint_dtype}.")
        #check if the Tensor in shape of Tensor(4)
        if t.dim() == 0:
            value = t.asnumpy()
            return Tensor([value], dtype=hint_dtype)
        #convert the type of tensor to dtype
        return t
    if isinstance(t, (list, np.ndarray)):
        return Tensor(t, dtype=hint_dtype)
    if np.isscalar(t):
        return Tensor([t], dtype=hint_dtype)
    raise RuntimeError("Input type is not supported.")

def convert_to_batch(t, batch_shape, hint_dtype):
    """
    Convert a Tensor to a given batch shape.

    Args:
        t (int, float, list, numpy.ndarray, Tensor, Parameter): Tensor to be converted.
        batch_shape (tuple): desired batch shape.
        dtype (mindspore.dtype): desired dtype.

    Raises:
        RuntimeError: if the converison cannot be done.

    Returns:
        Tensor, with shape of batch_shape.
    """
    if isinstance(t, Parameter):
        return t
    t = cast_to_tensor(t, hint_dtype)
    return Tensor(np.broadcast_to(t.asnumpy(), batch_shape), dtype=hint_dtype)

def check_scalar_from_param(params):
    """
    Check if params are all scalars.

    Args:
        params (dict): parameters used to initialize distribution.

    Notes: String parameters are excluded.
    """
    for value in params.values():
        if isinstance(value, (msp.bijector.Bijector, msp.distribution.Distribution)):
            return params['distribution'].is_scalar_batch
        if isinstance(value, Parameter):
            return False
        if isinstance(value, (str, type(params['dtype']))):
            continue
        elif np.isscalar(value):
            continue
        else:
            return False
    return True


def calc_broadcast_shape_from_param(params):
    """
    Calculate the broadcast shape from params.

    Args:
        params (dict): parameters used to initialize distribution.

    Returns:
        tuple.
    """
    broadcast_shape = []
    for value in params.values():
        if isinstance(value, (msp.bijector.Bijector, msp.distribution.Distribution)):
            return params['distribution'].broadcast_shape
        if isinstance(value, (str, type(params['dtype']))):
            continue
        if value is None:
            return None
        if isinstance(value, Parameter):
            value_t = value.default_input
        else:
            value_t = cast_to_tensor(value, params['dtype'])
        broadcast_shape = utils.get_broadcast_shape(broadcast_shape, list(value_t.shape), params['name'])
    return tuple(broadcast_shape)

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
        if not isinstance(value.default_input, Tensor):
            return
        value = value.default_input
    comp = np.less(value.asnumpy(), np.zeros(value.shape))
    if comp.any():
        raise ValueError(f'{name} should be greater than ot equal to zero.')

def check_greater_zero(value, name):
    """
    Check if the given Tensor is strictly greater than zero.

    Args:
        value (Tensor, Parameter): value to be checked.
        name (str) : name of the value.

    Raises:
        ValueError: if the input value is less than or equal to zero.

    """
    if isinstance(value, Parameter):
        if isinstance(value.default_input, MetaTensor):
            return
        value = value.default_input
    comp = np.less(np.zeros(value.shape), value.asnumpy())
    if not comp.all():
        raise ValueError(f'{name} should be greater than zero.')

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
    if isinstance(a, Parameter) or isinstance(b, Parameter):
        return
    comp = np.less(a.asnumpy(), b.asnumpy())
    if not comp.all():
        raise ValueError(f'{name_a} should be less than {name_b}')


def check_prob(p):
    """
    Check if p is a proper probability, i.e. 0 < p <1.

    Args:
        p (Tensor, Parameter): value to be checked.

    Raises:
        ValueError: if p is not a proper probability.
    """
    if isinstance(p, Parameter):
        if not isinstance(p.default_input, Tensor):
            return
        p = p.default_input
    comp = np.less(np.zeros(p.shape), p.asnumpy())
    if not comp.all():
        raise ValueError('Probabilities should be greater than zero')
    comp = np.greater(np.ones(p.shape), p.asnumpy())
    if not comp.all():
        raise ValueError('Probabilities should be less than one')


def logits_to_probs(logits, is_binary=False):
    """
    converts logits into probabilities.
    Args:
        logits (Tensor)
        is_binary (bool)
    """
    if is_binary:
        return nn.sigmoid()(logits)
    return nn.softmax(axis=-1)(logits)


def clamp_probs(probs):
    """
    clamp probs boundary
    Args:
        probs (Tensor)
    """
    eps = P.Eps()(probs)
    return C.clip_by_value(probs, eps, 1-eps)


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

def check_tensor_type(name, inputs, valid_type):
    """
   Check if inputs is proper.

   Args:
       inputs: Tensor to be checked.
       name: inputs name

   Raises:
       ValueError: if inputs is not a proper Tensor.
   """
    if not isinstance(inputs, Tensor):
        raise TypeError(f"{name} should be a Tensor")
    inputs = P.DType()(inputs)
    if inputs not in valid_type:
        raise TypeError(f"{name} dtype is invalid")

def check_type(data_type, value_type, name):
    if not data_type in value_type:
        raise TypeError(f"For {name}, valid type include {value_type}, {data_type} is invalid")
