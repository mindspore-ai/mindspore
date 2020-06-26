
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
from mindspore.ops import operations as P
from mindspore.ops import _utils as utils
from ....common.tensor import Tensor
from ....common import dtype as mstype


def check_scalar(value):
    """
    Check if input value is a scalar.
    """
    return np.isscalar(value)


def cast_to_tensor(t, dtype=mstype.float32):
    """
    Cast an user input value into a Tensor of dtype.

    Args:
        t (int/float/list/numpy.ndarray/Tensor).
        dtype (mindspore.dtype).

    Raises:
        RuntimeError: if t cannot be cast to Tensor.

    Outputs:
        Tensor.
    """
    if isinstance(t, Tensor):
        #check if the Tensor in shape of Tensor(4)
        if t.dim() == 0:
            value = t.asnumpy()
            return Tensor([t], dtype=dtype)
        #convert the type of tensor to dtype
        t.set_dtype(dtype)
        return t
    if isinstance(t, (list, np.ndarray)):
        return Tensor(t, dtype=dtype)
    if check_scalar(t):
        return Tensor([t], dtype=dtype)
    raise RuntimeError("Input type is not supported.")

def calc_batch_size(batch_shape):
    """
    Calculate the size of a given batch_shape.

    Args:
        batch_shape (tuple)

    Outputs:
        int.
    """
    return int(np.prod(batch_shape))

def convert_to_batch(t, batch_shape, dtype):
    """
    Convert a Tensor to a given batch shape.

    Args:
        t (Tensor)
        batch_shape (tuple)
        dtype (mindspore.dtype)
    Raises:
        RuntimeError: if the converison cannot be done.

    Outputs:
        Tensor, with shape of batch_shape.
    """
    t = cast_to_tensor(t, dtype)
    reshape = P.Reshape()
    if t.shape != batch_shape:
        mul = calc_batch_size(batch_shape) // t.size()
        if (calc_batch_size(batch_shape) % t.size()) != 0:
            raise RuntimeError("Cannot cast the tensor to the given batch shape.")
        temp = list(t.asnumpy()) * mul
        return reshape(Tensor(temp), batch_shape)
    return t

def check_scalar_from_param(params):
    """
    Check if params are all scalars.

    Args:
        params (dict): parameters used to initialized distribution.

    Notes: String parameters are excluded.
    """
    for value in params.values():
        if isinstance(value, (str, type(params['dtype']))):
            continue
        elif check_scalar(value):
            continue
        else:
            return False
    return True


def calc_broadcast_shape_from_param(params):
    """
    Calculate the broadcast shape from params.

    Args:
        params (dict): parameters used to initialized distribution.

    Outputs:
        tuple.
    """
    broadcast_shape = []
    for value in params.values():
        if isinstance(value, (str, type(params['dtype']))):
            continue
        if value is None:
            return None
        value_t = cast_to_tensor(value, params['dtype'])
        broadcast_shape = utils.get_broadcast_shape(broadcast_shape, list(value_t.shape), params['name'])
    return tuple(broadcast_shape)

def check_greater_equal_zero(value, name):
    """
    Check if the given Tensor is greater zero.

    Args:
        value (Tensor)
        name (str) : name of the value.

    Raises:
        ValueError: if the input value is less than zero.

    """
    less = P.Less()
    zeros = Tensor([0.0], dtype=value.dtype)
    value = less(value, zeros)
    if value.asnumpy().any():
        raise ValueError('{} should be greater than zero.'.format(name))

def check_greater(a, b, name_a, name_b):
    """
    Check if Tensor b is strictly greater than Tensor a.

    Args:
        a (Tensor)
        b (Tensor)
        name_a (str): name of Tensor_a.
        name_b (str): name of Tensor_b.

    Raises:
        ValueError: if b is less than or equal to a
    """
    less = P.Less()
    value = less(a, b)
    if not value.asnumpy().all():
        raise ValueError('{} should be less than {}'.format(name_a, name_b))


def check_prob(p):
    """
    Check if p is a proper probability, i.e. 0 <= p <=1.

    Args:
        p (Tensor): value to check.

    Raises:
        ValueError: if p is not a proper probability.
    """
    less = P.Less()
    greater = P.Greater()
    zeros = Tensor([0.0], dtype=p.dtype)
    ones = Tensor([1.0], dtype=p.dtype)
    comp = less(p, zeros)
    if comp.asnumpy().any():
        raise ValueError('Probabilities should be greater than or equal to zero')
    comp = greater(p, ones)
    if comp.asnumpy().any():
        raise ValueError('Probabilities should be less than or equal to one')
