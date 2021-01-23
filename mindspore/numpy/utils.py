# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""internal utility functions"""

import numpy as onp

import mindspore.context as context
from ..common import Tensor
from ..ops import functional as F

from .utils_const import _tile_size


def _deep_list(array_like):
    """convert nested tuple/list mixtures to pure nested list"""
    if isinstance(array_like, (list, tuple)):
        return list(map(_deep_list, array_like))
    return array_like


def _deep_tensor_to_nparray(array_like):
    """
    convert a nested list of tensor to nested list of np_array.

    Args:
        array_like(list(tensor)): In any format of nested lists that may contain
        tensors.

    Returns:
        array_like(list(np_array)): Formatted array that can be directly processed
            by numpy.array(), with all tensor elements converted to numpy_array.
    """
    # Recursively check whether each element is a tensor or not, if is tensor,
    # convert it to a numpy array in place
    if isinstance(array_like, Tensor):
        return array_like.asnumpy()

    if isinstance(array_like, list):
        for idx, value in enumerate(array_like):
            array_like[idx] = _deep_tensor_to_nparray(value)

    return array_like


def _check_input_for_asarray(array_like):
    """check whether array_like argument is a valid type for np.asarray conversion"""
    if isinstance(array_like, (Tensor, list, tuple, int, float, bool, onp.ndarray)):
        return True
    raise TypeError("input data must be `int`, `float`, `bool`, `Tensor`, `list`, `tuple`" + \
        f"or numpy.ndarray, but got {type(array_like)}")


def _is_scalar(shape):
    """check whether input shape is a scalar"""
    return F.shape_mul(shape) == 1


def _is_empty(shape):
    """Checks if the shape is empty"""
    return F.shape_mul(shape) == 0


def _get_device():
    """Get the current device (`GPU`, `CPU`, `Ascend`)"""
    return context.get_context('device_target')


def _convert_list_tensor_to_tuple_tensor(list_of_tensor):
    """Convert a list of tensor to a tuple of tensor"""
    if isinstance(list_of_tensor, list):
        tuple_of_tensor = ()
        for tensor in list_of_tensor:
            tuple_of_tensor += (tensor,)
        return tuple_of_tensor
    return list_of_tensor


def _get_mode():
    """Get the current mode (0 is Graph mode, 1 is PyNative mode)"""
    return context.get_context('mode')


def _expand(x, ndim, axis=0):
    """Expand x to ndim."""
    while F.rank(x) < ndim:
        x = F.expand_dims(x, axis)
    return x


def _broadcast_to(x, shape_cur, shape_to, ndim_to):
    """Broadcasts x from shape_cur to shape_to."""
    size = _tile_size(shape_cur, shape_to, ndim_to)
    return F.tile(x, size)
