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

"""utils for operator"""

from ..._checkparam import ParamValidator as validator
from ..._checkparam import Rel
from ...common import dtype as mstype

def _get_broadcast_shape(x_shape, y_shape, prim_name):
    """
    Doing broadcast between tensor x and tensor y.

    Args:
        x_shape (list): The shape of tensor x.
        y_shape (list): The shape of tensor y.
        prim_name (str): Primitive name.

    Returns:
        List, the shape that broadcast between tensor x and tensor y.

    Raises:
        ValueError: If tensor x and tensor y are not equal and could't broadcast.

    Examples:
        >>> x_shape = [1, 2, 3]
        >>> y_shape = [1, 2]
        >>> broadcast_shape = _get_broadcast_shape(x_shape, y_shape)
    """
    if x_shape == y_shape:
        return x_shape
    x_len = len(x_shape)
    y_len = len(y_shape)
    length = x_len if x_len < y_len else y_len
    broadcast_shape_back = []

    for i in range(-length, 0):
        if x_shape[i] == 1:
            broadcast_shape_back.append(y_shape[i])
        elif y_shape[i] == 1:
            broadcast_shape_back.append(x_shape[i])
        elif x_shape[i] == y_shape[i]:
            broadcast_shape_back.append(x_shape[i])
        else:
            raise ValueError("For '{}' the x_shape {} and y_shape {} can not broadcast.".format(
                prim_name, x_shape, y_shape))

    broadcast_shape_front = y_shape[0: y_len - length] if length == x_len else x_shape[0: x_len - length]
    broadcast_shape = broadcast_shape_front + broadcast_shape_back
    return broadcast_shape


def _get_concat_offset(x_shp, x_type, axis):
    """for concat and concatoffset check args and compute offset"""
    validator.check_type("shape", x_shp, [tuple])
    validator.check_integer("len of input_x shape", len(x_shp), 0, Rel.GT)
    validator.check_subclass("shape0", x_type[0], mstype.tensor)
    validator.check_integer("len of input_x0 shape", len(x_shp[0]), 0, Rel.GT)
    rank_base = len(x_shp[0])
    validator.check_int_range('axis', axis, -rank_base - 1, rank_base, Rel.INC_BOTH)
    if axis < 0:
        axis = axis + rank_base
    all_shp = x_shp[0][axis]
    offset = [0,]
    for i in range(1, len(x_shp)):
        v = x_shp[i]
        validator.check('len of x_shp[%d]' % i, len(v), 'len of base', len(x_shp[0]))
        validator.check('x_type[%d]' % i, x_type[i], 'base', x_type[0])
        for j in range(rank_base):
            if j != axis and v[j] != x_shp[0][j]:
                raise ValueError("Concat evaluator element %d shape in input can not concat with first element" % i)
        offset.append(all_shp)
        all_shp += v[axis]
    return offset, all_shp, axis
