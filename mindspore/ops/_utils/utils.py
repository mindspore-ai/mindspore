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

"""utils for operator"""

from ..._checkparam import Validator as validator
from ..._checkparam import Rel
from ...common import dtype as mstype


def get_broadcast_shape(x_shape, y_shape, prim_name):
    """
    Doing broadcast between tensor x and tensor y.

    Args:
        x_shape (list): The shape of tensor x.
        y_shape (list): The shape of tensor y.
        prim_name (str): Primitive name.

    Returns:
        List, the shape that broadcast between tensor x and tensor y.

    Raises:
        ValueError: If tensor x and tensor y are not equal and couldn't broadcast.

    Examples:
        >>> x_shape = [1, 2, 3]
        >>> y_shape = [1, 2]
        >>> broadcast_shape = get_broadcast_shape(x_shape, y_shape)
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
        elif x_shape[i] == -1 or y_shape[i] == -1:
            broadcast_shape_back.append(-1)
        else:
            raise ValueError(f"For '{prim_name}', the x_shape {x_shape} and y_shape {y_shape} can not broadcast.")

    broadcast_shape_front = y_shape[0: y_len - length] if length == x_len else x_shape[0: x_len - length]
    broadcast_shape = list(broadcast_shape_front) + broadcast_shape_back
    return broadcast_shape


def get_concat_offset(x_shp, x_type, axis, prim_name):
    """for concat and concatoffset check args and compute offset"""
    validator.check_value_type("shape", x_shp, [tuple, list], prim_name)
    validator.check_positive_int(len(x_shp), "input_x rank", prim_name)
    validator.check_subclass("shape0", x_type[0], mstype.tensor, prim_name)
    validator.check_positive_int(len(x_shp[0]), "len of x_shp[0]", prim_name)
    rank_base = len(x_shp[0])
    validator.check_int_range(axis, -rank_base - 1, rank_base, Rel.INC_BOTH, 'axis', prim_name)
    if axis < 0:
        axis = axis + rank_base
    all_shp = x_shp[0][axis]
    offset = [0]
    for i in range(1, len(x_shp)):
        v = x_shp[i]
        validator.check('len of x_shp[%d]' % i, len(v), 'len of x_shp[0]', len(x_shp[0]), Rel.EQ, prim_name)
        validator.check('x_type[%d]' % i, x_type[i], 'x_type[0]', x_type[0], Rel.EQ, prim_name)
        for j in range(rank_base):
            if j != axis and v[j] != x_shp[0][j]:
                raise ValueError(f"For \'{prim_name}\' element {i} shape in input can not concat with first element")
        offset.append(all_shp)
        if all_shp == -1 or v[axis] == -1:
            all_shp = -1
        else:
            all_shp += v[axis]
    return offset, all_shp, axis
