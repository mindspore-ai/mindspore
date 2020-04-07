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

"""broadcast"""


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
