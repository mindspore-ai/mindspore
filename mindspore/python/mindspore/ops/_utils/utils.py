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
from __future__ import absolute_import

import re

from mindspore.common.tensor import Tensor
from mindspore import _checkparam as validator
from mindspore.common import dtype as mstype
from mindspore.ops.primitive import _primexpr
from mindspore.common._utils import is_dim_unknown


def get_broadcast_shape(x_shape, y_shape, prim_name, arg_name1="x", arg_name2="y"):
    """
    Doing broadcast between tensor x and tensor y.

    Args:
        x_shape (list): The shape of tensor x.
        y_shape (list): The shape of tensor y.
        prim_name (str): Primitive name.
        arg_name1 (str): The arg name of x_shape.
        arg_name2 (str): The arg name of y_shape.

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
    if is_dim_unknown(x_shape) or is_dim_unknown(y_shape):
        return [-2]

    for i in range(-length, 0):
        if x_shape[i] == 1:
            broadcast_shape_back.append(y_shape[i])
        elif y_shape[i] == 1:
            broadcast_shape_back.append(x_shape[i])
        elif x_shape[i] == y_shape[i]:
            broadcast_shape_back.append(x_shape[i])
        elif (x_shape[i] == -1 and abs(y_shape[i]) != 1) or \
             (y_shape[i] == -1 and abs(x_shape[i]) != 1):
            broadcast_shape_back.append(max(x_shape[i], y_shape[i]))
        elif x_shape[i] == -1 or y_shape[i] == -1:
            broadcast_shape_back.append(-1)
        else:
            raise ValueError(f"For '{prim_name}', {arg_name1}.shape and {arg_name2}.shape need to "
                             f"broadcast. The value of {arg_name1}.shape[{i}] or {arg_name2}.shape[{i}]"
                             f" must be 1 or -1 when they are not the same, "
                             f"but got {arg_name1}.shape = {x_shape} "
                             f"and {arg_name2}.shape = {y_shape}.")

    broadcast_shape_front = y_shape[0: y_len - length] if length == x_len else x_shape[0: x_len - length]
    broadcast_shape = list(broadcast_shape_front) + broadcast_shape_back
    return broadcast_shape


def dim_not_equal(dim1, dim2):
    """Compare dim in shape"""
    return dim1 != dim2 and dim1 >= 0 and dim2 >= 0


def get_concat_offset(x_shp, x_type, axis, prim_name):
    """for concat and concatoffset check args and compute offset"""
    validator.check_value_type("shape", x_shp, [tuple, list], prim_name)
    validator.check_positive_int(len(x_shp), "input_x rank", prim_name)
    validator.check_subclass("shape0", x_type[0], mstype.tensor_type, prim_name)
    validator.check_positive_int(len(x_shp[0]), "len of x_shp[0]", prim_name)
    rank_base = len(x_shp[0])
    for i in range(1, len(x_shp)):
        validator.check('len of x_shp[%d]' % i, len(x_shp[i]), 'len of x_shp[0]',
                        len(x_shp[0]), validator.EQ, prim_name)
        validator.check('x_type[%d]' % i, x_type[i], 'x_type[0]', x_type[0], validator.EQ, prim_name)

    validator.check_int_range(axis, -rank_base, rank_base - 1, validator.INC_BOTH, 'axis', prim_name)
    if axis < 0:
        axis = axis + rank_base
    all_shp = x_shp[0][axis]
    offset = [0]
    for i in range(1, len(x_shp)):
        v = x_shp[i]
        for j in range(rank_base):
            if j != axis and dim_not_equal(v[j], x_shp[0][j]):
                raise ValueError(f"The shape of the two input elements of the Concat operator do not match:"
                                 f"shape[0] = {x_shp[0]} and shape[{i}] = {x_shp[i]}.")
        offset.append(all_shp)
        if all_shp == -1 or v[axis] == -1:
            all_shp = -1
        else:
            all_shp += v[axis]
    return offset, all_shp, axis


@_primexpr
def range_op(start, limit, delta, dtype):
    """helper function to get tensor in specified range."""
    output_tensor = Tensor(list(range(start, limit, delta)), dtype)
    return output_tensor


@_primexpr
def get_1d_shape(in_shape):
    """helper function to get 1d shape."""
    out_shape = 1
    for i in in_shape:
        out_shape *= i
    return (out_shape,)


@_primexpr
def generate_shape_index(out_shape, indices_shape, axis, batch_dims=0):
    out_rank = len(out_shape)
    ind_rank = len(indices_shape)
    if axis < 0:
        axis += out_rank - ind_rank + 1
    perm_part1 = tuple(range(axis, axis + ind_rank - batch_dims))
    index = tuple(range(out_rank))
    perm = index[:batch_dims] + perm_part1 + index[batch_dims:axis] + index[axis + ind_rank - batch_dims:]
    return perm


def ms_arrange(x):
    out = [i for i in range(x)]
    return out


def _parse_tcp_url_for_ipv4(url):
    """
    Parses a TCP URL (e.g., "tcp://100.100.1.1:1234") to extract the IPv4 address and port number.

    Args:
        url (str): The TCP URL string to be parsed.

    Returns:
        A tuple containing (ip_address, port).

    Raises:
        ValueError: If the URL format is incorrect, missing necessary parts, contains a non-numeric port,
                    or the port number is out of range.
        TypeError: If the provided `url` is not a string type.
    """

    def _is_valid_ipv4(ip):
        """
        Validates whether a string is a valid IPv4 address.
        """
        pattern = r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        return re.match(pattern, ip) is not None

    if not isinstance(url, str):
        raise TypeError("URL must be a string.")

    try:
        prefix = "tcp://"

        if not url.startswith(prefix):
            raise ValueError("must start with 'tcp://'")

        host_port = url[len(prefix):]
        if not host_port:
            raise ValueError("missing IP and port")

        colon_pos = host_port.find(':')
        if colon_pos == -1:
            raise ValueError("missing port separator ':'")

        if ':' in host_port[colon_pos + 1:]:
            raise ValueError("IPv6 addresses are not supported")

        ip = host_port[:colon_pos]
        if not ip:
            raise ValueError("missing IP address")
        if not _is_valid_ipv4(ip):
            raise ValueError(f"invalid IPv4 address - {ip}")

        port_str = host_port[colon_pos + 1:]
        if not port_str:
            raise ValueError("missing port number")
        if not port_str.isdigit():
            raise ValueError(f"port must be numeric - {port_str}")

        port = int(port_str)
        if not 1 <= port <= 65535:
            raise ValueError(f"port out of range [1, 65535] - {port}")

        return ip, port

    except ValueError as e:
        raise ValueError(f"Failed to parse TCP URL '{url}': {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error parsing TCP URL: {e}") from e
