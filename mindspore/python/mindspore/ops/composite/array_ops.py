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
"""array Operations."""
from mindspore.ops.composite.multitype_ops import _constexpr_utils as const_utils
from mindspore.common import dtype as mstype
from mindspore.common._register_for_tensor import tensor_operator_registry
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.ops.primitive import constexpr
from mindspore.ops import functional as F
from mindspore.ops import operations as P


@constexpr
def _check_is_int(arg_value, arg_name, op_name):
    arg_value = validator.check_is_int(arg_value, arg_name, op_name)
    return arg_value


@constexpr
def _check_positive_int(arg_value, arg_name, op_name):
    arg_value = validator.check_positive_int(arg_value, arg_name, op_name)
    return arg_value


@constexpr
def _check_axis_range(arg_value, limit, arg_name, op_name):
    arg_value = validator.check_int_range(arg_value, -limit, limit, Rel.INC_LEFT, arg_name, op_name)
    return arg_value


@constexpr
def _cal_repeat_dims(x_rank, rep, expand_axis):
    rep_dims = [1] * (x_rank + 1)
    rep_dims[expand_axis] = rep
    return tuple(rep_dims)


@constexpr
def _cal_reshape(x_shape, rep, axis):
    x_reshape = list(x_shape)
    x_reshape[axis] *= rep
    return tuple(x_reshape)


def repeat_interleave(x, repeats, dim=None):
    """
    Repeat elements of a tensor along an axis, like `numpy.repeat`.

    Args:
        x (Tensor): The tensor to repeat values for. Must be of type: float16,
            float32, int8, uint8, int16, int32, or int64.
        repeats (int): The number of times to repeat, must be positive.
        dim (int, optional): The axis along which to repeat, if None, defaults to 0.

    Returns:
        One tensor with values repeated along the specified axis. If x has shape
        (s1, s2, ..., sn) and axis is i, the output will have shape (s1, s2, ...,
        si * repeats, ..., sn). The output type will be the same as the type of `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), mindspore.int32)
        >>> output = ops.repeat_interleave(x, repeats=2, dim=0)
        >>> print(output)
        [[0 1 2]
         [0 1 2]
         [3 4 5]
         [3 4 5]]
    """
    return repeat_elements(x, repeats, dim)


def repeat_elements(x, rep, axis=0):
    """
    Repeat elements of a tensor along an axis, like `np.repeat` .

    Args:
        x (Tensor): The tensor to repeat values for. Must be of type: float16,
            float32, int8, uint8, int16, int32, or int64.
        rep (int): The number of times to repeat, must be positive.
        axis (int): The axis along which to repeat, default 0.

    Returns:
        One tensor with values repeated along the specified axis. If x has shape
        (s1, s2, ..., sn) and axis is i, the output will have shape (s1, s2, ...,
        si * rep, ..., sn). The output type will be the same as the type of `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore
        >>> import numpy as np
        >>> # case 1 : repeat on axis 0
        >>> x = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), mindspore.int32)
        >>> output = ops.repeat_elements(x, rep = 2, axis = 0)
        >>> print(output)
        [[0 1 2]
         [0 1 2]
         [3 4 5]
         [3 4 5]]
        >>> # case 2 : repeat on axis 1
        >>> x = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), mindspore.int32)
        >>> output = ops.repeat_elements(x, rep = 2, axis = 1)
        >>> print(output)
        [[0 0 1 1 2 2]
         [3 3 4 4 5 5]]
    """
    const_utils.check_type_valid(F.dtype(x), mstype.number_type, 'input x')
    rep = _check_positive_int(rep, "rep", "repeat_elements")
    axis = _check_is_int(axis, "axis", "repeat_elements")

    shape_op = P.Shape()
    rank_op = P.Rank()
    tile_op = P.Tile()
    expand_dims_op = P.ExpandDims()
    reshape_op = P.Reshape()

    x_rank = rank_op(x)
    axis = _check_axis_range(axis, x_rank, "axis", "repeat_elements")

    expand_axis = axis + 1
    x_expand = expand_dims_op(x, expand_axis)
    rep_dims = _cal_repeat_dims(x_rank, rep, expand_axis)
    x_expand = tile_op(x_expand, rep_dims)
    x_shape = shape_op(x)
    x_reshape = _cal_reshape(x_shape, rep, axis)
    x_rep = reshape_op(x_expand, x_reshape)

    return x_rep


tensor_operator_registry.register('repeat_elements', repeat_elements)


@constexpr
def _check_sequence_mask_input_len(input_shape, prim_name=None):
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if not input_shape:
        raise ValueError(f"{msg_prefix} input_shape must be greater than 0, but got {input_shape}.")
    # broadcast only supports 7d shape
    shape_size = len(input_shape)
    if shape_size >= 7:
        raise ValueError(f"{msg_prefix} dimension of input_shape must be less than 7, but got {shape_size}d.")


def sequence_mask(lengths, maxlen=None):
    """
    Returns a mask tensor representing the first N positions of each cell.

    If `lengths` has shape :math:`(d_1, d_2, ..., d_n)`, then the resulting tensor mask has type and shape
    :math:`(d_1, d_2, ..., d_n, maxlen)`, with mask :math:`[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])`.

    Args:
        lengths (Tensor): Tensor to calculate the mask for. All values in this tensor should be
            less than or equal to `maxlen`. Values greater than `maxlen` will be treated as `maxlen`.
        maxlen (int): size of the last dimension of returned tensor. Must be positive and same
            type as elements in `lengths`. Default is None.

    Returns:
        One mask tensor of shape `lengths.shape + (maxlen,)` .

    Raises:
        TypeError: If `lengths` is not a Tensor.
        TypeError: If `maxlen` is not an int.
        TypeError: If dtype of `lengths` is neither int32 nor int64.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore
        >>> import numpy as np
        >>> # case 1: When maxlen is assigned
        >>> x = Tensor(np.array([1, 2, 3, 4]))
        >>> output = ops.sequence_mask(x, 5)
        >>> print(output)
        [[ True False False False False]
         [ True  True False False False]
         [ True  True  True False False]
         [ True  True  True  True False]]
        >>> # case 2: When there is 0 in x
        >>> x = Tensor(np.array([[1, 3], [2, 0]]))
        >>> output = ops.sequence_mask(x, 5)
        >>> print(output)
        [[[ True False False False False]
          [ True  True  True False False]]
         [[ True  True False False False]
          [False False False False False]]]
        >>> # case 3: when the maxlen is not assigned
        >>> x = Tensor(np.array([[1, 3], [2, 4]]))
        >>> output = ops.sequence_mask(x)
        >>> print(output)
        [[[ True False False False ]
          [ True  True  True False ]]
         [[ True  True False False ]
          [ True  True  True  True ]]]
    """

    argmax_op = P.ArgMaxWithValue()
    reshape_op = P.Reshape()
    range_op = P.Range()
    expand_op = P.ExpandDims()
    cast_op = P.Cast()
    shape_op = P.Shape()
    to_tensor_op = P.ScalarToTensor()

    const_utils.check_type_valid(F.dtype(lengths), [mstype.int64, mstype.int32], 'lengths')
    _check_sequence_mask_input_len(shape_op(lengths), "sequence_mask")

    if maxlen is None:
        flatten_data = reshape_op(lengths, (-1,))
        flatten_data = cast_op(flatten_data, mstype.float32)
        _, value = argmax_op(flatten_data)
        maxlen = cast_op(value, mstype.int32)
    else:
        maxlen = _check_positive_int(maxlen, "maxlen", "sequence_mask")
        maxlen = to_tensor_op(maxlen, mstype.int32)

    range_vector = range_op(to_tensor_op(0, mstype.int32), maxlen
                            , to_tensor_op(1, mstype.int32))
    mask = expand_op(lengths, -1)
    result = range_vector < mask
    return result
