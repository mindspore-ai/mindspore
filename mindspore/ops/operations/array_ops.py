# coding: utf-8

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

"""Operators for array."""

import copy
import functools
import itertools
import numbers
import numpy as np

from ..._checkparam import Validator as validator
from ..._checkparam import Rel
from ...common import dtype as mstype
from ...common.tensor import Tensor
from ..operations.math_ops import _infer_shape_reduce
from .._utils import get_concat_offset
from ..primitive import Primitive, PrimitiveWithInfer, prim_attr_register, _run_op
from ..._c_expression import signature_rw as sig_rw
from ..._c_expression import signature_kind as sig_kind
from ..._c_expression import signature_dtype as sig_dtype

def _check_infer_attr_reduce(axis, keep_dims, prim_name):
    validator.check_value_type('keep_dims', keep_dims, [bool], prim_name)
    validator.check_value_type('axis', axis, [int, tuple], prim_name)
    if isinstance(axis, tuple):
        for index, value in enumerate(axis):
            validator.check_value_type('axis[%d]' % index, value, [int], prim_name)


class ExpandDims(PrimitiveWithInfer):
    """
    Adds an additional dimension at the given axis.

    Note:
        If the specified axis is a negative number, the index is counted
        backward from the end and starts at 1.

    Raises:
        ValueError: If axis is not an integer or not in the valid range.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        - **axis** (int) - Specifies the dimension index at which to expand
          the shape of `input_x`. The value of axis must be in the range
          `[-input_x.dim()-1, input_x.dim()]`. Only constant value is allowed.

    Outputs:
        Tensor, the shape of tensor is :math:`(1, x_1, x_2, ..., x_R)` if the
        value of `axis` is 0.

    Examples:
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> expand_dims = P.ExpandDims()
        >>> output = expand_dims(input_tensor, 0)
    """

    @prim_attr_register
    def __init__(self):
        """init ExpandDims"""
        self.init_prim_io_names(inputs=['x', 'axis'], outputs=['output'])

    def __infer__(self, x, axis):
        validator.check_subclass("input_x", x['dtype'], mstype.tensor, self.name)
        x_shape = list(x['shape'])
        axis_v = axis['value']
        rank = len(x_shape)
        validator.check_int_range('axis', axis_v, -rank - 1, rank, Rel.INC_BOTH, self.name)
        if axis_v < 0:
            axis_v = rank + 1 + axis_v
        x_shape.insert(axis_v, 1)
        out = {'shape': x_shape,
               'dtype': x['dtype'],
               'value': None}
        return out


class DType(PrimitiveWithInfer):
    """
    Returns the data type of input tensor as mindspore.dtype.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        mindspore.dtype, the data type of a tensor.

    Examples:
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> type = P.DType()(input_tensor)
    """

    @prim_attr_register
    def __init__(self):
        """init DType"""

    def __infer__(self, x):
        validator.check_subclass("input_x", x['dtype'], mstype.tensor, self.name)
        out = {'shape': (),
               'dtype': mstype.type_type,
               'value': x['dtype'].element_type()}
        return out


class SameTypeShape(PrimitiveWithInfer):
    """
    Checks whether data type and shape of two tensors are the same.

    Raises:
        TypeError - If data type not the same.
        ValueError - If shape of two tensors not the same.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        - **input_y** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_S)`.

    Outputs:
        Tensor, the shape of tensor is :math:`(x_1, x_2, ..., x_R)`,
        if data type and shape of `input_x` and `input_y` are the same.

    Examples:
        >>> input_x = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> input_y = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> out = P.SameTypeShape()(input_x, input_y)
    """

    @prim_attr_register
    def __init__(self):
        """init Same"""

    def __call__(self, x, y):
        """run in PyNative mode"""
        validator.check_value_type('x', x, Tensor, self.name)
        validator.check_value_type('y', y, Tensor, self.name)
        validator.check('x dtype', x.dtype, 'y dtype', y.dtype, Rel.EQ, self.name, TypeError)
        validator.check('x shape', x.shape, 'y shape', y.shape, Rel.EQ, self.name)
        return x

    def __infer__(self, x, y):
        validator.check_subclass('x', x['dtype'], mstype.tensor, self.name)
        validator.check_subclass('y', y['dtype'], mstype.tensor, self.name)
        validator.check('x dtype', x['dtype'], 'y dtype', y['dtype'], Rel.EQ, self.name, TypeError)
        validator.check('x shape', x['shape'], 'y shape', y['shape'], Rel.EQ, self.name)
        return x


class Cast(PrimitiveWithInfer):
    """
    Returns a tensor with the new specified data type.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
          The tensor to be casted.
        - **type** (dtype.Number) - The valid data type of the output tensor. Only constant value is allowed.

    Outputs:
        Tensor, the shape of tensor is :math:`(x_1, x_2, ..., x_R)`, same as `input_x`.

    Examples:
        >>> input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
        >>> input_x = Tensor(input_np)
        >>> type_dst = mindspore.float16
        >>> cast = P.Cast()
        >>> result = cast(input_x, type_dst)
    """

    @prim_attr_register
    def __init__(self):
        # if primitive need setattr in __infer__ need add this flag
        """init Cast"""
        self.init_prim_io_names(inputs=['x', 'dst_type'], outputs=['output'])

    def check_elim(self, x, dtype):
        if  isinstance(x, (Tensor, numbers.Number)):
            if isinstance(x, Tensor) and x.dtype == dtype:
                return (True, x)
            if isinstance(x, numbers.Number):
                return (True, Tensor(x, dtype=dtype))
            return (False, None)
        raise ValueError(f"Expecting (Tensor, dtype), got : ({x}, {dtype})")

    def __infer__(self, x, t):
        src_type = x['dtype']
        dst_type = t['value']

        validator.check_subclass("input_x", src_type, [mstype.tensor, mstype.number], self.name)
        validator.check_subclass("type", dst_type, mstype.number, self.name)

        if isinstance(src_type, type(mstype.tensor)):
            src_type = x['dtype'].element_type()
        if isinstance(dst_type, type(mstype.tensor)):
            dst_type = dst_type.element_type()
        self.add_prim_attr('DstT', dst_type)
        self.add_prim_attr('SrcT', src_type)

        value = None
        if x['value'] is not None:
            np_dst_type = mstype.dtype_to_nptype(dst_type)
            if isinstance(x['value'], (int, float)):
                value = Tensor(np.array(x['value']).astype(np_dst_type))
            else:
                value = Tensor(x['value'].asnumpy().astype(np_dst_type))

        out = {'shape': x['shape'],
               'dtype': mstype.tensor_type(t['value']),
               'value': value}
        return out


class IsSubClass(PrimitiveWithInfer):
    """
    Check whether one type is sub class of another type.

    Inputs:
        - **sub_type** (mindspore.dtype) - The type to be check. Only constant value is allowed.
        - **type_** (mindspore.dtype) - The target type. Only constant value is allowed.

    Outputs:
        bool, the check result.

    Examples:
        >>> result = P.IsSubClass()(mindspore.int32,  mindspore.intc)
    """

    @prim_attr_register
    def __init__(self):
        pass

    def __infer__(self, sub_type, type_):
        sub_type_t = sub_type['value']
        type_v = type_['value']

        validator.check_value_type("sub_type", sub_type_t, [mstype.Type], self.name)
        validator.check_value_type("type_", type_v, [mstype.Type], self.name)

        value = mstype.issubclass_(sub_type_t, type_v)

        out = {'shape': (),
               'dtype': mstype.type_type,
               'value': value}
        return out


class IsInstance(PrimitiveWithInfer):
    """
    Check whether an object is an instance of a target type.

    Inputs:
        - **inst** (Any Object) - The instance to be check. Only constant value is allowed.
        - **type_** (mindspore.dtype) - The target type. Only constant value is allowed.

    Outputs:
        bool, the check result.

    Examples:
        >>> a = 1
        >>> result = P.IsInstance()(a, mindspore.int32)
    """

    @prim_attr_register
    def __init__(self):
        pass

    def __infer__(self, inst, type_):
        sub_type_t = inst['dtype']
        type_v = type_['value']

        validator.check_const_input("inst", inst['value'], self.name)
        validator.check_value_type("type_", type_v, [mstype.Type], self.name)

        value = mstype.issubclass_(sub_type_t, type_v)

        out = {'shape': (),
               'dtype': mstype.type_type,
               'value': value}
        return out


class Reshape(PrimitiveWithInfer):
    """
    Reshapes input tensor with the same values based on a given shape tuple.

    Raises:
        ValueError: Given a shape tuple, if it has more than one -1; or if the product
            of its elements is less than or equal to 0 or cannot be divided by the product
            of the input tensor shape; or if it does not match the input's array size.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        - **input_shape** (tuple[int]) - The input tuple is constructed by multiple
          integers, i.e., :math:`(y_1, y_2, ..., y_S)`. Only constant value is allowed.

    Outputs:
        Tensor, the shape of tensor is :math:`(y_1, y_2, ..., y_S)`.

    Examples:
        >>> input_tensor = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> reshape = P.Reshape()
        >>> output = reshape(input_tensor, (3, 2))
    """

    @prim_attr_register
    def __init__(self):
        """init Reshape"""
        self.init_prim_io_names(inputs=['tensor', 'shape'], outputs=['output'])

    def __infer__(self, x, shape):
        shape_v = shape['value']
        x_shp = x['shape']
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        validator.check_value_type("shape", shape_v, [tuple], self.name)
        shape_v = list(shape_v)
        neg_index = -1
        dim_prod = 1
        for i, shp_i in enumerate(shape_v):
            validator.check_value_type("shape[%d]" % i, shp_i, [int], self.name)
            if shp_i == -1:
                if neg_index != -1:
                    raise ValueError(f'The shape can only has one -1 at most, but {shape_v}.')
                neg_index = i
            else:
                dim_prod *= shp_i
        arr_prod = np.prod(x_shp)
        if dim_prod <= 0 or arr_prod % dim_prod != 0:
            raise ValueError(f'For \'{self.name}\' the product of shape should > 0 and'
                             f' can be divided by prod of input {arr_prod},'
                             f' but shape {shape}, product of shape {dim_prod}.')

        if neg_index != -1:
            shape_v[neg_index] = int(arr_prod / dim_prod)
            dim_prod *= shape_v[neg_index]
        if dim_prod != arr_prod:
            raise ValueError(f'For \'{self.name}\' The shape arg for reshape must match array''s size'
                             f' input shape {arr_prod}, shape {dim_prod}.')

        value = None
        if x['value'] is not None:
            value = Tensor(x['value'].asnumpy().reshape(shape_v))

        out = {'shape': tuple(shape_v),
               'dtype': x['dtype'],
               'value': value}
        return out


class Shape(Primitive):
    """
    Returns the shape of input tensor.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        tuple[int], the output tuple is constructed by multiple integers,
        :math:`(x_1, x_2, ..., x_R)`.

    Examples:
        >>> input_tensor = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> shape = P.Shape()
        >>> output = shape(input_tensor)
    """

    @prim_attr_register
    def __init__(self):
        """init Shape"""


class Squeeze(PrimitiveWithInfer):
    """
    Returns a tensor with the same type but dimensions of 1 being removed based on axis.

    Note:
        The dimension index starts at 0 and must be in the range `[-input.dim(), input.dim())`.

    Raises:
        ValueError: If the corresponding dimension of the specified axis does not equal to 1.

    Args:
        axis (int): Specifies the dimension indexes of shape to be removed, which will remove
            all the dimensions that are equal to 1. If specified, it must be int32 or int64.
            Default: (), an empty tuple.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, the shape of tensor is :math:`(x_1, x_2, ..., x_S)`.

    Examples:
        >>> input_tensor = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> squeeze = P.Squeeze(2)
        >>> output = squeeze(input_tensor)
    """

    @prim_attr_register
    def __init__(self, axis=()):
        """init Squeeze"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        validator.check_value_type('axis', axis, [int, tuple], self.name)
        if isinstance(axis, tuple):
            for idx, item in enumerate(axis):
                validator.check_value_type("axis[%d]" % idx, item, [int], self.name)
        else:
            self.axis = (axis,)
            self.add_prim_attr("axis", (axis,))

    def infer_shape(self, x_shape):
        axis = self.axis
        x_shape = list(x_shape)
        ndim = len(x_shape)
        if not axis:
            ret = [d for d in x_shape if d != 1]
        else:
            for a in axis:
                validator.check_int_range('axis or its elements', a, -ndim, ndim - 1, Rel.INC_BOTH, self.name)
                if x_shape[a] != 1:
                    raise ValueError('Cannot select an axis to squeeze out which has size not equal to one.')
            ret = [x_shape[i] for i in range(ndim) if not (i in axis or (i - ndim) in axis)]
        return ret

    def infer_dtype(self, x_dtype):
        validator.check_subclass("x", x_dtype, mstype.tensor, self.name)
        return x_dtype


class Transpose(PrimitiveWithInfer):
    """
    Permutes the dimensions of input tensor according to input perm.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        - **input_perm** (tuple[int]) - The permutation to be converted. The input tuple is constructed by multiple
          indexes. The length of `input_perm` and the shape of `input_x` should be the same. Only constant value is
          allowed.

    Outputs:
        Tensor, the type of output tensor is same as `input_x` and the shape of output tensor is decided by the
        shape of `input_x` and the value of `input_perm`.

    Examples:
        >>> input_tensor = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
        >>> perm = (0, 2, 1)
        >>> transpose = P.Transpose()
        >>> output = transpose(input_tensor, perm)
    """

    @prim_attr_register
    def __init__(self):
        """init Transpose"""
        self.init_prim_io_names(inputs=['x', 'perm'], outputs=['output'])

    def __infer__(self, x, perm):
        x_shape = x['shape']
        p_value = perm['value']
        x_type = x['dtype']
        validator.check_value_type("p_value", p_value, [tuple], self.name)
        validator.check_subclass("x_type", x_type, mstype.tensor, self.name)

        if len(x_shape) != len(p_value):
            raise ValueError('The dimension of x and perm must be equal.')

        tmp = list(p_value)
        for i, dim in enumerate(p_value):
            validator.check_integer("perm[%d]" % i, dim, 0, Rel.GE, self.name)
            validator.check_integer("perm[%d]" % i, dim, len(p_value), Rel.LT, self.name)
            tmp.remove(dim)
            if dim in tmp:
                raise ValueError('The value of perm is wrong.')

        out_shapes = []
        for i in p_value:
            out_shapes.append(x_shape[i])
        out = {'shape': tuple(out_shapes),
               'dtype': x['dtype'],
               'value': None}
        return out


class GatherV2(PrimitiveWithInfer):
    """
    Returns a slice of input tensor based on the specified indices and axis.

    Inputs:
        - **input_params** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
          The original Tensor.
        - **input_indices** (Tensor) - The shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
          Specifies the indices of elements of the original Tensor. Must be in the range
          `[0, input_param.shape[axis])`.
        - **axis** (int) - Specifies the dimension index to gather indices.

    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`.

    Examples:
        >>> input_params = Tensor(np.array([[1, 2, 7, 42], [3, 4, 54, 22], [2, 2, 55, 3]]), mindspore.float32)
        >>> input_indices = Tensor(np.array([1, 2]), mindspore.int32)
        >>> axis = 1
        >>> out = P.GatherV2()(input_params, input_indices, axis)
    """

    @prim_attr_register
    def __init__(self):
        """init index_select"""
        self.init_prim_io_names(inputs=['params', 'indices', 'axis'], outputs=['output'])

    def __infer__(self, params, indices, axis):
        validator.check_subclass("params", params['dtype'], mstype.tensor, self.name)
        validator.check_tensor_type_same({"indices": indices['dtype']}, mstype.int_type, self.name)
        validator.check_subclass("axis", axis['dtype'], mstype.int_, self.name)
        axis_v = axis['value']
        params_shp = params['shape']
        rank = len(params_shp)
        validator.check_int_range("axis", axis_v, -rank, rank, Rel.INC_LEFT, self.name)
        if axis_v < 0:
            axis_v += rank
        out_shape = params_shp[:axis_v] + indices['shape'] + params_shp[axis_v + 1:]
        out = {'shape': out_shape,
               'dtype': params['dtype'],
               'value': None}
        return out


class SparseGatherV2(GatherV2):
    """
    Returns a slice of input tensor based on the specified indices and axis.

    Inputs:
        - **input_params** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
          The original Tensor.
        - **input_indices** (Tensor) - The shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
          Specifies the indices of elements of the original Tensor. Must be in the range
          `[0, input_param.shape[axis])`.
        - **axis** (int) - Specifies the dimension index to gather indices.

    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`.

    Examples:
        >>> input_params = Tensor(np.array([[1, 2, 7, 42], [3, 4, 54, 22], [2, 2, 55, 3]]), mindspore.float32)
        >>> input_indices = Tensor(np.array([1, 2]), mindspore.int32)
        >>> axis = 1
        >>> out = P.GatherV2()(input_params, input_indices, axis)
    """


class EmbeddingLookup(PrimitiveWithInfer):
    """
    Returns a slice of input tensor based on the specified indices.

    This Primitive has the similar functionality as GatherV2 operating on `axis = 0`, but has three more inputs:
    `offset`, `reduce_scatter_flag` and `split_num`. This primitive runs on the host instead of devices.

    Inputs:
        - **input_params** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
          The Tensor slice, instead of the entire Tensor.
        - **input_indices** (Tensor) - The shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
          Specifies the indices of elements of the original Tensor. Values can be out of range of `input_params`,
          and the exceeding part will be filled with 0 in the output.
        - **offset** (int) - Specifies the offset value of this `input_params` slice. Thus the real indices
          are equal to `input_indices` minus `offset`.
        - **reduce_scatter_flag** (bool) - Specifies whether perform reduce_scatter on host or not.
          Only constant value is allowed.
        - **split_num** (int) - Specifies the number of partitions of the reduce_scatter produces. This variable
          is used only if `reduce_scatter_flag` is True. Only constant value is allowed.


    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`.

    Examples:
        >>> input_params = Tensor(np.array([[8, 9], [10, 11], [12, 13], [14, 15]]), mindspore.float32)
        >>> input_indices = Tensor(np.array([[5, 2], [8, 5]]), mindspore.int32)
        >>> offset = 4
        >>> reduce_scatter_flag = False
        >>> split_num = 1
        >>> out = P.EmbeddingLookup()(input_params, input_indices, offset, reduce_scatter_flag, split_num)
        [[[10, 11], [0 ,0]], [[0, 0], [10, 11]]]
    """
    @prim_attr_register
    def __init__(self):
        """init index_select"""
        self.__setattr_flag__ = True
        self.init_prim_io_names(inputs=['params', 'indices', 'offset', 'reduce_scatter_flag', 'split_num'],
                                outputs=['output'])
        self.add_prim_attr('primitive_target', 'CPU')

    def __infer__(self, params, indices, offset, reduce_scatter_flag=False, split_num=2):
        validator.check_subclass("params", params['dtype'], mstype.tensor, self.name)
        validator.check_tensor_type_same({"indices": indices['dtype']}, mstype.int_type, self.name)
        validator.check_subclass("offset", offset['dtype'], mstype.int_, self.name)
        validator.check_subclass("split_num", split_num['dtype'], mstype.int_, self.name)
        if split_num['value'] < 1:
            raise ValueError("The parameter 'split_num' must be positive, but got %d." % split_num)
        params_shp = params['shape']
        out_shape = indices['shape'] + params_shp[1:]
        if reduce_scatter_flag is None:
            raise ValueError("The value of 'reduce_scatter_flag' is None.")
        reduce_scatter_flag_value = reduce_scatter_flag['value']
        if split_num is None:
            raise ValueError("The value of 'split_num_value' is None.")
        split_num_value = split_num['value']
        if reduce_scatter_flag_value is True:
            # Partition the tensor along the dimension 0. The shape size of dimension 0 should be divisible by
            # (split_num * 8)
            if out_shape[0] % (split_num_value * 8) != 0:
                raise ValueError("The dimension 0 of the shape: %d, is not divisible by: %d." %
                                 (out_shape[0], (split_num_value * 8)))
            # After 'Concat' on host, the shape size of dimension 0 is: out_shape[0] // 8
            out_shape[0] = out_shape[0] // 8
        out = {'shape': out_shape,
               'dtype': params['dtype'],
               'value': None}
        return out


class Split(PrimitiveWithInfer):
    """
    Splits input tensor into output_num of tensors along the given axis and output numbers.

    Args:
        axis (int): Index of the split position. Default: 0.
        output_num (int): The number of output tensors. Default: 1.

    Raises:
        ValueError: If axis is out of the range [-len(input_x.shape), len(input_x.shape)),
            or if the output_num is less than or equal to 0, or if the
            dimension which to split cannot be evenly divided by output_num.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        tuple[Tensor], the shape of each output tensor is same, which is
        :math:`(y_1, y_2, ..., y_S)`.

    Examples:
        >>> split = P.Split(1, 2)
        >>> x = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]))
        >>> output = split(x)
    """

    @prim_attr_register
    def __init__(self, axis=0, output_num=1):
        """init Split"""
        validator.check_value_type("axis", axis, [int], self.name)
        validator.check_value_type("output_num", output_num, [int], self.name)
        self.axis = axis
        self.output_num = output_num

    def __infer__(self, x):
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        x_shape = list(x['shape'])
        dim = len(x_shape)
        validator.check_int_range('axis value', self.axis, -dim, dim, Rel.INC_LEFT, self.name)
        validator.check_integer("output_num", self.output_num, 0, Rel.GT, self.name)
        output_valid_check = x_shape[self.axis] % self.output_num
        validator.check_integer("the dimension which to split divides output_num", output_valid_check, 0, Rel.EQ,
                                self.name)
        x_shape[self.axis] = int(x_shape[self.axis] / self.output_num)
        out_shapes = []
        out_dtypes = []
        for _ in range(self.output_num):
            out_shapes.append(tuple(x_shape))
            out_dtypes.append(x['dtype'])
        out_shapes = tuple(out_shapes)
        out_dtypes = tuple(out_dtypes)
        out = {'shape': out_shapes,
               'dtype': out_dtypes,
               'value': None}
        return out


class Rank(PrimitiveWithInfer):
    """
    Returns the rank of a tensor.

    Returns a 0-D int32 Tensor representing the rank of input; the rank of a tensor
    is the number of indices required to uniquely select each element of the tensor.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor. 0-D int32 Tensor representing the rank of input, i.e., :math:`R`.

    Examples:
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> rank = P.Rank()
        >>> rank(input_tensor)
    """

    @prim_attr_register
    def __init__(self):
        """init Rank"""

    def __infer__(self, x):
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        out = {'shape': None,
               'dtype': None,
               'value': len(x['shape'])}
        return out


class TruncatedNormal(PrimitiveWithInfer):
    """
    Returns a tensor of the specified shape filled with truncated normal values.

    The generated values follow a normal distribution.

    Args:
        seed (int): A int number used to create random seed. Default: 0.
        dtype (:class:`mindspore.dtype`): Data type. Default: mindspore.float32.

    Inputs:
        - **shape** (tuple[int]) - Shape of output tensor, is a tuple of positive int.

    Outputs:
        Tensor, type of output tensor is same as attribute `dtype`.

    Examples:
        >>> shape = (1, 2, 3)
        >>> truncated_normal = P.TruncatedNormal()
        >>> output = truncated_normal(shape)
    """

    @prim_attr_register
    def __init__(self, seed=0, dtype=mstype.float32):
        """init TruncatedNormal"""
        validator.check_value_type('seed', seed, [int], self.name)
        validator.check_type_same({'dtype': dtype}, mstype.number_type, self.name)

    def __infer__(self, shape):
        shape_value = shape['value']
        validator.check_value_type("shape", shape_value, [tuple], self.name)
        for i, value in enumerate(shape_value):
            validator.check_integer(f'{i}th value of shape', value, 0, Rel.GT, self.name)
        out = {'shape': shape_value,
               'dtype': mstype.tensor_type(self.dtype),
               'value': None}
        return out


class Size(PrimitiveWithInfer):
    r"""
    Returns the elements count size of a tensor.

    Returns an int scalar representing the elements size of input, the total number of elements in the tensor.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        int, a scalar representing the elements size of `input_x`, tensor is the number of elements
        in a tensor, :math:`size=x_1*x_2*...x_R`.

    Examples:
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> size = P.Size()
        >>> output = size(input_tensor)
    """

    @prim_attr_register
    def __init__(self):
        """init Size"""

    def __infer__(self, x):
        size = 1
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        shp = x['shape']
        if not shp:
            size = 0
        else:
            size = functools.reduce(lambda x, y: x * y, x['shape'])
        out = {'shape': None,
               'dtype': mstype.int32,
               'value': size}
        return out


class Fill(PrimitiveWithInfer):
    """
    Creates a tensor filled with a scalar value.

    Creates a tensor with shape described by the first argument and fills it with values in the second argument.

    Inputs:
        - **type** (mindspore.dtype) - The specified type of output tensor. Only constant value is allowed.
        - **shape** (tuple) - The specified shape of output tensor. Only constant value is allowed.
        - **value** (scalar) - Value to fill the returned tensor. Only constant value is allowed.

    Outputs:
        Tensor, has the same type and shape as input value.

    Examples:
        >>> fill = P.Fill()
        >>> fill(mindspore.float32, (2, 2), 1)
    """

    @prim_attr_register
    def __init__(self):
        """init Fill"""

    def __infer__(self, dtype, dims, x):
        validator.check_value_type("shape", dims['value'], [tuple], self.name)
        validator.check_value_type("value", x['value'], [numbers.Number, bool], self.name)
        for idx, item in enumerate(dims['value']):
            validator.check_integer("dims[%d]" % idx, item, 0, Rel.GT, self.name)
        valid_types = [mstype.bool_, mstype.int8, mstype.int32, mstype.int64,
                       mstype.uint8, mstype.uint32, mstype.uint64,
                       mstype.float16, mstype.float32, mstype.float64]
        validator.check_type_same({"value": dtype['value']}, valid_types, self.name)
        x_nptype = mstype.dtype_to_nptype(dtype['value'])
        ret = np.full(dims['value'], x['value'], x_nptype)
        out = {
            'value': Tensor(ret),
            'shape': dims['value'],
            'dtype': x['dtype'],
        }
        return out


class OnesLike(PrimitiveWithInfer):
    """
    Creates a new tensor. All elements' value are 1.

    Returns a tensor of ones with the same shape and type as the input.

    Inputs:
        - **input_x** (Tensor) - Input tensor.

    Outputs:
        Tensor, has the same shape and type as `input_x` but filled with ones.

    Examples:
        >>> oneslike = P.OnesLike()
        >>> x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
        >>> output = oneslike(x)
    """

    @prim_attr_register
    def __init__(self):
        """Init OnesLike"""

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_type_same({'x': x_dtype}, mstype.number_type + (mstype.bool_,), self.name)
        return x_dtype


class ZerosLike(PrimitiveWithInfer):
    """
    Creates a new tensor. All elements value are 0.

    Returns a tensor of zeros with the same shape and type as the input tensor.

    Inputs:
        - **input_x** (Tensor) - Input tensor.

    Outputs:
        Tensor, has the same shape and type as `input_x` but filled with zeros.

    Examples:
        >>> zeroslike = P.ZerosLike()
        >>> x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
        >>> output = zeroslike(x)
    """

    @prim_attr_register
    def __init__(self):
        """Init ZerosLike"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_type_same({'x': x_dtype}, mstype.number_type + (mstype.bool_,), self.name)
        return x_dtype


class TupleToArray(PrimitiveWithInfer):
    """
    Converts a tuple to tensor.

    If the first number type of tuple is int, the output tensor type is int. Else, the output tensor type is float.

    Inputs:
        - **input_x** (tuple) - A tuple of numbers. These numbers have the same type. Only constant value is allowed.

    Outputs:
        Tensor, if the input tuple contain `N` numbers, then the output tensor shape is (N,).

    Examples:
        >>> type = P.TupleToArray()((1,2,3))
    """

    @prim_attr_register
    def __init__(self):
        """init TupleToArray"""

    def infer_value(self, x):
        validator.check_value_type("x", x, [tuple], self.name)
        validator.check("size of x", len(x), '', 0, Rel.GT, self.name)
        dtype = type(x[0])
        for i, item in enumerate(x):
            validator.check_value_type(f"x[{i}]", item, [numbers.Number], self.name)
        if not all(isinstance(item, dtype) for item in x):
            raise TypeError("For \'{self.name}\' all elements of input x must be have same type.")
        if isinstance(x[0], int):
            ret = np.array(x, np.int32)
        else:
            ret = np.array(x, np.float32)
        return Tensor(ret)

    def __call__(self, x):
        args = list()
        if isinstance(x, range):
            args.append(tuple(x))
        else:
            args.append(x)
        return _run_op(self, self.name, args)


class ScalarToArray(PrimitiveWithInfer):
    """
    Converts scalar to `Tensor`.

    Inputs:
        - **input_x** (Union[int, float]) - The input is a scalar. Only constant value is allowed.

    Outputs:
        Tensor. 0-D Tensor and the content is the input.

    Examples:
        >>> op = P.ScalarToArray()
        >>> data = 1.0
        >>> output = op(data)
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_value(self, x):
        validator.check_value_type("x", x, [int, float], self.name)
        if isinstance(x, int):
            ret = np.array(x, np.int32)
        else:
            ret = np.array(x, np.float32)
        return Tensor(ret)


class ScalarToTensor(PrimitiveWithInfer):
    """
    Converts scalar to `Tensor`, and convert data type to specified type.

    Inputs:
        - **input_x** (Union[int, float]) - The input is a scalar. Only constant value is allowed.
        - **dtype** (mindspore.dtype) - The target data type. Default: mindspore.float32. Only
          constant value is allowed.

    Outputs:
        Tensor. 0-D Tensor and the content is the input.

    Examples:
        >>> op = P.ScalarToTensor()
        >>> data = 1
        >>> output = op(data, mindspore.float32)
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_value(self, x, dtype=mstype.float32):
        validator.check_value_type("x", x, [int, float], self.name)
        validator.check_subclass("dtype", dtype, mstype.number, self.name)
        data_type = mstype.dtype_to_nptype(dtype)
        return Tensor(np.array(x, data_type))


class InvertPermutation(PrimitiveWithInfer):
    r"""
    Computes the inverse of an index permutation.

    Given a tuple input, this operation inserts a dimension of 1 at the dimension
    This operation calculates the inverse of the index replacement. It requires a
    1-dimensional tuple x, which represents the array starting at zero,
    and swaps each value with its index position. In other words, for the output
    tuple y and the input tuple x, this operation calculates the following:
    :math:`y[x[i]] = i, \quad i \in [0, 1, \ldots, \text{len}(x)-1]`.

    Note:
        These values must include 0. There must be no duplicate values and the
        values can not be negative.

    Inputs:
        - **input_x** (Union(tuple[int], Tensor[int])) - The input tuple is constructed by multiple
          integers, i.e., :math:`(y_1, y_2, ..., y_S)` representing the indices.
          The values must include 0. There can be no duplicate values or negative values.
          If the input is Tensor, it must be 1-d and the dtype is int. Only constant value is allowed.


    Outputs:
        tuple[int]. the lenth is same as input.

    Examples:
        >>> invert = P.InvertPermutation()
        >>> input_data = (3, 4, 0, 2, 1)
        >>> output = invert(input_data)
        >>> output == (2, 4, 3, 0, 1)
    """

    @prim_attr_register
    def __init__(self):
        """init InvertPermutation"""

    def __infer__(self, x):
        x_shp = x['shape']
        x_value = x['value']
        if x_value is None:
            raise ValueError(f'For \'{self.name}\' the input value must be const.')
        validator.check_value_type("shape", x_shp, [tuple, list], self.name)
        if mstype.issubclass_(x['dtype'], mstype.tensor):
            validator.check('x dimension', len(x_shp), '', 1, Rel.EQ, self.name)
            validator.check_tensor_type_same({'x dtype': x['dtype']}, mstype.int_type, self.name)
            x_value = [int(i) for i in x_value.asnumpy()]
        z = [x_value[i] for i in range(len(x_value))]
        z.sort()

        for i in range(1, len(z)):
            if z[i-1] == z[i]:
                raise ValueError(f"For {self.name}, {z[i]} is duplicated in the input.")
        validator.check(f'value min', min(x_value), '', 0, Rel.EQ, self.name)
        validator.check(f'value max', max(x_value), '', len(x_value)-1, Rel.EQ, self.name)

        y = [None] * len(x_value)
        for i, value in enumerate(x_value):
            validator.check_value_type("input[%d]" % i, value, [int], self.name)
            validator.check(f'value', z[i], f'index', i, Rel.EQ, self.name)
            y[value] = i
            z.append(value)
        return {'shape': x_shp,
                'dtype': x['dtype'],
                'value': tuple(y)}


class Argmax(PrimitiveWithInfer):
    """
    Returns the indices of the max value of a tensor across the axis.

    If the shape of input tensor is :math:`(x_1, ..., x_N)`, the output tensor shape is
    :math:`(x_1, ..., x_{axis-1}, x_{axis+1}, ..., x_N)`.

    Args:
        axis (int): Axis on which Argmax operation applies. Default: -1.
        output_type (:class:`mindspore.dtype`): An optional data type of `mindspore.dtype.int32`.
            Default: `mindspore.dtype.int32`.

    Inputs:
        - **input_x** (Tensor) - Input tensor.

    Outputs:
        Tensor, indices of the max value of input tensor across the axis.

    Examples:
        >>> input_x = Tensor(np.array([2.0, 3.1, 1.2]), mindspore.float32)
        >>> index = P.Argmax(output_type=mindspore.int32)(input_x)
    """

    @prim_attr_register
    def __init__(self, axis=-1, output_type=mstype.int32):
        """init Argmax"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        validator.check_value_type("axis", axis, [int], self.name)
        validator.check_type_same({'output': output_type}, [mstype.int32, mstype.int64], self.name)
        self.axis = axis
        self.add_prim_attr('output_type', output_type)

    def infer_shape(self, x_shape):
        axis = self.axis
        if axis is None:
            axis = 0
        x_rank = len(x_shape)
        validator.check_int_range("axis", axis, -x_rank, x_rank, Rel.INC_LEFT, self.name)
        axis = axis + x_rank if axis < 0 else axis
        ouput_shape = [x_shape[i] for i in range(x_rank) if i != axis]
        return ouput_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("input_x", x_dtype, mstype.tensor, self.name)
        return mstype.tensor_type(self.output_type)


class Argmin(PrimitiveWithInfer):
    """
    Returns the indices of the min value of a tensor across the axis.

    If the shape of input tensor is :math:`(x_1, ..., x_N)`, the output tensor shape is
    :math:`(x_1, ..., x_{axis-1}, x_{axis+1}, ..., x_N)`.

    Args:
        axis (int): Axis on which Argmin operation applies. Default: -1.
        output_type (:class:`mindspore.dtype`): An optional data type of `mindspore.dtype.int32`.
            Default: `mindspore.dtype.int32`.

    Inputs:
        - **input_x** (Tensor) - Input tensor.

    Outputs:
        Tensor, indices of the min value of input tensor across the axis.

    Examples:
        >>> input_x = Tensor(np.array([2.0, 3.1, 1.2]), mindspore.float32)
        >>> index = P.Argmin()(input_x)
        >>> assert index == Tensor(2, mindspore.int64)
    """

    @prim_attr_register
    def __init__(self, axis=-1, output_type=mstype.int32):
        """init Argmin"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        validator.check_value_type("axis", axis, [int], self.name)
        validator.check_type_name("output_type", output_type, [mstype.int32, mstype.int64], self.name)
        self.axis = axis
        self.add_prim_attr('output_type', output_type)

    def infer_shape(self, x_shape):
        axis = self.axis
        if axis is None:
            axis = 0
        x_rank = len(x_shape)
        validator.check_int_range("axis", axis, -x_rank, x_rank, Rel.INC_LEFT, self.name)
        axis = axis + x_rank if axis < 0 else axis
        ouput_shape = [x_shape[i] for i in range(x_rank) if i != axis]
        return ouput_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("input_x", x_dtype, mstype.tensor, self.name)
        return mstype.tensor_type(self.output_type)


class ArgMaxWithValue(PrimitiveWithInfer):
    """
    Calculates maximum value with corresponding index.

    Calculates maximum value along with given axis for the input tensor. Returns the maximum values and indices.

    Note:
        In auto_parallel and semi_auto_parallel mode, the first output index can not be used.

    Args:
        axis (int): The dimension to reduce. Default: 0.
        keep_dims (bool): Whether to reduce dimension, if true the output will keep same dimension with the input,
                          the output will reduce dimension if false. Default: False.

    Inputs:
        - **input_x** (Tensor) - The input tensor, can be any dimension. Set the shape of input tensor as
          :math:`(x_1, x_2, ..., x_N)`.

    Outputs:
        tuple(Tensor), tuple of 2 tensors, corresponding index and maximum value of input tensor.
        - index (Tensor) - The index for maximum value of input tensor. If `keep_dims` is true, the output tensors shape
        is :math:`(x_1, x_2, ..., x_{axis-1}, 1, x_{axis+1}, ..., x_N)`. Else, the shape is
        :math:`(x_1, x_2, ..., x_{axis-1}, x_{axis+1}, ..., x_N)`.
        - output_x (Tensor) - The maximum value of input tensor, the shape same as index.

    Examples:
        >>> input_x = Tensor(np.random.rand(5))
        >>> index, output = P.ArgMaxWithValue()(input_x)
    """

    @prim_attr_register
    def __init__(self, axis=0, keep_dims=False):
        """init ArgMaxWithValue"""
        self.axis = axis
        self.keep_dims = keep_dims
        _check_infer_attr_reduce(axis, keep_dims, self.name)

    def infer_shape(self, x_shape):
        axis = self.axis
        x_rank = len(x_shape)
        validator.check_int_range("axis", axis, -x_rank, x_rank, Rel.INC_LEFT, self.name)
        ouput_shape = _infer_shape_reduce(x_shape, self.axis, self.keep_dims, self.name)
        return ouput_shape, ouput_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("input_x", x_dtype, mstype.tensor, self.name)
        return mstype.tensor_type(mstype.int32), x_dtype


class ArgMinWithValue(PrimitiveWithInfer):
    """
    Calculates minimum value with corresponding index, return indices and values.

    Calculates minimum value along with given axis for the input tensor. Returns the minimum values and indices.

    Note:
        In auto_parallel and semi_auto_parallel mode, the first output index can not be used.

    Args:
        axis (int): The dimension to reduce. Default: 0.
        keep_dims (bool): Whether to reduce dimension, if true the output will keep same dimension as the input,
                          the output will reduce dimension if false. Default: False.

    Inputs:
        - **input_x** (Tensor) - The input tensor, can be any dimension. Set the shape of input tensor as
          :math:`(x_1, x_2, ..., x_N)`.

    Outputs:
        Tensor, corresponding index and minimum value of input tensor. If `keep_dims` is true, the output tensors shape
        is :math:`(x_1, x_2, ..., x_{axis-1}, 1, x_{axis+1}, ..., x_N)`. Else, the shape is
        :math:`(x_1, x_2, ..., x_{axis-1}, x_{axis+1}, ..., x_N)`.

    Examples:
        >>> input_x = Tensor(np.random.rand(5))
        >>> index, output = P.ArgMinWithValue()(input_x)
    """

    @prim_attr_register
    def __init__(self, axis=0, keep_dims=False):
        """init ArgMinWithValue"""
        self.axis = axis
        self.keep_dims = keep_dims
        _check_infer_attr_reduce(axis, keep_dims, self.name)

    def infer_shape(self, x_shape):
        axis = self.axis
        x_rank = len(x_shape)
        validator.check_int_range("axis", axis, -x_rank, x_rank, Rel.INC_LEFT, self.name)
        ouput_shape = _infer_shape_reduce(x_shape, self.axis, self.keep_dims, self.name)
        return ouput_shape, ouput_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("input_x", x_dtype, mstype.tensor, self.name)
        return mstype.tensor_type(mstype.int32), x_dtype


class Tile(PrimitiveWithInfer):
    r"""
    Replicates a tensor with given multiples times.

    Creates a new tensor by replicating input multiples times. The dimension of
    output tensor is the larger of the dimension length of input and the length of multiples.

    Inputs:
        - **input_x** (Tensor) - 1-D or higher Tensor. Set the shape of input tensor as
          :math:`(x_1, x_2, ..., x_S)`.

        - **multiples** (tuple[int]) - The input tuple is constructed by multiple
          integers, i.e., :math:`(y_1, y_2, ..., y_S)`. The length of `multiples`
          can't be smaller than the length of shape in `input_x`.

    Outputs:
        Tensor, has the same type as the `input_x`.

        - If the length of `multiples` is the same as the length of shape in `input_x`,
          then the shape of their corresponding positions can be multiplied, and
          the shape of Outputs is :math:`(x_1*y_1, x_2*y_2, ..., x_S*y_R)`.
        - If the length of `multiples` is larger than the length of shape in `input_x`,
          fill in multiple 1 in front of the shape in `input_x` until their lengths are consistent.
          Such as set the shape of `input_x` as :math:`(1, ..., x_1, x_2, ..., x_S)`,
          then the shape of their corresponding positions can be multiplied, and
          the shape of Outputs is :math:`(1*y_1, ..., x_S*y_R)`.

    Examples:
        >>> tile = P.Tile()
        >>> input_x = Tensor(np.array([[1, 2], [3, 4]]), mindspore.float32)
        >>> multiples = (2, 3)
        >>> result = tile(input_x, multiples)
        [[1.  2.  1.  2.  1.  2.]
         [3.  4.  3.  4.  3.  4.]
         [1.  2.  1.  2.  1.  2.]
         [3.  4.  3.  4.  3.  4.]]
    """

    @prim_attr_register
    def __init__(self):
        """init Tile"""
        self.init_prim_io_names(inputs=['x', 'multiples'], outputs=['output'])

    def check_elim(self, base_tensor, multiplier):
        if (not isinstance(base_tensor, Tensor)) or (not isinstance(multiplier, tuple)):
            raise ValueError("Expecting (Tensor, tuple), got: ({}, {})".format(base_tensor, multiplier))
        def is_all_zeros(v_tuple):
            return all(v == 1 for v in v_tuple)
        if is_all_zeros(multiplier):
            return (True, base_tensor)
        return (False, None)

    def __infer__(self, x, multiples):
        multiples_v = multiples['value']
        x_shp = x['shape']
        validator.check_value_type("shape", multiples_v, [tuple], self.name)
        for i, multiple in enumerate(multiples_v):
            validator.check_value_type("multiples[%d]" % i, multiple, [int], self.name)
        valid_types = [mstype.int16, mstype.int32, mstype.bool_, mstype.float16, mstype.float32]
        validator.check_tensor_type_same({'x': x['dtype']}, valid_types, self.name)
        len_sub = len(multiples_v) - len(x_shp)
        multiples_w = None
        if len_sub == 0:
            multiples_w = multiples_v
        if len_sub > 0:
            for i in range(0, len_sub):
                x_shp.insert(0, 1)
            multiples_w = multiples_v
        elif len_sub < 0:
            raise ValueError(f'For \'{self.name}\' the length of multiples can not be smaller than '
                             f'the length of dimension in input_x.')
        for i, a in enumerate(multiples_w):
            x_shp[i] *= a
        value = None
        if x['value'] is not None:
            value = Tensor(np.tile(x['value'].asnumpy(), multiples_w))
        return {'shape': x_shp,
                'dtype': x['dtype'],
                'value': value}


class UnsortedSegmentSum(PrimitiveWithInfer):
    r"""
    Computes the sum along segments of a tensor.

    Calculates a tensor such that :math:`\text{output}[i] = \sum_{segment\_ids[j] == i} \text{data}[j, \ldots]`, where
    :math:`j` is a tuple describing the index of element in data.  `segment_ids` selects which elements in data to sum
    up. Segment_ids does not need to be sorted, and it does not need to cover all values in the entire valid value
    range.

    If the sum of the given segment_ids :math:`i` is empty, then :math:`\text{output}[i] = 0`. If the given segment_ids
    is negative, the value will be ignored. 'num_segments' should be equal to the number of different segment_ids.

    Inputs:
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_R)`.
        - **segment_ids** (Tensor) - Set the shape as :math:`(x_1, x_2, ..., x_N)`, where 0 < N <= R. Type must be int.
        - **num_segments** (int) - Set :math:`z` as num_segments.

    Outputs:
        Tensor, the shape is :math:`(z, x_{N+1}, ..., x_R)`.

    Examples:
        >>> input_x = Tensor([1, 2, 3, 4], mindspore.float32)
        >>> segment_ids = Tensor([0, 0, 1, 2], mindspore.int32)
        >>> num_segments = 4
        >>> P.UnsortedSegmentSum()(input_x, segment_ids, num_segments)
        [3, 3, 4, 0]
    """

    @prim_attr_register
    def __init__(self):
        """init UnsortedSegmentSum"""
        self.init_prim_io_names(inputs=['x', 'segment_ids', 'num_segments'], outputs=['y'])

    def __infer__(self, x, segment_ids, num_segments):
        x_type = x['dtype']
        x_shp = x['shape']
        validator.check_subclass("input_x", x_type, mstype.tensor, self.name)
        validator.check_value_type("x_shape", x_shp, [list], self.name)
        x_shp_len = len(x_shp)
        validator.check_integer("rank of input_x", x_shp_len, 0, Rel.GT, self.name)
        segment_ids_shp = segment_ids['shape']
        segment_ids_type = segment_ids['dtype']
        validator.check_subclass("segment_ids", segment_ids_type, mstype.tensor, self.name)
        validator.check_value_type("segment_ids", segment_ids_shp, [list], self.name)
        segment_ids_shp_len = len(segment_ids_shp)
        validator.check_integer("rank of segment_ids", segment_ids_shp_len, 0, Rel.GT, self.name)
        validator.check(f'rank of input_x', len(x_shp),
                        'rank of segments_id', len(segment_ids_shp), Rel.GE, self.name)
        for i, value in enumerate(segment_ids_shp):
            validator.check("ids[%d]" % i, value, 'input[%d]' % i, x_shp[i], Rel.EQ, self.name)
        num_segments_v = num_segments['value']
        validator.check_value_type('num_segments', num_segments_v, [int], self.name)
        validator.check_integer("num_segments", num_segments_v, 0, Rel.GT, self.name)
        shp = [num_segments_v]
        shp += x_shp[segment_ids_shp_len:]
        out = {'shape': shp,
               'dtype': mstype.tensor_type(x_type.element_type()),
               'value': None}
        return out


class UnsortedSegmentMin(PrimitiveWithInfer):
    """
    Computes the minimum along segments of a tensor.

    If the given segment_ids is negative, the value will be ignored.

    Inputs:
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_R)`.
        - **segment_ids** (Tensor) - A `1-D` tensor whose shape is :math:`(x_1)`.
        - **num_segments** (int) - The value spcifies the number of distinct `segment_ids`.

    Outputs:
        Tensor, Set the number of `num_segments` as `N`, the shape is :math:`(N, x_2, ..., x_R)`.

    Examples:
        >>> input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
        >>> segment_ids = Tensor(np.array([0, 1, 1]).astype(np.int32))
        >>> num_segments = 2
        >>> unsorted_segment_min = P.UnsortedSegmentMin()
        >>> unsorted_segment_min(input_x, segment_ids, num_segments)
        [[1., 2., 3.], [4., 2., 1.]]
    """

    @prim_attr_register
    def __init__(self):
        """init UnsortedSegmentMin"""
        self.init_prim_io_names(inputs=['x', 'segment_ids', 'num_segments'], outputs=['y'])

    def __infer__(self, x, segment_ids, num_segments):
        x_type = x['dtype']
        x_shape = x['shape']
        segment_ids_shape = segment_ids['shape']
        valid_type = [mstype.float16, mstype.float32, mstype.int32]
        validator.check_tensor_type_same({"x": x['dtype']}, valid_type, self.name)
        validator.check_tensor_type_same({"segment_ids": segment_ids['dtype']}, [mstype.int32], self.name)
        validator.check_integer("rank of segment_ids_shape", len(segment_ids_shape), 1, Rel.EQ, self.name)
        validator.check(f'first shape of input_x', x_shape[0],
                        'length of segments_id', segment_ids_shape[0], Rel.EQ, self.name)
        num_segments_v = num_segments['value']
        validator.check_value_type('num_segments', num_segments_v, [int], self.name)
        validator.check_integer("num_segments", num_segments_v, 0, Rel.GT, self.name)
        segment_ids_shape_len = len(segment_ids_shape)
        out_shape = [num_segments_v]
        out_shape += x_shape[segment_ids_shape_len:]
        out = {'shape': out_shape,
               'dtype': x_type,
               'value': None}
        return out


class Concat(PrimitiveWithInfer):
    r"""
    Concat tensor in specified axis.

    Concat input tensors along with the given axis.

    Note:
        The input data is a tuple of tensors. These tensors have the same rank `R`. Set the given axis as `m`, and
        :math:`0 \le m < N`. Set the number of input tensors as `N`. For the :math:`i`-th tensor :math:`t_i` has
        the shape :math:`(x_1, x_2, ..., x_{mi}, ..., x_R)`. :math:`x_{mi}` is the :math:`m`-th dimension of the
        :math:`i`-th tensor. Then, the output tensor shape is

        .. math::
            (x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)

    Args:
        axis (int): The specified axis. Default: 0.

    Inputs:
        - **input_x** (tuple, list) - Tuple or list of input tensors.

    Outputs:
        Tensor, the shape is :math:`(x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)`.

    Examples:
        >>> data1 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
        >>> data2 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
        >>> op = P.Concat()
        >>> output = op((data1, data2))
    """

    @prim_attr_register
    def __init__(self, axis=0):
        """init Tile"""
        validator.check_value_type("axis", axis, [int], self.name)

    def __infer__(self, input_x):
        axis = self.axis
        x_shp = input_x['shape']
        x_type = input_x['dtype']
        _, all_shp, _ = get_concat_offset(x_shp, x_type, axis, self.name)
        self.add_prim_attr('T', x_type[0].element_type())
        self.add_prim_attr('inputNums', len(x_shp))
        ret_shp = x_shp[0].copy()
        ret_shp[axis] = all_shp
        out = {'shape': ret_shp,
               'dtype': x_type[0],
               'value': None}
        return out


def _get_pack_shape(x_shape, x_type, axis, prim_name):
    """for pack output shape"""
    validator.check_value_type("shape", x_shape, [tuple, list], prim_name)
    validator.check_integer("len of input_x", len(x_shape), 1, Rel.GT, prim_name)
    validator.check_subclass("input_x[0]", x_type[0], mstype.tensor, prim_name)
    rank_base = len(x_shape[0])
    N = len(x_shape)
    out_shape = x_shape[0]
    validator.check_int_range('axis', axis, -rank_base - 1, rank_base, Rel.INC_BOTH, prim_name)
    if axis < 0:
        axis = axis + rank_base + 1
    for i in range(1, N):
        validator.check('x_type[%d]' % i, x_type[i], 'base', x_type[0], Rel.EQ, prim_name, TypeError)
        if x_shape[i] != x_shape[0]:
            raise ValueError(f"For \'{prim_name}\' element {i} shape in input can not pack with first element")
    out_shape.insert(axis, N)
    return out_shape


class Pack(PrimitiveWithInfer):
    r"""
    Packs a list of tensors in specified axis.

    Packs the list of input tensors with the same rank `R`, output is a tensor of rank `(R+1)`.

    Given input tensors of shape :math:`(x_1, x_2, ..., x_R)`. Set the number of input tensors as `N`.
    If :math:`0 \le axis`, the output tensor shape is :math:`(x_1, x_2, ..., x_{axis}, N, x_{axis+1}, ..., x_R)`.

    Args:
        axis (int): Dimension along which to pack. Default: 0.
                    Negative values wrap around. The range is [-(R+1), R+1).

    Inputs:
        - **input_x** (Union[tuple, list]) - A Tuple or list of Tensor objects with the same shape and type.

    Outputs:
        Tensor. A packed Tensor with the same type as `input_x`.

    Raises:
        TypeError: If the data types of elements in input_x are not the same.
        ValueError: If length of input_x is not greater than 1;
                    or if axis is out of the range [-(R+1), R+1);
                    or if the shapes of elements in input_x are not the same.

    Examples:
        >>> data1 = Tensor(np.array([0, 1]).astype(np.float32))
        >>> data2 = Tensor(np.array([2, 3]).astype(np.float32))
        >>> pack = P.Pack()
        >>> output = pack([data1, data2])
        [[0, 1], [2, 3]]
    """

    @prim_attr_register
    def __init__(self, axis=0):
        """init Pack"""
        validator.check_value_type("axis", axis, [int], self.name)
        self.axis = axis

    def __infer__(self, value):
        x_shape = value['shape']
        x_type = value['dtype']
        self.add_prim_attr('num', len(x_shape))
        all_shape = _get_pack_shape(x_shape, x_type, self.axis, self.name)
        out = {'shape': all_shape,
               'dtype': x_type[0],
               'value': None}
        return out


class Unpack(PrimitiveWithInfer):
    r"""
    Unpacks tensor in specified axis.

    Unpacks a tensor of rank `R` along axis dimension, output tensors will have rank `(R-1)`.

    Given a tensor of shape :math:`(x_1, x_2, ..., x_R)`. If :math:`0 \le axis`,
    the shape of tensor in output is :math:`(x_1, x_2, ..., x_{axis}, x_{axis+2}, ..., x_R)`.

    This is the opposite of pack.

    Args:
        axis (int): Dimension along which to pack. Default: 0.
                    Negative values wrap around. The range is [-R, R).

    Inputs:
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_R)`.
          A rank R > 0 Tensor to be unpacked.

    Outputs:
        A tuple of Tensors, the shape of each objects is same.

    Raises:
        ValueError: If axis is out of the range [-len(input_x.shape), len(input_x.shape)).

    Examples:
        >>> unpack = P.Unpack()
        >>> input_x = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]))
        >>> output = unpack(input_x)
        ([1, 1, 1, 1], [2, 2, 2, 2])
    """

    @prim_attr_register
    def __init__(self, axis=0):
        """init Unpack"""
        validator.check_value_type("axis", axis, [int], self.name)
        self.axis = axis

    def __infer__(self, x):
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        x_shape = list(x['shape'])
        dim = len(x_shape)
        validator.check_int_range('axis value', self.axis, -dim, dim, Rel.INC_LEFT, self.name)
        if self.axis < 0:
            self.axis = self.axis + dim
        output_num = x_shape[self.axis]
        validator.check_value_type("num", output_num, [int], self.name)
        validator.check_integer("output_num", output_num, 0, Rel.GT, self.name)
        self.add_prim_attr('num', output_num)
        output_valid_check = x_shape[self.axis] - output_num
        validator.check_integer("The dimension which to unpack divides output_num", output_valid_check, 0, Rel.EQ,
                                self.name)
        out_shapes = []
        out_dtypes = []
        out_shape = x_shape[:self.axis] + x_shape[self.axis + 1:]
        for _ in range(output_num):
            out_shapes.append(tuple(out_shape))
            out_dtypes.append(x['dtype'])
        out_shapes = tuple(out_shapes)
        out_dtypes = tuple(out_dtypes)
        out = {'shape': out_shapes,
               'dtype': out_dtypes,
               'value': None}
        return out


class Slice(PrimitiveWithInfer):
    """
    Slice a tensor in specified shape.

    Args:
        x (Tensor): The target tensor.
        begin (tuple): The beginning of the slice. Only constant value is allowed.
        size (tuple): The size of the slice. Only constant value is allowed.

    Returns:
        Tensor.

    Examples:
        >>> data = Tensor(np.array([[[1, 1, 1], [2, 2, 2]],
        >>>                         [[3, 3, 3], [4, 4, 4]],
        >>>                         [[5, 5, 5], [6, 6, 6]]]).astype(np.int32))
        >>> type = P.Slice()(data, (1, 0, 0), (1, 1, 3))
    """

    @prim_attr_register
    def __init__(self):
        """init slice"""
        self.init_prim_io_names(inputs=['x', 'begin', 'size'], outputs=['output'])

    def __infer__(self, x, begin, size):
        x_shape = x['shape']
        x_shp_len = len(x_shape)
        validator.check_const_input('begin', begin['value'], self.name)
        validator.check_const_input('size', size['value'], self.name)
        begin_v, size_v = begin['value'], size['value']
        if begin_v is None or size_v is None:
            return {'shape': None,
                    'dtype': x['dtype'],
                    'value': None}
        for key, value in zip(('begin', 'size'), (begin_v, size_v)):
            validator.check(f'len of {key}', len(value),
                            'len x\'s dim', x_shp_len)
        for i in range(x_shp_len):
            if x_shape[i] < begin_v[i] + size_v[i]:
                y = begin_v[i] + size_v[i]
                raise ValueError("For '%s' slice shape can not bigger than orign shape %d, %d." %
                                 (self.name, x_shape[i], y))
        return {'shape': size_v,
                'dtype': x['dtype'],
                'value': None}


class Select(PrimitiveWithInfer):
    r"""

    Return the selected elements, either from input :math:`x` or input :math:`y`, depending on the `condition`.

    Given a tensor as input, this operation inserts a dimension of 1 at the dimension,
    if both :math:`x` and :math:`y` are none, the operation returns the coordinates of the true
    element in the condition, the coordinates are returned as a two-dimensional
    tensor, where the first dimension (row) represents the number of true elements
    and the second dimension (columns) represents the coordinates of the true
    elements. Keep in mind that the shape of the output tensor can vary depending
    on how much of the true value is in the input. Indexes are output in row-first
    order.

    If neither is None, :math:`x` and :math:`y` must have the same shape. If :math:`x` and :math:`y` are
    scalars, the conditional tensor must be a scalar. If :math:`x` and :math:`y` are
    higher-demensional vectors, the condition must be a vector whose size matches the
    first dimension of :math:`x`, or must have the same shape as :math:`y`.

    The conditional tensor acts as an optional compensation (mask), which
    determines whether the corresponding element / row in the output should be
    selected from :math:`x` (if true) or :math:`y` (if false) based on the value of each
    element.

    If condition is a vector, then :math:`x` and :math:`y` are higher-demensional matrices, then it
    chooses to copy that row (external dimensions) from :math:`x` and :math:`y`. If condition has
    the same shape as :math:`x` and :math:`y`, you can choose to copy these elements from :math:`x`
    and :math:`y`.

    Inputs:
        - **input_x** (Tensor[bool]) - The shape is :math:`(x_1, x_2, ..., x_N)`.
          The condition tensor, decides whose element is chosen.
        - **input_y** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_N, ..., x_R)`.
          The first input tensor.
        - **input_z** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_N, ..., x_R)`.
          The second input tensor.

    Outputs:
        Tensor, has the same shape as input_y. The shape is :math:`(x_1, x_2, ..., x_N, ..., x_R)`.

    Examples:
        >>> select = P.Select()
        >>> input_x = Tensor([True, False])
        >>> input_y = Tensor([2,3], mindspore.float32)
        >>> input_z = Tensor([1,2], mindspore.float32)
        >>> select(input_x, input_y, input_z)
    """

    @prim_attr_register
    def __init__(self):
        """init"""

    def infer_shape(self, cond_shape, x_shape, y_shape):
        if cond_shape != x_shape or x_shape != y_shape:
            raise ValueError('The x_shape and y_shape must be the same as cond_shape.')
        return x_shape

    def infer_dtype(self, cond_type, x_type, y_type):
        self.add_prim_attr('T', x_type)
        validator.check_subclass("x_type", x_type, mstype.tensor, self.name)
        validator.check_subclass("y_type", y_type, mstype.tensor, self.name)
        validator.check_tensor_type_same({"cond": cond_type}, [mstype.bool_], self.name)
        if x_type != y_type:
            raise TypeError('\'%s\' the x_type %s must be the same as y_type %s.' % (self.name, x_type, y_type))
        return x_type


class StridedSlice(PrimitiveWithInfer):
    r"""

    Extracts a strided slice of a tensor.

    Given an input tensor, this operation inserts a dimension of length 1 at the dimension.
    This operation extracts a fragment of size (end-begin)/stride from the given
    'input_tensor'. Starting from the position specified by the begin, the fragment
    continues adding stride to the index until all dimensions are not less than end.

    Note:
        The stride may be negative value, which causes reverse slicing.
        The shape of `begin`, `end` and `strides` should be the same.

    Args:
        begin_mask (int): Starting index of the slice. Default: 0.
        end_mask (int): Ending index of the slice. Default: 0.
        ellipsis_mask (int): An int mask. Default: 0.
        new_axis_mask (int): An int mask. Default: 0.
        shrink_axis_mask (int): An int mask. Default: 0.

    Inputs:
        - **input_x** (Tensor) - The input Tensor.
        - **begin** (tuple[int]) - A tuple which represents the location where to start. Only
          constant value is allowed.
        - **end** (tuple[int]) - A tuple or which represents the maximum location where to stop.
          Only constant value is allowed.
        - **strides** (tuple[int]) - A tuple which represents the stride continuously added
          before reach the maximum location. Only constant value is allowed.

    Outputs:
        Tensor.
        Explain with the following example.
            - In the 0th dim, begin is 1, end is 2, and strides is 1,
              because :math:`1+1=2\geq2`, the interval is :math:`[1,2)`.
              Thus, return the element with :math:`index = 1` in 0th dim, i.e., [[3, 3, 3], [4, 4, 4]].
            - In the 1st dim, similarly, the interval is :math:`[0,1)`.
              Based on the return value of the 0th dim, return the element with :math:`index = 0`,
              i.e., [3, 3, 3].
            - In the 2nd dim, similarly, the interval is :math:`[0,3)`.
              Based on the return value of the 1st dim, return the element with :math:`index = 0,1,2`,
              i.e., [3, 3, 3].
            - Finally, the output is [3, 3, 3].

    Examples
        >>> input_x = Tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]],
        >>>                   [[5, 5, 5], [6, 6, 6]]], mindspore.float32)
        >>> slice = P.StridedSlice()
        >>> output = slice(input_x, (1, 0, 0), (2, 1, 3), (1, 1, 1))
        >>> output.shape
        (1, 1, 3)
        >>> output
        [[[3, 3, 3]]]
    """

    @prim_attr_register
    def __init__(self,
                 begin_mask=0,
                 end_mask=0,
                 ellipsis_mask=0,
                 new_axis_mask=0,
                 shrink_axis_mask=0):
        """init StrideSlice"""
        self.init_prim_io_names(inputs=['x', 'begin', 'end', 'strides'], outputs=['output'])
        validator.check_value_type('begin_mask', begin_mask, [int], self.name)
        validator.check_value_type('end_mask', end_mask, [int], self.name)
        validator.check_value_type('ellipsis_mask', ellipsis_mask, [int], self.name)
        validator.check_value_type('new_axis_mask', new_axis_mask, [int], self.name)
        validator.check_value_type('shrink_axis_mask', shrink_axis_mask, [int], self.name)

    def __infer__(self, x, begin, end, strides):
        begin_v, end_v, strides_v = begin['value'], end['value'], strides['value']
        validator.check_value_type("begin", begin_v, [tuple], self.name)
        validator.check_value_type("end", end_v, [tuple], self.name)
        validator.check_value_type("strides", strides_v, [tuple], self.name)

        x_shape = x['shape']
        x_shp_len = len(x_shape)
        if len(begin_v) != x_shp_len or len(end_v) != x_shp_len or len(strides_v) != x_shp_len:
            raise ValueError(f"For \'{self.name}\' the length of begin index{begin_v}, end index{end_v} and "
                             f"strides{strides_v} must be equal to the dims({x_shp_len}) of input.")

        ret_shape = []
        append_dimensions = []
        shrink_pos = bin(self.shrink_axis_mask)[::-1]
        new_pos = bin(self.new_axis_mask)[::-1]
        for i in range(x_shp_len):
            # After the integer is converted to binary, it is a str and the first two chars are the flag char '0b'
            if i < (len(new_pos) - 2) and new_pos[i] == '1':
                ret_shape.append(1)
                append_dimensions.append(x_shape[x_shp_len - 1 - len(append_dimensions)])
                continue
            if i < (len(shrink_pos) - 2) and shrink_pos[i] == '1':
                validator.check_integer(f'begin[{i}]', begin_v[i], -x_shape[i], Rel.GE, self.name)
                validator.check_integer(f'begin[{i}]', begin_v[i], x_shape[i], Rel.LT, self.name)
                continue

            begin_idx = begin_v[i]
            end_idx = end_v[i]
            strides_idx = strides_v[i]
            if self.begin_mask:
                begin_idx = 0
            if self.end_mask:
                end_idx = x_shape[i]
            validator.check_integer(f'begin[{i}]', begin_idx, x_shape[i], Rel.LE, self.name)
            validator.check_integer(f'end[{i}]', end_idx, x_shape[i], Rel.LE, self.name)
            validator.check_integer(f'strides[{i}]', strides_idx, 0, Rel.NE, self.name)
            if strides_idx > 0:
                # If sliced forward , end_idx >= begin_idx
                validator.check(f'begin[{i}]', begin_idx, f'end[{i}]', end_idx, Rel.LE)
                if begin_idx < 0 < end_idx:
                    # Turn negative begin_idx into positive values
                    begin_idx = x_shape[i] + begin_idx
                num_elems = (end_idx - begin_idx + strides_idx - 1) // strides_idx
            else:
                # If sliced backwards, end_idx <= begin_idx
                validator.check(f'begin[{i}]', begin_idx, f'end[{i}]', end_idx, Rel.GE)
                if end_idx < 0 < begin_idx:
                    # Turn negative end_idx into positive values
                    end_idx = x_shape[i] + end_idx
                num_elems = (end_idx - begin_idx + strides_idx + 1) // strides_idx

            ret_shape.append(num_elems)
        if append_dimensions:
            ret_shape += append_dimensions[::-1]
        return {'shape': ret_shape,
                'dtype': x['dtype'],
                'value': None}


class Diag(PrimitiveWithInfer):
    r"""

    Construct a diagonal tensor with a given diagonal values.

    Assume `input_x` has dimensions :math:`[D_1,... D_k]`, the output is a tensor of
    rank 2k with dimensions :math:`[D_1,..., D_k, D_1,..., D_k]` where:
    :math:`output[i_1,..., i_k, i_1,..., i_k] = input_x[i_1,..., i_k]` and 0 everywhere else.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor.

    Examples:
        >>> input_x = Tensor([1, 2, 3, 4])
        >>> diag = P.Diag()
        >>> diag(input_x)
        [[1, 0, 0, 0],
         [0, 2, 0, 0],
         [0, 0, 3, 0],
         [0, 0, 0, 4]]
    """

    @prim_attr_register
    def __init__(self):
        """init Diag"""

    def infer_dtype(self, x_type):
        validator.check_subclass('input_x', x_type, mstype.tensor, self.name)
        return x_type

    def infer_shape(self, x_shape):
        validator.check("x rank", len(x_shape), "", 1, Rel.GE)
        ret_shape = copy.deepcopy(x_shape)
        ret_shape = ret_shape + ret_shape
        return ret_shape

    def infer_value(self, x):
        if x is None:
            return None
        # do constant-folding only when x rank is 1
        if len(x.shape) != 1:
            return None
        ret = np.diag(x.asnumpy())
        return Tensor(ret)


class DiagPart(PrimitiveWithInfer):
    r"""

    Extract the diagonal part from given tensor.

    Assume input has dimensions :math:`[D_1,..., D_k, D_1,..., D_k]`, the output is a tensor
    of rank k with dimensions :math:`[D_1,..., D_k]` where:
    :math:`output[i_1,..., i_k] = input[i_1,..., i_k, i_1,..., i_k]`.

    Inputs:
        - **input_x** (Tensor) - The input Tensor.

    Outputs:
        Tensor.

    Examples
        >>> input_x = Tensor([[1, 0, 0, 0],
        >>>                   [0, 2, 0, 0],
        >>>                   [0, 0, 3, 0],
        >>>                   [0, 0, 0, 4]])
        >>> diag_part = P.DiagPart()
        >>> diag_part(input_x)
        [1, 2, 3, 4]
    """

    @prim_attr_register
    def __init__(self):
        """init DiagPart"""

    def infer_dtype(self, x_type):
        validator.check_subclass('input_x', x_type, mstype.tensor, self.name)
        return x_type

    def infer_shape(self, x_shape):
        if len(x_shape) % 2 != 0 or \
                not x_shape:
            raise ValueError(f"For \'{self.name}\' input rank must be non-zero and even, but got rank {len(x_shape)}, "
                             f"with shapes {x_shape}")
        length = len(x_shape) // 2
        ret_shape = x_shape[0:length]
        return ret_shape

    def infer_value(self, x):
        if x is None:
            return None
        # do constant-folding only when x rank is 2
        if len(x.shape) != 2:
            return None
        ret = np.diag(x.asnumpy())
        return Tensor(ret)


class Eye(PrimitiveWithInfer):
    """

    Creates a tensor with ones on the diagonal and zeros elsewhere.

    Inputs:
        - **n** (int) - Number of rows of returned tensor
        - **m** (int) - Number of columns of returned tensor
        - **t** (mindspore.dtype) - Mindspore's dtype, The data type of the returned tensor.

    Outputs:
        Tensor, a tensor with ones on the diagonal and zeros elsewhere.

    Examples:
        >>> eye = P.Eye()
        >>> out_tensor = eye(2, 2, mindspore.int32)
    """

    @prim_attr_register
    def __init__(self):
        """init Eye"""

    def infer_value(self, n, m, t):
        validator.check_integer("n", n, 0, Rel.GT, self.name)
        validator.check_integer("m", m, 0, Rel.GT, self.name)
        args = {"dtype": t}
        validator.check_type_same(args, mstype.number_type + (mstype.bool_,), self.name)
        np_type = mstype.dtype_to_nptype(t)
        ret = np.eye(n, m, dtype=np_type)
        return Tensor(ret)


class ScatterNd(PrimitiveWithInfer):
    """
    Scatters a tensor into a new tensor depending on the specified indices.

    Creates an empty tensor, and set values by scattering the update tensor depending on indices.

    Inputs:
        - **indices** (Tensor) - The index of scattering in the new tensor. With int32 data type.
        - **update** (Tensor) - The source Tensor to be scattered.
        - **shape** (tuple[int]) - Define the shape of the output tensor. Has the same type as indices.

    Outputs:
        Tensor, the new tensor, has the same type as `update` and the same shape as `shape`.

    Examples:
        >>> op = P.ScatterNd()
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
        >>> update = Tensor(np.array([3.2, 1.1]), mindspore.float32)
        >>> shape = (3, 3)
        >>> output = op(indices, update, shape)
    """

    @prim_attr_register
    def __init__(self):
        """Init ScatterNd"""
        self.init_prim_io_names(inputs=['indices', 'update', 'shape'], outputs=['output'])

    def __infer__(self, indices, update, shape):
        shp = shape['value']
        validator.check_subclass("update_dtype", update['dtype'], mstype.tensor, self.name)
        validator.check_tensor_type_same({"indices": indices['dtype']}, [mstype.int32], self.name)
        validator.check_value_type("shape", shp, [tuple], self.name)
        for i, x in enumerate(shp):
            validator.check_integer("shape[%d]" % i, x, 0, Rel.GT, self.name)

        indices_shape, update_shape = indices["shape"], update["shape"]
        if indices_shape[0] != update_shape[0]:
            raise ValueError(f'For \'{self.name}\' The indices_shape[0] and update_shape[0] must be equal.')

        return {'shape': shp,
                'dtype': update['dtype'],
                'value': None}


class ResizeNearestNeighbor(PrimitiveWithInfer):
    r"""
    Resize the input tensor by using nearest neighbor algorithm.

    Resize input tensor to given size by using nearest neighbor algorithm. The nearest
    neighbor algorithm selects the value of the nearest point and does not consider the
    values of neighboring points at all, yielding a piecewise-constant interpolant.

    Args:
        size (Union[tuple, list]): The target size. The dimension of size must be 2.
        align_corners (bool): Whether the centers of the 4 corner pixels of the input
                              and output tensors are aligned. Default: False.

    Inputs:
         - **input_x** (Tensor) - The input tensor. The shape of the tensor is :math:`(N, C, H, W)`.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, NEW\_C, NEW\_H, W)`.

    Examples:
        >>> input_tensor = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> resize = P.ResizeNearestNeighbor((2, 2))
        >>> output = resize(input_tensor)
    """

    @prim_attr_register
    def __init__(self, size, align_corners=False):
        """Init ResizeNearestNeighbor"""
        validator.check_value_type("size", size, [tuple, list], self.name)
        validator.check_value_type("align_corners", align_corners, [bool], self.name)
        validator.check_integer("length of size", len(size), 2, Rel.EQ, self.name)
        for i, value in enumerate(size):
            validator.check_integer(f'{i}th value of size', value, 0, Rel.GE, self.name)
        self.init_prim_io_names(inputs=['image_in'], outputs=['image_out'])

    def infer_shape(self, x):
        validator.check('the dimension of input_x', len(x), '', 2, Rel.GE, self.name)
        return tuple(x)[:-2] + tuple(self.size)

    def infer_dtype(self, x):
        return x


class GatherNd(PrimitiveWithInfer):
    """
    Gathers slices from a tensor by indices.

    Using given indices to gather slices from a tensor with a specified shape.

    Inputs:
        - **input_x** (Tensor) - The target tensor to gather values.
        - **indices** (Tensor) - The index tensor.

    Outputs:
        Tensor, has the same type as `input_x` and the shape is indices_shape[:-1] + x_shape[indices_shape[-1]:].

    Examples:
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> op = P.GatherNd()
        >>> output = op(input_x, indices)
    """

    @prim_attr_register
    def __init__(self):
        """Init GatherNd"""
        self.init_prim_io_names(inputs=['input_x', 'indices'], outputs=['y'])

    def infer_shape(self, x_shape, indices_shape):
        validator.check('the dimension of x', len(x_shape),
                        'the dimension of indices', indices_shape[-1], Rel.GE, self.name)
        return indices_shape[:-1] + x_shape[indices_shape[-1]:]

    def infer_dtype(self, x_dtype, indices_dtype):
        validator.check_subclass("x_dtype", x_dtype, mstype.tensor, self.name)
        validator.check_tensor_type_same({"indices": indices_dtype}, mstype.int_type, self.name)
        return x_dtype


class TensorScatterUpdate(PrimitiveWithInfer):
    """
    Update tensor value by using input indices and value.

    Using given values to update tensor value, along with the input indices.

    Inputs:
        - **input_x** (Tensor) - The target tensor.
        - **indices** (Tensor) - The index of input tensor whose data type is int32.
        - **update** (Tensor) - The tensor to update the input tensor, has the same type as input,
          and update.shape = indices.shape + input_x.shape[1:].

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> update = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> op = P.TensorScatterUpdate()
        >>> output = op(input_x, indices, update)
    """
    @prim_attr_register
    def __init__(self):
        """Init TensorScatterUpdate"""
        self.init_prim_io_names(inputs=['x', 'indices', 'value'], outputs=['y'])

    def infer_shape(self, x_shape, indices_shape, value_shape):
        validator.check('the dimension of x', len(x_shape),
                        'the dimension of indices', indices_shape[-1], Rel.GE)
        if indices_shape[:-1] + x_shape[indices_shape[-1]:] != value_shape:
            raise ValueError("For 'TensorScatterUpdate', input value are not match with input indices.")
        return x_shape

    def infer_dtype(self, x_dtype, indices_dtype, value_dtype):
        validator.check_tensor_type_same({'indices': indices_dtype}, [mstype.int32], self.name)
        args = {"x": x_dtype, "value": value_dtype}
        validator.check_tensor_type_same(args, (mstype.bool_,) + mstype.number_type, self.name)
        return x_dtype


class ScatterUpdate(PrimitiveWithInfer):
    """
    Update tensor value by using input indices and value.

    Using given values to update tensor value, along with the input indices.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: True.

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
        - **indices** (Tensor) - The index of input tensor. With int32 data type.
        - **update** (Tensor) - The tensor to update the input tensor, has the same type as input,
          and update.shape = indices.shape + input_x.shape[1:].

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Examples:
        >>> np_x = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]])
        >>> input_x = mindspore.Parameter(Tensor(np_x, mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> np_update = np.array([[[1.0, 2.2, 1.0], [2.0, 1.2, 1.0]], [[2.0, 2.2, 1.0], [3.0, 1.2, 1.0]]])
        >>> update = Tensor(np_update, mindspore.float32)
        >>> op = P.ScatterUpdate()
        >>> output = op(input_x, indices, update)
    """
    __mindspore_signature__ = (
        ('x', sig_rw.RW_WRITE, sig_kind.KIND_POSITIONAL_KEYWORD, sig_kind.KIND_EMPTY_DEFAULT_VALUE, sig_dtype.T),
        ('indices', sig_rw.RW_READ, sig_kind.KIND_POSITIONAL_KEYWORD, sig_kind.KIND_EMPTY_DEFAULT_VALUE, sig_dtype.T1),
        ('value', sig_rw.RW_READ, sig_kind.KIND_POSITIONAL_KEYWORD, sig_kind.KIND_EMPTY_DEFAULT_VALUE, sig_dtype.T)
    )
    @prim_attr_register
    def __init__(self, use_locking=True):
        """Init ScatterUpdate"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'value'], outputs=['y'])

    def infer_shape(self, x_shape, indices_shape, value_shape):
        if indices_shape + x_shape[1:] != value_shape:
            raise ValueError("For 'ScatterUpdate', input value are not match with input indices.")
        return x_shape

    def infer_dtype(self, x_dtype, indices_dtype, value_dtype):
        validator.check_tensor_type_same({'indices': indices_dtype}, [mstype.int32], self.name)
        args = {"x": x_dtype, "value": value_dtype}
        validator.check_tensor_type_same(args, (mstype.bool_,) + mstype.number_type, self.name)
        return x_dtype


class ScatterNdUpdate(PrimitiveWithInfer):
    """
    Update tensor value by using input indices and value.

    Using given values to update tensor value, along with the input indices.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: True.

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
        - **indices** (Tensor) - The index of input tensor, with int32 data type.
        - **update** (Tensor) - The tensor to add to the input tensor, has the same type as input.

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Examples:
        >>> np_x = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]])
        >>> input_x = mindspore.Parameter(Tensor(np_x, mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> update = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> op = P.ScatterNdUpdate()
        >>> output = op(input_x, indices, update)
    """
    __mindspore_signature__ = (
        ('x', sig_rw.RW_WRITE, sig_kind.KIND_POSITIONAL_KEYWORD, sig_kind.KIND_EMPTY_DEFAULT_VALUE, sig_dtype.T),
        ('indices', sig_rw.RW_READ, sig_kind.KIND_POSITIONAL_KEYWORD, sig_kind.KIND_EMPTY_DEFAULT_VALUE, sig_dtype.T1),
        ('value', sig_rw.RW_READ, sig_kind.KIND_POSITIONAL_KEYWORD, sig_kind.KIND_EMPTY_DEFAULT_VALUE, sig_dtype.T)
    )
    @prim_attr_register
    def __init__(self, use_locking=True):
        """Init ScatterNdUpdate"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'value'], outputs=['y'])

    def infer_shape(self, x_shape, indices_shape, value_shape):
        validator.check('the dimension of x', len(x_shape),
                        'the dimension of indices', indices_shape[-1], Rel.GE)
        if indices_shape[:-1] + x_shape[indices_shape[-1]:] != value_shape:
            raise ValueError("For 'ScatterNdUpdate', input value are not match with input indices.")
        return x_shape

    def infer_dtype(self, x_dtype, indices_dtype, value_dtype):
        validator.check_tensor_type_same({'indices': indices_dtype}, [mstype.int32], self.name)
        args = {"x": x_dtype, "value": value_dtype}
        validator.check_tensor_type_same(args, (mstype.bool_,) + mstype.number_type, self.name)
        return x_dtype

def _check_scatter_shape(x_shape, indices_shape, updates_shape, prim_name):
    if updates_shape and updates_shape != indices_shape + x_shape[1:]:
        raise ValueError(f"For '{prim_name}', the shape of updates should be [] or "
                         f"updates_shape = indices_shape + x_shape[1:], but got x_shape: {x_shape}, "
                         f"indices_shape: {indices_shape}, updates_shape: {updates_shape}.")


class ScatterMax(PrimitiveWithInfer):
    """
    Update the value of the input tensor through the max operation.

    Using given values to update tensor value through the max operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: True.

    Inputs:
        - **input_x** (Parameter) - The target parameter.
        - **indices** (Tensor) - The index to do max operation whose data type should be mindspore.int32.
        - **updates** (Tensor) - The tensor doing the maximum operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices_shape + x_shape[1:]`.

    Outputs:
        Parameter, the updated `input_x`.

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32), name="input_x")
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> update = Tensor(np.ones([2, 2, 3]) * 88, mindspore.float32)
        >>> scatter_max = P.ScatterMax()
        >>> output = scatter_max(input_x, indices, update)
        [[88.0, 88.0, 88.0], [88.0, 88.0, 88.0]]
    """

    @prim_attr_register
    def __init__(self, use_locking=True):
        """Init ScatterMax"""
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])
        validator.check_value_type('use_locking', use_locking, (bool,), self.name)

    def infer_shape(self, x_shape, indices_shape, updates_shape):
        _check_scatter_shape(x_shape, indices_shape, updates_shape, self.name)
        return x_shape

    def infer_dtype(self, x_dtype, indices_dtype, updates_dtype):
        validator.check_tensor_type_same({'indices': indices_dtype}, (mstype.int32,), self.name)
        args = {"x": x_dtype, "updates": updates_dtype}
        validator.check_tensor_type_same(args, mstype.number_type, self.name)
        return x_dtype


class ScatterAdd(PrimitiveWithInfer):
    """
    Update the value of the input tensor through the add operation.

    Using given values to update tensor value through the add operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: False.

    Inputs:
        - **input_x** (Parameter) - The target parameter.
        - **indices** (Tensor) - The index to do add operation whose data type should be mindspore.int32.
        - **updates** (Tensor) - The tensor doing the add operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices_shape + x_shape[1:]`.

    Outputs:
        Parameter, the updated `input_x`.

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.ones([2, 2, 3]), mindspore.float32)
        >>> scatter_add = P.ScatterAdd()
        >>> output = scatter_add(input_x, indices, updates)
        [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]
    """

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Init ScatterAdd"""
        validator.check_value_type('use_locking', use_locking, (bool,), self.name)

    def infer_shape(self, x_shape, indices_shape, updates_shape):
        _check_scatter_shape(x_shape, indices_shape, updates_shape, self.name)
        return x_shape

    def infer_dtype(self, x_dtype, indices_dtype, updates_dtype):
        validator.check_tensor_type_same({'indices': indices_dtype}, (mstype.int32,), self.name)
        args = {'x': x_dtype, 'updates': updates_dtype}
        validator.check_tensor_type_same(args, mstype.number_type, self.name)
        return x_dtype


class SpaceToDepth(PrimitiveWithInfer):
    r"""
    Rearrange blocks of spatial data into depth.

    The output tensor's `height` dimension is :math:`height / block\_size`.

    The output tensor's `weight` dimension is :math:`weight / block\_size`.

    The depth of output tensor is :math:`block\_size * block\_size * input\_depth`.

    The input tensor's height and width must be divisible by `block_size`.
    The data format is "NCHW".

    Args:
        block_size (int): The block size used to divide spatial data. It must be >= 2.

    Inputs:
        - **x** (Tensor) - The target tensor.

    Outputs:
        Tensor, the same type as `x`.

    Examples:
        >>> x = Tensor(np.random.rand(1,3,2,2), mindspore.float32)
        >>> block_size = 2
        >>> op = P.SpaceToDepth(block_size)
        >>> output = op(x)
        >>> output.asnumpy().shape == (1,12,1,1)
    """

    @prim_attr_register
    def __init__(self, block_size):
        """Init SpaceToDepth"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        validator.check_value_type('block_size', block_size, [int], self.name)
        validator.check('block_size', block_size, '', 2, Rel.GE)
        self.block_size = block_size

    def infer_shape(self, x_shape):
        validator.check('x dimension', len(x_shape), '', 4, Rel.EQ)
        out_shape = copy.deepcopy(x_shape)
        for i in range(2):
            if out_shape[i + 2] % self.block_size != 0:
                raise ValueError(f'For \'{self.name}\' input shape[{i + 2}] {out_shape[i + 2]} should be '
                                 f'fully divided by block_size {self.block_size}')
            out_shape[i + 2] //= self.block_size

        out_shape[1] *= self.block_size * self.block_size
        return out_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("x_dtype", x_dtype, mstype.tensor, self.name)
        return x_dtype


class DepthToSpace(PrimitiveWithInfer):
    r"""
    Rearrange blocks of depth data into spatial dimensions.

    This is the reverse operation of SpaceToDepth.

    The output tensor's `height` dimension is :math:`height * block\_size`.

    The output tensor's `weight` dimension is :math:`weight * block\_size`.

    The depth of output tensor is :math:`input\_depth / (block\_size * block\_size)`.

    The input tensor's depth must be divisible by `block_size * block_size`.
    The data format is "NCHW".

    Args:
        block_size (int): The block size used to divide depth data. It must be >= 2.

    Inputs:
        - **x** (Tensor) - The target tensor.

    Outputs:
        Tensor, the same type as `x`.

    Examples:
        >>> x = Tensor(np.random.rand(1,12,1,1), mindspore.float32)
        >>> block_size = 2
        >>> op = P.DepthToSpace(block_size)
        >>> output = op(x)
        >>> output.asnumpy().shape == (1,3,2,2)
    """

    @prim_attr_register
    def __init__(self, block_size):
        """Init DepthToSpace"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        validator.check_value_type('block_size', block_size, [int], self.name)
        validator.check('block_size', block_size, '', 2, Rel.GE, self.name)
        self.block_size = block_size

    def infer_shape(self, x_shape):
        validator.check('x dimension', len(x_shape), '', 4, Rel.EQ)
        out_shape = copy.deepcopy(x_shape)
        for i in range(2):
            out_shape[i + 2] *= self.block_size

        validator.check_integer('x_shape[1] % (block_size*block_size)',
                                x_shape[1] % (self.block_size * self.block_size),
                                0, Rel.EQ, self.name)
        out_shape[1] //= self.block_size * self.block_size
        return out_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("x_dtype", x_dtype, mstype.tensor, self.name)
        return x_dtype


class SpaceToBatch(PrimitiveWithInfer):
    r"""
    Divide spatial dimensions into blocks and combine the block size with the original batch.

    This operation will divide spatial dimensions (H, W) into blocks with block_size, the output tensor's H and W
    dimension is the corresponding number of blocks after division. The output tensor's batch dimension is the
    product of the original batch and the square of block_size. Prior to division into blocks, the spatial dimensions
    of the input are zero padded according to paddings if necessary.

    Args:
        block_size (int): The block size of dividing block with value >= 2.
        paddings (list): The padding value for H and W dimension, containing 2 sub list, each containing 2 int value.
            All values must be >= 0. paddings[i] specifies the paddings for spatial dimension i, which corresponds to
            input dimension i+2. It is required that input_shape[i+2]+paddings[i][0]+paddings[i][1] is divisible
            by block_size.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, the output tensor with the same type as input. Assume input shape is :math:`(n, c, h, w)` with
        :math:`block\_size` and :math:`padddings`. The output tensor shape will be :math:`(n', c', h', w')`, where

            :math:`n' = n*(block\_size*block\_size)`

            :math:`c' = c`

            :math:`h' = (h+paddings[0][0]+paddings[0][1])//block\_size`

            :math:`w' = (w+paddings[1][0]+paddings[1][1])//block\_size`

    Examples:
        >>> block_size = 2
        >>> paddings = [[0, 0], [0, 0]]
        >>> space_to_batch = P.SpaceToBatch(block_size, paddings)
        >>> input_x = Tensor(np.array([[[[1, 2], [3, 4]]]]), mindspore.float32)
        >>> space_to_batch(input_x)
        [[[[1.]]], [[[2.]]], [[[3.]]], [[[4.]]]]

    """

    @prim_attr_register
    def __init__(self, block_size, paddings):
        """Init SpaceToBatch"""
        validator.check_value_type('block_size', block_size, [int], self.name)
        validator.check('block_size', block_size, '', 2, Rel.GE, self.name)
        self.block_size = block_size
        validator.check('paddings shape', np.array(paddings).shape, '', (2, 2), Rel.EQ, self.name)
        for elem in itertools.chain(*paddings):
            validator.check_integer('paddings element', elem, 0, Rel.GE, self.name)
            validator.check_value_type('paddings element', elem, [int], self.name)
        self.paddings = paddings

    def infer_dtype(self, x_dtype):
        validator.check_tensor_type_same({'input_x': x_dtype}, mstype.number_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape):
        validator.check_integer('rank of input_x', len(x_shape), 4, Rel.EQ, self.name)
        out_shape = copy.deepcopy(x_shape)
        for i in range(2):
            padded = out_shape[i + 2] + self.paddings[i][0] + \
                     self.paddings[i][1]
            if padded % self.block_size != 0:
                raise ValueError(f'For \'{self.name}\' padded[{i}] {padded} should be divisible by '
                                 f'block_size {self.block_size}')
            out_shape[i + 2] = padded // self.block_size
        out_shape[0] *= self.block_size * self.block_size
        return out_shape


class BatchToSpace(PrimitiveWithInfer):
    r"""
    Divide batch dimension with blocks and interleaves these blocks back into spatial dimensions.

    This operation will divide batch dimension N into blocks with block_size, the output tensor's N dimension
    is the corresponding number of blocks after division. The output tensor's H, W dimension is product of original H, W
    dimension and block_size with given amount to crop from dimension, respectively.

    Args:
        block_size (int): The block size of dividing block with value >= 1.
        crops (list): The crop value for H and W dimension, containing 2 sub list, each containing 2 int value.
            All values must be >= 0. crops[i] specifies the crop values for spatial dimension i, which corresponds to
            input dimension i+2. It is required that input_shape[i+2]*block_size >= crops[i][0]+crops[i][1].

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, the output tensor with the same type as input. Assume input shape is (n, c, h, w) with block_size
        and crops. The output shape will be (n', c', h', w'), where

                :math:`n' = n//(block\_size*block\_size)`

                :math:`c' = c`

                :math:`h' = h*block\_size-crops[0][0]-crops[0][1]`

                :math:`w' = w*block\_size-crops[1][0]-crops[1][1]`

    Examples:
        >>> block_size = 2
        >>> crops = [[0, 0], [0, 0]]
        >>> op = P.BatchToSpace(block_size, crops)
        >>> input_x = Tensor(np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]]), mindspore.float32)
        >>> output = op(input_x)
        [[[[1., 2.], [3., 4.]]]]

    """

    @prim_attr_register
    def __init__(self, block_size, crops):
        """Init BatchToSpace"""
        validator.check_value_type('block_size', block_size, [int], self.name)
        validator.check('block_size', block_size, '', 1, Rel.GE, self.name)
        self.block_size = block_size
        validator.check('crops shape', np.array(crops).shape, '', (2, 2))
        for elem in itertools.chain(*crops):
            validator.check_integer('crops element', elem, 0, Rel.GE, self.name)
            validator.check_value_type('crops element', elem, [int], self.name)
        self.crops = crops

    def infer_dtype(self, x_dtype):
        validator.check_tensor_type_same({'input_x': x_dtype}, mstype.number_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape):
        validator.check('rank of input_x', len(x_shape), '', 4)
        out_shape = copy.deepcopy(x_shape)
        for i in range(2):
            x_block_prod = out_shape[i + 2] * self.block_size
            crops_sum = self.crops[i][0] + self.crops[i][1]
            validator.check("x block shape prod", x_block_prod, 'crops sum', crops_sum, Rel.GT, self.name)
            out_shape[i + 2] = x_block_prod - crops_sum
        block_size_prod = self.block_size * self.block_size
        if out_shape[0] % block_size_prod != 0:
            raise ValueError(f'For \'{self.name}\' input_x dimension 0 {out_shape[0]}  should be divisible by '
                             f'block_size_prod {block_size_prod}')
        out_shape[0] = out_shape[0] // block_size_prod
        return out_shape


class SpaceToBatchND(PrimitiveWithInfer):
    r"""
    Divide spatial dimensions into blocks and combine the block size with the original batch.

    This operation will divide spatial dimensions (H, W) into blocks with block_shape, the output tensor's H and W
    dimension is the corresponding number of blocks after division. The output tensor's batch dimension is the
    product of the original batch and the product of block_shape. Prior to division into blocks, the spatial dimensions
    of the input are zero padded according to paddings if necessary.

    Args:
        block_shape (Union[list(int), tuple(int)]): The block shape of dividing block with all value >= 1.
            The length of block_shape is M correspoding to the number of spatial dimensions.
        paddings (list): The padding value for H and W dimension, containing M sub list, each containing 2 int value.
            All values must be >= 0. paddings[i] specifies the paddings for spatial dimension i, which corresponds to
            input dimension i+2. It is required that input_shape[i+2]+paddings[i][0]+paddings[i][1] is divisible
            by block_shape[i].

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, the output tensor with the same type as input. Assume input shape is :math:`(n, c, h, w)` with
        :math:`block\_shape` and :math:`padddings`. The output tensor shape will be :math:`(n', c', h', w')`, where

            :math:`n' = n*(block\_shape[0]*block\_shape[1])`

            :math:`c' = c`

            :math:`h' = (h+paddings[0][0]+paddings[0][1])//block\_shape[0]`

            :math:`w' = (w+paddings[1][0]+paddings[1][1])//block\_shape[1]`

    Examples:
        >>> block_shape = [2, 2]
        >>> paddings = [[0, 0], [0, 0]]
        >>> space_to_batch_nd = P.SpaceToBatchND(block_shape, paddings)
        >>> input_x = Tensor(np.array([[[[1, 2], [3, 4]]]]), mindspore.float32)
        >>> space_to_batch_nd(input_x)
        [[[[1.]]], [[[2.]]], [[[3.]]], [[[4.]]]]

    """

    @prim_attr_register
    def __init__(self, block_shape, paddings):
        """Init SpaceToBatchND"""
        validator.check_value_type('block_shape type', block_shape, [list, tuple], self.name)
        validator.check('block_shape shape', len(np.array(block_shape).shape), '', 1, Rel.EQ, self.name)
        block_rank = len(block_shape)

        for elem in block_shape:
            validator.check('block_shape element', elem, '', 1, Rel.GE, self.name)
        self.block_shape = block_shape

        validator.check('paddings shape', np.array(paddings).shape, '', (block_rank, 2), Rel.EQ, self.name)
        for elem in itertools.chain(*paddings):
            validator.check_integer('paddings element', elem, 0, Rel.GE, self.name)
            validator.check_value_type('paddings element', elem, [int], self.name)
        self.paddings = paddings

    def infer_dtype(self, x_dtype):
        validator.check_tensor_type_same({'input_x': x_dtype}, mstype.number_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape):
        x_rank = len(x_shape)
        out_shape = copy.deepcopy(x_shape)

        block_shape_prod = 1
        for i in range(x_rank - 2):
            padded = out_shape[i + 2] + self.paddings[i][0] + \
                     self.paddings[i][1]
            if padded % self.block_shape[i] != 0:
                raise ValueError(f'For \'{self.name}\' padded[{i}] {padded} should be divisible by '
                                 f'block_shape[{i}] {self.block_shape[i]}')
            out_shape[i + 2] = padded // self.block_shape[i]
            block_shape_prod = block_shape_prod * self.block_shape[i]
        out_shape[0] *= block_shape_prod
        return out_shape


class BatchToSpaceND(PrimitiveWithInfer):
    r"""
    Divide batch dimension with blocks and interleaves these blocks back into spatial dimensions.

    This operation will divide batch dimension N into blocks with block_shape, the output tensor's N dimension
    is the corresponding number of blocks after division. The output tensor's H, W dimension is product of original H, W
    dimension and block_shape with given amount to crop from dimension, respectively.

    Args:
        block_shape (Union[list(int), tuple(int)]): The block shape of dividing block with all value >= 1.
            The length of block_shape is M correspoding to the number of spatial dimensions.
        crops (list): The crop value for H and W dimension, containing 2 sub list, each containing 2 int value.
            All values must be >= 0. crops[i] specifies the crop values for spatial dimension i, which corresponds to
            input dimension i+2. It is required that input_shape[i+2]*block_size[i] >= crops[i][0]+crops[i][1].

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, the output tensor with the same type as input. Assume input shape is (n, c, h, w) with block_shape
        and crops. The output shape will be (n', c', h', w'), where

                :math:`n' = n//(block\_shape[0]*block\_shape[1])`

                :math:`c' = c`

                :math:`h' = h*block\_shape[0]-crops[0][0]-crops[0][1]`

                :math:`w' = w*block\_shape[1]-crops[1][0]-crops[1][1]`

    Examples:
        >>> block_shape = [2, 2]
        >>> crops = [[0, 0], [0, 0]]
        >>> batch_to_space_nd = P.BatchToSpaceND(block_shape, crops)
        >>> input_x = Tensor(np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]]), mindspore.float32)
        >>> output = batch_to_space_nd(input_x)
        [[[[1., 2.], [3., 4.]]]]

    """

    @prim_attr_register
    def __init__(self, block_shape, crops):
        """Init BatchToSpaceND"""
        validator.check_value_type('block_shape type', block_shape, [list, tuple], self.name)
        validator.check('block_shape shape', len(np.array(block_shape).shape), '', 1, Rel.EQ, self.name)
        block_rank = len(block_shape)

        for elem in block_shape:
            validator.check('block_shape element', elem, '', 1, Rel.GE, self.name)
        self.block_shape = block_shape

        validator.check('crops shape', np.array(crops).shape, '', (block_rank, 2), Rel.EQ, self.name)
        for elem in itertools.chain(*crops):
            validator.check_integer('crops element', elem, 0, Rel.GE, self.name)
            validator.check_value_type('crops element', elem, [int], self.name)
        self.crops = crops

    def infer_dtype(self, x_dtype):
        validator.check_tensor_type_same({'input_x': x_dtype}, mstype.number_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape):
        x_rank = len(x_shape)
        out_shape = copy.deepcopy(x_shape)

        block_shape_prod = 1
        for i in range(x_rank - 2):
            block_shape_prod = block_shape_prod * self.block_shape[i]
            x_block_prod = out_shape[i + 2] * self.block_shape[i]
            crops_sum = self.crops[i][0] + self.crops[i][1]
            validator.check("x block shape prod", x_block_prod, 'crops sum', crops_sum, Rel.GT, self.name)
            out_shape[i + 2] = x_block_prod - crops_sum

        if out_shape[0] % block_shape_prod != 0:
            raise ValueError(f'For \'{self.name}\' input_x dimension 0 {out_shape[0]}  should be divisible by '
                             f'block_shape_prod {block_shape_prod}')
        out_shape[0] = out_shape[0] // block_shape_prod
        return out_shape


class BroadcastTo(PrimitiveWithInfer):
    """
    Broadcasts input tensor to a given shape.

    Args:
        shape (tuple): The target shape to broadcast.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, with the given `shape` and the same data type as `input_x`.

    Examples:
        >>> shape = (2, 3)
        >>> input_x = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> broadcast_to = P.BroadcastTo(shape)
        >>> broadcast_to(input_x)
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    """

    @prim_attr_register
    def __init__(self, shape):
        """Init BroadcastTo"""
        validator.check_value_type("shape", shape, (tuple), self.name)
        for i in shape:
            validator.check_integer("shape element", i, 0, Rel.GT, self.name)
        self.shape = shape

    def infer_shape(self, x_shape):
        return self.shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("input_x", x_dtype, mstype.tensor, self.name)
        return x_dtype


class InplaceUpdate(PrimitiveWithInfer):
    r"""
    Updates specified rows with values in `v`.

    Args:
        indices (Union[int, tuple]): Indices into the left-most dimension of `x`.

    Inputs:
        - **x** (Tensor) - A tensor which to be inplace updated. It can be of the following data types:
          float32, float16, int32.
        - **v** (Tensor) - A tensor of the same type as `x`. Same dimension size as `x` except
          the first dimension, which must be the same as the size of `indices`.

    Outputs:
        Tensor, with the same type and shape as the input `x`.

    Examples:
        >>> x = Tensor(np.arange(24).reshape(3, 4, 2), mindspore.float32)
        >>> v = Tensor(np.arange(-8, 8).reshape(2, 4, 2), mindspore.float32)
        >>> inplace_update = P.InplaceUpdate((0, 2))
        >>> result = inplace_update(x, v)
        [[[-8.  -7.]
          [-6.  -5.]
          [-4.  -3.]
          [-2.  -1.]]
         [[ 8.   9.]
          [10.  11.]
          [12.  13.]
          [14.  15.]]
         [[ 0.   1.]
          [ 2.   3.]
          [ 4.   5.]
          [ 6.   7.]]]
    """
    @prim_attr_register
    def __init__(self, indices):
        """Init InplaceUpdate"""
        self.init_prim_io_names(inputs=['x', 'indices', 'v'], outputs=['y'])
        validator.check_value_type("indices", indices, [int, tuple], self.name)
        if isinstance(indices, int):
            self.add_prim_attr('indices', (indices,))
        for item in self.indices:
            validator.check_value_type("item of indices", item, [int], self.name)

    def infer_dtype(self, x_dtype, v_dtype):
        valid_type = [mstype.int32, mstype.float16, mstype.float32]
        validator.check_tensor_type_same(
            {
                "x": x_dtype,
                "v": v_dtype
            }, valid_type, self.name)

        return x_dtype

    def infer_shape(self, x_shape, v_shape):
        validator.check("x", len(x_shape), "v", len(v_shape), Rel.EQ, self.name)

        x_rank = len(x_shape)
        for idx in range(x_rank)[1:]:
            validator.check("x dim %d" % idx, x_shape[idx], 'v dim %d' % idx, v_shape[idx], Rel.EQ, self.name)

        validator.check("size of indices", len(self.indices), "v's first dimension", v_shape[0],
                        Rel.EQ, self.name)

        return x_shape
