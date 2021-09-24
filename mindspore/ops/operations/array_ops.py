# coding: utf-8

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

"""Operators for array."""
from collections import Counter
import copy
import functools
import itertools
import numbers
import numpy as np

from mindspore import log as logger
from mindspore.common.initializer import Zero
from .._utils import get_broadcast_shape
from .._utils import get_concat_offset
from ..operations.math_ops import _infer_shape_reduce
from ..primitive import Primitive, PrimitiveWithInfer, PrimitiveWithCheck, prim_attr_register, _run_op
from .. import signature as sig
from ..._checkparam import Rel
from ..._checkparam import Validator as validator
from ...common import dtype as mstype
from ...common._decorator import deprecated
from ...common.parameter import Parameter
from ...common.tensor import Tensor


class _ScatterOp(PrimitiveWithInfer):
    """
    Defines Scatter operators
    """
    __mindspore_signature__ = (
        sig.make_sig('x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T)
    )

    def _check_scatter_shape(self, x_shape, indices_shape, updates_shape, prim_name):
        if indices_shape != [-1] and updates_shape and updates_shape != indices_shape + x_shape[1:]:
            raise ValueError(f"For '{prim_name}', "
                             f"updates_shape = indices_shape + x_shape[1:], but got x_shape: {x_shape}, "
                             f"indices_shape: {indices_shape}, updates_shape: {updates_shape}.")

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize _ScatterOp"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, x_shape, indices_shape, updates_shape):
        self._check_scatter_shape(x_shape, indices_shape, updates_shape, self.name)
        return x_shape

    def infer_dtype(self, x_dtype, indices_dtype, updates_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32], self.name)
        args = {"x": x_dtype, "updates": updates_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        return x_dtype


class _ScatterOpDynamic(PrimitiveWithCheck):
    """
    Defines Scatter operators with dynamic shape
    """
    __mindspore_signature__ = (
        sig.make_sig('x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T)
    )

    def _check_scatter_shape(self, x_shape, indices_shape, updates_shape, prim_name):
        # x_shape cannot be dynamic
        if np.any(np.array(x_shape) == -1):
            raise ValueError(f"For '{prim_name}', the 'input_x' does not support dynamic shape, "
                             f"but got the shape of 'input_x' is {x_shape}.")
        # support indices and updates dynamic
        if np.any(np.array(indices_shape) == -1) or np.any(np.array(updates_shape) == -1):
            pass
        elif indices_shape != [-1] and updates_shape and updates_shape != indices_shape + x_shape[1:]:
            raise ValueError(f"For '{prim_name}', "
                             f"updates_shape = indices_shape + x_shape[1:], but got x_shape: {x_shape}, "
                             f"indices_shape: {indices_shape}, updates_shape: {updates_shape}.")

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize _ScatterOpDynamic"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)

    def check_shape(self, x_shape, indices_shape, updates_shape):
        self._check_scatter_shape(x_shape, indices_shape, updates_shape, self.name)

    def check_dtype(self, x_dtype, indices_dtype, updates_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32], self.name)
        args = {"x": x_dtype, "updates": updates_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)


class _ScatterNdOp(_ScatterOp):
    """
    Defines _ScatterNd operators
    """

    def _check_scatter_shape(self, x_shape, indices_shape, updates_shape, prim_name):
        validator.check('the dimension of x', len(x_shape),
                        'the dimension of indices', indices_shape[-1], Rel.GE)
        if indices_shape[:-1] + x_shape[indices_shape[-1]:] != updates_shape:
            raise ValueError(f"For '{prim_name}', updates_shape = "
                             f"indices_shape[:-1] + x_shape[indices_shape[-1]:], but got x_shape: {x_shape}, "
                             f"indices_shape: {indices_shape}, updates_shape: {updates_shape}.")


def _check_infer_attr_reduce(axis, keep_dims, prim_name):
    validator.check_value_type('keep_dims', keep_dims, [bool], prim_name)
    validator.check_value_type('axis', axis, [int, tuple], prim_name)
    if isinstance(axis, tuple):
        for index, value in enumerate(axis):
            validator.check_value_type('axis[%d]' % index, value, [int], prim_name)


class ExpandDims(PrimitiveWithInfer):
    """
    Adds an additional dimension to 'input_x` at the given axis.

    Note:
        If the specified axis is a negative number, the index is counted
        backward from the end and starts at 1.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        - **axis** (int) - Specifies the dimension index at which to expand
          the shape of `input_x`. The value of axis must be in the range
          `[-input_x.ndim-1, input_x.ndim]`. Only constant value is allowed.

    Outputs:
        Tensor, the shape of tensor is :math:`(1, x_1, x_2, ..., x_R)` if the
        value of `axis` is 0. It has the same data type as `input_x`.

    Raises:
        ValueError: If `axis` is not an int or not in the valid range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> expand_dims = ops.ExpandDims()
        >>> output = expand_dims(input_tensor, 0)
        >>> print(output)
        [[[2. 2.]
          [2. 2.]]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ExpandDims"""
        self.init_prim_io_names(inputs=['x', 'axis'], outputs=['output'])

    def __infer__(self, x, axis):
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        x_shape = list(x['shape'])
        axis_v = axis['value']
        rank = len(x_shape)
        validator.check_int_range(axis_v, -rank - 1, rank, Rel.INC_BOTH, 'axis', self.name)
        value = None
        if x['value'] is not None:
            value = x['value'].asnumpy()
            value = np.expand_dims(value, axis_v)
            value = Tensor(value)
        if axis_v < 0:
            axis_v = rank + 1 + axis_v
        x_shape.insert(axis_v, 1)
        out = {'shape': x_shape,
               'dtype': x['dtype'],
               'value': value}
        if 'min_shape' in x and 'max_shape' in x:
            out['min_shape'] = x['min_shape']
            out['min_shape'].insert(axis_v, 1)
            out['max_shape'] = x['max_shape']
            out['max_shape'].insert(axis_v, 1)
        return out


class DType(Primitive):
    """
    Returns the data type of the input tensor as mindspore.dtype.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        mindspore.dtype, the data type of a tensor.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> output = ops.DType()(input_tensor)
        >>> print(output)
        Float32
    """

    @prim_attr_register
    def __init__(self):
        """Initialize DType"""


class SameTypeShape(PrimitiveWithInfer):
    """
    Checks whether the data type and shape of two tensors are the same.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        - **input_y** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_S)`.

    Outputs:
        Tensor, the shape of tensor is :math:`(x_1, x_2, ..., x_R)`,
        if data type and shape of `input_x` and `input_y` are the same.

    Raises:
        TypeError: If the data types of `input_x` and `input_y` are not the same.
        ValueError: If the shapes of `input_x` and `input_y` are not the same.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> input_y = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> output = ops.SameTypeShape()(input_x, input_y)
        >>> print(output)
        [[2. 2.]
         [2. 2.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Same"""

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
          The tensor to be cast.
        - **type** (dtype.Number) - The valid data type of the output tensor. Only constant value is allowed.

    Outputs:
        Tensor, the shape of tensor is the same as `input_x`, :math:`(x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If `input_x` is neither Tensor nor Number.
        TypeError: If `type` is not a Number.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
        >>> input_x = Tensor(input_np)
        >>> type_dst = mindspore.int32
        >>> cast = ops.Cast()
        >>> output = cast(input_x, type_dst)
        >>> print(output.dtype)
        Int32
        >>> print(output.shape)
        (2, 3, 4, 5)
    """

    @prim_attr_register
    def __init__(self):
        # if primitive need setattr in __infer__ need add this flag
        """Initialize Cast"""
        self.init_prim_io_names(inputs=['x', 'dst_type'], outputs=['output'])

    def check_elim(self, x, dtype):
        if isinstance(x, (Tensor, numbers.Number, Parameter)):
            if isinstance(x, Parameter):
                data = x.data
                if data.dtype == dtype:
                    return (True, x)
            if isinstance(x, Tensor) and x.dtype == dtype:
                x = Tensor(x)
                x.set_cast_dtype()
                return (True, x)
            if isinstance(x, numbers.Number):
                return (True, Tensor(x, dtype=dtype))
        return (False, None)

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
        self.add_prim_attr('dst_type', dst_type)

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
        if 'min_shape' in x and 'max_shape' in x:
            out['min_shape'] = x['min_shape']
            out['max_shape'] = x['max_shape']
        return out


class IsSubClass(PrimitiveWithInfer):
    """
    Checks whether this type is a sub-class of another type.

    Inputs:
        - **sub_type** (mindspore.dtype) - The type to be checked. Only constant value is allowed.
        - **type_** (mindspore.dtype) - The target type. Only constant value is allowed.

    Outputs:
        bool, the check result.

    Raises:
        TypeError: If `sub_type` or `type_` is not a Type.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = ops.IsSubClass()(mindspore.int32,  mindspore.intc)
        >>> print(output)
        True
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
    Checks whether an object is an instance of a target type.

    Inputs:
        - **inst** (Any Object) - The instance to be checked. Only constant value is allowed.
        - **type_** (mindspore.dtype) - The target type. Only constant value is allowed.

    Outputs:
        bool, the check result.

    Raises:
        TypeError: If `type_` is not a Type.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> inst = 1
        >>> output = ops.IsInstance()(inst, mindspore.int32)
        >>> print(output)
        False
    """

    @prim_attr_register
    def __init__(self):
        pass

    def __infer__(self, inst, type_):
        sub_type_t = inst['dtype']
        type_v = type_['value']

        validator.check_value_type("type_", type_v, [mstype.Type], self.name)

        if type_v == mstype.list_:
            value = isinstance(sub_type_t, list)
        elif type_v == mstype.tuple_:
            value = isinstance(sub_type_t, tuple)
        else:
            value = mstype.issubclass_(sub_type_t, type_v)

        out = {'shape': (),
               'dtype': mstype.type_type,
               'value': value}
        return out


class Reshape(PrimitiveWithInfer):
    """
    Reshapes the input tensor with the same values based on a given shape tuple.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        - **input_shape** (tuple[int]) - The input tuple is constructed by multiple
          integers, i.e., :math:`(y_1, y_2, ..., y_S)`. Only constant value is allowed.

    Outputs:
        Tensor, the shape of tensor is :math:`(y_1, y_2, ..., y_S)`.

    Raises:
        ValueError: Given a shape tuple, if it has several -1; or if the product
            of its elements is less than or equal to 0 or cannot be divided by the product
            of the input tensor shape; or if it does not match the input's array size.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> reshape = ops.Reshape()
        >>> output = reshape(input_x, (3, 2))
        >>> print(output)
        [[-0.1  0.3]
         [ 3.6  0.4]
         [ 0.5 -3.2]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Reshape"""
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
                    raise ValueError(f"For '{self.name}', the 'input_shape' can only has one -1 at most, "
                                     f"but got {shape_v}.")
                neg_index = i
            else:
                dim_prod *= shp_i
        arr_prod = np.prod(x_shp)
        if arr_prod <= 0:
            if 'max_shape' in x:
                x_max_shape = x['max_shape']
            else:
                x_max_shape = x['shape']
            if 'min_shape' in x:
                x_min_shape = x['min_shape']
            else:
                x_min_shape = x['shape']
            max_arr_prod = np.prod(x_max_shape)
            min_arr_prod = np.prod(x_min_shape)
            max_shape = list(shape_v)
            min_shape = list(shape_v)
            if neg_index != -1:
                max_shape[neg_index] = int(max_arr_prod / dim_prod)
                min_shape[neg_index] = int(min_arr_prod / dim_prod)
            else:
                raise ValueError(f"For '{self.name}', the 'input_shape' must have -1 in the case of dynamic shape, "
                                 f"but got {shape_v}.")
            out = {'shape': shape['value'],
                   'dtype': x['dtype'],
                   'value': None,
                   'max_shape': tuple(max_shape),
                   'min_shape': tuple(min_shape)}
        else:
            if dim_prod <= 0:
                raise ValueError(f"For '{self.name}', the shape of 'input_x' is {x_shp}, "
                                 f"the value of 'input_shape' is {shape_v}. "
                                 f"The product of 'input_shape' should > 0, but got {dim_prod}.")
            if neg_index != -1:
                shape_v[neg_index] = int(arr_prod / dim_prod)
                dim_prod *= shape_v[neg_index]
            if dim_prod != arr_prod:
                raise ValueError(f"For '{self.name}', the shape of 'input_x' is {x_shp}, "
                                 f"the value of 'input_shape' value is {shape_v}. "
                                 f"The product of the shape of 'input_x' should be equal to product of 'input_shape', "
                                 f"but product of the shape of 'input_x' is {arr_prod}, "
                                 f"product of 'input_shape' is {dim_prod}.")
            value = None
            if x['value'] is not None:
                value = Tensor(x['value'].asnumpy().reshape(shape_v))

            out = {'shape': tuple(shape_v),
                   'dtype': x['dtype'],
                   'value': value}
        return out


class Shape(Primitive):
    """
    Returns the shape of the input tensor. And it used to be static shape.

    static shape: A shape that can be obtained without running the graph. It is an inherent property of tensor and
    may be unknown. The static shape information can be completed by artificial setting.
    No matter what the input of the graph is, the static shape is not affected.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        tuple[int], the output tuple is constructed by multiple integers,
        :math:`(x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> shape = ops.Shape()
        >>> output = shape(input_x)
        >>> print(output)
        (3, 2, 1)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Shape"""


class DynamicShape(Primitive):
    """
    Returns the shape of the input tensor. And it used to be dynamic shape.

    Note:
        Dynamic shape: After the graph is running, as the tensor flows in the graph, the specific shape of the tensor
        on each node on the graph can be inferred according to the structure of the graph.
        This shape is called a dynamic shape. As the input shape of the graph is different,
        the dynamic shape of the tensor in the graph will change.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor[int], 1-dim Tensor of type int32

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> shape = ops.DynamicShape()
        >>> output = shape(input_x)
        >>> print(output)
        [3 2 1]
    """

    @prim_attr_register
    def __init__(self):
        """init Shape"""
        self.init_prim_io_names(inputs=['tensor'], outputs=['output'])
        self.add_prim_attr('is_dynamic_shape', True)


class Squeeze(PrimitiveWithInfer):
    """
    Returns a tensor with the same data type but dimensions of 1 are removed based on `axis`.

    If `axis` is specified, it will remove the dimensions of size 1 in the given `axis`.
    It `axis` is None, it will remove all the dimensions of size 1.

    Note:
        The dimension index starts at 0 and must be in the range `[-input.ndim, input.ndim)`.

    Args:
        axis (Union[int, tuple(int)]): Specifies the dimension indexes of shape to be removed, which will remove
            all the dimensions that are equal to 1. If specified, it must be int32 or int64.
            Default: (), an empty tuple.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, the shape of tensor is :math:`(x_1, x_2, ..., x_S)`.

    Raises:
        TypeError: If `axis` is neither an int nor tuple.
        TypeError: If `axis` is a tuple whose elements are not all int.
        ValueError: If the corresponding dimension of the specified axis does not equal to 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> squeeze = ops.Squeeze(2)
        >>> output = squeeze(input_x)
        >>> print(output)
        [[1. 1.]
         [1. 1.]
         [1. 1.]]
    """

    @prim_attr_register
    def __init__(self, axis=()):
        """Initialize Squeeze"""
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
                validator.check_int_range(a, -ndim, ndim - 1, Rel.INC_BOTH, 'axis or its elements', self.name)
                if x_shape[a] != 1:
                    raise ValueError(f"For '{self.name}', the shape of 'input_x' at {a} dimension should be 1, "
                                     f"but got {x_shape[a]}.")
            ret = [x_shape[i] for i in range(ndim) if not (i in axis or (i - ndim) in axis)]
        return ret

    def infer_dtype(self, x_dtype):
        validator.check_subclass("x", x_dtype, mstype.tensor, self.name)
        return x_dtype


class Transpose(Primitive):
    """
    Permutes the dimensions of the input tensor according to input permutation.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        - **input_perm** (tuple[int]) - The permutation to be converted. The elements in `input_perm` are composed of
          the indexes of each dimension of `input_x`. The length of `input_perm` and the shape of `input_x` must be
          the same. Only constant value is allowed. Must be in the range [0, rank(input_x)).

    Outputs:
        Tensor, the type of output tensor is the same as `input_x` and the shape of output tensor is decided by the
        shape of `input_x` and the value of `input_perm`.

    Raises:
        TypeError: If `input_perm` is not a tuple.
        ValueError: If length of shape of `input_x` is not equal to length of shape of `input_perm`.
        ValueError: If the same element exists in `input_perm`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
        >>> input_perm = (0, 2, 1)
        >>> transpose = ops.Transpose()
        >>> output = transpose(input_x, input_perm)
        >>> print(output)
        [[[ 1.  4.]
          [ 2.  5.]
          [ 3.  6.]]
         [[ 7. 10.]
          [ 8. 11.]
          [ 9. 12.]]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Transpose"""
        self.init_prim_io_names(inputs=['x', 'perm'], outputs=['output'])


class Unique(Primitive):
    """
    Returns the unique elements of input tensor and also return a tensor containing the index of each value of input
    tensor corresponding to the output unique tensor.

    The output contains Tensor `y` and Tensor `idx`, the format is probably similar to (`y`, `idx`).
    The shape of Tensor `y` and Tensor `idx` is different in most cases, because Tensor `y` will be deduplicated,
    and the shape of Tensor `idx` is consistent with the input.

    To get the same shape between `idx` and `y`, please ref to 'UniqueWithPad' operator.

    Inputs:
        - **input_x** (Tensor) - The input tensor.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tuple, containing Tensor objects `(y, idx), `y` is a tensor with the
        same type as `input_x`, and contains the unique elements in `x`, sorted in
        ascending order. `idx` is a tensor containing indices of elements in
        the input corresponding to the output tensor.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 5, 2]), mindspore.int32)
        >>> output = ops.Unique()(input_x)
        >>> print(output)
        (Tensor(shape=[3], dtype=Int32, value= [1, 2, 5]), Tensor(shape=[4], dtype=Int32, value= [0, 1, 2, 1]))
        >>> y = output[0]
        >>> print(y)
        [1 2 5]
        >>> idx = output[1]
        >>> print(idx)
        [0 1 2 1]
        >>> # As can be seen from the above, y and idx shape
        >>> # note that for GPU, this operator must be wrapped inside a model, and executed in graph mode.
        >>> class UniqueNet(nn.Cell):
        ...     def __init__(self):
        ...         super(UniqueNet, self).__init__()
        ...         self.unique_op = ops.Unique()
        ...
        ...     def construct(self, x):
        ...         output, indices = self.unique_op(x)
        ...         return output, indices
        ...
        >>> input_x = Tensor(np.array([1, 2, 5, 2]), mindspore.int32)
        >>> net = UniqueNet()
        >>> output = net(input_x)
        >>> print(output)
        (Tensor(shape=[3], dtype=Int32, value= [1, 2, 5]), Tensor(shape=[4], dtype=Int32, value= [0, 1, 2, 1]))
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


class Gather(Primitive):
    r"""
    Returns a slice of the input tensor based on the specified indices and axis.

    Slices the input tensor base on the indices at specified axis. See the following example for more clear.

    Inputs:
        - **input_params** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
          The original Tensor.
        - **input_indices** (Tensor) - The shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
          Specifies the indices of elements of the original Tensor. Must be in the range
          `[0, input_param.shape[axis])` which are only validated on CPU. The data type can be int32 or int64.
        - **axis** (int) - Specifies the dimension index to gather indices.

    Outputs:
        Tensor, the shape of tensor is
        :math:`input\_params.shape[:axis] + input\_indices.shape + input\_params.shape[axis + 1:]`.

    Raises:
        TypeError: If `axis` is not an int.
        TypeError: If `input_indices` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_params = Tensor(np.array([[1, 2, 7, 42], [3, 4, 54, 22], [2, 2, 55, 3]]), mindspore.float32)
        >>> input_indices = Tensor(np.array([1, 2]), mindspore.int32)
        >>> axis = 1
        >>> output = ops.Gather()(input_params, input_indices, axis)
        >>> print(output)
        [[ 2.  7.]
         [ 4. 54.]
         [ 2. 55.]]
        >>> axis = 0
        >>> output = ops.Gather()(input_params, input_indices, axis)
        >>> print(output)
        [[3. 4. 54. 22.]
         [2. 2. 55.  3.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Gather"""
        self.init_prim_io_names(inputs=['params', 'indices', 'axis'], outputs=['output'])


class GatherV2(PrimitiveWithCheck):
    """
    Same as operator Gather. GatherV2 will be deprecated in the future.
    Please use Gather instead.
    """

    @deprecated("1.1", "Gather", True)
    @prim_attr_register
    def __init__(self):
        """Initialize GatherV2"""
        self.init_prim_io_names(inputs=['params', 'indices', 'axis'], outputs=['output'])

    def __check__(self, params, indices, axis):
        validator.check_subclass("params", params['dtype'], mstype.tensor, self.name)
        validator.check_tensor_dtype_valid("indices", indices['dtype'], mstype.int_type, self.name)
        validator.check_subclass("axis", axis['dtype'], [mstype.number], self.name)
        axis_v = axis['value']
        validator.check_value_type('axis', axis_v, [int], self.name)
        rank = len(params['shape'])
        validator.check_int_range(axis_v, -rank, rank, Rel.INC_LEFT, "axis", self.name)


class SparseGatherV2(PrimitiveWithCheck):
    """
    Returns a slice of input tensor based on the specified indices and axis.

    Inputs:
        - **input_params** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        - **input_indices** (Tensor) - The shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
          Specifies the indices of elements of the original Tensor, must be in the range
          `[0, input_param.shape[axis])`.
        - **axis** (int) - Specifies the dimension index to gather indices.

    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`.

    Raises:
        TypeError: If `axis` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> input_params = Tensor(np.array([[1, 2, 7, 42], [3, 4, 54, 22], [2, 2, 55, 3]]), mindspore.float32)
        >>> input_indices = Tensor(np.array([1, 2]), mindspore.int32)
        >>> axis = 1
        >>> out = ops.SparseGatherV2()(input_params, input_indices, axis)
        >>> print(out)
        [[2. 7.]
         [4. 54.]
         [2. 55.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseGatherV2"""
        self.init_prim_io_names(inputs=['params', 'indices', 'axis'], outputs=['output'])


    def __check__(self, params, indices, axis):
        validator.check_subclass("params", params['dtype'], mstype.tensor, self.name)
        validator.check_tensor_dtype_valid("indices", indices['dtype'], mstype.int_type, self.name)
        validator.check_subclass("axis", axis['dtype'], [mstype.number], self.name)
        axis_v = axis['value']
        validator.check_value_type('axis', axis_v, [int], self.name)
        rank = len(params['shape'])
        validator.check_int_range(axis_v, -rank, rank, Rel.INC_LEFT, "axis", self.name)



class Padding(PrimitiveWithInfer):
    """
    Extends the last dimension of the input tensor from 1 to pad_dim_size, by filling with 0.

    Args:
        pad_dim_size (int): The value of the last dimension of `x` to be extended, which must be positive. Default: 8.

    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The rank of `x` must be at least 2.
          The last dimension of `x` must be 1. The data type is Number.

    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`.

    Raises:
        TypeError: If `pad_dim_size` is not an int.
        ValueError: If `pad_dim_size` is less than 1.
        ValueError: If last dim of `x` is not equal 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.array([[8], [10]]), mindspore.float32)
        >>> pad_dim_size = 4
        >>> output = ops.Padding(pad_dim_size)(x)
        >>> print(output)
        [[ 8.  0.  0.  0.]
         [10.  0.  0.  0.]]
    """

    @prim_attr_register
    def __init__(self, pad_dim_size=8):
        """Initialize padding"""
        validator.check_value_type("pad_dim_size", pad_dim_size, [int], self.name)
        validator.check_positive_int(pad_dim_size, "pad_dim_size", self.name)
        self.pad_dim_size = pad_dim_size

    def __infer__(self, x):
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        x_shape = list(x['shape'])
        validator.check_int(len(x_shape), 1, Rel.GT, "rank of x", self.name)
        validator.check_int(x_shape[-1], 1, Rel.EQ, "last dim of x", self.name)
        out_shape = x_shape
        out_shape[-1] = self.pad_dim_size
        out = {'shape': out_shape,
               'dtype': x['dtype'],
               'value': None}
        return out


class UniqueWithPad(PrimitiveWithInfer):
    """
    Returns unique elements and relative indexes in 1-D tensor, filled with padding num.

    The basic function is the same as the Unique operator, but the UniqueWithPad operator adds a Pad function.
    The returned tuple(`y`,`idx`) after the input Tensor `x` is processed by the unique operator,
    in which the shapes of `y` and `idx` are mostly not equal. Therefore, in order to solve the above situation,
    the UniqueWithPad operator will fill the `y` Tensor with the `pad_num` specified by the user
    to make it have the same shape as the Tensor `idx`.

    Inputs:
        - **x** (Tensor) - The tensor need to be unique. Must be 1-D vector with types: int32, int64.
        - **pad_num** (int) - Pad num. The data type is an int.

    Outputs:
        tuple(Tensor), tuple of 2 tensors, `y` and `idx`.
        - y (Tensor) - The unique elements filled with pad_num, the shape and data type same as `x`.
        - idx (Tensor) - The index of each value of `x` in the unique output `y`, the shape and data type same as `x`.

    Raises:
        TypeError: If dtype of `x` is neither int32 nor int64.
        ValueError: If length of shape of `x` is not equal to 1.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 1, 5, 5, 4, 4, 3, 3, 2, 2,]), mindspore.int32)
        >>> pad_num = 8
        >>> output = ops.UniqueWithPad()(x, pad_num)
        >>> print(output)
        (Tensor(shape=[10], dtype=Int32, value= [1, 5, 4, 3, 2, 8, 8, 8, 8, 8]),
         Tensor(shape=[10], dtype=Int32, value= [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]))
    """

    @prim_attr_register
    def __init__(self):
        """init UniqueWithPad"""

    def __infer__(self, x, pad_num):
        validator.check_tensor_dtype_valid("x", x['dtype'], [mstype.int32, mstype.int64], self.name)
        validator.check_subclass("pad_num", pad_num['dtype'], [mstype.int32, mstype.int64], self.name)
        x_shape = list(x['shape'])
        validator.check("rank of x", len(x_shape), "expected", 1, Rel.EQ, self.name)
        out_shape = x_shape
        out = {'shape': (out_shape, out_shape),
               'dtype': (x['dtype'], x['dtype']),
               'value': None}
        return out


class Split(PrimitiveWithCheck):
    """
    Splits the input tensor into output_num of tensors along the given axis and output numbers.

    The `input_x` tensor will be split into equally sized sub-tensors.
    This requires that `input_x.shape(axis)` is divisible by `output_num`.

    Args:
        axis (int): Index of the split position. Default: 0.
        output_num (int): The number of output tensors. Must be positive int. Default: 1.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        tuple[Tensor], the shape of each output tensor is the same, which is
        :math:`(y_1, y_2, ..., y_S)`. And the data type is the same with `input_x`.

    Raises:
        TypeError: If `axis` or `output_num` is not an int.
        ValueError: If `axis` is out of the range [-len(`input_x.shape`), len(`input_x.shape`)),
            or if the `output_num` is less than or equal to 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> split = ops.Split(1, 2)
        >>> x = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]), mindspore.int32)
        >>> print(x)
        [[1 1 1 1]
         [2 2 2 2]]
        >>> output = split(x)
        >>> print(output)
        (Tensor(shape=[2, 2], dtype=Int32, value=
        [[1, 1],
         [2, 2]]), Tensor(shape=[2, 2], dtype=Int32, value=
        [[1, 1],
         [2, 2]]))
        >>> split = ops.Split(1, 4)
        >>> output = split(x)
        >>> print(output)
        (Tensor(shape=[2, 1], dtype=Int32, value=
        [[1],
         [2]]), Tensor(shape=[2, 1], dtype=Int32, value=
        [[1],
         [2]]), Tensor(shape=[2, 1], dtype=Int32, value=
        [[1],
         [2]]), Tensor(shape=[2, 1], dtype=Int32, value=
        [[1],
         [2]]))
    """

    @prim_attr_register
    def __init__(self, axis=0, output_num=1):
        """Initialize Split"""
        validator.check_value_type("axis", axis, [int], self.name)
        validator.check_value_type("output_num", output_num, [int], self.name)
        validator.check_positive_int(output_num, "output_num", self.name)
        self.axis = axis
        self.output_num = output_num

    def __check__(self, x):
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        x_shape = list(x['shape'])
        dim = len(x_shape)
        validator.check_int_range(self.axis, -dim, dim, Rel.INC_LEFT, 'axis value', self.name)
        if -1 not in x_shape:
            # only validate when shape fully known
            output_valid_check = x_shape[self.axis] % self.output_num
            if output_valid_check != 0:
                raise ValueError(f"For '{self.name}', the specified axis of 'input_x' should be divided exactly by "
                                 f"'output_num', but got the shape of 'input_x' in 'axis' {self.axis} is "
                                 f"{x_shape[self.axis]}, 'output_num': {self.output_num}.")
        size_splits = [x_shape[self.axis] // self.output_num] * self.output_num
        self.add_prim_attr('size_splits', size_splits)


class Rank(PrimitiveWithInfer):
    """
    Returns the rank of a tensor.

    Returns a 0-D int32 Tensor representing the rank of input; the rank of a tensor
    is the number of indices required to uniquely select each element of the tensor.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The data type is Number.

    Outputs:
        Tensor. 0-D int32 Tensor representing the rank of input, i.e., :math:`R`. The data type is an int.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> rank = ops.Rank()
        >>> output = rank(input_tensor)
        >>> print(output)
        2
        >>> print(type(output))
        <class 'int'>
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Rank"""

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
        seed (int): A integer number used to create random seed. Default: 0.
        dtype (:class:`mindspore.dtype`): Data type. Default: mindspore.float32.

    Inputs:
        - **shape** (tuple[int]) - The shape of the output tensor, is a tuple of positive integer.

    Outputs:
        Tensor, the data type of output tensor is the same as attribute `dtype`.

    Examples:
        >>> shape = (1, 2, 3)
        >>> truncated_normal = ops.TruncatedNormal()
        >>> output = truncated_normal(shape)
    """

    @prim_attr_register
    def __init__(self, seed=0, dtype=mstype.float32):
        """Initialize TruncatedNormal"""
        validator.check_value_type('seed', seed, [int], self.name)
        validator.check_types_same_and_valid({'dtype': dtype}, mstype.number_type, self.name)

    def __infer__(self, shape):
        shape_value = shape['value']
        validator.check_value_type("shape", shape_value, [tuple], self.name)
        for i, value in enumerate(shape_value):
            validator.check_positive_int(value, f'{i}th value of shape', self.name)
        out = {'shape': shape_value,
               'dtype': mstype.tensor_type(self.dtype),
               'value': None}
        return out


class Size(PrimitiveWithInfer):
    r"""
    Returns the size of a tensor.

    Returns an int scalar representing the elements size of input, the total number of elements in the tensor.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The data type is Number.

    Outputs:
        int, a scalar representing the elements size of `input_x`, tensor is the number of elements
        in a tensor, :math:`size=x_1*x_2*...x_R`. The data type is an int.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> size = ops.Size()
        >>> output = size(input_x)
        >>> print(output)
        4
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Size"""

    def __infer__(self, x):
        size = 1
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        shp = x['shape']
        if not shp:
            size = 0
        else:
            size = functools.reduce(lambda x, y: x * y, x['shape'])
        out = {'shape': None,
               'dtype': mstype.int64,
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

    Raises:
        TypeError: If `shape` is not a tuple.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> fill = ops.Fill()
        >>> output = fill(mindspore.float32, (2, 2), 1)
        >>> print(output)
        [[1. 1.]
         [1. 1.]]
        >>> output = fill(mindspore.float32, (3, 3), 0)
        >>> print(output)
        [[0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Fill"""

    def __infer__(self, dtype, dims, x):
        validator.check_value_type("shape", dims['value'], [tuple], self.name)
        validator.check_value_type("value", x['value'], [numbers.Number, bool], self.name)
        for i, item in enumerate(dims['value']):
            validator.check_positive_int(item, f'dims[{i}]', self.name)
        valid_dtypes = [mstype.bool_, mstype.int8, mstype.int16, mstype.int32, mstype.int64,
                        mstype.uint8, mstype.uint16, mstype.uint32, mstype.uint64,
                        mstype.float16, mstype.float32, mstype.float64]
        validator.check_types_same_and_valid({"value": dtype['value']}, valid_dtypes, self.name)
        x_nptype = mstype.dtype_to_nptype(dtype['value'])
        ret = np.full(dims['value'], x['value'], x_nptype)
        out = {
            'value': Tensor(ret),
            'shape': dims['value'],
            'dtype': x['dtype'],
        }
        return out


class Ones(Primitive):
    r"""
    Creates a tensor filled with value ones.

    Creates a tensor with shape described by the first argument and
    fills it with value ones in type of the second argument.

    Inputs:
        - **shape** (Union[tuple[int], int]) - The specified shape of output tensor.
          Only constant positive int is allowed.
        - **type** (mindspore.dtype) - The specified type of output tensor. Only constant value is allowed.

    Outputs:
        Tensor, has the same type and shape as input shape value.

    Raises:
        TypeError: If `shape` is neither tuple nor int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> ones = ops.Ones()
        >>> output = ones((2, 2), mindspore.float32)
        >>> print(output)
        [[1. 1.]
         [1. 1.]]
        >>> output = ones((3, 3), mindspore.float32)
        >>> print(output)
        [[1. 1. 1.]
         [1. 1. 1.]
         [1. 1. 1.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Ones"""


class Zeros(Primitive):
    r"""
    Creates a tensor filled with value zeros.

    Creates a tensor with shape described by the first argument and
    fills it with value zeros in type of the second argument.

    Inputs:
        - **shape** (Union[tuple[int], int]) - The specified shape of output tensor.
          Only constant positive int is allowed.
        - **type** (mindspore.dtype) - The specified type of output tensor. Only constant value is allowed.

    Outputs:
        Tensor, has the same type and shape as input shape value.

    Raises:
        TypeError: If `shape` is neither int nor tuple.
        TypeError: If `shape` is a tuple whose elements are not all int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> zeros = ops.Zeros()
        >>> output = zeros((2, 2), mindspore.float32)
        >>> print(output)
        [[0. 0.]
         [0. 0.]]

    """

    @prim_attr_register
    def __init__(self):
        """Initialize Zeros"""


class OnesLike(Primitive):
    """
    Creates a new tensor. The values of all elements are 1.

    Returns a tensor of ones with the same shape and type as the input.

    Inputs:
        - **input_x** (Tensor) - Input tensor.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, has the same shape and type as `input_x` but filled with ones.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> oneslike = ops.OnesLike()
        >>> input_x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
        >>> output = oneslike(input_x)
        >>> print(output)
        [[1 1]
         [1 1]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize OnesLike"""


class ZerosLike(Primitive):
    """
    Creates a new tensor. All elements value are 0.

    Returns a tensor of zeros with the same shape and data type as the input tensor.

    Inputs:
        - **input_x** (Tensor) - Input tensor. The data type is int32, int64, float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, has the same shape and data type as `input_x` but filled with zeros.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> zeroslike = ops.ZerosLike()
        >>> input_x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
        >>> output = zeroslike(input_x)
        >>> print(output)
        [[0. 0.]
         [0. 0.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ZerosLike"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class TupleToArray(PrimitiveWithInfer):
    """
    Converts a tuple to a tensor.

    If the type of the first number in the tuple is integer, the data type of the output tensor is int.
    Otherwise, the data type of the output tensor is float.

    Inputs:
        - **input_x** (tuple) - A tuple of numbers. These numbers have the same type. Only constant value is allowed.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.

    Outputs:
        Tensor, if the input tuple contains `N` numbers, then the shape of the output tensor is (N,).

    Raises:
        TypeError: If `input_x` is not a tuple.
        ValueError: If length of `input_x` is less than or equal to 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = (1,2,3)
        >>> print(type(input_x))
        <class 'tuple'>
        >>> output = ops.TupleToArray()(input_x)
        >>> print(type(output))
        <class 'mindspore.common.tensor.Tensor'>
        >>> print(output)
        [1 2 3]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize TupleToArray"""

    def infer_value(self, x):
        validator.check_value_type("x", x, [tuple], self.name)
        validator.check("size of x", len(x), '', 0, Rel.GT, self.name)
        dtype = type(x[0])
        for i, item in enumerate(x):
            validator.check_value_type(f"x[{i}]", item, [numbers.Number], self.name)
        if not all(isinstance(item, dtype) for item in x):
            raise TypeError(f"For \'{self.name}\', all elements of 'input_x' must be have same type.")
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
    Converts a scalar to a `Tensor`.

    Inputs:
        - **input_x** (Union[int, float]) - The input is a scalar. Only constant value is allowed.

    Outputs:
        Tensor. 0-D Tensor and the content is the input.

    Raises:
        TypeError: If `input_x` is neither int nor float.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> op = ops.ScalarToArray()
        >>> input_x = 1.0
        >>> print(type(input_x))
        <class 'float'>
        >>> output = op(input_x)
        >>> print(type(output))
        <class 'mindspore.common.tensor.Tensor'>
        >>> print(output)
        1.0
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
    Converts a scalar to a `Tensor`, and converts the data type to the specified type.

    Inputs:
        - **input_x** (Union[int, float]) - The input is a scalar. Only constant value is allowed.
        - **dtype** (mindspore.dtype) - The target data type. Default: mindspore.float32. Only
          constant value is allowed.

    Outputs:
        Tensor. 0-D Tensor and the content is the input.

    Raises:
        TypeError: If `input_x` is neither int nor float.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> op = ops.ScalarToTensor()
        >>> data = 1
        >>> output = op(data, mindspore.float32)
        >>> print(output)
        1.0
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

    This operator is mainly used to calculate the inverse of index permutation.
    It requires a 1-dimensional integer tensor x, which represents the index of a zero-based array,
    and exchanges each value with its index position. In other words, For output tensor y and input tensor x,
    this operation calculates the following values:

    :math:`y[x[i]] = i, \quad i \in [0, 1, \ldots, \text{len}(x)-1]`.

    Note:
        These values must include 0. There must be no duplicate values and the
        values can not be negative.

    Inputs:
        - **input_x** (Union(tuple[int], list[int]) - The input is constructed by multiple
          integers, i.e., :math:`(y_1, y_2, ..., y_S)` representing the indices.
          The values must include 0. There can be no duplicate values or negative values.
          Only constant value is allowed. The maximum value must be equal to length of input_x.

    Outputs:
        tuple[int]. It has the same length as the input.

    Raises:
        TypeError: If `input_x` is neither tuple nor list.
        TypeError: If element of `input_x` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> invert = ops.InvertPermutation()
        >>> input_data = (3, 4, 0, 2, 1)
        >>> output = invert(input_data)
        >>> print(output)
        (2, 4, 3, 0, 1)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize InvertPermutation"""
        self.set_const_prim(True)

    def __infer__(self, x):
        x_shp = x['shape']
        x_value = x['value']
        if x_value is None:
            raise ValueError(f"For '{self.name}', the value of 'input_x' can not be None, but got {x_value}.")
        validator.check_value_type("shape", x_shp, [tuple, list], self.name)
        if mstype.issubclass_(x['dtype'], mstype.tensor):
            raise ValueError(f"For \'{self.name}\', the value of 'input_x' must be non-Tensor, but got {x['dtype']}")
        for shp in x_shp:
            if shp:
                x_rank = len(np.array(x_value, np.int64).shape)
                raise ValueError(f"For \'{self.name}\', the dimension of 'input_x' must be 1, but got {x_rank}.")
        for i, value in enumerate(x_value):
            validator.check_value_type("input[%d]" % i, value, [int], self.name)
        z = [x_value[i] for i in range(len(x_value))]
        z.sort()

        for i in range(1, len(z)):
            if z[i - 1] == z[i]:
                raise ValueError(f"For '{self.name}', the 'input_x' can not contain duplicate values, "
                                 f"but got duplicated {z[i]} in the 'input_x'.")
        validator.check(f'value min', min(x_value), '', 0, Rel.EQ, self.name)
        validator.check(f'value max', max(x_value), '', len(x_value) - 1, Rel.EQ, self.name)

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
    Returns the indices of the maximum value of a tensor across the axis.

    If the shape of input tensor is :math:`(x_1, ..., x_N)`, the shape of the output tensor will be
    :math:`(x_1, ..., x_{axis-1}, x_{axis+1}, ..., x_N)`.

    Args:
        axis (int): Axis where the Argmax operation applies to. Default: -1.
        output_type (:class:`mindspore.dtype`): An optional data type of `mindspore.dtype.int32`.
            Default: `mindspore.dtype.int32`.

    Inputs:
        - **input_x** (Tensor) - Input tensor. :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Support data type list as follows:

          - Ascend: Float16, Float32.
          - GPU: Float16, Float32.
          - CPU: Float16, Float32, Float64.

    Outputs:
        Tensor, indices of the max value of input tensor across the axis.

    Raises:
        TypeError: If `axis` is not an int.
        TypeError: If `output_type` is neither int32 nor int64.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[1, 20, 5], [67, 8, 9], [130, 24, 15]]).astype(np.float32))
        >>> output = ops.Argmax(output_type=mindspore.int32)(input_x)
        >>> print(output)
        [1 0 0]
    """

    @prim_attr_register
    def __init__(self, axis=-1, output_type=mstype.int32):
        """Initialize Argmax"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        validator.check_value_type("axis", axis, [int], self.name)
        validator.check_types_same_and_valid({'output': output_type}, [mstype.int32], self.name)
        self.axis = axis
        self.add_prim_attr('output_type', output_type)

    def infer_shape(self, x_shape):
        axis = self.axis
        if axis is None:
            axis = 0
        x_rank = len(x_shape)
        validator.check_int_range(axis, -x_rank, x_rank, Rel.INC_LEFT, "axis", self.name)
        axis = axis + x_rank if axis < 0 else axis
        ouput_shape = [x_shape[i] for i in range(x_rank) if i != axis]
        return ouput_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("input_x", x_dtype, mstype.tensor, self.name)
        return mstype.tensor_type(self.output_type)


class Argmin(PrimitiveWithInfer):
    """
    Returns the indices of the minimum value of a tensor across the axis.

    If the shape of input tensor is :math:`(x_1, ..., x_N)`, the shape of the output tensor is
    :math:`(x_1, ..., x_{axis-1}, x_{axis+1}, ..., x_N)`.

    Args:
        axis (int): Axis where the Argmin operation applies to. Default: -1.
        output_type (:class:`mindspore.dtype`): An optional data type of `mindspore.dtype.int32`.
            Default: `mindspore.dtype.int32`.

    Inputs:
        - **input_x** (Tensor) - Input tensor.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, indices of the min value of input tensor across the axis.

    Raises:
        TypeError: If `axis` is not an int.
        TypeError: If `output_type` is neither int32 nor int64.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.array([2.0, 3.1, 1.2]), mindspore.float32)
        >>> index = ops.Argmin()(input_x)
        >>> print(index)
        2
    """

    @prim_attr_register
    def __init__(self, axis=-1, output_type=mstype.int32):
        """Initialize Argmin"""
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
        validator.check_int_range(axis, -x_rank, x_rank, Rel.INC_LEFT, "axis", self.name)
        axis = axis + x_rank if axis < 0 else axis
        ouput_shape = [x_shape[i] for i in range(x_rank) if i != axis]
        return ouput_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("input_x", x_dtype, mstype.tensor, self.name)
        return mstype.tensor_type(self.output_type)


class ArgMaxWithValue(PrimitiveWithInfer):
    """
    Calculates the maximum value with the corresponding index.

    Calculates the maximum value along with the given axis for the input tensor. It returns the maximum values and
    indices.

    Note:
        In auto_parallel and semi_auto_parallel mode, the first output index can not be used.

    .. warning::
        - If there are multiple maximum values, the index of the first maximum value is used.
        - The value range of "axis" is [-dims, dims - 1]. "dims" is the dimension length of "input_x".

    Args:
        axis (int): The dimension to reduce. Default: 0.
        keep_dims (bool): Whether to reduce dimension, if true, the output will keep same dimension with the input,
                          the output will reduce dimension if false. Default: False.

    Inputs:
        - **input_x** (Tensor) - The input tensor, can be any dimension. Set the shape of input tensor as
          :math:`(x_1, x_2, ..., x_N)`. And the data type only support mindspore.float16 or float32.

    Outputs:
        tuple (Tensor), tuple of 2 tensors, containing the corresponding index and the maximum value of the input
        tensor.
        - index (Tensor) - The index for the maximum value of the input tensor. If `keep_dims` is true, the shape of
        output tensors is :math:`(x_1, x_2, ..., x_{axis-1}, 1, x_{axis+1}, ..., x_N)`. Otherwise, the shape is
        :math:`(x_1, x_2, ..., x_{axis-1}, x_{axis+1}, ..., x_N)`.
        - output_x (Tensor) - The maximum value of input tensor, with the same shape as index.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `axis` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), mindspore.float32)
        >>> index, output = ops.ArgMaxWithValue()(input_x)
        >>> print(index, output)
        3 0.7
        >>> index, output = ops.ArgMaxWithValue(keep_dims=True)(input_x)
        >>> print(index, output)
        [3] [0.7]
    """

    @prim_attr_register
    def __init__(self, axis=0, keep_dims=False):
        """Initialize ArgMaxWithValue"""
        self.axis = axis
        self.keep_dims = keep_dims
        validator.check_value_type('keep_dims', keep_dims, [bool], self.name)
        validator.check_value_type('axis', axis, [int], self.name)

    def infer_shape(self, x_shape):
        axis = self.axis
        x_rank = len(x_shape)
        validator.check_int_range(axis, -x_rank, x_rank, Rel.INC_LEFT, "axis", self.name)
        ouput_shape = _infer_shape_reduce(x_shape, self.axis, self.keep_dims, self.name)
        return ouput_shape, ouput_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("input_x", x_dtype, mstype.tensor, self.name)
        return mstype.tensor_type(mstype.int32), x_dtype


class ArgMinWithValue(PrimitiveWithInfer):
    """
    Calculates the minimum value with corresponding index, and returns indices and values.

    Calculates the minimum value along with the given axis for the input tensor. It returns the minimum values and
    indices.

    Note:
        In auto_parallel and semi_auto_parallel mode, the first output index can not be used.

    .. warning::
        - If there are multiple minimum values, the index of the first minimum value is used.
        - The value range of "axis" is [-dims, dims - 1]. "dims" is the dimension length of "input_x".

    Args:
        axis (int): The dimension to reduce. Default: 0.
        keep_dims (bool): Whether to reduce dimension, if true the output will keep the same dimension as the input,
                          the output will reduce dimension if false. Default: False.

    Inputs:
        - **input_x** (Tensor) - The input tensor, can be any dimension. Set the shape of input tensor as
          :math:`(x_1, x_2, ..., x_N)`.

    Outputs:
        tuple (Tensor), tuple of 2 tensors, containing the corresponding index and the minimum value of the input
        tensor.
        - index (Tensor) - The index for the minimum value of the input tensor. If `keep_dims` is true, the shape of
        output tensors is :math:`(x_1, x_2, ..., x_{axis-1}, 1, x_{axis+1}, ..., x_N)`. Otherwise, the shape is
        :math:`(x_1, x_2, ..., x_{axis-1}, x_{axis+1}, ..., x_N)`.
        - output_x (Tensor) - The minimum value of input tensor, with the same shape as index.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `axis` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), mindspore.float32)
        >>> output = ops.ArgMinWithValue()(input_x)
        >>> print(output)
        (Tensor(shape=[], dtype=Int32, value= 0), Tensor(shape=[], dtype=Float32, value= 0))
        >>> output = ops.ArgMinWithValue(keep_dims=True)(input_x)
        >>> print(output)
        (Tensor(shape=[1], dtype=Int32, value= [0]), Tensor(shape=[1], dtype=Float32, value= [ 0.00000000e+00]))
    """

    @prim_attr_register
    def __init__(self, axis=0, keep_dims=False):
        """Initialize ArgMinWithValue"""
        self.axis = axis
        self.keep_dims = keep_dims
        validator.check_value_type('keep_dims', keep_dims, [bool], self.name)
        validator.check_value_type('axis', axis, [int], self.name)

    def infer_shape(self, x_shape):
        axis = self.axis
        x_rank = len(x_shape)
        validator.check_int_range(axis, -x_rank, x_rank, Rel.INC_LEFT, "axis", self.name)
        ouput_shape = _infer_shape_reduce(x_shape, self.axis, self.keep_dims, self.name)
        return ouput_shape, ouput_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("input_x", x_dtype, mstype.tensor, self.name)
        return mstype.tensor_type(mstype.int32), x_dtype


class Tile(PrimitiveWithInfer):
    r"""
    Replicates a tensor with given multiples times.

    Creates a new tensor by replicating `input_x` `multiples` times. The i'th dimension of
    output tensor has `input_x.shape(i) * multiples[i]` elements, and the values of `input_x`
    are replicated `multiples[i]` times along the i'th dimension.

    Note:
        The length of `multiples` must be greater or equal to the length of dimension in `input_x`.

    Inputs:
        - **input_x** (Tensor) - 1-D or higher Tensor. Set the shape of input tensor as
          :math:`(x_1, x_2, ..., x_S)`.

        - **multiples** (tuple[int]) - The input tuple is constructed by multiple
          integers, i.e., :math:`(y_1, y_2, ..., y_S)`. The length of `multiples`
          cannot be smaller than the length of the shape of `input_x`.
          Only constant value is allowed.

    Outputs:
        Tensor, has the same data type as the `input_x`.

        - If the length of `multiples` is the same as the length of shape of `input_x`,
          then the shape of their corresponding positions can be multiplied, and
          the shape of Outputs is :math:`(x_1*y_1, x_2*y_2, ..., x_S*y_R)`.
        - If the length of `multiples` is larger than the length of shape of `input_x`,
          fill in multiple 1 in the length of the shape of `input_x` until their lengths are consistent.
          Such as set the shape of `input_x` as :math:`(1, ..., x_1, x_2, ..., x_S)`,
          then the shape of their corresponding positions can be multiplied, and
          the shape of Outputs is :math:`(1*y_1, ..., x_S*y_R)`.

    Raises:
        TypeError: If `multiples` is not a tuple or its elements are not all int.
        ValueError: If the elements of `multiples` are not all greater than 0.
        ValueError: If the length of `multiples` are smaller than the length of dimension in `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> tile = ops.Tile()
        >>> input_x = Tensor(np.array([[1, 2], [3, 4]]), mindspore.float32)
        >>> multiples = (2, 3)
        >>> output = tile(input_x, multiples)
        >>> print(output)
        [[1.  2.  1.  2.  1.  2.]
         [3.  4.  3.  4.  3.  4.]
         [1.  2.  1.  2.  1.  2.]
         [3.  4.  3.  4.  3.  4.]]
        >>> multiples = (2, 3, 2)
        >>> output = tile(input_x, multiples)
        >>> print(output)
        [[[1. 2. 1. 2.]
          [3. 4. 3. 4.]
          [1. 2. 1. 2.]
          [3. 4. 3. 4.]
          [1. 2. 1. 2.]
          [3. 4. 3. 4.]]
         [[1. 2. 1. 2.]
          [3. 4. 3. 4.]
          [1. 2. 1. 2.]
          [3. 4. 3. 4.]
          [1. 2. 1. 2.]
          [3. 4. 3. 4.]]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Tile"""
        self.init_prim_io_names(inputs=['x', 'multiples'], outputs=['output'])

    def check_elim(self, base_tensor, multiplier):
        if (not isinstance(base_tensor, Tensor)) or (not isinstance(multiplier, tuple)):
            raise TypeError(f"For '{self.name}', the type of ('input_x', 'multiples') should be (Tensor, tuple), "
                            f"but got ({type(base_tensor).__name__}, {type(multiplier).__name__}).")
        if all(v == 1 for v in multiplier):
            return (True, base_tensor)
        return (False, None)

    def __infer__(self, x, multiples):
        multiples_v = multiples['value']
        x_shp = x['shape']
        validator.check_value_type("multiples", multiples_v, [tuple], self.name)
        for i, multiple in enumerate(multiples_v):
            validator.check_positive_int(multiple, "multiples[%d]" % i, self.name)
        validator.check_value_type("x[\'dtype\']", x["dtype"], mstype.tensor_type, self.name)
        len_sub = len(multiples_v) - len(x_shp)
        multiples_w = None
        if len_sub == 0:
            multiples_w = multiples_v
        if len_sub > 0:
            for i in range(0, len_sub):
                x_shp.insert(0, 1)
            multiples_w = multiples_v
        elif len_sub < 0:
            raise ValueError(f"For '{self.name}', the length of 'multiples' can not be smaller than "
                             f"the dimension of 'input_x', but got length of 'multiples': {len(multiples_v)} "
                             f"and dimension of 'input_x': {len(x_shp)}.")
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
    Computes the sum of a tensor along segments.

    Calculates a tensor such that :math:`\text{output}[i] = \sum_{segment\_ids[j] == i} \text{data}[j, \ldots]`, where
    :math:`j` is a tuple describing the index of element in data.  `segment_ids` selects which elements in data to sum
    up. Segment_ids does not need to be sorted, and it does not need to cover all values in the entire valid value
    range.

    The following figure shows the calculation process of UnsortedSegmentSum:

    .. image:: api_img/UnsortedSegmentSum.png

    Note:
        If the segment_id i is absent in the segment_ids, then output[i] will be filled with 0.

    If the sum of the given segment_ids :math:`i` is empty, then :math:`\text{output}[i] = 0`. If the given segment_ids
    is negative, the value will be ignored. 'num_segments' must be equal to the number of different segment_ids.

    Inputs:
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_R)`.
        - **segment_ids** (Tensor) - Set the shape as :math:`(x_1, x_2, ..., x_N)`, where 0 < N <= R.
        - **num_segments** (int) - Set :math:`z` as num_segments.

    Outputs:
        Tensor, the shape is :math:`(z, x_{N+1}, ..., x_R)`.

    Raises:
        TypeError: If `num_segments` is not an int.
        ValueError: If length of shape of `segment_ids` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor([1, 2, 3, 4], mindspore.float32)
        >>> segment_ids = Tensor([0, 0, 1, 2], mindspore.int32)
        >>> num_segments = 4
        >>> output = ops.UnsortedSegmentSum()(input_x, segment_ids, num_segments)
        >>> print(output)
        [3. 3. 4. 0.]
        >>> input_x = Tensor([1, 2, 3, 4, 2, 5], mindspore.float32)
        >>> segment_ids = Tensor([0, 0, 1, 2, 3, 4], mindspore.int32)
        >>> num_segments = 6
        >>> output = ops.UnsortedSegmentSum()(input_x, segment_ids, num_segments)
        >>> print(output)
        [3. 3. 4. 2. 5. 0.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize UnsortedSegmentSum"""
        self.init_prim_io_names(inputs=['x', 'segment_ids', 'num_segments'], outputs=['y'])

    def __infer__(self, x, segment_ids, num_segments):
        x_type = x['dtype']
        x_shp = x['shape']
        validator.check_subclass("input_x", x_type, mstype.tensor, self.name)
        validator.check_value_type("x_shape", x_shp, [list], self.name)
        x_shp_len = len(x_shp)
        validator.check_positive_int(x_shp_len, "rank of input_x", self.name)
        segment_ids_shp = segment_ids['shape']
        segment_ids_type = segment_ids['dtype']
        validator.check_subclass("segment_ids", segment_ids_type, mstype.tensor, self.name)
        validator.check_value_type("segment_ids", segment_ids_shp, [list], self.name)
        segment_ids_shp_len = len(segment_ids_shp)
        validator.check_positive_int(segment_ids_shp_len, "rank of segment_ids", self.name)
        validator.check(f'rank of input_x', len(x_shp),
                        'rank of segments_id', len(segment_ids_shp), Rel.GE, self.name)
        if -1 not in x_shp and -1 not in segment_ids_shp:
            # only validate when both shapes fully known
            for i, value in enumerate(segment_ids_shp):
                validator.check("ids[%d]" % i, value, 'input[%d]' % i, x_shp[i], Rel.EQ, self.name)
        num_segments_v = num_segments['value']
        num_segments_type = num_segments['dtype']
        validator.check_subclass("num_segments", num_segments_type, [mstype.tensor, mstype.number], self.name)
        if isinstance(num_segments_type, type(mstype.tensor)):
            validator.check_tensor_dtype_valid("num_segments", num_segments_type, [mstype.int32, mstype.int64],
                                               self.name)
            shp = [-1]
        else:
            validator.check_value_type('num_segments', num_segments_v, [int], self.name)
            validator.check_positive_int(num_segments_v, "num_segments", self.name)
            shp = [num_segments_v]

        shp += x_shp[segment_ids_shp_len:]
        if "max_value" in num_segments and "min_value" in num_segments:
            output_max_shape = list(num_segments['max_value'])
            output_min_shape = list(num_segments['min_value'])
        else:
            if isinstance(num_segments_type, type(mstype.tensor)):
                raise ValueError(f"For '{self.name}', the dtype of 'num_segments' only support int type "
                                 f"when it is not a dynamic value, but got type of 'num_segments': "
                                 f"{num_segments_type}.")
            output_max_shape = [num_segments_v]
            output_min_shape = [num_segments_v]
        if 'max_shape' in x and 'min_shape' in x:
            max_output_incoming = x['max_shape']
            min_output_incoming = x['min_shape']
        else:
            max_output_incoming = x_shp
            min_output_incoming = x_shp
        output_max_shape += max_output_incoming[segment_ids_shp_len:]
        output_min_shape += min_output_incoming[segment_ids_shp_len:]
        return {'shape': shp,
                'max_shape': output_max_shape,
                'min_shape': output_min_shape,
                'dtype': mstype.tensor_type(x_type.element_type()),
                'value': None}


class UnsortedSegmentMin(PrimitiveWithCheck):
    """
    Computes the minimum of a tensor along segments.

    The following figure shows the calculation process of UnsortedSegmentMin:

    .. image:: api_img/UnsortedSegmentMin.png

    Note:
        If the segment_id i is absent in the segment_ids, then output[i] will be filled with
        the maximum value of the input_x's type.

    Inputs:
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_R)`.
          The data type must be float16, float32 or int32.
        - **segment_ids** (Tensor) - A `1-D` tensor whose shape is :math:`(x_1)`, the value must be non-negative tensor.
          The data type must be int32.
        - **num_segments** (int) - The value specifies the number of distinct `segment_ids`.

    Outputs:
        Tensor, set the number of `num_segments` as `N`, the shape is :math:`(N, x_2, ..., x_R)`.

    Raises:
        TypeError: If `num_segments` is not an int.
        ValueError: If length of shape of `segment_ids` is not equal to 1.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
        >>> segment_ids = Tensor(np.array([0, 1, 1]).astype(np.int32))
        >>> num_segments = 2
        >>> unsorted_segment_min = ops.UnsortedSegmentMin()
        >>> output = unsorted_segment_min(input_x, segment_ids, num_segments)
        >>> print(output)
        [[1. 2. 3.]
         [4. 2. 1.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize UnsortedSegmentMin"""
        self.init_prim_io_names(inputs=['x', 'segment_ids', 'num_segments'], outputs=['y'])

    def __check__(self, x, segment_ids, num_segments):
        x_shape = x['shape']
        segment_ids_shape = segment_ids['shape']
        valid_type = [mstype.float16, mstype.float32, mstype.int32]
        validator.check_tensor_dtype_valid("x", x['dtype'], valid_type, self.name)
        validator.check_tensor_dtype_valid("segment_ids", segment_ids['dtype'], [mstype.int32], self.name)
        validator.check_equal_int(len(segment_ids_shape), 1, "rank of segment_ids_shape", self.name)
        num_segments_type = num_segments['dtype']
        validator.check_subclass("num_segments", num_segments_type, [mstype.number], self.name)
        if -1 not in x_shape and -1 not in segment_ids_shape:
            # only validate when both shapes fully known
            validator.check(f'first shape of input_x', x_shape[0],
                            'length of segments_id', segment_ids_shape[0], Rel.EQ, self.name)
        num_segments_v = num_segments['value']
        validator.check_value_type('num_segments', num_segments_v, [int], self.name)
        validator.check_positive_int(num_segments_v, "num_segments", self.name)


class UnsortedSegmentMax(PrimitiveWithCheck):
    """
    Computes the maximum along segments of a tensor.

    The following figure shows the calculation process of UnsortedSegmentMax:

    .. image:: api_img/UnsortedSegmentMax.png

    Note:
        If the segment_id i is absent in the segment_ids, then output[i] will be filled with
        the minimum value of the input_x's type.

    Inputs:
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_R)`.
          The data type must be float16, float32 or int32.
        - **segment_ids** (Tensor) - A `1-D` tensor whose shape is :math:`(x_1)`, the value must be non-negative tensor.
          The data type must be int32.
        - **num_segments** (int) - The value specifies the number of distinct `segment_ids`.

    Outputs:
        Tensor, set the number of `num_segments` as `N`, the shape is :math:`(N, x_2, ..., x_R)`.

    Raises:
        TypeError: If `num_segments` is not an int.
        ValueError: If length of shape of `segment_ids` is not equal to 1.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
        >>> segment_ids = Tensor(np.array([0, 1, 1]).astype(np.int32))
        >>> num_segments = 2
        >>> unsorted_segment_max = ops.UnsortedSegmentMax()
        >>> output = unsorted_segment_max(input_x, segment_ids, num_segments)
        >>> print(output)
        [[1. 2. 3.]
         [4. 5. 6.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize UnsortedSegmentMax"""
        self.init_prim_io_names(inputs=['x', 'segment_ids', 'num_segments'], outputs=['y'])

    def __check__(self, x, segment_ids, num_segments):
        x_shape = x['shape']
        segment_ids_shape = segment_ids['shape']
        valid_type = [mstype.float16, mstype.float32, mstype.int32]
        validator.check_tensor_dtype_valid("x", x['dtype'], valid_type, self.name)
        validator.check_tensors_dtypes_same_and_valid({"segment_ids": segment_ids['dtype']},
                                                      [mstype.int32, mstype.int64], self.name)
        validator.check_equal_int(len(segment_ids_shape), 1, "rank of segment_ids_shape", self.name)
        num_segments_type = num_segments['dtype']
        validator.check_subclass("num_segments", num_segments_type, [mstype.number], self.name)
        if -1 not in x_shape and -1 not in segment_ids_shape:
            # only validate when both shapes fully known
            validator.check(f'first shape of input_x', x_shape[0],
                            'length of segments_id', segment_ids_shape[0], Rel.EQ, self.name)
        num_segments_v = num_segments['value']
        validator.check_value_type('num_segments', num_segments_v, [int], self.name)
        validator.check_positive_int(num_segments_v, "num_segments", self.name)


class UnsortedSegmentProd(PrimitiveWithInfer):
    """
    Computes the product of a tensor along segments.

    The following figure shows the calculation process of UnsortedSegmentProd:

    .. image:: api_img/UnsortedSegmentProd.png

    Inputs:
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_R)`.
          With float16, float32 or int32 data type.
        - **segment_ids** (Tensor) - A `1-D` tensor whose shape is :math:`(x_1)`, the value must be non-negative tensor.
          Data type must be int32.
        - **num_segments** (int) - The value specifies the number of distinct `segment_ids`,
          must be greater than 0.

    Outputs:
        Tensor, set the number of `num_segments` as `N`, the shape is :math:`(N, x_2, ..., x_R)`.

    Raises:
        TypeError: If `num_segments` is not an int.
        ValueError: If length of shape of `segment_ids` is not equal to 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
        >>> segment_ids = Tensor(np.array([0, 1, 0]).astype(np.int32))
        >>> num_segments = 2
        >>> unsorted_segment_prod = ops.UnsortedSegmentProd()
        >>> output = unsorted_segment_prod(input_x, segment_ids, num_segments)
        >>> print(output)
        [[4. 4. 3.]
         [4. 5. 6.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize UnsortedSegmentProd"""
        self.init_prim_io_names(inputs=['x', 'segment_ids', 'num_segments'], outputs=['y'])

    def __infer__(self, x, segment_ids, num_segments):
        x_type = x['dtype']
        x_shape = x['shape']
        segment_ids_shape = segment_ids['shape']
        validator.check_subclass("input_x", x_type, mstype.tensor, self.name)
        validator.check_value_type("x_shape", x_shape, [list], self.name)
        valid_type = [mstype.float16, mstype.float32, mstype.int32]
        validator.check_tensor_dtype_valid("x", x['dtype'], valid_type, self.name)
        validator.check_tensor_dtype_valid("segment_ids", segment_ids['dtype'], [mstype.int32], self.name)
        validator.check_equal_int(len(segment_ids_shape), 1, "rank of segment_ids_shape", self.name)
        validator.check(f'first shape of input_x', x_shape[0],
                        'length of segments_id', segment_ids_shape[0], Rel.EQ, self.name)
        num_segments_v = num_segments['value']
        validator.check_value_type('num_segments', num_segments_v, [int], self.name)
        validator.check_positive_int(num_segments_v, "num_segments", self.name)
        segment_ids_shape_len = len(segment_ids_shape)
        out_shape = [num_segments_v]
        out_shape += x_shape[segment_ids_shape_len:]
        out = {'shape': out_shape,
               'dtype': mstype.tensor_type(x_type.element_type()),
               'value': None}
        return out


class Concat(PrimitiveWithInfer):
    r"""
    Connect tensor in the specified axis.

    Connect input tensors along with the given axis.

    The input data is a tuple of tensors. These tensors have the same rank `R`. Set the given axis as `m`, and
    :math:`0 \le m < R`. Set the number of input tensors as `N`. For the :math:`i`-th tensor :math:`t_i`, it has
    the shape of :math:`(x_1, x_2, ..., x_{mi}, ..., x_R)`. :math:`x_{mi}` is the :math:`m`-th dimension of the
    :math:`i`-th tensor. Then, the shape of the output tensor is

    .. math::

        (x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)

    .. warning::
        "axis" is in the range [-len(x.shape), len(x.shape)].

    Args:
        axis (int): The specified axis. Default: 0.

    Inputs:
        - **input_x** (tuple, list) - A tuple or a list of input tensors.
          Suppose there are two tensors in this tuple or list, namely x1 and x2.
          To perform `Concat` in the axis 0 direction, except for the 0th axis, all other axes should be equal,
          that is, :math:`x1.shape[1] == x2.shape[1], x1.shape[2] == x2.shape[2], ..., x1.shape[R] == x2.shape[R]`,
          where the :math:`R` indicates the last axis.

    Outputs:
        Tensor, the shape is :math:`(x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)`.
        The data type is the same with `input_x`.

    Raises:
        TypeError: If `axis` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x1 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
        >>> input_x2 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
        >>> op = ops.Concat()
        >>> output = op((input_x1, input_x2))
        >>> print(output)
        [[0. 1.]
         [2. 1.]
         [0. 1.]
         [2. 1.]]
        >>> op = ops.Concat(1)
        >>> output = op((input_x1, input_x2))
        >>> print(output)
        [[0. 1. 0. 1.]
         [2. 1. 2. 1.]]
    """

    @prim_attr_register
    def __init__(self, axis=0):
        """Initialize Concat"""
        validator.check_value_type("axis", axis, [int], self.name)

    def __infer__(self, input_x):
        axis = self.axis
        x_shp = input_x['shape']
        x_type = input_x['dtype']
        _, all_shp, _ = get_concat_offset(x_shp, x_type, axis, self.name)
        self.add_prim_attr('inputNums', len(x_shp))
        ret_shp = x_shp[0].copy()
        value = None
        if input_x['value'] is not None:
            value = Tensor(np.concatenate([x.asnumpy() for x in input_x['value']], axis=axis))
        ret_shp[axis] = all_shp
        out = {'shape': ret_shp,
               'dtype': x_type[0],
               'value': value}
        if -1 in x_shp[0]:
            x_min_shp = input_x['min_shape']
            ret_min_shp = x_min_shp[0].copy()
            ret_min_shp[axis] = 0
            for all_min_shp in x_min_shp:
                ret_min_shp[axis] += all_min_shp[axis]
            out['min_shape'] = ret_min_shp
            x_max_shp = input_x['max_shape']
            ret_max_shp = x_max_shp[0].copy()
            ret_max_shp[axis] = 0
            for all_max_shp in x_max_shp:
                ret_max_shp[axis] += all_max_shp[axis]
            out['max_shape'] = ret_max_shp
        return out


class ParallelConcat(PrimitiveWithInfer):
    r"""
    Concats tensor in the first dimension.

    Concats input tensors along with the first dimension.

    The difference between Concat and ParallelConcat is that Concat requires all of the inputs be computed
    before the operation will begin but doesn't require that the input shapes be known during graph construction.
    Parallel concat will copy pieces of the input into the output as they become available, in some situations
    this can provide a performance benefit.

    Note:
        The input tensors are all required to have size 1 in the first dimension.

    Inputs:
        - **values** (tuple, list) - A tuple or a list of input tensors. The data type and shape of these
          tensors must be the same. The data type is Number except float64.

    Outputs:
        Tensor, data type is the same as `values`.

    Raises:
        ValueError: If length of shape of `values` is less than 1.
        ValueError: The data type and shape of these tensors are not the same.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> data1 = Tensor(np.array([[0, 1]]).astype(np.int32))
        >>> data2 = Tensor(np.array([[2, 1]]).astype(np.int32))
        >>> op = ops.ParallelConcat()
        >>> output = op((data1, data2))
        >>> print(output)
        [[0 1]
         [2 1]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ParallelConcat"""

    def __infer__(self, values):
        x_shp = values['shape']
        x_type = values['dtype']

        validator.check_int(len(x_shp), 1, Rel.GE, f'x_shp length', self.name)

        args = {f"x_type[{i}]": elem for i, elem in enumerate(x_type)}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type + (mstype.bool_,), self.name)

        first_elem = x_shp[0]
        for i, elem in enumerate(x_shp[1:]):
            j = i + 1
            validator.check_equal_int(elem[0], 1, f'x_shp[{j}][0]', self.name)
            validator.check(f"x_shp[0] shape", first_elem, f"x_shp[{j}] shape", elem, Rel.EQ, self.name)

        ret_shp = x_shp[0].copy()
        ret_shp[0] = len(x_shp)
        self.add_prim_attr('shape', ret_shp)
        self.add_prim_attr('N', len(x_shp))

        out = {'shape': ret_shp,
               'dtype': x_type[0],
               'value': None}
        return out


def _get_stack_shape(x_shape, x_type, axis, prim_name):
    """for stack output shape"""
    validator.check_value_type("shape", x_shape, [tuple, list], prim_name)
    validator.check_int(len(x_shape), 1, Rel.GE, "len of input_x", prim_name)
    validator.check_subclass("input_x[0]", x_type[0], mstype.tensor, prim_name)
    rank_base = len(x_shape[0])
    n = len(x_shape)
    out_shape = x_shape[0]
    validator.check_int_range(axis, -rank_base - 1, rank_base, Rel.INC_BOTH, 'axis', prim_name)
    if axis < 0:
        axis = axis + rank_base + 1
    for i in range(1, n):
        validator.check('x_type[%d]' % i, x_type[i], 'base', x_type[0], Rel.EQ, prim_name, TypeError)
        if x_shape[i] != x_shape[0]:
            raise ValueError(f"For \'{prim_name}\' element {i} shape in input can not pack with first element")
    out_shape.insert(axis, n)
    return out_shape


class Pack(PrimitiveWithInfer):
    """
    Same as operator Stack. Pack will be deprecated in the future.
    Please use Stack instead.
    """

    @deprecated("1.1", "Stack", True)
    @prim_attr_register
    def __init__(self, axis=0):
        """Initialize Pack"""
        validator.check_value_type("axis", axis, [int], self.name)
        self.axis = axis

    def __infer__(self, value):
        x_shape = value['shape']
        x_type = value['dtype']
        self.add_prim_attr('num', len(x_shape))
        all_shape = _get_stack_shape(x_shape, x_type, self.axis, self.name)
        out = {'shape': all_shape,
               'dtype': x_type[0],
               'value': None}
        return out


class Stack(PrimitiveWithInfer):
    r"""
    Stacks a list of tensors in specified axis.

    Stacks the list of input tensors with the same rank `R`, output is a tensor of rank `(R+1)`.

    Given input tensors of shape :math:`(x_1, x_2, ..., x_R)`. Set the number of input tensors as `N`.
    If :math:`0 \le axis`, the shape of the output tensor is
    :math:`(x_1, x_2, ..., x_{axis}, N, x_{axis+1}, ..., x_R)`.

    Args:
        axis (int): Dimension to stack. Default: 0.
                    Negative values wrap around. The range is [-(R+1), R+1).

    Inputs:
        - **input_x** (Union[tuple, list]) - A Tuple or list of Tensor objects with the same shape and type.

    Outputs:
        Tensor. A stacked Tensor with the same type as `input_x`.

    Raises:
        TypeError: If the data types of elements in `input_x` are not the same.
        ValueError: If the length of `input_x` is not greater than 1;
                    or if axis is out of the range [-(R+1), R+1);
                    or if the shapes of elements in input_x are not the same.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> data1 = Tensor(np.array([0, 1]).astype(np.float32))
        >>> data2 = Tensor(np.array([2, 3]).astype(np.float32))
        >>> stack = ops.Stack()
        >>> output = stack([data1, data2])
        >>> print(output)
        [[0. 1.]
         [2. 3.]]
    """

    @prim_attr_register
    def __init__(self, axis=0):
        """Initialize Stack"""
        validator.check_value_type("axis", axis, [int], self.name)
        self.axis = axis

    def __infer__(self, value):
        x_shape = value['shape']
        x_type = value['dtype']
        self.add_prim_attr('num', len(x_shape))
        all_shape = _get_stack_shape(x_shape, x_type, self.axis, self.name)
        out = {'shape': all_shape,
               'dtype': x_type[0],
               'value': None}
        return out


class Unpack(PrimitiveWithInfer):
    """
    Same as operator Unstack. Unpack will be deprecated in the future.
    Please use Unstack instead.
    """

    @deprecated("1.1", "Unstack", True)
    @prim_attr_register
    def __init__(self, axis=0):
        """Initialize Unpack"""
        validator.check_value_type("axis", axis, [int], self.name)
        self.axis = axis

    def __infer__(self, x):
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        x_shape = list(x['shape'])
        dim = len(x_shape)
        validator.check_int_range(self.axis, -dim, dim, Rel.INC_LEFT, 'axis value', self.name)
        if self.axis < 0:
            self.axis = self.axis + dim
        output_num = x_shape[self.axis]
        validator.check_value_type("num", output_num, [int], self.name)
        validator.check_positive_int(output_num, "output_num", self.name)
        self.add_prim_attr('num', output_num)
        output_valid_check = x_shape[self.axis] - output_num
        validator.check_int(output_valid_check, 0, Rel.EQ,
                            "The dimension which to unstack divides output_num", self.name)
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


class Unstack(PrimitiveWithInfer):
    r"""
    Unstacks tensor in specified axis.

    Unstacks a tensor of rank `R` along axis dimension, output tensors will have rank `(R-1)`.

    Given a tensor of shape :math:`(x_1, x_2, ..., x_R)`. If :math:`0 \le axis`,
    the shape of tensor in output is :math:`(x_1, x_2, ..., x_{axis}, x_{axis+2}, ..., x_R)`.

    This is the opposite of pack.

    Args:
        axis (int): Dimension along which to pack. Default: 0.
                    Negative values wrap around. The range is [-R, R).

    Inputs:
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_R)`.
          A tensor to be unstacked and the rank of the tensor must be greater than 0.

    Outputs:
        A tuple of tensors, the shape of each objects is the same.

    Raises:
        ValueError: If axis is out of the range [-len(input_x.shape), len(input_x.shape)).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> unstack = ops.Unstack()
        >>> input_x = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]))
        >>> output = unstack(input_x)
        >>> print(output)
        (Tensor(shape=[4], dtype=Int64, value= [1, 1, 1, 1]), Tensor(shape=[4], dtype=Int64, value= [2, 2, 2, 2]))
    """

    @prim_attr_register
    def __init__(self, axis=0):
        """Initialize Unstack"""
        validator.check_value_type("axis", axis, [int], self.name)
        self.axis = axis

    def __infer__(self, x):
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        x_shape = list(x['shape'])
        dim = len(x_shape)
        validator.check_int_range(self.axis, -dim, dim, Rel.INC_LEFT, 'axis value', self.name)
        if self.axis < 0:
            self.axis = self.axis + dim
        output_num = x_shape[self.axis]
        validator.check_value_type("num", output_num, [int], self.name)
        validator.check_positive_int(output_num, "output_num", self.name)
        self.add_prim_attr('num', output_num)
        output_valid_check = x_shape[self.axis] - output_num
        validator.check_int(output_valid_check, 0, Rel.EQ,
                            "The dimension which to unstack divides output_num", self.name)
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
    Slices a tensor in the specified shape.

    Slice the tensor 'input_x` in shape of `size` and starting at the location specified by `begin`,
    The slice `begin` represents the offset in each dimension of `input_x`,
    The slice `size` represents the size of the output tensor.

    Note that `begin` is zero-based and `size` is one-based.

    If `size[i]` is -1, all remaining elements in dimension i are included in the slice.
    This is equivalent to setting :math:`size[i] = input_x.shape(i) - begin[i]`

    Inputs:
        - **input_x** (Tensor): The target tensor.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        - **begin** (Union[tuple, list]): The beginning of the slice. Only constant value(>=0) is allowed.
        - **size** (Union[tuple, list]): The size of the slice. Only constant value is allowed.

    Outputs:
        Tensor, the shape is : input `size`, the data type is the same as `input_x`.

    Raises:
        TypeError: If `begin` or `size` is neither tuple nor list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> data = Tensor(np.array([[[1, 1, 1], [2, 2, 2]],
        ...                         [[3, 3, 3], [4, 4, 4]],
        ...                         [[5, 5, 5], [6, 6, 6]]]).astype(np.int32))
        >>> slice_op = ops.Slice()
        >>> output = slice_op(data, (1, 0, 0), (1, 1, 3))
        >>> print(output)
        [[[3 3 3]]]
        >>> output = slice_op(data, (1, 0, 0), (1, 1, 2))
        >>> print(output)
        [[[3 3]]]
        >>> output = slice_op(data, (1, 0, 0), (1, 1, 1))
        >>> print(output)
        [[[3]]]
        >>> output = slice_op(data, (1, 1, 0), (1, 1, 3))
        >>> print(output)
        [[[4 4 4]]]
        >>> output = slice_op(data, (1, 0, 1), (1, 1, 2))
        >>> print(output)
        [[[3 3]]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize slice"""
        self.init_prim_io_names(inputs=['x', 'begin', 'size'], outputs=['output'])

    def __infer__(self, x, begin, size):
        x_shape = x['shape']
        x_shp_len = len(x_shape)
        validator.check_valid_input('begin', begin['value'], self.name)
        validator.check_valid_input('size', size['value'], self.name)
        begin_v, size_v = begin['value'], size['value']
        if begin_v is None or size_v is None:
            return {'shape': None,
                    'dtype': x['dtype'],
                    'value': None}
        validator.check_value_type("input begin", begin_v, [tuple, list], self.name)
        validator.check_value_type("input size", size_v, [tuple, list], self.name)
        for key, value in zip(('begin', 'size'), (begin_v, size_v)):
            validator.check(f'len of {key}', len(value),
                            'len x\'s dim', x_shp_len)
        for i in range(x_shp_len):
            validator.check_positive_int(size_v[i], f'input size[{i}]')
            validator.check_non_negative_int(begin_v[i], f'input begin[{i}]')
            if x_shape[i] < begin_v[i] + size_v[i]:
                y = begin_v[i] + size_v[i]
                raise ValueError(f"For '{self.name}', the sliced shape can not be greater than origin shape, but got "
                                 f"sliced shape is {y}, and origin shape is {x_shape}.")
        return {'shape': size_v,
                'dtype': x['dtype'],
                'value': None}


class ReverseV2(PrimitiveWithInfer):
    """
    Reverses specific dimensions of a tensor.

    .. warning::
        "axis" must be within the rank of "input_x".

    Args:
        axis (Union[tuple(int), list(int)): The indices of the dimensions to reverse.

    Inputs:
        - **input_x** (Tensor) - The target tensor. The data type is Number except float64.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `axis` is neither list nor tuple.
        TypeError: If element of `axis` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> input_x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), mindspore.int32)
        >>> op = ops.ReverseV2(axis=[1])
        >>> output = op(input_x)
        >>> print(output)
        [[4 3 2 1]
         [8 7 6 5]]
        >>> op = ops.ReverseV2(axis=[1, 0])
        >>> output = op(input_x)
        >>> print(output)
        [[8 7 6 5]
         [4 3 2 1]]
    """

    @prim_attr_register
    def __init__(self, axis):
        """Initialize ReverseV2."""
        validator.check_value_type('axis', axis, [list, tuple], self.name)
        for i, each in enumerate(axis):
            validator.check_value_type(f'axis[{i}]', each, [int], self.name)
        self.axis = axis
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


    def infer_shape(self, x_shape):
        dim = len(x_shape)
        for i, each in enumerate(self.axis):
            validator.check_int_range(each, -dim, dim, Rel.INC_LEFT, f'axis[{i}]', self.name)
        normalized_axis = []
        for i, v in enumerate(self.axis):
            if v < 0:
                normalized_axis.append(v + dim)
            else:
                normalized_axis.append(v)

        if len(normalized_axis) != len(set(normalized_axis)):
            duplicated = [item for item, count in Counter(normalized_axis).items() if count > 1]
            raise ValueError(f"For '{self.name}', the 'axis' cannot contain duplicate dimensions,"
                             f" but got duplicated elements {duplicated}.")

        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, (mstype.bool_,) + mstype.number_type, self.name)
        return x_dtype


class Rint(PrimitiveWithInfer):
    """
    Returns an integer that is closest to x element-wise.

    Inputs:
        - **input_x** (Tensor) - The target tensor, which must be one of the following types:
          float16, float32. The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([-1.6, -0.1, 1.5, 2.0]), mindspore.float32)
        >>> op = ops.Rint()
        >>> output = op(input_x)
        >>> print(output)
        [-2.  0.  2.  2.]
        >>> input_x = Tensor(np.array([[-2.0, -1.9, -1.8, -1.7, -1.6],
        ...                            [-2.0, -1.9, -1.8, -1.7, -1.6]]), mindspore.float32)
        >>> output = op(input_x)
        >>> print(output)
        [[-2. -2. -2. -2. -2.]
         [-2. -2. -2. -2. -2.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Rint."""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, [mstype.float16, mstype.float32], self.name)
        return x_dtype


class Select(Primitive):
    r"""

    Returns the selected elements, either from input :math:`x` or input :math:`y`, depending on the `condition`.

    Given a tensor as input, this operation inserts a dimension of 1 at the dimension,
    it was invalid when both math: 'x' and math: 'y' are none.
    Keep in mind that the shape of the output tensor can vary depending
    on how many true values are in the input. Indexes are output in row-first
    order.

    The conditional tensor acts as an optional compensation (mask), which
    determines whether the corresponding element / row in the output must be
    selected from :math:`x` (if true) or :math:`y` (if false) based on the value of each
    element.

    It can be defined as:

    .. math::
        out_i = \begin{cases}
        x_i, & \text{if } condition_i \\
        y_i, & \text{otherwise}
        \end{cases}

    If condition is a vector, then :math:`x` and :math:`y` are higher-dimensional matrices, then it
    chooses to copy that row (external dimensions) from :math:`x` and :math:`y`. If condition has
    the same shape as :math:`x` and :math:`y`, you can choose to copy these elements from :math:`x`
    and :math:`y`.

    Inputs:
        - **input_cond** (Tensor[bool]) - The shape is :math:`(x_1, x_2, ..., x_N, ..., x_R)`.
          The condition tensor, decides which element is chosen.
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_N, ..., x_R)`.
          The first input tensor.
        - **input_y** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_N, ..., x_R)`.
          The second input tensor.

    Outputs:
        Tensor, has the same shape as `input_x`. The shape is :math:`(x_1, x_2, ..., x_N, ..., x_R)`.

    Raises:
        TypeError: If `input_x` or `input_y` is not a Tensor.
        ValueError: If shape of `input_x` is not equal to shape of `input_y` or shape of `input_cond`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> select = ops.Select()
        >>> input_cond = Tensor([True, False])
        >>> input_x = Tensor([2,3], mindspore.float32)
        >>> input_y = Tensor([1,2], mindspore.float32)
        >>> output = select(input_cond, input_x, input_y)
        >>> print(output)
        [2. 2.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Select."""
        self.init_prim_io_names(inputs=['condition', 'x', 'y'], outputs=['output'])


def _compute_slicing_length(begin, end, stride, x_shape, i):
    """Computes the length of the slicing."""
    if i >= len(x_shape):
        raise ValueError(f"For 'StridedSlice', the index must be less than or equal to "
                         f"the dimension of 'input_x', but got the dimension of 'input_x': {len(x_shape)} "
                         f"and the index: {i}.")
    x_dim = x_shape[i]
    if stride > 0:
        # When slicing forward, convert begin and end to positive numbers.
        if begin >= x_dim or end < -x_dim:
            # When slicing forward, if begin >= x_dim or end < -x_dim, the length of the slicing is 0.
            slicing_length = 0
        else:
            if -x_dim <= begin < 0:
                begin += x_dim
            if begin < -x_dim:
                # When slicing forward, if begin < -x_dim, set begin = 0, which means start from the 0th element.
                begin = 0
            if -x_dim <= end < 0:
                end += x_dim
            if end > x_dim:
                # When slicing forward, if end > x_dim, set end = x_dims, which means slice to the last element.
                end = x_dim
            if begin >= end:
                # When slicing forward, if begin >= end, the length of the slicing is 0.
                slicing_length = 0
            else:
                slicing_length = 1 + (end - 1 - begin) // stride
    else:
        # When slicing backward, convert begin and end to negative numbers.
        if begin < -x_dim or end >= x_dim:
            # When slicing backward, if begin < -x_dim or end >= x_dim, the length of the slicing is 0.
            slicing_length = 0
        else:
            if 0 <= begin < x_dim:
                begin += -x_dim
            if begin >= x_dim:
                begin = -1
            if 0 <= end < x_dim:
                end += -x_dim
            if end < -x_dim - 1:
                # Slicing to the 0th element.
                end = -x_dim - 1
            if begin <= end:
                slicing_length = 0
            else:
                slicing_length = 1 + (end + 1 - begin) // stride
    return slicing_length


class StridedSlice(PrimitiveWithInfer):
    r"""

    Extracts a strided slice of a tensor.

    Given an input tensor, this operation inserts a dimension of length 1 at the dimension.
    This operation extracts a fragment of size (end-begin)/stride from the given 'input_tensor'.
    Starting from the beginning position, the fragment continues adding stride to the index until
    all dimensions are not less than the ending position.

    Given a `input_x[m1, m2, ..., mn]`, `begin`, `end` and `strides` will be vectors of length n.

    In each mask field (`begin_mask`, `end_mask`, `ellipsis_mask`, `new_axis_mask`, `shrink_axis_mask`)
    the ith bit will correspond to the ith m.

    If the ith bit of `begin_mask` is set, `begin[i]` is ignored and the fullest possible range in that dimension
    is used instead. `end_mask` is analogous, except with the end range.

    As for a 5*6*7 tensor, `x[2:,:3,:]` is equivalent to `x[2:5,0:3,0:7]`.

    If the ith bit of `ellipsis_mask` is set, as many unspecified dimensions as needed will be inserted between
    other dimensions. Only one non-zero bit is allowed in `ellipsis_mask`.

    As for a 5*6*7*8 tensor, `x[2:,...,:6]` is equivalent to `x[2:5,:,:,0:6]`.
    `x[2:,...]` is equivalent to `x[2:5,:,:,:]`.

    If the ith bit of `new_axis_mask` is set, `begin`, `end` and `strides` are ignored and a new length 1
    dimension is added at the specified position in tthe output tensor.

    As for a 5*6*7 tensor, `x[:2, newaxis, :6]` will produce a tensor with shape (2, 1, 7).

    If the ith bit of `shrink_axis_mask` is set, ith size shrinks the dimension by 1, taking on the value
    at index `begin[i]`, `end[i]` and `strides[i]` are ignored.

    As for a 5*6*7 tensor, `x[:, 5, :]` will result in `shrink_axis_mask` equal to 4.

    Note:
        The stride may be negative value, which causes reverse slicing.
        The shape of `begin`, `end` and `strides` must be the same.
        `begin` and `end` are zero-indexed. The element of `strides` must be non-zero.

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
        - **end** (tuple[int]) - A tuple or which represents the maximum location where to end.
          Only constant value is allowed.
        - **strides** (tuple[int]) - A tuple which represents the stride is continuously added
          before reaching the maximum location. Only constant value is allowed.

    Outputs:
        Tensor, The output is explained by following example.

        In the 0th dimension, begin is 1, end is 2, and strides is 1,
        because :math:`1+1=2\geq2`, the interval is :math:`[1,2)`.
        Thus, return the element with :math:`index = 1` in 0th dimension, i.e., [[3, 3, 3], [4, 4, 4]].

        In the 1st dimension, similarly, the interval is :math:`[0,1)`.
        Based on the return value of the 0th dimension, return the element with :math:`index = 0`,
        i.e., [3, 3, 3].

        In the 2nd dimension, similarly, the interval is :math:`[0,3)`.
        Based on the return value of the 1st dimension, return the element with :math:`index = 0,1,2`,
        i.e., [3, 3, 3].

        Finally, the output is [3, 3, 3].

    Raises:
        TypeError: If `begin_mask`, `end_mask`, `ellipsis_mask`, `new_axis_mask` or `shrink_axis_mask` is not an int.
        TypeError: If `begin`, `end` or `strides` is not a tuple.
        ValueError: If `begin_mask`, `end_mask`, `ellipsis_mask`, `new_axis_mask` or `shrink_axis_mask` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]],
        ...                   [[5, 5, 5], [6, 6, 6]]], mindspore.float32)
        >>> #         [[[1. 1. 1.]
        >>> #           [2. 2. 2.]]
        >>> #
        >>> #          [[3. 3. 3.]
        >>> #           [4. 4. 4.]]
        >>> #
        >>> #          [[5. 5. 5.]
        >>> #           [6. 6. 6.]]]
        >>> # In order to visually view the multi-dimensional array, write the above as follows：
        >>> #         [
        >>> #             [
        >>> #                 [1,1,1]
        >>> #                 [2,2,2]
        >>> #             ]
        >>> #             [
        >>> #                 [3,3,3]
        >>> #                 [4,4,4]
        >>> #             ]
        >>> #             [
        >>> #                 [5,5,5]
        >>> #                 [6,6,6]
        >>> #             ]
        >>> #         ]
        >>> strided_slice = ops.StridedSlice()
        >>> output = strided_slice(input_x, (1, 0, 2), (3, 1, 3), (1, 1, 1))
        >>> # Take this " output = strided_slice(input_x, (1, 0, 2), (3, 1, 3), (1, 1, 1)) " as an example,
        >>> # start = [1, 0, 2] , end = [3, 1, 3], stride = [1, 1, 1], Find a segment of (start, end),
        >>> # note that end is an open interval
        >>> # To facilitate understanding, this operator can be divided into three steps:
        >>> # Step 1: Calculation of the first dimension:
        >>> # start = 1, end = 3, stride = 1, So can take 1st, 2nd rows, and then gets the final output at this time.
        >>> # output_1th =
        >>> # [
        >>> #     [
        >>> #         [3,3,3]
        >>> #         [4,4,4]
        >>> #     ]
        >>> #     [
        >>> #         [5,5,5]
        >>> #         [6,6,6]
        >>> #     ]
        >>> # ]
        >>> # Step 2: Calculation of the second dimension
        >>> # 2nd dimension, start = 0, end = 1, stride = 1. So only 0th rows can be taken, and the output at this time.
        >>> # output_2nd =
        >>> # [
        >>> #     [
        >>> #         [3,3,3]
        >>> #     ]
        >>> #     [
        >>> #         [5,5,5]
        >>> #     ]
        >>> # ]
        >>> # Step 3: Calculation of the third dimension
        >>> # 3nd dimension,start = 2, end = 3, stride = 1, So can take 2th cols,
        >>> # and you get the final output at this time.
        >>> # output_3ed =
        >>> # [
        >>> #     [
        >>> #         [3]
        >>> #     ]
        >>> #     [
        >>> #         [5]
        >>> #     ]
        >>> # ]
        >>> # The final output after finishing is:
        >>> print(output)
        [[[3.]]
         [[5.]]]
        >>> # another example like :
        >>> output = strided_slice(input_x, (1, 0, 0), (2, 1, 3), (1, 1, 1))
        >>> print(output)
        [[[3. 3. 3.]]]
    """

    @prim_attr_register
    def __init__(self,
                 begin_mask=0,
                 end_mask=0,
                 ellipsis_mask=0,
                 new_axis_mask=0,
                 shrink_axis_mask=0):
        """Initialize StridedSlice"""
        self.init_prim_io_names(inputs=['x', 'begin', 'end', 'strides'], outputs=['output'])
        validator.check_non_negative_int(begin_mask, 'begin_mask', self.name)
        validator.check_non_negative_int(end_mask, 'end_mask', self.name)
        validator.check_non_negative_int(ellipsis_mask, 'ellipsis_mask', self.name)
        if len(tuple(filter(lambda x: x == '1', bin(ellipsis_mask)[-1:1:-1]))) > 1:
            raise ValueError(f"For '{self.name}', only support one ellipsis in the index, but got {end_mask}.")
        validator.check_non_negative_int(new_axis_mask, 'new_axis_mask', self.name)
        validator.check_non_negative_int(shrink_axis_mask, 'shrink_axis_mask', self.name)

    def __infer__(self, x, begin, end, strides):
        begin_v, end_v, strides_v = begin['value'], end['value'], strides['value']
        validator.check_value_type("begin", begin_v, [tuple], self.name)
        validator.check_value_type("end", end_v, [tuple], self.name)
        validator.check_value_type("strides", strides_v, [tuple], self.name)

        if tuple(filter(lambda x: not isinstance(x, int), begin_v + end_v + strides_v)):
            raise TypeError(f"For {self.name}, both the 'begin', 'end', and 'strides' must be a tuple of int, "
                            f"but got 'begin': {begin_v}, 'end': {end_v}, 'strides': {strides_v}.")

        if tuple(filter(lambda x: x == 0, strides_v)):
            raise ValueError(f"For '{self.name}', the 'strides' cannot contain 0, but got 'strides': {strides_v}.")

        if len(end_v) != len(begin_v) or len(strides_v) != len(begin_v):
            raise ValueError(f"For '{self.name}', the length of 'begin' index and the length of 'end' index "
                             f"must be the same as 'strides', but got the length of 'begin': {begin_v}, "
                             f"'end' index: {end_v} and 'strides': {strides_v}.")

        ret_shape = self._compute_slicing_shape(x['shape'], begin_v, end_v, strides_v)

        if all(ret_shape):
            value = None
        else:
            init_func = Zero()
            init_func.__enable_zero_dim__ = True
            value = Tensor(dtype=x['dtype'].element_type(), shape=ret_shape, init=init_func)

        if "max_value" in x and "min_value" in x:
            validator.check_value_type("min_value", x["min_value"], [tuple, list], self.name)
            validator.check_value_type("max_value", x["max_value"], [tuple, list], self.name)
            max_value_np = np.array(x["max_value"])
            min_value_np = np.array(x["min_value"])
            slice_index = []
            for begin_i, end_i, strides_i in zip(begin_v, end_v, strides_v):
                s = slice(begin_i, end_i, strides_i)
                slice_index.append(s)
            slice_index = tuple(slice_index)
            max_value_slice = max_value_np[slice_index]
            min_value_slice = min_value_np[slice_index]
            max_value_slice = tuple(max_value_slice.tolist())
            min_value_slice = tuple(min_value_slice.tolist())
            return {'shape': ret_shape,
                    'dtype': x['dtype'],
                    'value': value,
                    'max_value': max_value_slice,
                    'min_value': min_value_slice}

        return {'shape': ret_shape,
                'dtype': x['dtype'],
                'value': value}

    def _compute_slicing_shape(self, x_shape, begin_v, end_v, strides_v):
        """Computes the shape of the slicing."""
        x_rank = len(x_shape)
        slice_len = len(begin_v)

        # After the integer is converted to binary, it is a str and the first two chars are the flag char '0b'.
        begin_pos = bin(self.begin_mask)[-1:1:-1]
        end_pos = bin(self.end_mask)[-1:1:-1]
        ellipsis_pos = bin(self.ellipsis_mask)[-1:1:-1]
        new_axis_pos = bin(self.new_axis_mask)[-1:1:-1]
        shrink_axis_pos = bin(self.shrink_axis_mask)[-1:1:-1]

        ret_shape = []
        i, j = 0, 0
        has_ellipsis = False
        while i < x_rank or j < slice_len:
            if j < slice_len:
                begin, end, stride = begin_v[j], end_v[j], strides_v[j]

                if j < len(ellipsis_pos) and ellipsis_pos[j] == '1':
                    # When there is ellipsis, the latter part of the ellipsis will be processed separately.
                    has_ellipsis = True
                    break
                if j < len(begin_pos) and begin_pos[j] == '1':
                    begin = -1 if strides_v[j] < 0 else 0
                if j < len(end_pos) and end_pos[j] == '1':
                    end = -(x_shape[i] + 1) if strides_v[j] < 0 else x_shape[i]
                if j < len(new_axis_pos) and new_axis_pos[j] == '1':
                    ret_shape.append(1)
                    j += 1
                    continue
                if j < len(shrink_axis_pos) and shrink_axis_pos[j] == '1':
                    if (not -x_shape[i] <= begin < x_shape[i]) or stride < 0:
                        raise IndexError(f"For '{self.name}', the 'strides' cannot be negative number and "
                                         f"'begin' should be in [-{x_shape[i]}, {x_shape[i]}) "
                                         f"when 'shrink_axis_mask' is greater than 0, "
                                         f"but got 'shrink_axis_mask': {self.shrink_axis_mask}, 'strides': {stride}, "
                                         f"'begin': {begin}.")
                    j += 1
                    i += 1
                    continue
            else:
                begin, end, stride = 0, x_shape[i], 1

            slicing_length = _compute_slicing_length(begin, end, stride, x_shape, i)
            ret_shape.append(slicing_length)
            i += 1
            j += 1
        if has_ellipsis:
            # When there is ellipsis, handle the second half of the ellipsis split.
            ellipsis_occupied_dims = x_rank - i - (slice_len - (j + 1)) + \
                                     len(tuple(filter(lambda x: x == '1', new_axis_pos[j + 1:slice_len])))
            ret_shape.extend(x_shape[i:i + ellipsis_occupied_dims])
            j += 1
            i += ellipsis_occupied_dims

            while i < x_rank or j < slice_len:
                begin, end, stride = begin_v[j], end_v[j], strides_v[j]

                if j < len(begin_pos) and begin_pos[j] == '1':
                    begin = -1 if strides_v[j] < 0 else 0
                if j < len(end_pos) and end_pos[j] == '1':
                    end = -(x_shape[i] + 1) if strides_v[j] < 0 else x_shape[i]
                if j < len(new_axis_pos) and new_axis_pos[j] == '1':
                    ret_shape.append(1)
                    j += 1
                    continue
                if j < len(shrink_axis_pos) and shrink_axis_pos[j] == '1':
                    if (not -x_shape[i] <= begin < x_shape[i]) or stride < 0:
                        raise IndexError(f"For '{self.name}', the 'strides' cannot be negative number and "
                                         f"'begin' should be in [-{x_shape[i]}, {x_shape[i]}) "
                                         f"when 'shrink_axis_mask' is greater than 0, "
                                         f"but got 'shrink_axis_mask': {self.shrink_axis_mask}, 'strides': {stride}, "
                                         f"'begin': {begin}.")
                    j += 1
                    i += 1
                    continue

                slicing_length = _compute_slicing_length(begin, end, stride, x_shape, i)
                ret_shape.append(slicing_length)
                i += 1
                j += 1
        return ret_shape


class Diag(PrimitiveWithInfer):
    r"""

    Constructs a diagonal tensor with a given diagonal values.

    Assume `input_x` has dimensions :math:`[D_1,... D_k]`, the output is a tensor of
    rank 2k with dimensions :math:`[D_1,..., D_k, D_1,..., D_k]` where:
    :math:`output[i_1,..., i_k, i_1,..., i_k] = input_x[i_1,..., i_k]` and 0 everywhere else.

    Inputs:
        - **input_x** (Tensor) - The input tensor. The input shape must be less than 5d.

    Outputs:
        Tensor, has the same dtype as the `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        ValueError: If rank of `input_x` is less than 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor([1, 2, 3, 4])
        >>> diag = ops.Diag()
        >>> output = diag(input_x)
        >>> print(output)
        [[1, 0, 0, 0],
         [0, 2, 0, 0],
         [0, 0, 3, 0],
         [0, 0, 0, 4]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Diag"""

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

    Extracts the diagonal part from given tensor.

    Assume input has dimensions :math:`[D_1,..., D_k, D_1,..., D_k]`, the output is a tensor
    of rank k with dimensions :math:`[D_1,..., D_k]` where:
    :math:`output[i_1,..., i_k] = input[i_1,..., i_k, i_1,..., i_k]`.

    Inputs:
        - **input_x** (Tensor) - The input tensor of rank 2k, k is not zero.

    Outputs:
        Tensor, the extracted diagonal has the same dtype as the `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        ValueError: If rank of `input_x` is not even or zero.
        ValueError: If input_shape[i] is not equal to input_shape[i + len(input_shape)/2].

    Supported Platforms:
        ``Ascend``

    Examples
        >>> input_x = Tensor([[1, 0, 0, 0],
        ...                   [0, 2, 0, 0],
        ...                   [0, 0, 3, 0],
        ...                   [0, 0, 0, 4]])
        >>> diag_part = ops.DiagPart()
        >>> output = diag_part(input_x)
        >>> print(output)
        [1 2 3 4]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize DiagPart"""

    def infer_dtype(self, x_type):
        validator.check_subclass('input_x', x_type, mstype.tensor, self.name)
        return x_type

    def infer_shape(self, x_shape):
        if len(x_shape) % 2 != 0 or \
                not x_shape:
            raise ValueError(f"For \'{self.name}\', the rank of 'input_x' must be non-zero and even, "
                             f"but got rank {len(x_shape)}, with shapes {x_shape}.")
        length = len(x_shape) // 2
        for i in range(length):
            validator.check('input_shape[i + len(input_shape)/2]', x_shape[i + length],
                            'input_shape[i]', x_shape[i], Rel.EQ, self.name)
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

    Creates a tensor with ones on the diagonal and zeros the rest.

    Inputs:
        - **n** (int) - The number of rows of returned tensor. only constant value.
        - **m** (int) - The number of columns of returned tensor. only constant value.
        - **t** (mindspore.dtype) - MindSpore's dtype, The data type of the returned tensor.
          The data type can be Number.

    Outputs:
        Tensor, a tensor with ones on the diagonal and the rest of elements are zero. The shape of `output` depends on
        the user's Inputs `n` and `m`. And the data type depends on Inputs `t`.

    Raises:
        TypeError: If `m` or `n` is not an int.
        ValueError: If `m` or `n` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> eye = ops.Eye()
        >>> output = eye(2, 2, mindspore.int32)
        >>> print(output)
        [[1 0]
         [0 1]]
        >>> print(output.dtype)
        Int32
        >>> output = eye(1, 2, mindspore.float64)
        >>> print(output)
        [[1. 0.]]
        >>> print(output.dtype)
        Float64
        >>> # if wants a anti-diagonal
        >>> anti_diagonal_input = eye(2, 2, mindspore.int32)
        >>> # Note that ReverseV2 only supports "Ascend" at this time
        >>> reverse = ops.ReverseV2([1])
        >>> anti_diagonal_output = reverse(anti_diagonal_input)
        >>> print(anti_diagonal_output)
        [[0 1]
         [1 0]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Eye"""

    def infer_value(self, n, m, t):
        validator.check_positive_int(n, "n", self.name)
        validator.check_positive_int(m, "m", self.name)
        args = {"dtype": t}
        validator.check_types_same_and_valid(args, mstype.number_type + (mstype.bool_,), self.name)
        np_type = mstype.dtype_to_nptype(t)
        ret = np.eye(n, m, dtype=np_type)
        return Tensor(ret)


class ScatterNd(PrimitiveWithInfer):
    r"""
    Scatters a tensor into a new tensor depending on the specified indices.

    Creates an empty tensor with the given `shape`, and set values by scattering the update tensor
    depending on indices.

    The empty tensor has rank P and `indices` has rank Q where `Q >= 2`.

    `indices` has shape :math:`(i_0, i_1, ..., i_{Q-2}, N)` where `N <= P`.

    The last dimension of `indices` (with length `N` ) indicates slices along the `N` th dimension of the empty tensor.

    `updates` is a tensor of rank `Q-1+P-N`. Its shape is: :math:`(i_0, i_1, ..., i_{Q-2}, shape_N, ..., shape_{P-1})`.

    The following figure shows the calculation process of inserting two slices in the first dimension of a rank-3
    with two matrices of new values:

    .. image:: api_img/ScatterNd.png

    Inputs:
        - **indices** (Tensor) - The index of scattering in the new tensor with int32 or int64 data type.
          The rank of indices must be at least 2 and `indices_shape[-1] <= len(shape)`.
        - **updates** (Tensor) - The source Tensor to be scattered.
          It has shape `indices_shape[:-1] + shape[indices_shape[-1]:]`.
        - **shape** (tuple[int]) - Define the shape of the output tensor, has the same data type as indices.
          The shape of `shape` is :math:`(x_1, x_2, ..., x_R)`, and length of 'shape' is greater than or equal 2.
          In other words, the shape of `shape` is at least :math:`(x_1, x_2)`.
          And the value of any element in `shape` must be greater than or equal 1.
          In other words, :math:`x_1` >= 1, :math:`x_2` >= 1.

    Outputs:
        Tensor, the new tensor, has the same type as `update` and the same shape as `shape`.

    Raises:
        TypeError: If `shape` is not a tuple.
        ValueError: If any element of `shape` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> op = ops.ScatterNd()
        >>> indices = Tensor(np.array([[0], [2]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2],
        ...                             [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[1, 1, 1, 1], [2, 2, 2, 2],
        ...                             [3, 3, 3, 3], [4, 4, 4, 4]]]), mindspore.float32)
        >>> shape = (4, 4, 4)
        >>> output = op(indices, updates, shape)
        >>> print(output)
        [[[1. 1. 1. 1.]
          [2. 2. 2. 2.]
          [3. 3. 3. 3.]
          [4. 4. 4. 4.]]
         [[0. 0. 0. 0.]
          [0. 0. 0. 0.]
          [0. 0. 0. 0.]
          [0. 0. 0. 0.]]
         [[1. 1. 1. 1.]
          [2. 2. 2. 2.]
          [3. 3. 3. 3.]
          [4. 4. 4. 4.]]
         [[0. 0. 0. 0.]
          [0. 0. 0. 0.]
          [0. 0. 0. 0.]
          [0. 0. 0. 0.]]]
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([3.2, 1.1]), mindspore.float32)
        >>> shape = (3, 3)
        >>> output = op(indices, updates, shape)
        >>> # In order to facilitate understanding, explain the operator pseudo-operation process step by step:
        >>> # Step 1: Generate an empty Tensor of the specified shape according to the shape
        >>> # [
        >>> #     [0. 0. 0.]
        >>> #     [0. 0. 0.]
        >>> #     [0. 0. 0.]
        >>> # ]
        >>> # Step 2: Modify the data at the specified location according to the indicators
        >>> # 0th row of indices is [0, 1], 0th row of updates is 3.2.
        >>> # means that the empty tensor in the 0th row and 1st col set to 3.2
        >>> # [
        >>> #     [0. 3.2. 0.]
        >>> #     [0. 0.   0.]
        >>> #     [0. 0.   0.]
        >>> # ]
        >>> # 1th row of indices is [1, 1], 1th row of updates is 1.1.
        >>> # means that the empty tensor in the 1th row and 1st col set to 1.1
        >>> # [
        >>> #     [0. 3.2. 0.]
        >>> #     [0. 1.1  0.]
        >>> #     [0. 0.   0.]
        >>> # ]
        >>> # The final result is as follows:
        >>> print(output)
        [[0. 3.2 0.]
         [0. 1.1 0.]
         [0. 0.  0.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ScatterNd"""
        self.init_prim_io_names(inputs=['indices', 'update', 'shape'], outputs=['output'])

    def __infer__(self, indices, update, shape):
        shp = shape['value']
        validator.check_subclass("update_dtype", update['dtype'], mstype.tensor, self.name)
        validator.check_tensor_dtype_valid("indices", indices['dtype'], [mstype.int32, mstype.int64], self.name)
        validator.check_value_type("shape", shp, [tuple], self.name)
        for i, x in enumerate(shp):
            validator.check_positive_int(x, f'shape[{i}]', self.name)

        indices_shape, update_shape = indices["shape"], update["shape"]
        if indices_shape[0] != update_shape[0]:
            raise ValueError(f"For '{self.name}', the first shape of 'indices' must be the same as the first shape of "
                             f"'updates', but got the first shape of 'indices': {indices_shape[0]}, "
                             f"the first shape of 'updates': {update_shape[0]}.")

        return {'shape': shp,
                'dtype': update['dtype'],
                'value': None}


class ResizeNearestNeighbor(PrimitiveWithInfer):
    r"""
    Resizes the input tensor by using the nearest neighbor algorithm.

    Resizes the input tensor to a given size by using the nearest neighbor algorithm. The nearest
    neighbor algorithm selects the value of the nearest point and does not consider the
    values of neighboring points at all, yielding a piecewise-constant interpolant.

    Args:
        size (Union[tuple, list]): The target size. The dimension of size must be 2.
        align_corners (bool): Whether the centers of the 4 corner pixels of the input
                              and output tensors are aligned. Default: False.

    Inputs:
        - **input_x** (Tensor) - The input tensor. The shape of the tensor is :math:`(N, C, H, W)`.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, C, NEW\_H, NEW\_W)`.
          The data type is same as the `input_x`.

    Raises:
        TypeError: If `size` is neither tuple nor list.
        TypeError: If `align_corners` is not a bool.
        ValueError: If length of `size` is not equal to 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_tensor = Tensor(np.array([[[[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]]]), mindspore.float32)
        >>> resize = ops.ResizeNearestNeighbor((2, 2))
        >>> output = resize(input_tensor)
        >>> print(output)
        [[[[-0.1  0.3]
           [ 0.4  0.5]]]]
    """

    @prim_attr_register
    def __init__(self, size, align_corners=False):
        """Initialize ResizeNearestNeighbor"""
        validator.check_value_type("size", size, [tuple, list], self.name)
        validator.check_value_type("align_corners", align_corners, [bool], self.name)
        validator.check_equal_int(len(size), 2, "length of size", self.name)
        for i, value in enumerate(size):
            validator.check_non_negative_int(value, f'{i}th value of size', self.name)
        self.init_prim_io_names(inputs=['image_in'], outputs=['image_out'])

    def infer_shape(self, x_shape):
        validator.check('the dimension of input_x', len(x_shape), '', 4, Rel.EQ, self.name)
        return tuple(x_shape)[:-2] + tuple(self.size)

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x", x_dtype, mstype.number_type, self.name)
        return x_dtype


class GatherNd(PrimitiveWithInfer):
    r"""
    Gathers slices from a tensor by indices.

    Using given indices to gather slices from a tensor with a specified shape.

    `indices` is an K-dimensional integer tensor. Supposes it as a (K-1)-dimensional tensor and each element of it
    defines a slice of `input_x`:

    .. math::
        output[(i_0, ..., i_{K-2})] = input\_x[indices[(i_0, ..., i_{K-2})]]

    The last dimension of `indices` can not more than the rank of `input_x`:
    :math:`indices.shape[-1] <= input\_x.rank`.

    Inputs:
        - **input_x** (Tensor) - The target tensor to gather values.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        - **indices** (Tensor) - The index tensor, with int32 or int64 data type.
          The dimension of `indices` should be <= the dimension of `input_x`.

    Outputs:
        Tensor, has the same type as `input_x` and the shape is indices_shape[:-1] + x_shape[indices_shape[-1]:].

    Raises:
        ValueError: If length of shape of `input_x` is less than the last dimension of `indices`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> op = ops.GatherNd()
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> output = op(input_x, indices)
        >>> print(output)
        [-0.1  0.5]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize GatherNd"""
        self.init_prim_io_names(inputs=['input_x', 'indices'], outputs=['y'])

    def infer_shape(self, x_shape, indices_shape):
        validator.check('the dimension of x', len(x_shape),
                        'the dimension of indices', indices_shape[-1], Rel.GE, self.name)
        return indices_shape[:-1] + x_shape[indices_shape[-1]:]

    def infer_dtype(self, x_dtype, indices_dtype):
        validator.check_tensor_dtype_valid("indices", indices_dtype, mstype.int_type, self.name)
        return x_dtype


class TensorScatterUpdate(PrimitiveWithInfer):
    """
    Creates a new tensor by updating the positions in `input_x` indicicated by
    `indices`, with values from `update`. This operation is almost equivalent to using
    ScatterNd, except that the updates are applied on `input_x` instead of a zero tensor.

    `indices` must have rank at least 2, the last axis is the depth of each index
    vectors. For each index vector, there must be a corresponding value in `update`. If
    the depth of each index tensor matches the rank of `input_x`, then each index
    vector corresponds to a scalar in `input_x` and each update updates a scalar. If
    the depth of each index tensor is less than the rank of `input_x`, then each index
    vector corresponds to a slice in `input_x`, and each update updates a slice.

    The order in which updates are applied is nondeterministic, meaning that if there
    are multiple index vectors in `indices` that correspond to the same position, the
    value of that position in the output will be nondeterministic.

    Inputs:
        - **input_x** (Tensor) - The target tensor. The dimension of input_x must be no less than indices.shape[-1].
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
          The data type is Number.
        - **indices** (Tensor) - The index of input tensor whose data type is int32 or int64.
          The rank must be at least 2.
        - **update** (Tensor) - The tensor to update the input tensor, has the same type as input,
          and update.shape = indices.shape[:-1] + input_x.shape[indices.shape[-1]:].

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If dtype of `indices` is neither int32 nor int64.
        ValueError: If length of shape of `input_x` is less than the last dimension of shape of `indices`.
        ValueError: If the value of `input_x` are not match with input `indices`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> update = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> op = ops.TensorScatterUpdate()
        >>> output = op(input_x, indices, update)
        >>> print(output)
        [[ 1.   0.3  3.6]
         [ 0.4  2.2 -3.2]]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['input_x', 'indices', 'updates'], outputs=['y'])

    def infer_shape(self, input_x_shape, indices_shape, updates_shape):
        if len(indices_shape) < 2:
            raise ValueError(f"For '{self.name}', the dimension of 'indices' cannot be less than 2,"
                             f" but got {len(indices_shape)}.")

        if indices_shape[-1] > len(input_x_shape):
            raise ValueError(f"For '{self.name}', the last dimension of 'indices' must be less than or equal to "
                             f"the dimension of 'input_x', but got the "
                             f"last dimension of 'indices': {indices_shape[-1]} and the dimension of 'input_x': "
                             f"{len(indices_shape)}.")

        updates_shape_check = indices_shape[:-1] + input_x_shape[indices_shape[-1]:]
        if updates_shape_check != updates_shape:
            raise ValueError(f"For '{self.name}', the shape of 'update' must be equal to updates_shape_check, "
                             f"where updates_shape_check = indices_shape[:-1] + input_x_shape[indices_shape[-1]:] "
                             f"but got the shape of 'update': {updates_shape}, "
                             f"updates_shape_check: {updates_shape_check}, indices_shape: {indices_shape} and "
                             f"input_x_shape: {input_x_shape}. Please check input_x_shape and indices_shape.")

        return input_x_shape

    def infer_dtype(self, input_x_dtype, indices_dtype, updates_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32, mstype.int64], self.name)
        args = {"input_x": input_x_dtype, "updates": updates_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.bool_,) + mstype.number_type, self.name)
        return input_x_dtype


class TensorScatterAdd(PrimitiveWithInfer):
    """
    Creates a new tensor by adding the values from the positions in `input_x` indicicated by
    `indices`, with values from `updates`. When multiple values are given for the same
    index, the updated result will be the sum of all values. This operation is almost
    equivalent to using ScatterNdAdd, except that the updates are applied on `Tensor`
    instead of `Parameter`.

    The last axis of `indices` is the depth of each index vectors. For each index vector,
    there must be a corresponding value in `updates`. The shape of `updates` should be
    equal to the shape of `input_x[indices]`. For more details, see use cases.

    Note:
        If some values of the `indices` are out of bound, instead of raising an index error,
        the corresponding `updates` will not be updated to `input_x`.

    Inputs:
        - **input_x** (Tensor) - The target tensor. The dimension of input_x must be no less than indices.shape[-1].
        - **indices** (Tensor) - The index of input tensor whose data type is int32 or int64.
          The rank must be at least 2.
        - **updates** (Tensor) - The tensor to update the input tensor, has the same type as input,
          and updates.shape should be equal to indices.shape[:-1] + input_x.shape[indices.shape[-1]:].

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If dtype of `indices` is neither int32 nor int64.
        ValueError: If length of shape of `input_x` is less than the last dimension of shape of `indices`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [0, 0]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> # Next, demonstrate the approximate operation process of this operator:
        >>> # 1, indices[0] = [0, 0], indices[1] = [0, 0]
        >>> # 2, And input_x[0, 0] = -0.1
        >>> # 3, So input_x[indices] = [-0.1, -0.1]
        >>> # 4, Satisfy the above formula: input_x[indices].shape=(2) == updates.shape=(2)
        >>> op = ops.TensorScatterAdd()
        >>> # 5, Perform the addition operation for the first time:
        >>> #      first_input_x = input_x[0][0] + updates[0] = [[0.9, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> # 6, Perform the addition operation for the second time:
        >>> #      second_input_x = input_x[0][0] + updates[1] = [[3.1, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> output = op(input_x, indices, updates)
        >>> print(output)
        [[ 3.1  0.3  3.6]
         [ 0.4  0.5 -3.2]]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['input_x', 'indices', 'updates'], outputs=['y'])

    def infer_shape(self, input_x_shape, indices_shape, updates_shape):
        if len(indices_shape) < 2:
            raise ValueError(f"For '{self.name}', the dimension of 'indices' cannot be less than 2,"
                             f" but got {len(indices_shape)}.")

        if indices_shape[-1] > len(input_x_shape):
            raise ValueError(f"For '{self.name}', the last dimension of 'indices' must be less than or equal to "
                             f"the dimension of 'input_x', but got the "
                             f"last dimension of 'indices': {indices_shape[-1]} and the dimension of 'input_x': "
                             f"{len(indices_shape)}.")

        updates_shape_check = indices_shape[:-1] + input_x_shape[indices_shape[-1]:]
        if updates_shape_check != updates_shape:
            raise ValueError(f"For '{self.name}', the shape of 'update' must be equal to updates_shape_check, "
                             f"where updates_shape_check = indices_shape[:-1] + input_x_shape[indices_shape[-1]:] "
                             f"but got the shape of 'update': {updates_shape}, "
                             f"updates_shape_check: {updates_shape_check}, indices_shape: {indices_shape} and "
                             f"input_x_shape: {input_x_shape}. Please check input_x_shape and indices_shape.")

        return input_x_shape

    def infer_dtype(self, input_x_dtype, indices_dtype, updates_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32, mstype.int64], self.name)
        args = {"input_x": input_x_dtype, "updates": updates_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.bool_,) + mstype.number_type, self.name)
        return input_x_dtype


class ScatterUpdate(_ScatterOpDynamic):
    r"""
    Updates tensor values by using input indices and value.

    Using given values to update tensor value, along with the input indices.

    for each `i, ..., j` in `indices.shape`:

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :] = \text{updates}[i, ..., j, :]

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: True.

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        - **indices** (Tensor) - The index of input tensor. With int32 data type.
          If there are duplicates in indices, the order for updating is undefined.
        - **updates** (Tensor) - The tensor to update the input tensor, has the same type as input,
          and updates.shape = indices.shape + input_x.shape[1:].

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> np_x = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]])
        >>> input_x = mindspore.Parameter(Tensor(np_x, mindspore.float32), name="x")
        >>> indices = Tensor(np.array([0, 1]), mindspore.int32)
        >>> np_updates = np.array([[2.0, 1.2, 1.0], [3.0, 1.2, 1.0]])
        >>> updates = Tensor(np_updates, mindspore.float32)
        >>> op = ops.ScatterUpdate()
        >>> output = op(input_x, indices, updates)
        >>> print(output)
        [[2. 1.2  1.]
         [3. 1.2  1.]]
    """

    @prim_attr_register
    def __init__(self, use_locking=True):
        """Initialize ScatterUpdate"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)


class ScatterNdUpdate(_ScatterNdOp):
    r"""
    Updates tensor values by using input indices and value.

    Using given values to update tensor value, along with the input indices.

    `input_x` has rank P and `indices` has rank Q where `Q >= 2`.

    `indices` has shape :math:`(i_0, i_1, ..., i_{Q-2}, N)` where `N <= P`.

    The last dimension of `indices` (with length `N` ) indicates slices along the `N` th dimension of `input_x`.

    `updates` is a tensor of rank `Q-1+P-N`. Its shape is:
    :math:`(i_0, i_1, ..., i_{Q-2}, x\_shape_N, ..., x\_shape_{P-1})`.

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: True.

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        - **indices** (Tensor) - The index of input tensor, with int32 data type.
        - **updates** (Tensor) - The tensor to be updated to the input tensor, has the same type as input.
          The shape is `indices_shape[:-1] + x_shape[indices_shape[-1]:]`.

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> np_x = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]])
        >>> input_x = mindspore.Parameter(Tensor(np_x, mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> op = ops.ScatterNdUpdate()
        >>> output = op(input_x, indices, updates)
        >>> print(output)
        [[1.   0.3   3.6]
         [0.4  2.2  -3.2]]
    """

    @prim_attr_register
    def __init__(self, use_locking=True):
        """Initialize ScatterNdUpdate"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'value'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)

    def infer_dtype(self, x_dtype, indices_dtype, value_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32], self.name)
        args = {"x": x_dtype, "value": value_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.bool_,) + mstype.number_type, self.name)
        return x_dtype


class ScatterMax(_ScatterOp):
    r"""
    Updates the value of the input tensor through the maximum operation.

    Using given values to update tensor value through the max operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    for each `i, ..., j` in `indices.shape`:

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :]
        = max(\text{input_x}[\text{indices}[i, ..., j], :], \text{updates}[i, ..., j, :])

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: True.

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        - **indices** (Tensor) - The index to do max operation whose data type must be mindspore.int32.
        - **updates** (Tensor) - The tensor that performs the maximum operation with `input_x`,
          the data type is the same as `input_x`, the shape is `indices_shape + x_shape[1:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32.
        ValueError: If the shape of `updates` is not equal to `indices_shape + x_shape[1:]`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32),
        ...                     name="input_x")
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.ones([2, 2, 3]) * 88, mindspore.float32)
        >>> scatter_max = ops.ScatterMax()
        >>> output = scatter_max(input_x, indices, updates)
        >>> print(output)
        [[88. 88. 88.]
         [88. 88. 88.]]
    """


class ScatterMin(_ScatterOp):
    r"""
    Updates the value of the input tensor through the minimum operation.

    Using given values to update tensor value through the min operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    for each `i, ..., j` in `indices.shape`:

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :]
        = min(\text{input_x}[\text{indices}[i, ..., j], :], \text{updates}[i, ..., j, :])

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: False.

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        - **indices** (Tensor) - The index to do min operation whose data type must be mindspore.int32.
        - **updates** (Tensor) - The tensor doing the min operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices_shape + x_shape[1:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32.
        ValueError: If the shape of `updates` is not equal to `indices_shape + x_shape[1:]`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]), mindspore.float32),
        ...                     name="input_x")
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> update = Tensor(np.ones([2, 2, 3]), mindspore.float32)
        >>> scatter_min = ops.ScatterMin()
        >>> output = scatter_min(input_x, indices, update)
        >>> print(output)
        [[0. 1. 1.]
         [0. 0. 0.]]
    """


class ScatterAdd(_ScatterOpDynamic):
    r"""
    Updates the value of the input tensor through the addition operation.

    Using given values to update tensor value through the add operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    for each `i, ..., j` in `indices.shape`:

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :] \mathrel{+}= \text{updates}[i, ..., j, :]

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Note:
        This is an in-place update operator. Therefore, the `input_x` will be updated after the operation is completed.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: False.

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        - **indices** (Tensor) - The index to do min operation whose data type must be mindspore.int32.
        - **updates** (Tensor) - The tensor doing the min operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices_shape + x_shape[1:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32.
        ValueError: If the shape of `updates` is not equal to `indices_shape + x_shape[1:]`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.ones([2, 2, 3]), mindspore.float32)
        >>> scatter_add = ops.ScatterAdd()
        >>> output = scatter_add(input_x, indices, updates)
        >>> print(output)
        [[1. 1. 1.]
         [3. 3. 3.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [1, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [0.0, 0.0, 0.0] + [1.0, 1.0, 1.0] = [1.0, 1.0, 1.0]
        >>> # input_x[1] = [0.0, 0.0, 0.0] + [3.0, 3.0, 3.0] = [3.0, 3.0, 3.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [3.0, 3.0, 3.0] + [7.0, 7.0, 7.0] = [10.0, 10.0, 10.0]
        >>> # input_x[1] = [10.0, 10.0, 10.0] + [9.0, 9.0, 9.0] = [19.0, 19.0, 19.0]
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> scatter_add = ops.ScatterAdd()
        >>> output = scatter_add(input_x, indices, updates)
        >>> print(output)
        [[ 1.  1.  1.]
         [19. 19. 19.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="x")
        >>> # for indices = [[1, 0], [1, 1]]
        >>> # step 1: [1, 0]
        >>> # input_x[0] = [0.0, 0.0, 0.0] + [3.0, 3.0, 3.0] = [3.0, 3.0, 3.0]
        >>> # input_x[1] = [0.0, 0.0, 0.0] + [1.0, 1.0, 1.0] = [1.0, 1.0, 1.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [1.0, 1.0, 1.0] + [7.0, 7.0, 7.0] = [8.0, 8.0, 8.0]
        >>> # input_x[1] = [8.0, 8.0, 8.0] + [9.0, 9.0, 9.0] = [17.0, 17.0, 17.0]
        >>> indices = Tensor(np.array([[1, 0], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> scatter_add = ops.ScatterAdd()
        >>> output = scatter_add(input_x, indices, updates)
        >>> print(output)
        [[ 3.  3.  3.]
         [17. 17. 17.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [0, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [0.0, 0.0, 0.0] + [1.0, 1.0, 1.0] = [1.0, 1.0, 1.0]
        >>> # input_x[1] = [0.0, 0.0, 0.0] + [3.0, 3.0, 3.0] = [3.0, 3.0, 3.0]
        >>> # step 2: [0, 1]
        >>> # input_x[0] = [1.0, 1.0, 1.0] + [7.0, 7.0, 7.0] = [8.0, 8.0, 8.0]
        >>> # input_x[1] = [3.0, 3.0, 3.0] + [9.0, 9.0, 9.0] = [12.0, 12.0, 12.0]
        >>> indices = Tensor(np.array([[0, 1], [0, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> scatter_add = ops.ScatterAdd()
        >>> output = scatter_add(input_x, indices, updates)
        >>> print(output)
        [[ 8.  8.  8.]
         [12. 12. 12.]]
    """

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterAdd"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)


class ScatterSub(_ScatterOpDynamic):
    r"""
    Updates the value of the input tensor through the subtraction operation.

    Using given values to update tensor value through the subtraction operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    for each `i, ..., j` in `indices.shape`:

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :] \mathrel{-}= \text{updates}[i, ..., j, :]

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: False.

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        - **indices** (Tensor) - The index to do min operation whose data type must be mindspore.int32.
        - **updates** (Tensor) - The tensor doing the min operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices_shape + x_shape[1:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32.
        ValueError: If the shape of `updates` is not equal to `indices_shape + x_shape[1:]`.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[0, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]), mindspore.float32)
        >>> scatter_sub = ops.ScatterSub()
        >>> output = scatter_sub(input_x, indices, updates)
        >>> print(output)
        [[-1. -1. -1.]
         [-1. -1. -1.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [1, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [0.0, 0.0, 0.0] - [1.0, 1.0, 1.0] = [-1.0, -1.0, -1.0]
        >>> # input_x[1] = [0.0, 0.0, 0.0] - [3.0, 3.0, 3.0] = [-3.0, -3.0, -3.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [-3.0, -3.0, -3.0] - [7.0, 7.0, 7.0] = [-10.0, -10.0, -10.0]
        >>> # input_x[1] = [-10.0, -10.0, -10.0] - [9.0, 9.0, 9.0] = [-19.0, -19.0, -19.0]
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> scatter_sub = ops.ScatterSub()
        >>> output = scatter_sub(input_x, indices, updates)
        >>> print(output)
        [[ -1.  -1.  -1.]
         [-19. -19. -19.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="x")
        >>> # for indices = [[1, 0], [1, 1]]
        >>> # step 1: [1, 0]
        >>> # input_x[0] = [0.0, 0.0, 0.0] - [3.0, 3.0, 3.0] = [-3.0, -3.0, -3.0]
        >>> # input_x[1] = [0.0, 0.0, 0.0] - [1.0, 1.0, 1.0] = [-1.0, -1.0, -1.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [-1.0, -1.0, -1.0] - [7.0, 7.0, 7.0] = [-8.0, -8.0, -8.0]
        >>> # input_x[1] = [-8.0, -8.0, -8.0] - [9.0, 9.0, 9.0] = [-17.0, -17.0, -17.0]
        >>> indices = Tensor(np.array([[1, 0], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> scatter_sub = ops.ScatterSub()
        >>> output = scatter_sub(input_x, indices, updates)
        >>> print(output)
        [[ -3.  -3.  -3.]
         [-17. -17. -17.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [0, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [0.0, 0.0, 0.0] - [1.0, 1.0, 1.0] = [-1.0, -1.0, -1.0]
        >>> # input_x[1] = [0.0, 0.0, 0.0] - [3.0, 3.0, 3.0] = [-3.0, -3.0, -3.0]
        >>> # step 2: [0, 1]
        >>> # input_x[0] = [-1.0, -1.0, -1.0] - [7.0, 7.0, 7.0] = [-8.0, -8.0, -8.0]
        >>> # input_x[1] = [-3.0, -3.0, -3.0] - [9.0, 9.0, 9.0] = [-12.0, -12.0, -12.0]
        >>> indices = Tensor(np.array([[0, 1], [0, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> scatter_sub = ops.ScatterSub()
        >>> output = scatter_sub(input_x, indices, updates)
        >>> print(output)
        [[ -8.  -8.  -8.]
         [-12. -12. -12.]]
    """

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterSub"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)


class ScatterMul(_ScatterOp):
    r"""
    Updates the value of the input tensor through the multiply operation.

    Using given values to update tensor value through the mul operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    for each `i, ..., j` in `indices.shape`:

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :] \mathrel{*}= \text{updates}[i, ..., j, :]

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: False.

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        - **indices** (Tensor) - The index to do min operation whose data type must be mindspore.int32.
        - **updates** (Tensor) - The tensor doing the min operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices_shape + x_shape[1:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32.
        ValueError: If the shape of `updates` is not equal to `indices_shape + x_shape[1:]`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([0, 1]), mindspore.int32)
        >>> updates = Tensor(np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]), mindspore.float32)
        >>> scatter_mul = ops.ScatterMul()
        >>> output = scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[2. 2. 2.]
         [4. 4. 4.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [1, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [1.0, 1.0, 1.0] * [1.0, 1.0, 1.0] = [1.0, 1.0, 1.0]
        >>> # input_x[1] = [2.0, 2.0, 2.0] * [3.0, 3.0, 3.0] = [6.0, 6.0, 6.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [6.0, 6.0, 6.0] * [7.0, 7.0, 7.0] = [42.0, 42.0, 42.0]
        >>> # input_x[1] = [42.0, 42.0, 42.0] * [9.0, 9.0, 9.0] = [378.0, 378.0, 378.0]
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> scatter_mul = ops.ScatterMul()
        >>> output = scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[  1.   1.   1.]
         [378. 378. 378.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), mindspore.float32), name="x")
        >>> # for indices = [[1, 0], [1, 1]]
        >>> # step 1: [1, 0]
        >>> # input_x[0] = [1.0, 1.0, 1.0] * [3.0, 3.0, 3.0] = [3.0, 3.0, 3.0]
        >>> # input_x[1] = [2.0, 2.0, 2.0] * [1.0, 1.0, 1.0] = [2.0, 2.0, 2.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [2.0, 2.0, 2.0] * [7.0, 7.0, 7.0] = [14.0, 14.0, 14.0]
        >>> # input_x[1] = [14.0, 14.0, 14.0] * [9.0, 9.0, 9.0] = [126.0, 126.0, 126.0]
        >>> indices = Tensor(np.array([[1, 0], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> scatter_mul = ops.ScatterMul()
        >>> output = scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[  3.   3.   3.]
         [126. 126. 126.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [0, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [1.0, 1.0, 1.0] * [1.0, 1.0, 1.0] = [1.0, 1.0, 1.0]
        >>> # input_x[1] = [2.0, 2.0, 2.0] * [3.0, 3.0, 3.0] = [6.0, 6.0, 6.0]
        >>> # step 2: [0, 1]
        >>> # input_x[0] = [1.0, 1.0, 1.0] * [7.0, 7.0, 7.0] = [7.0, 7.0, 7.0]
        >>> # input_x[1] = [6.0, 6.0, 6.0] * [9.0, 9.0, 9.0] = [54.0, 54.0, 54.0]
        >>> indices = Tensor(np.array([[0, 1], [0, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> scatter_mul = ops.ScatterMul()
        >>> output = scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[ 7.  7.  7.]
         [54. 54. 54.]]
    """


class ScatterDiv(_ScatterOp):
    r"""
    Updates the value of the input tensor through the divide operation.

    Using given values to update tensor value through the div operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    for each `i, ..., j` in `indices.shape`:

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :] \mathrel{/}= \text{updates}[i, ..., j, :]

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: False.

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        - **indices** (Tensor) - The index to do min operation whose data type must be mindspore.int32.
        - **updates** (Tensor) - The tensor doing the min operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices_shape + x_shape[1:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32.
        ValueError: If the shape of `updates` is not equal to `indices_shape + x_shape[1:]`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[6.0, 6.0, 6.0], [2.0, 2.0, 2.0]]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([0, 1]), mindspore.int32)
        >>> updates = Tensor(np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]), mindspore.float32)
        >>> scatter_div = ops.ScatterDiv()
        >>> output = scatter_div(input_x, indices, updates)
        >>> print(output)
        [[3. 3. 3.]
         [1. 1. 1.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[105.0, 105.0, 105.0],
        ...                                      [315.0, 315.0, 315.0]]), mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [1, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [105.0, 105.0, 105.0] / [1.0, 1.0, 1.0] = [105.0, 105.0, 105.0]
        >>> # input_x[1] = [315.0, 315.0, 315.0] / [3.0, 3.0, 3.0] = [105.0, 105.0, 105.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [105.0, 105.0, 105.0] / [5.0, 5.0, 5.0] = [21.0, 21.0, 21.0]
        >>> # input_x[1] = [21.0, 21.0, 21.0] / [7.0, 7.0, 7.0] = [3.0, 3.0, 3.0]
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[5.0, 5.0, 5.0], [7.0, 7.0, 7.0]]]), mindspore.float32)
        >>> scatter_div = ops.ScatterDiv()
        >>> output = scatter_div(input_x, indices, updates)
        >>> print(output)
        [[105. 105. 105.]
         [  3.   3.   3.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[105.0, 105.0, 105.0],
        ...                                      [315.0, 315.0, 315.0]]), mindspore.float32), name="x")
        >>> # for indices = [[1, 0], [1, 1]]
        >>> # step 1: [1, 0]
        >>> # input_x[0] = [105.0, 105.0, 105.0] / [3.0, 3.0, 3.0] = [35.0, 35.0, 35.0]
        >>> # input_x[1] = [315.0, 315.0, 315.0] / [1.0, 1.0, 1.0] = [315.0, 315.0, 315.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [315.0, 315.0, 315.0] / [5.0, 5.0, 5.0] = [63.0 63.0 63.0]
        >>> # input_x[1] = [63.0 63.0 63.0] / [7.0, 7.0, 7.0] = [9.0, 9.0, 9.0]
        >>> indices = Tensor(np.array([[1, 0], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[5.0, 5.0, 5.0], [7.0, 7.0, 7.0]]]), mindspore.float32)
        >>> scatter_div = ops.ScatterDiv()
        >>> output = scatter_div(input_x, indices, updates)
        >>> print(output)
        [[35. 35. 35.]
         [ 9.  9.  9.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[105.0, 105.0, 105.0],
        ...                                      [315.0, 315.0, 315.0]]), mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [0, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [105.0, 105.0, 105.0] / [1.0, 1.0, 1.0] = [105.0, 105.0, 105.0]
        >>> # input_x[1] = [315.0, 315.0, 315.0] / [3.0, 3.0, 3.0] = [105.0, 105.0, 105.0]
        >>> # step 2: [0, 1]
        >>> # input_x[0] = [105.0, 105.0, 105.0] / [5.0, 5.0, 5.0] = [21.0, 21.0, 21.0]
        >>> # input_x[1] = [105.0, 105.0, 105.0] / [7.0, 7.0, 7.0] = [15.0, 15.0, 15.0]
        >>> indices = Tensor(np.array([[0, 1], [0, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[5.0, 5.0, 5.0], [7.0, 7.0, 7.0]]]), mindspore.float32)
        >>> scatter_div = ops.ScatterDiv()
        >>> output = scatter_div(input_x, indices, updates)
        >>> print(output)
        [[21. 21. 21.]
         [15. 15. 15.]]
    """


class ScatterNdAdd(_ScatterNdOp):
    r"""
    Applies sparse addition to individual values or slices in a tensor.

    Using given values to update tensor value through the add operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    `input_x` has rank P and `indices` has rank Q where `Q >= 2`.

    `indices` has shape :math:`(i_0, i_1, ..., i_{Q-2}, N)` where `N <= P`.

    The last dimension of `indices` (with length `N` ) indicates slices along the `N` th dimension of `input_x`.

    `updates` is a tensor of rank `Q-1+P-N`. Its shape is:
    :math:`(i_0, i_1, ..., i_{Q-2}, x\_shape_N, ..., x\_shape_{P-1})`.

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: False.

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        - **indices** (Tensor) - The index to do min operation whose data type must be mindspore.int32.
          The rank of indices must be at least 2 and `indices_shape[-1] <= len(shape)`.
        - **updates** (Tensor) - The tensor doing the min operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices_shape[:-1] + x_shape[indices_shape[-1]:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32.
        ValueError: If the shape of `updates` is not equal to `indices_shape[:-1] + x_shape[indices_shape[-1]:]`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> scatter_nd_add = ops.ScatterNdAdd()
        >>> output = scatter_nd_add(input_x, indices, updates)
        >>> print(output)
        [ 1. 10.  9.  4. 12.  6.  7. 17.]
        >>> input_x = Parameter(Tensor(np.zeros((4, 4, 4)), mindspore.int32))
        >>> indices = Tensor(np.array([[0], [2]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]), mindspore.int32)
        >>> scatter_nd_add = ops.ScatterNdAdd()
        >>> output = scatter_nd_add(input_x, indices, updates)
        >>> print(output)
        [[[1 1 1 1]
          [2 2 2 2]
          [3 3 3 3]
          [4 4 4 4]]
         [[0 0 0 0]
          [0 0 0 0]
          [0 0 0 0]
          [0 0 0 0]]
         [[5 5 5 5]
          [6 6 6 6]
          [7 7 7 7]
          [8 8 8 8]]
         [[0 0 0 0]
          [0 0 0 0]
          [0 0 0 0]
          [0 0 0 0]]]
    """


class ScatterNdSub(_ScatterNdOp):
    r"""
    Applies sparse subtraction to individual values or slices in a tensor.

    Using given values to update tensor value through the subtraction operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    `input_x` has rank P and `indices` has rank Q where `Q >= 2`.

    `indices` has shape :math:`(i_0, i_1, ..., i_{Q-2}, N)` where `N <= P`.

    The last dimension of `indices` (with length `N` ) indicates slices along the `N` th dimension of `input_x`.

    `updates` is a tensor of rank `Q-1+P-N`. Its shape is:
    :math:`(i_0, i_1, ..., i_{Q-2}, x\_shape_N, ..., x\_shape_{P-1})`.

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: False.

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        - **indices** (Tensor) - The index of input tensor, with int32 data type.
          The rank of indices must be at least 2 and `indices_shape[-1] <= len(shape)`.
        - **updates** (Tensor) - The tensor to be updated to the input tensor, has the same type as input.
          The shape is `indices_shape[:-1] + x_shape[indices_shape[-1]:]`.

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32.
        ValueError: If the shape of `updates` is not equal to `indices_shape[:-1] + x_shape[indices_shape[-1]:]`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> scatter_nd_sub = ops.ScatterNdSub()
        >>> output = scatter_nd_sub(input_x, indices, updates)
        >>> print(output)
        [ 1. -6. -3.  4. -2.  6.  7. -1.]
        >>> input_x = Parameter(Tensor(np.zeros((4, 4, 4)), mindspore.int32))
        >>> indices = Tensor(np.array([[0], [2]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]), mindspore.int32)
        >>> scatter_nd_sub = ops.ScatterNdSub()
        >>> output = scatter_nd_sub(input_x, indices, updates)
        >>> print(output)
        [[[-1 -1 -1 -1]
          [-2 -2 -2 -2]
          [-3 -3 -3 -3]
          [-4 -4 -4 -4]]
         [[ 0  0  0  0]
          [ 0  0  0  0]
          [ 0  0  0  0]
          [ 0  0  0  0]]
         [[-5 -5 -5 -5]
          [-6 -6 -6 -6]
          [-7 -7 -7 -7]
          [-8 -8 -8 -8]]
         [[ 0  0  0  0]
          [ 0  0  0  0]
          [ 0  0  0  0]
          [ 0  0  0  0]]]
    """


class ScatterNonAliasingAdd(_ScatterNdOp):
    """
    Applies sparse addition to the input using individual values or slices.

    Using given values to update tensor value through the add operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Inputs:
        - **input_x** (Parameter) - The target parameter. The data type must be float16, float32 or int32.
        - **indices** (Tensor) - The index to perform the addition operation whose data type must be mindspore.int32.
        - **updates** (Tensor) - The tensor that performs the addition operation with `input_x`,
          the data type is the same as `input_x`, the shape is `indices_shape[:-1] + x_shape[indices_shape[-1]:]`.

    Outputs:
        Parameter, the updated `input_x`.

    Raises:
        TypeError: If dtype of `indices` is not int32.
        TypeError: If dtype of `input_x` is not one of float16, float32, int32.
        ValueError: If the shape of `updates` is not equal to `indices_shape[:-1] + x_shape[indices_shape[-1]:]`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> scatter_non_aliasing_add = ops.ScatterNonAliasingAdd()
        >>> output = scatter_non_aliasing_add(input_x, indices, updates)
        >>> print(output)
        [ 1. 10.  9.  4. 12.  6.  7. 17.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ScatterNonAliasingAdd"""
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)

    def infer_dtype(self, x_dtype, indices_dtype, updates_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32], self.name)
        args = {"x": x_dtype, "updates": updates_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32, mstype.int32], self.name)
        return x_dtype


class SpaceToDepth(PrimitiveWithInfer):
    r"""
    Rearranges blocks of spatial data into depth.

    The output tensor's `height` dimension is :math:`height / block\_size`.

    The output tensor's `weight` dimension is :math:`weight / block\_size`.

    The depth of output tensor is :math:`block\_size * block\_size * input\_depth`.

    The input tensor's height and width must be divisible by `block_size`.
    The data format is "NCHW".

    Args:
        block_size (int): The block size used to divide spatial data. It must be >= 2.

    Inputs:
        - **x** (Tensor) - The target tensor. The data tyoe is Number. It must be a 4-D tensor.

    Outputs:
        Tensor, the same data type as `x`. It must be a 4-D tensor.Tensor of shape
          :math:`(N, ( C_{in} * \text{block_size} * 2), H_{in} / \text{block_size}, W_{in} / \text{block_size})`.

    Raises:
        TypeError: If `block_size` is not an int.
        ValueError: If `block_size` is less than 2.
        ValueError: If length of shape of `x` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.random.rand(1,3,2,2), mindspore.float32)
        >>> block_size = 2
        >>> space_to_depth = ops.SpaceToDepth(block_size)
        >>> output = space_to_depth(x)
        >>> print(output.shape)
        (1, 12, 1, 1)
    """

    @prim_attr_register
    def __init__(self, block_size):
        """Initialize SpaceToDepth"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        validator.check_value_type('block_size', block_size, [int], self.name)
        validator.check('block_size', block_size, '', 2, Rel.GE)
        self.block_size = block_size
        self.add_prim_attr("data_format", "NCHW")

    def infer_shape(self, x_shape):
        validator.check('x dimension', len(x_shape), '', 4, Rel.EQ)
        out_shape = copy.deepcopy(x_shape)
        for i in range(2):
            if out_shape[i + 2] % self.block_size != 0:
                msg_prefix = "2nd" if i + 2 == 2 else "3rd"
                raise ValueError(f"For '{self.name}', the shape of output with index {i + 2} must be divided "
                                 f"exactly by 'block_size', but got the {msg_prefix} dimension "
                                 f"of output: {out_shape[i + 2]} and "
                                 f"'block_size': {self.block_size}.")
            out_shape[i + 2] //= self.block_size

        out_shape[1] *= self.block_size * self.block_size
        return out_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("x_dtype", x_dtype, mstype.tensor, self.name)
        return x_dtype


class DepthToSpace(PrimitiveWithInfer):
    r"""
    Rearranges blocks of depth data into spatial dimensions.

    This is the reverse operation of SpaceToDepth.

    The depth of output tensor is :math:`input\_depth / (block\_size * block\_size)`.

    The output tensor's `height` dimension is :math:`height * block\_size`.

    The output tensor's `weight` dimension is :math:`weight * block\_size`.

    The input tensor's depth must be divisible by `block_size * block_size`.
    The data format is "NCHW".

    Args:
        block_size (int): The block size used to divide depth data. It must be >= 2.

    Inputs:
        - **x** (Tensor) - The target tensor. It must be a 4-D tensor with shape :math:`(N, C_{in}, H_{in}, W_{in})`.
          The data type is Number.

    Outputs:
        Tensor of shape :math:`(N, C_{in} / \text{block_size} ^ 2, H_{in} * \text{block_size},
        W_{in} * \text{block_size})`.

    Raises:
        TypeError: If `block_size` is not an int.
        ValueError: If `block_size` is less than 2.
        ValueError: If length of shape of `x` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.random.rand(1, 12, 1, 1), mindspore.float32)
        >>> block_size = 2
        >>> depth_to_space = ops.DepthToSpace(block_size)
        >>> output = depth_to_space(x)
        >>> print(output.shape)
        (1, 3, 2, 2)
    """

    @prim_attr_register
    def __init__(self, block_size):
        """Initialize DepthToSpace"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        validator.check_value_type('block_size', block_size, [int], self.name)
        validator.check('block_size', block_size, '', 2, Rel.GE, self.name)
        self.block_size = block_size
        self.add_prim_attr("data_format", "NCHW")

    def infer_shape(self, x_shape):
        validator.check('x dimension', len(x_shape), '', 4, Rel.EQ)
        out_shape = copy.deepcopy(x_shape)
        for i in range(2):
            out_shape[i + 2] *= self.block_size

        validator.check_int(x_shape[1] % (self.block_size * self.block_size),
                            0, Rel.EQ, 'x_shape[1] % (block_size*block_size)', self.name)
        out_shape[1] //= self.block_size * self.block_size
        return out_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("x_dtype", x_dtype, mstype.tensor, self.name)
        return x_dtype


class SpaceToBatch(PrimitiveWithInfer):
    r"""
    Divides spatial dimensions into blocks and combines the block size with the original batch.

    This operation will divide spatial dimensions (H, W) into blocks with `block_size`, the output tensor's H and W
    dimension is the corresponding number of blocks after division. The output tensor's batch dimension is the
    product of the original batch and the square of block_size. Before division, the spatial dimensions
    of the input are zero padded according to paddings if necessary.

    Args:
        block_size (int): The block size of dividing blocks with value greater than or euqual to 2.
        paddings (Union[tuple, list]): The padding values for H and W dimension, containing 2 subtraction lists.
            Each subtraction list contains 2 integer value. All values must be greater than 0.
            paddings[i] specifies the paddings for the spatial dimension i, which corresponds to the
            input dimension i+2. It is required that input_shape[i+2]+paddings[i][0]+paddings[i][1]
            is divisible by block_size.

    Inputs:
        - **input_x** (Tensor) - The input tensor. It must be a 4-D tensor. The data type is Number.

    Outputs:
        Tensor, the output tensor with the same data type as input. Assume input shape is :math:`(n, c, h, w)` with
        :math:`block\_size` and :math:`paddings`. The shape of the output tensor will be :math:`(n', c', h', w')`,
        where

        :math:`n' = n*(block\_size*block\_size)`

        :math:`c' = c`

        :math:`h' = (h+paddings[0][0]+paddings[0][1])//block\_size`

        :math:`w' = (w+paddings[1][0]+paddings[1][1])//block\_size`

    Raises:
        TypeError: If `block_size` is not an int.
        ValueError: If `block_size` is less than 2.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> block_size = 2
        >>> paddings = [[0, 0], [0, 0]]
        >>> space_to_batch = ops.SpaceToBatch(block_size, paddings)
        >>> input_x = Tensor(np.array([[[[1, 2], [3, 4]]]]), mindspore.float32)
        >>> output = space_to_batch(input_x)
        >>> print(output)
        [[[[1.]]]
         [[[2.]]]
         [[[3.]]]
         [[[4.]]]]
    """

    @prim_attr_register
    def __init__(self, block_size, paddings):
        """Initialize SpaceToBatch"""
        logger.warning("WARN_DEPRECATED: The usage of SpaceToBatch is deprecated."
                       " Please use SpaceToBatchND.")
        validator.check_value_type('block_size', block_size, [int], self.name)
        validator.check('block_size', block_size, '', 2, Rel.GE, self.name)
        self.block_size = block_size
        validator.check('paddings shape', np.array(paddings).shape, '', (2, 2), Rel.EQ, self.name)
        for elem in itertools.chain(*paddings):
            validator.check_non_negative_int(elem, 'paddings element', self.name)
            validator.check_value_type('paddings element', elem, [int], self.name)
        self.paddings = paddings

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('input_x', x_dtype, mstype.number_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape):
        validator.check_equal_int(len(x_shape), 4, 'rank of input_x', self.name)
        out_shape = copy.deepcopy(x_shape)
        for i in range(2):
            padded = out_shape[i + 2] + self.paddings[i][0] + self.paddings[i][1]
            if padded % self.block_size != 0:
                msg_ndim = "2nd" if i + 2 == 2 else "3rd"
                raise ValueError(f"For '{self.name}', the shape of the output tensor should be "
                                 f"divisible by 'block_size', but got the {msg_ndim} dimension of output: {padded} and "
                                 f"'block_size': {self.block_size}. Please check the official homepage "
                                 f"for more information about the output tensor.")
            out_shape[i + 2] = padded // self.block_size
        out_shape[0] *= self.block_size * self.block_size
        return out_shape


class BatchToSpace(PrimitiveWithInfer):
    r"""
    Divides batch dimension with blocks and interleaves these blocks back into spatial dimensions.

    This operation will divide batch dimension N into blocks with block_size, the output tensor's N dimension
    is the corresponding number of blocks after division. The output tensor's H, W dimension is product of
    original H, W dimension and block_size with given amount to crop from dimension, respectively.

    Args:
        block_size (int): The block size of division, has the value not less than 2.
        crops (Union[list(int), tuple(int)]): The crop value for H and W dimension, containing 2 subtraction lists.
            Each list contains 2 integers.
            All values must be not less than 0. crops[i] specifies the crop values for the spatial dimension i, which
            corresponds to the input dimension i+2. It is required that
            input_shape[i+2]*block_size >= crops[i][0]+crops[i][1].

    Inputs:
        - **input_x** (Tensor) - The input tensor. It must be a 4-D tensor, dimension 0 must be divisible by
          product of `block_shape`. The data type is float16 or float32.

    Outputs:
        Tensor, the output tensor with the same type as input. Assume input shape is (n, c, h, w) with block_size
        and crops. The output shape will be (n', c', h', w'), where

        :math:`n' = n//(block\_size*block\_size)`

        :math:`c' = c`

        :math:`h' = h*block\_size-crops[0][0]-crops[0][1]`

        :math:`w' = w*block\_size-crops[1][0]-crops[1][1]`

    Raises:
        TypeError: If `block_size` or element of `crops` is not an int.
        TypeError: If `crops` is neither list nor tuple.
        ValueError: If `block_size` is less than 2.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> block_size = 2
        >>> crops = [[0, 0], [0, 0]]
        >>> batch_to_space = ops.BatchToSpace(block_size, crops)
        >>> input_x = Tensor(np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]]), mindspore.float32)
        >>> output = batch_to_space(input_x)
        >>> print(output)
        [[[[1.  2.]
           [3.  4.]]]]

    """

    @prim_attr_register
    def __init__(self, block_size, crops):
        """Initialize BatchToSpace"""
        logger.warning("WARN_DEPRECATED: The usage of BatchToSpace is deprecated."
                       " Please use BatchToSpaceND.")
        validator.check_value_type('block_size', block_size, [int], self.name)
        validator.check('block_size', block_size, '', 2, Rel.GE, self.name)
        self.block_size = block_size
        validator.check_value_type('crops type', crops, [list, tuple], self.name)
        validator.check('crops shape', np.array(crops).shape, '', (2, 2))
        for elem in itertools.chain(*crops):
            validator.check_non_negative_int(elem, 'crops element', self.name)
            validator.check_value_type('crops element', elem, [int], self.name)
        self.crops = crops

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('input_x', x_dtype, mstype.number_type, self.name)
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
            raise ValueError(f"For '{self.name}', the shape of output with index 0 must be divided exactly "
                             f"by block_size_prod, but got the shape of output: {out_shape} and "
                             f"block_size_prod: {block_size_prod}.")
        out_shape[0] = out_shape[0] // block_size_prod
        return out_shape


class SpaceToBatchND(PrimitiveWithInfer):
    r"""
    Divides spatial dimensions into blocks and combines the block size with the original batch.

    This operation will divide spatial dimensions (H, W) into blocks with block_shape, the output tensor's H and W
    dimension is the corresponding number of blocks after division. The output tensor's batch dimension is the
    product of the original batch and the product of `block_shape`. Before division,
    the spatial dimensions of the input are zero padded according to paddings if necessary.

    Args:
        block_shape (Union[list(int), tuple(int), int]): The block shape of dividing block with all value greater
            than 1. If `block_shape` is a tuple or list, the length of `block_shape` is M corresponding to the
            number of spatial dimensions. If `block_shape` is a int, the block size of M dimendions are the same,
            equal to `block_shape`. M must be 2.
        paddings (Union[tuple, list]): The padding values for H and W dimension, containing 2 subtraction list.
            Each contains 2 integer value. All values must be greater than 0.
            `paddings[i]` specifies the paddings for the spatial dimension i,
            which corresponds to the input dimension i+2.
            It is required that input_shape[i+2]+paddings[i][0]+paddings[i][1] is divisible by block_shape[i].

    Inputs:
        - **input_x** (Tensor) - The input tensor. It must be a 4-D tensor.

    Outputs:
        Tensor, the output tensor with the same data type as input. Assume input shape is :math:`(n, c, h, w)` with
        :math:`block\_shape` and :math:`padddings`. The shape of the output tensor will be :math:`(n', c', h', w')`,
        where

        :math:`n' = n*(block\_shape[0]*block\_shape[1])`

        :math:`c' = c`

        :math:`h' = (h+paddings[0][0]+paddings[0][1])//block\_shape[0]`

        :math:`w' = (w+paddings[1][0]+paddings[1][1])//block\_shape[1]`

    Raises:
        TypeError: If `block_shape` is not one of list, tuple, int.
        TypeError: If `paddings` is neither list nor tuple.
        ValueError: If length of shape of `block_shape` is not equal to 1.
        ValueError: If length of `block_shape` or `paddings` is not equal to 2.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> block_shape = [2, 2]
        >>> paddings = [[0, 0], [0, 0]]
        >>> space_to_batch_nd = ops.SpaceToBatchND(block_shape, paddings)
        >>> input_x = Tensor(np.array([[[[1, 2], [3, 4]]]]), mindspore.float32)
        >>> output = space_to_batch_nd(input_x)
        >>> print(output)
        [[[[1.]]]
         [[[2.]]]
         [[[3.]]]
         [[[4.]]]]
    """

    @prim_attr_register
    def __init__(self, block_shape, paddings):
        """Initialize SpaceToBatchND"""
        if isinstance(block_shape, int):
            block_shape = (block_shape,) * 2
        self.add_prim_attr("block_shape", block_shape)
        validator.check_value_type('block_shape type', block_shape, [list, tuple], self.name)
        validator.check('block_shape shape', len(np.array(block_shape).shape), '', 1, Rel.EQ, self.name)
        block_rank = len(block_shape)
        validator.check('block_shape length', block_rank, '', 2, Rel.EQ, self.name)
        for elem in block_shape:
            validator.check('block_shape element', elem, '', 1, Rel.GE, self.name)
            validator.check_value_type('block_shape element', elem, [int], self.name)
        self.block_shape = block_shape

        validator.check_value_type('paddings type', paddings, [list, tuple], self.name)
        validator.check('paddings length', len(paddings), '', 2, Rel.EQ, self.name)
        validator.check('paddings shape', np.array(paddings).shape, '', (block_rank, 2), Rel.EQ, self.name)
        for elem in itertools.chain(*paddings):
            validator.check_non_negative_int(elem, 'paddings element', self.name)
            validator.check_value_type('paddings element', elem, [int], self.name)
        self.paddings = paddings

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('input_x', x_dtype, mstype.number_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape):
        x_rank = len(x_shape)
        validator.check_equal_int(x_rank, 4, 'x_shape rank', self.name)
        out_shape = copy.deepcopy(x_shape)

        block_shape_prod = 1
        offset = 2
        for i in range(len(self.block_shape)):
            padded = out_shape[i + offset] + self.paddings[i][0] + \
                     self.paddings[i][1]
            if padded % self.block_shape[i] != 0:
                raise ValueError(f"For '{self.name}', the padded should be divisible by 'block_shape', "
                                 f"where padded = input_x_shape[i + 2] + paddings[i][0] + paddings[i][1], "
                                 f"but got input_x_shape[{i + 2}]: {out_shape[i + offset]}, "
                                 f"paddings[{i}][0]: {self.paddings[i][0]} and paddings[{i}][1]: {self.paddings[i][1]}."
                                 f" Please check the official api documents for "
                                 f"more information about the output tensor.")
            out_shape[i + offset] = padded // self.block_shape[i]
            block_shape_prod = block_shape_prod * self.block_shape[i]
        out_shape[0] *= block_shape_prod
        return out_shape


class BatchToSpaceND(PrimitiveWithInfer):
    r"""
    Divides batch dimension with blocks and interleaves these blocks back into spatial dimensions.

    This operation will divide batch dimension N into blocks with block_shape, the output tensor's N dimension
    is the corresponding number of blocks after division. The output tensor's H, W dimension is product of
    original H, W dimension and block_shape with given amount to crop from dimension, respectively.

    Args:
        block_shape (Union[list(int), tuple(int), int]): The block shape of dividing block with all value greater
            than 1. If `block_shape` is a tuple or list, the length of `block_shape` is M corresponding to the
            number of spatial dimensions. If `block_shape` is a int, the block size of M dimendions are the same,
            equal to `block_shape`. M must be 2.
        crops (Union[list(int), tuple(int)]): The crop value for H and W dimension, containing 2 subtraction list,
            each containing 2 int value.
            All values must be >= 0. crops[i] specifies the crop values for spatial dimension i, which corresponds to
            input dimension i+2. It is required that input_shape[i+2]*block_shape[i] > crops[i][0]+crops[i][1].

    Inputs:
        - **input_x** (Tensor) - The input tensor. It must be a 4-D tensor, dimension 0 must be divisible by
          product of `block_shape`. The data type is float16 or float32.

    Outputs:
        Tensor, the output tensor with the same type as input. Assume input shape is (n, c, h, w) with block_shape
        and crops. The output shape will be (n', c', h', w'), where

        :math:`n' = n//(block\_shape[0]*block\_shape[1])`

        :math:`c' = c`

        :math:`h' = h*block\_shape[0]-crops[0][0]-crops[0][1]`

        :math:`w' = w*block\_shape[1]-crops[1][0]-crops[1][1]`

    Raises:
        TypeError: If `block_shape` is not one of list, tuple, int.
        TypeError: If `crops` is neither list nor tuple.
        ValueError: If length of `block_shape` or `crops` is not equal to 2.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> block_shape = [2, 2]
        >>> crops = [[0, 0], [0, 0]]
        >>> batch_to_space_nd = ops.BatchToSpaceND(block_shape, crops)
        >>> input_x = Tensor(np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]]), mindspore.float32)
        >>> output = batch_to_space_nd(input_x)
        >>> print(output)
        [[[[1.  2.]
           [3.  4.]]]]

    """

    @prim_attr_register
    def __init__(self, block_shape, crops):
        """Initialize BatchToSpaceND"""
        if isinstance(block_shape, int):
            block_shape = (block_shape,) * 2
        self.add_prim_attr("block_shape", block_shape)
        validator.check_value_type('block_shape type', block_shape, [list, tuple], self.name)
        validator.check('block_shape shape', len(np.array(block_shape).shape), '', 1, Rel.EQ, self.name)
        block_rank = len(block_shape)
        validator.check('block_shape length', block_rank, '', 2, Rel.EQ, self.name)
        for elem in block_shape:
            validator.check('block_shape element', elem, '', 1, Rel.GE, self.name)
            validator.check_value_type('block_shape element', elem, [int], self.name)
        self.block_shape = block_shape

        validator.check_value_type('crops type', crops, [list, tuple], self.name)
        validator.check('crops length', len(crops), '', 2, Rel.EQ, self.name)
        validator.check('crops shape', np.array(crops).shape, '', (block_rank, 2), Rel.EQ, self.name)
        for elem in itertools.chain(*crops):
            validator.check_non_negative_int(elem, 'crops element', self.name)
            validator.check_value_type('crops element', elem, [int], self.name)
        self.crops = crops

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('input_x', x_dtype, mstype.number_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape):
        x_rank = len(x_shape)
        validator.check_int(x_rank, 4, Rel.EQ, 'x_shape rank', self.name)
        out_shape = copy.deepcopy(x_shape)

        block_shape_prod = 1
        offset = 2
        for i in range(len(self.block_shape)):
            block_shape_prod = block_shape_prod * self.block_shape[i]
            x_block_prod = out_shape[i + offset] * self.block_shape[i]
            crops_sum = self.crops[i][0] + self.crops[i][1]
            validator.check("x block shape prod", x_block_prod, 'crops sum', crops_sum, Rel.GT, self.name)
            out_shape[i + offset] = x_block_prod - crops_sum

        if out_shape[0] % block_shape_prod != 0:
            raise ValueError(f"For '{self.name}', the 0th dimension of the 'input_x' should be "
                             f"divisible by block_shape_prod, where block_shape_prod = "
                             f"'block_shape[0]' * 'block_shape[1]', "
                             f"but got 0th dimension of the 'input_x': "
                             f"{out_shape[0]} and the block_shape_prod: {block_shape_prod}.")
        out_shape[0] = out_shape[0] // block_shape_prod
        return out_shape


class BroadcastTo(Primitive):
    """
    Broadcasts input tensor to a given shape.

    Input shape can be broadcast to target shape if for each dimension pair they are either equal or input is one or
    the target dimension is -1. In case of -1 in target shape, it will be replaced by the input shape's value
    in that dimension.

    When input shape is broadcast to target shape, it starts with the trailing
    dimensions. If there is a -1 in the target shape, the -1 cannot be in a leading,
    non-existing dimension.

    Args:
        shape (tuple): The target shape to broadcast. Can be fully specified, or have -1 in one position
            where it will be substituted by the input tensor's shape in that position, see example.

    Inputs:
        - **input_x** (Tensor) - The input tensor. The data type should be one of the following types:
          float16, float32, int32, int8, uint8.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.

    Outputs:
        Tensor, with the given `shape` and the same data type as `input_x`.

    Raises:
        TypeError: If `shape` is not a tuple.
        ValueError: if the target and input shapes are incompatible, or if a - 1 in the target shape is in an invalid
                    location.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> shape = (2, 3)
        >>> input_x = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> broadcast_to = ops.BroadcastTo(shape)
        >>> output = broadcast_to(input_x)
        >>> print(output)
        [[1. 2. 3.]
         [1. 2. 3.]]

        >>> shape = (-1, 2)
        >>> input_x = Tensor(np.array([[1], [2]]).astype(np.float32))
        >>> broadcast_to = ops.BroadcastTo(shape)
        >>> output = broadcast_to(input_x)
        >>> print(output)
        [[1. 1.]
         [2. 2.]]
    """

    @prim_attr_register
    def __init__(self, shape):
        """Initialize BroadcastTo"""
        validator.check_value_type("shape", shape, (tuple), self.name)
        validator.check("shape length", len(shape), "", 0, Rel.GT, self.name)
        for ix, i in enumerate(shape):
            validator.check_value_type('target shape index -> ' + str(ix), i, [int], self.name)
            validator.check("shape element", i, "shape element min limit", -1, Rel.GE, self.name)
        self.shape = shape


class Meshgrid(PrimitiveWithInfer):
    """
    Generates coordinate matrices from given coordinate tensors.

    Given N one-dimensional coordinate tensors, returns a tuple outputs of N N-D
    coordinate tensors for evaluating expressions on an N-D grid.


    Args:
        indexing (str): Either 'xy' or 'ij'. Default: 'xy'.
          When the indexing argument is set to 'xy' (the default), the broadcasting
          instructions for the first two dimensions are swapped.

    Inputs:
        - **input** (Union[tuple]) - A Tuple of N 1-D Tensor objects.
          The length of input should be greater than 1. The data type is Number.

    Outputs:
        Tensors, A Tuple of N N-D Tensor objects. The data type is the same with the Inputs.

    Raises:
        TypeError: If `indexing` is not a str or `input` is not a tuple.
        ValueError: If `indexing` is neither 'xy' nor 'ij'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
        >>> y = Tensor(np.array([5, 6, 7]).astype(np.int32))
        >>> z = Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
        >>> inputs = (x, y, z)
        >>> meshgrid = ops.Meshgrid(indexing="xy")
        >>> output = meshgrid(inputs)
        >>> print(output)
        (Tensor(shape=[3, 4, 5], dtype=Int32, value=
         [[[1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]],
          [[1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]],
          [[1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]]]),
         Tensor(shape=[3, 4, 5], dtype=Int32, value=
         [[[5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5]],
          [[6, 6, 6, 6, 6],
           [6, 6, 6, 6, 6],
           [6, 6, 6, 6, 6],
           [6, 6, 6, 6, 6]],
          [[7, 7, 7, 7, 7],
           [7, 7, 7, 7, 7],
           [7, 7, 7, 7, 7],
           [7, 7, 7, 7, 7]]]),
         Tensor(shape=[3, 4, 5], dtype=Int32, value=
         [[[8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2]],
          [[8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2]],
          [[8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2]]]))
    """

    @prim_attr_register
    def __init__(self, indexing="xy"):
        """Initialize Meshgrid."""
        validator.check_value_type("indexing", indexing, (str), self.name)
        validator.check_string(indexing.lower(), ["xy", "ij"], "indexing", self.name)
        self.indexing = indexing

    def infer_shape(self, x_shape):
        validator.check_value_type("shape", x_shape, [tuple], self.name)
        validator.check_int(len(x_shape), 2, Rel.GE, "len of input", self.name)
        n = len(x_shape)
        shape_0 = []
        for s in x_shape:
            validator.check_int(len(s), 1, Rel.EQ, 'each input rank', self.name)
            shape_0.append(s[0])
        if self.indexing == "xy":
            shape_0[0], shape_0[1] = shape_0[1], shape_0[0]
        out_shape = tuple(tuple(shape_0) for _ in range(n))
        return out_shape

    def infer_dtype(self, x_type):
        validator.check_subclass("input[0]", x_type[0], mstype.tensor, self.name)
        n = len(x_type)
        for i in range(1, n):
            validator.check('x_type[%d]' % i, x_type[i], 'base', x_type[0], Rel.EQ, self.name, TypeError)
        return x_type


class InplaceUpdate(PrimitiveWithInfer):
    r"""
    Updates specified rows with values in `v`.

    Args:
        indices (Union[int, tuple]): Indices into the left-most dimension of `x`, and determines which rows of x
            to update with v. It is a int or tuple, whose value is in [0, the first dimension size of x).

    Inputs:
        - **x** (Tensor) - A tensor which to be inplace updated. It can be one of the following data types:
          float32, float16 and int32.
        - **v** (Tensor) - A tensor with the same type as `x` and the same dimension size as `x` except
          the first dimension, which must be the same as the size of `indices`.

    Outputs:
        Tensor, with the same type and shape as the input `x`.

    Raises:
        TypeError: If `indices` is neither int nor tuple.
        TypeError: If `indices` is a tuple and its element is not an int.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> indices = (0, 1)
        >>> x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]), mindspore.float32)
        >>> v = Tensor(np.array([[0.5, 1.0], [1.0, 1.5]]), mindspore.float32)
        >>> inplace_update = ops.InplaceUpdate(indices)
        >>> output = inplace_update(x, v)
        >>> print(output)
        [[0.5 1. ]
         [1.  1.5]
         [5.  6. ]]
    """

    @prim_attr_register
    def __init__(self, indices):
        """Initialize InplaceUpdate"""
        self.init_prim_io_names(inputs=['x', 'v'], outputs=['y'])
        self.indices = indices
        validator.check_value_type("indices", indices, [int, tuple], self.name)
        if isinstance(indices, int):
            self.indices = (indices,)
        for item in self.indices:
            validator.check_value_type("item of indices", item, [int], self.name)

    def infer_dtype(self, x_dtype, v_dtype):
        args = {'x': x_dtype, 'v': v_dtype}
        valid_type = [mstype.int32, mstype.float16, mstype.float32]
        validator.check_tensors_dtypes_same_and_valid(args, valid_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape, v_shape):
        validator.check("x", len(x_shape), "v", len(v_shape), Rel.EQ, self.name)
        validator.check("size of indices", len(self.indices), "v's first dimension", v_shape[0],
                        Rel.EQ, self.name)
        for i in self.indices:
            if i < 0 or i >= x_shape[0]:
                raise ValueError(f"For '{self.name}', the value of indices must be in [0, {x_shape[0]}), "
                                 f"but got {i}.")
        x_rank = len(x_shape)
        for idx in range(x_rank)[1:]:
            validator.check('v dim %d' % idx, v_shape[idx], "x dim %d" % idx, x_shape[idx], Rel.EQ, self.name)
        return x_shape


class ReverseSequence(PrimitiveWithInfer):
    """
    Reverses variable length slices.

    Args:
        seq_dim (int): The dimension where reversal is performed. Required.
        batch_dim (int): The input is sliced in this dimension. Default: 0.

    Inputs:
        - **x** (Tensor) - The input to reverse, supporting all number types including bool.
        - **seq_lengths** (Tensor) - Must be a 1-D vector with int32 or int64 types.

    Outputs:
        Reversed tensor with the same shape and data type as input.

    Raises:
        TypeError: If `seq_dim` or `batch_dim` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([1, 2, 3]))
        >>> reverse_sequence = ops.ReverseSequence(seq_dim=1)
        >>> output = reverse_sequence(x, seq_lengths)
        >>> print(output)
        [[1. 2. 3.]
         [5. 4. 6.]
         [9. 8. 7.]]
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([1, 2, 3]))
        >>> reverse_sequence = ops.ReverseSequence(seq_dim=0, batch_dim=1)
        >>> output = reverse_sequence(x, seq_lengths)
        >>> print(output)
        [[1. 5. 9.]
         [4. 2. 6.]
         [7. 8. 3.]]
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([2, 2, 3]))
        >>> reverse_sequence = ops.ReverseSequence(seq_dim=1)
        >>> output = reverse_sequence(x, seq_lengths)
        >>> print(output)
        [[2. 1. 3.]
         [5. 4. 6.]
         [9. 8. 7.]]
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([3, 2, 3]))
        >>> reverse_sequence = ops.ReverseSequence(seq_dim=1)
        >>> output = reverse_sequence(x, seq_lengths)
        >>> print(output)
        [[3. 2. 1.]
         [5. 4. 6.]
         [9. 8. 7.]]
        >>> x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([4, 4]))
        >>> reverse_sequence = ops.ReverseSequence(seq_dim=1)
        >>> output = reverse_sequence(x, seq_lengths)
        >>> print(output)
        [[4. 3. 2. 1.]
         [8. 7. 6. 5.]]
    """

    @prim_attr_register
    def __init__(self, seq_dim, batch_dim=0):
        """Initialize ReverseSequence"""
        self.init_prim_io_names(inputs=['x', 'seq_lengths'], outputs=['y'])
        validator.check_value_type("seq_dim", seq_dim, [int], self.name)
        self.seq_dim_ = seq_dim
        validator.check_value_type("batch_dim", batch_dim, [int], self.name)
        self.batch_dim_ = batch_dim

    def infer_shape(self, x, seq_lengths):
        validator.check("seq_dim", self.seq_dim_, "x rank", len(x), Rel.LE, self.name)
        validator.check("batch_dim", self.batch_dim_, "x rank", len(x), Rel.LE, self.name)
        validator.check("batch_dim", self.batch_dim_, "seq_dim", self.seq_dim_, Rel.NE, self.name)
        validator.check("seq_lengths rank", len(seq_lengths), "expected", 1, Rel.EQ, self.name)
        validator.check("seq_lengths vector size", seq_lengths[0],
                        "input size along batch_dim", x[self.batch_dim_], Rel.EQ, self.name)
        return x

    def infer_dtype(self, x, seq_lengths):
        validator.check_tensor_dtype_valid("x_dtype", x, mstype.number_type + (mstype.bool_,), self.name)
        validator.check_tensor_dtype_valid("seq_lengths_dtype", seq_lengths, [mstype.int32, mstype.int64], self.name)
        return x


class EditDistance(PrimitiveWithInfer):
    """
    Computes the Levenshtein Edit Distance. It is used to measure the similarity of two sequences. The inputs are
    variable-length sequences provided by SparseTensors (hypothesis_indices, hypothesis_values, hypothesis_shape)
    and (truth_indices, truth_values, truth_shape).

    Args:
        normalize (bool): If true, edit distances are normalized by length of truth. Default: True.

    Inputs:
        - **hypothesis_indices** (Tensor) - The indices of the hypothesis list SparseTensor. With int64 data type.
          The shape of tensor is :math:`(N, R)`.
        - **hypothesis_values** (Tensor) - The values of the hypothesis list SparseTensor. With float32 data type.
          Must be 1-D vector with length of N.
        - **hypothesis_shape** (Tensor) - The shape of the hypothesis list SparseTensor.
          Must be R-length vector with int64 data type. Only constant value is allowed.
        - **truth_indices** (Tensor) - The indices of the truth list SparseTensor. With int64 data type.
          The shape of tensor is :math:`(M, R)`.
        - **truth_values** (Tensor) - The values of the truth list SparseTensor. Must be 1-D vector with length of M.
          With float32 data type.
        - **truth_shape** (Tensor) - The shape of the truth list SparseTensor.
          Must be R-length vector with int64 data type. Only constant value is allowed.

    Outputs:
        Tensor, a dense tensor with rank `R-1` and float32 data type.

    Raises:
        TypeError: If `normalize` is not a bool.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import context
        >>> from mindspore import Tensor
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> class EditDistance(nn.Cell):
        ...     def __init__(self, hypothesis_shape, truth_shape, normalize=True):
        ...         super(EditDistance, self).__init__()
        ...         self.edit_distance = ops.EditDistance(normalize)
        ...         self.hypothesis_shape = hypothesis_shape
        ...         self.truth_shape = truth_shape
        ...
        ...     def construct(self, hypothesis_indices, hypothesis_values, truth_indices, truth_values):
        ...         return self.edit_distance(hypothesis_indices, hypothesis_values, self.hypothesis_shape,
        ...                                   truth_indices, truth_values, self.truth_shape)
        ...
        >>> hypothesis_indices = Tensor(np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]]).astype(np.int64))
        >>> hypothesis_values = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> hypothesis_shape = Tensor(np.array([1, 1, 2]).astype(np.int64))
        >>> truth_indices = Tensor(np.array([[0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]).astype(np.int64))
        >>> truth_values = Tensor(np.array([1, 3, 2, 1]).astype(np.float32))
        >>> truth_shape = Tensor(np.array([2, 2, 2]).astype(np.int64))
        >>> edit_distance = EditDistance(hypothesis_shape, truth_shape)
        >>> output = edit_distance(hypothesis_indices, hypothesis_values, truth_indices, truth_values)
        >>> print(output)
        [[1. 1.]
         [1. 1.]]
    """

    @prim_attr_register
    def __init__(self, normalize=True):
        """Initialize EditDistance"""
        self.normalize = validator.check_value_type("normalize", normalize, [bool], self.name)
        self.set_const_input_indexes([2, 5])

    def __infer__(self, h_indices, h_values, h_shape, truth_indices, truth_values, truth_shape):
        validator.check_valid_input('hypothesis_shape', h_shape['value'], self.name)
        validator.check_valid_input('truth_shape', truth_shape['value'], self.name)
        args_int = {"hypothesis_indices": h_indices['dtype'], "hypothesis_shape": h_shape['dtype'],
                    "truth_indices": truth_indices['dtype'], "truth_shape": truth_shape['dtype']}
        validator.check_tensors_dtypes_same_and_valid(args_int, [mstype.int64], self.name)
        args = {"hypothesis_values": h_values['dtype'], "truth_values": truth_values['dtype']}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)

        hypothesis_indices_shp, truth_indices_shp = h_indices['shape'], truth_indices['shape']
        validator.check("hypothesis_indices rank", len(hypothesis_indices_shp), "expected", 2, Rel.EQ, self.name)
        validator.check("truth_indices rank", len(truth_indices_shp), "expected", 2, Rel.EQ, self.name)
        validator.check("hypothesis_values rank", len(h_values['shape']), "expected", 1, Rel.EQ, self.name)
        validator.check("hypothesis_shape rank", len(h_shape['shape']), "expected", 1, Rel.EQ, self.name)
        validator.check("truth_values rank", len(truth_values['shape']), "expected", 1, Rel.EQ, self.name)
        validator.check("truth_shape rank", len(truth_shape['shape']), "expected", 1, Rel.EQ, self.name)
        validator.check("hypothesis_values shape", h_values['shape'][0],
                        "hypothesis_indices shape[0]", hypothesis_indices_shp[0], Rel.EQ, self.name)
        validator.check("hypothesis_shape", h_shape['shape'][0],
                        "hypothesis_indices shape[1]", hypothesis_indices_shp[1], Rel.EQ, self.name)
        validator.check("truth_values shape", truth_values['shape'][0],
                        "truth_indices shape[0]", truth_indices_shp[0], Rel.EQ, self.name)
        validator.check("hypothesis_shape", h_shape['shape'][0],
                        "truth_shape", truth_shape['shape'][0], Rel.EQ, self.name)
        hypothesis_shape_v = h_shape['value'].asnumpy()
        truth_shape_v = truth_shape['value'].asnumpy()
        out_shape_rank = len(hypothesis_shape_v) - 1
        out_shape = []
        for i in range(out_shape_rank):
            out_shape.append(max(hypothesis_shape_v[i], truth_shape_v[i]))

        return {'shape': tuple(out_shape),
                'dtype': mstype.tensor_type(mstype.float32),
                'value': None}


class TransShape(PrimitiveWithInfer):
    """
    Transforms the shape of input tensor to target shape.

    Inputs:
        - **input_x** (Tensor) - A input tensor.
        - **out_shape** (tuple[int]) - The shape of output data.

    Outputs:
        Tensor, a tensor whose data type is same as 'input_x', and the shape is the same as the `out_shape`.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize TransShape."""
        self.__setattr_flag__ = True

    def __infer__(self, x, shape):
        shp = shape['value']
        dtype = x['dtype']
        validator.check_tensor_dtype_valid('x', dtype, mstype.number_type + (mstype.bool_,), self.name)
        self.add_prim_attr('out_shape', tuple(shp))
        return {'shape': shp,
                'dtype': dtype,
                'value': None}


class Sort(PrimitiveWithInfer):
    """
    Sorts the elements of the input tensor along a given dimension in ascending order by value.

    Args:
        axis (int): The dimension to sort along. Default: -1.
        descending (bool): Controls the sorting order. If descending is True then the elements
            are sorted in descending order by value. Default: False.

    Inputs:
        - **x** (Tensor) - The input to sort, with float16 or float32 data type.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.

    Outputs:
        - **y1** (Tensor) - A tensor whose values are the sorted values, with the same shape and data type as input.
        - **y2** (Tensor) - The indices of the elements in the original input tensor. Data type is int32.

    Raises:
        TypeError: If `axis` is not an int.
        TypeError: If `descending` is not a bool.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]), mindspore.float16)
        >>> sort = ops.Sort()
        >>> output = sort(x)
        >>> print(output)
        (Tensor(shape=[3, 3], dtype=Float16, value=
        [[ 1.0000e+00,  2.0000e+00,  8.0000e+00],
         [ 3.0000e+00,  5.0000e+00,  9.0000e+00],
         [ 4.0000e+00,  6.0000e+00,  7.0000e+00]]), Tensor(shape=[3, 3], dtype=Int32, value=
        [[2, 1, 0],
         [2, 0, 1],
         [0, 1, 2]]))
    """

    @prim_attr_register
    def __init__(self, axis=-1, descending=False):
        """Initialize Sort"""
        self.axis = validator.check_value_type("axis", axis, [int], self.name)
        self.descending = validator.check_value_type("descending", descending, [bool], self.name)

    def infer_shape(self, x_shape):
        return x_shape, x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x_dtype", x_dtype, [mstype.float32, mstype.float16], self.name)
        return x_dtype, mstype.tensor_type(mstype.int32)


class EmbeddingLookup(PrimitiveWithCheck):
    """
    Returns a slice of input tensor based on the specified indices.

    This Primitive has the similar functionality as GatherV2 operating on `axis = 0`, but has one more inputs:
    `offset`.

    Inputs:
        - **input_params** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
          This represents a Tensor slice, instead of the entire Tensor. Currently, the dimension is restricted to be 2.
        - **input_indices** (Tensor) - The shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
          Specifies the indices of elements of the original Tensor. Values can be out of range of `input_params`,
          and the exceeding part will be filled with 0 in the output. Values does not support negative and the result
          is undefined if values are negative. The data type should be int32 or int64.
        - **offset** (int) - Specifies the offset value of this `input_params` slice. Thus the real indices
          are equal to `input_indices` minus `offset`.

    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`. The data type is the same with `input_params`.

    Raises:
        TypeError: If dtype of `input_indices` is not int.
        ValueError: If length of shape of `input_params` is greater than 2.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> input_params = Tensor(np.array([[8, 9], [10, 11], [12, 13], [14, 15]]), mindspore.float32)
        >>> input_indices = Tensor(np.array([[5, 2], [8, 5]]), mindspore.int32)
        >>> offset = 4
        >>> output = ops.EmbeddingLookup()(input_params, input_indices, offset)
        >>> print(output)
        [[[10. 11.]
          [ 0.  0.]]
         [[ 0.  0.]
          [10. 11.]]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize EmbeddingLookup."""
        self.__setattr_flag__ = True
        self.init_prim_io_names(inputs=['params', 'indices', 'offset'],
                                outputs=['output'])

    def __check__(self, params, indices, offset):
        validator.check_subclass("params", params['dtype'], mstype.tensor, self.name)
        validator.check_tensor_dtype_valid("indices", indices['dtype'], mstype.int_type, self.name)
        validator.check_subclass("offset", offset['dtype'], mstype.int_, self.name)
        indices_shp = indices['shape']
        if not indices_shp:
            raise ValueError(f"For '{self.name}', the 'input_indices' should not be a scalar, but got {indices_shp}.")
        params_shp = params['shape']
        if len(params_shp) > 2:
            raise ValueError(f"For '{self.name}', the dimension of 'input_params' must <= 2, "
                             f"but got {len(params_shp)}.")


class GatherD(Primitive):
    """
    Gathers values along an axis specified by dim.

    For a 3-D tensor, the output is:

    .. code-block::

        output[i][j][k] = x[index[i][j][k]][j][k]  # if dim == 0

        output[i][j][k] = x[i][index[i][j][k]][k]  # if dim == 1

        output[i][j][k] = x[i][j][index[i][j][k]]  # if dim == 2

    If `x` is an n-D tensor with shape :math:`(z_0, z_1, ..., z_i, ..., z_{n-1})` and `dim` = i,
    the `index` must be an n-D tensor with shape :math:`(z_0, z_1, ..., y, ..., z_{n-1})`
    where `y`>=1 and the output will have the same shape as `index`.

    Inputs:
        - **x** (Tensor) - The source tensor.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        - **dim** (int) - The axis along which to index. It must be int32 or int64. Only constant value is allowed.
        - **index** (Tensor) - The indices of elements to gather. It can be one of the following data types:
          int32, int64. The value range of each index element is [-x_rank[dim], x_rank[dim]).

    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`, has the same data type with `x`.

    Raises:
        TypeError: If dtype of `dim` or `index` is neither int32 nor int64.
        ValueError: If length of shape of `x` is not equal to length of shape of `index`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1, 2], [3, 4]]), mindspore.int32)
        >>> index = Tensor(np.array([[0, 0], [1, 0]]), mindspore.int32)
        >>> dim = 1
        >>> output = ops.GatherD()(x, dim, index)
        >>> print(output)
        [[1 1]
         [4 3]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize GatherD"""
        self.init_prim_io_names(inputs=['x', 'dim', 'index'], outputs=['output'])


class Identity(PrimitiveWithInfer):
    """
    Returns a Tensor with the same shape and contents as input.

    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The data type is Number.

    Outputs:
        Tensor, the shape of tensor and the data type are the same as `input_x`, :math:`(x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3, 4]), mindspore.int64)
        >>> output = ops.Identity()(x)
        >>> print(output)
        [1 2 3 4]
    """

    # Side effect is identity with input.
    side_effect_propagate = 1

    @prim_attr_register
    def __init__(self):
        """Initialize identity"""
        self.add_prim_attr('side_effect_propagate', 1)

    def __infer__(self, x):
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        validator.check_tensor_dtype_valid('x', x['dtype'], mstype.number_type + (mstype.bool_,), self.name)
        out = {'shape': x['shape'],
               'dtype': x['dtype'],
               'value': None}
        return out


class Range(PrimitiveWithCheck):
    r"""
    Creates a sequence of numbers that begins at `start` and extends by increments of
    `delta` up to but not including `limit`.

    The types of all 3 inputs must be the same. The type of the resulting tensor is
    the same as the type of the inputs.

    Args:
        maxlen (int): Memory that can fit `maxlen` many elements
            will be allocated for the output. Optional, must be positive, defaults to 1000000.
            If the output has more than `maxlen` elements, a runtime error
            will occur.

    Inputs:
        - **start** (Tensor) - A scalar Tensor. The first number in the sequence. Must have
          type: int32 or float32
        - **limit** (Tensor) - A scalar Tensor. Upper limit of the sequence, exclusive. Must
          have type: int32 or float32
        - **delta** (Tensor) - A scalar Tensor. Number that increments `start`. Must have
          type: int32 or float32

    Outputs:
       A 1-D Tensor, with the same type as the inputs.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> start = Tensor(0, mstype.int32)
        >>> limit = Tensor(10, mstype.int32)
        >>> delta = Tensor(4, mstype.int32)
        >>> output = ops.Range()(start, limit, delta)
        >>> print(output)
        [0, 4, 8]
    """

    @prim_attr_register
    def __init__(self, maxlen=1000000):
        self.init_prim_io_names(inputs=['start', 'limit', 'delta'], outputs=['output'])
        validator.check_value_type("maxlen", maxlen, [int], self.name)
        validator.check_positive_int(maxlen, "maxlen", self.name)
        self.maxlen = maxlen
        self.add_prim_attr('maxlen', maxlen)

    def check_shape(self, start_shape, limit_shape, delta_shape):
        validator.check("start_shape", len(start_shape), "", 0, Rel.EQ, self.name)
        validator.check("limit_shape", len(limit_shape), "", 0, Rel.EQ, self.name)
        validator.check("delta_shape", len(delta_shape), "", 0, Rel.EQ, self.name)

    def check_dtype(self, start_dtype, limit_dtype, delta_dtype):
        valid_dtypes = [mstype.int32, mstype.float32]
        inputs = {"start": start_dtype, "limit": limit_dtype, "delta": delta_dtype}
        validator.check_tensors_dtypes_same_and_valid(inputs, valid_dtypes, self.name)

    def infer_value(self, start_value, limit_value, delat_value):
        """Infer the value of input for Range."""
        if start_value is not None and limit_value is not None and delat_value is not None:
            start = np.asscalar(start_value.asnumpy())
            limit = np.asscalar(limit_value.asnumpy())
            delat = np.asscalar(delat_value.asnumpy())
            return Tensor(np.arange(start, limit, delat), dtype=start_value.dtype)
        return None


class MaskedFill(Primitive):
    """
    Fills elements of self tensor with value where mask is True.

    The shapes of `input` and `mask` need to be the same or broadcast.

    Inputs:
        - **input** (Tensor) - The source tensor whose data type is one of float16, float32, int8, int32.
        - **mask** (Tensor[bool]) - The boolean mask.
        - **value** (Union[float, Tensor]) – The value to fill in with, which only supports
          a 0-dimensional tensor or a float number.

    Outputs:
        Tensor, has the same type and shape as `input`.

    Raises:
        TypeError: If `input` or `mask` is not a tensor.
        TypeError: If `value` is neither float number nor tensor.
        TypeError: If dtype of `input` or `value` is not one of float16, float32, int8, int32.
        TypeError: If dtype of `value` is different from that of `input`.
        TypeError: If dtype of `mask` is not bool.
        ValueError: If the shapes of `input` and `mask` could not be broadcast.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input = Tensor(np.array([1., 2., 3., 4.]), mindspore.float32)
        >>> mask = Tensor(np.array([True, True, False, True]), mindspore.bool_)
        >>> output = ops.MaskedFill()(input, mask, 0.5)
        >>> print(output)
        [0.5 0.5 3.  0.5]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['input', 'mask', 'value'], outputs=['output'])


class MaskedSelect(PrimitiveWithCheck):
    """
    Returns a new 1-D Tensor which indexes the input tensor according to the boolean mask.
    The shapes of the mask tensor and the input tensor don’t need to match, but they must be broadcastable.

    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        - **mask** (Tensor[bool]) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        A 1-D Tensor, with the same type as x.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3, 4]), mindspore.int64)
        >>> mask = Tensor(np.array([1, 0, 1, 0]), mindspore.bool_)
        >>> output = ops.MaskedSelect()(x, mask)
        >>> print(output)
        [1 3]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x', 'mask'], outputs=['output'])

    def check_shape(self, x_shape, mask_shape):
        get_broadcast_shape(x_shape, mask_shape, self.name)

    def check_dtype(self, x_dtype, mask_dtype):
        validator.check_tensor_dtype_valid('mask', mask_dtype, [mstype.bool_], self.name)


class SearchSorted(PrimitiveWithInfer):
    """
    Find the indices from the innermost dimension of `sequence` such that the order of the innermost dimension
    within `sequence` would be preserved when the corresponding values in `values` were inserted before the indices.

    Args:
        out_int32 (bool): Output datatype. Optional. If True, the output datatype will be int32;
                          if False, the output datatype will be int64. Default is False.
        right (bool): Search Strategy. Optional. If True, return the last suitable index found.
                      If False, return the first such index. Default is False.

    Inputs:
        - **sequence** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R-1, x_R)` or `(x_1)`.
                                  It must contain monitonically increasing sequence on the innermost dimension.
        - **values** (Tensor) - The shape of tensor is : math:`(x_1, x_2, ..., x_R-1, x_S)`.

    Outputs:
        Tensor containing the indices from the innermost dimension of the input sequence such that,
        if insert the corresponding value in the values tensor, the order of the tensor sequence would be preserved.
        The shape of tensor is :math:`(x_1, x_2, ..., x_R-1, x_S)`,
        whose datatype is int32 if out_int32 is True, otherwise int64, and shape is the same as the shape of values.

    Raises:
        ValueError: If `sequence` and `values` do not have proper shapes.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> sequence = Tensor(np.array([[0, 1, 3, 5, 7], [2, 4, 6, 8, 10]]), mindspore.float32)
        >>> values = Tensor(np.array([[3, 6, 9], [3, 6, 9]]), mindspore.float32)
        >>> output = ops.SearchSorted()(sequence, values)
        >>> print(output)
        [[2, 4, 5]
         [1, 2, 4]]
    """
    @prim_attr_register
    def __init__(self, out_int32=False, right=False):
        """Initialize SearchSorted"""
        self.out_int32 = validator.check_value_type("out_int32", out_int32, [bool], self.name)
        self.right = validator.check_value_type("right", right, [bool], self.name)
        self.init_prim_io_names(inputs=['sequence', 'values'], outputs=['positions'])

    def infer_shape(self, sequence_shape, values_shape):
        if len(sequence_shape) != 1 and sequence_shape[:-1] != values_shape[:-1]:
            raise ValueError(f"For '{self.name}', the 'sequence' should be 1 dimensional or "
                             f"all dimensions except the last dimension of 'sequence' "
                             f"must be the same as all dimensions except the last dimension of 'values'. "
                             f"but got shape of 'sequence': {sequence_shape} "
                             f"and shape of 'values': {values_shape}.")
        return values_shape

    def infer_dtype(self, sequence_dtype, values_dtype):
        args = {"sequence_dtype": sequence_dtype, "values_dtype": values_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        return mstype.tensor_type(mstype.int32) if self.out_int32 else mstype.tensor_type(mstype.int64)


class TensorScatterMax(PrimitiveWithInfer):
    """
    By comparing the value at the position indicated by the index in input_x with the value in the update,
    the value at the index will eventually be equal to the largest one to create a new tensor.

    The last axis of the index is the depth of each index vector. For each index vector,
    there must be a corresponding value in `updates`. The shape of `updates` should be
    equal to the shape of input_x[indices].
    For more details, see use cases.

    Note:
        If some values of the `indices` are out of bound, instead of raising an index error,
        the corresponding `updates` will not be updated to `input_x`.

    Inputs:
        - **input_x** (Tensor) - The target tensor. The dimension of input_x must be no less than indices.shape[-1].
        - **indices** (Tensor) - The index of input tensor whose data type is int32 or int64.
          The rank must be at least 2.
        - **updates** (Tensor) - The tensor to update the input tensor, has the same type as input,
          and updates.shape should be equal to indices.shape[:-1] + input_x.shape[indices.shape[-1]:].

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If dtype of `indices` is neither int32 nor int64.
        ValueError: If length of shape of `input_x` is less than the last dimension of shape of `indices`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [0, 0]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> # Next, demonstrate the approximate operation process of this operator:
        >>> # 1, indices[0] = [0, 0], indices[1] = [0, 0]
        >>> # 2, And input_x[0, 0] = -0.1
        >>> # 3, So input_x[indices] = [-0.1, -0.1]
        >>> # 4, Satisfy the above formula: input_x[indices].shape=(2) == updates.shape=(2)
        >>> op = ops.TensorScatterMax()
        >>> # 5, Perform the max operation for the first time:
        >>> #      first_input_x = Max(input_x[0][0], updates[0]) = [[2.2, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> # 6, Perform the max operation for the second time:
        >>> #      second_input_x = Max(input_x[0][0], updates[0]) = [[2.2, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> output = op(input_x, indices, updates)
        >>> print(output)
        [[ 2.2  0.3  3.6]
         [ 0.4  0.5 -3.2]]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['input_x', 'indices', 'updates'], outputs=['y'])

    def infer_shape(self, input_x_shape, indices_shape, updates_shape):
        if len(indices_shape) < 2:
            raise ValueError(f"For '{self.name}', the dimension of 'indices' cannot be less than 2,"
                             f" but got {len(indices_shape)}.")

        if indices_shape[-1] > len(input_x_shape):
            raise ValueError(f"For '{self.name}', the last dimension of 'indices' must be less than or equal to "
                             f"the dimension of 'input_x', but got the "
                             f"last dimension of 'indices': {indices_shape[-1]} and the dimension of 'input_x': "
                             f"{len(indices_shape)}.")

        updates_shape_check = indices_shape[:-1] + input_x_shape[indices_shape[-1]:]
        if updates_shape_check != updates_shape:
            raise ValueError(f"For '{self.name}', the shape of 'update' must be equal to updates_shape_check, "
                             f"where updates_shape_check = indices_shape[:-1] + input_x_shape[indices_shape[-1]:] "
                             f"but got the shape of 'update': {updates_shape}, "
                             f"updates_shape_check: {updates_shape_check}, indices_shape: {indices_shape} and "
                             f"input_x_shape: {input_x_shape}. Please check input_x_shape and indices_shape.")

        return input_x_shape

    def infer_dtype(self, input_x_dtype, indices_dtype, updates_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32, mstype.int64], self.name)
        args = {"input_x": input_x_dtype, "updates": updates_dtype}
        valid_input_types = (mstype.float16, mstype.float32, mstype.int8, mstype.uint8, mstype.int32)
        validator.check_tensors_dtypes_same_and_valid(args, valid_input_types, self.name)
        return input_x_dtype


class TensorScatterMin(PrimitiveWithInfer):
    """
    By comparing the value at the position indicated by the index in input_x with the value in the `updates`,
    the value at the index will eventually be equal to the smallest one to create a new tensor.

    The last axis of the index is the depth of each index vector. For each index vector,
    there must be a corresponding value in `updates`. The shape of `updates` should be
    equal to the shape of input_x[indices].
    For more details, see use cases.

    Note:
        If some values of the `indices` are out of bound, instead of raising an index error,
        the corresponding `updates` will not be updated to `input_x`.

    Inputs:
        - **input_x** (Tensor) - The target tensor. The dimension of input_x must be no less than indices.shape[-1].
        - **indices** (Tensor) - The index of input tensor whose data type is int32 or int64.
          The rank must be at least 2.
        - **updates** (Tensor) - The tensor to update the input tensor, has the same type as input,
          and updates.shape should be equal to indices.shape[:-1] + input_x.shape[indices.shape[-1]:].

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If dtype of `indices` is neither int32 nor int64.
        ValueError: If length of shape of `input_x` is less than the last dimension of shape of `indices`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [0, 0]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> # Next, demonstrate the approximate operation process of this operator:
        >>> # 1, indices[0] = [0, 0], indices[1] = [0, 0]
        >>> # 2, And input_x[0, 0] = -0.1
        >>> # 3, So input_x[indices] = [-0.1, -0.1]
        >>> # 4, Satisfy the above formula: input_x[indices].shape=(2) == updates.shape=(2)
        >>> op = ops.TensorScatterMin()
        >>> # 5, Perform the min operation for the first time:
        >>> #      first_input_x = Min(input_x[0][0], updates[0]) = [[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> # 6, Perform the min operation for the second time:
        >>> #      second_input_x = Min(input_x[0][0], updates[1]) = [[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> output = op(input_x, indices, updates)
        >>> print(output)
        [[ -0.1  0.3  3.6]
         [ 0.4  0.5 -3.2]]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['input_x', 'indices', 'updates'], outputs=['y'])

    def infer_shape(self, input_x_shape, indices_shape, updates_shape):
        if len(indices_shape) < 2:
            raise ValueError(f"For '{self.name}', the dimension of 'indices' cannot be less than 2,"
                             f" but got {len(indices_shape)}.")

        if indices_shape[-1] > len(input_x_shape):
            raise ValueError(f"For '{self.name}', the last dimension of 'indices' must be less than or equal to "
                             f"the dimension of 'input_x', but got the "
                             f"last dimension of 'indices': {indices_shape[-1]} and the dimension of 'input_x': "
                             f"{len(indices_shape)}.")

        updates_shape_check = indices_shape[:-1] + input_x_shape[indices_shape[-1]:]
        if updates_shape_check != updates_shape:
            raise ValueError(f"For '{self.name}', the shape of 'update' must be equal to updates_shape_check, "
                             f"where updates_shape_check = indices_shape[:-1] + input_x_shape[indices_shape[-1]:] "
                             f"but got the shape of 'update': {updates_shape}, "
                             f"updates_shape_check: {updates_shape_check}, indices_shape: {indices_shape} and "
                             f"input_x_shape: {input_x_shape}. Please check input_x_shape and indices_shape.")

        return input_x_shape

    def infer_dtype(self, input_x_dtype, indices_dtype, updates_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32, mstype.int64], self.name)
        args = {"input_x": input_x_dtype, "updates": updates_dtype}
        valid_input_types = (mstype.float16, mstype.float32, mstype.int8, mstype.uint8, mstype.int32)
        validator.check_tensors_dtypes_same_and_valid(args, valid_input_types, self.name)
        return input_x_dtype


class TensorScatterSub(PrimitiveWithInfer):
    """
    Creates a new tensor by subtracting the values from the positions in `input_x` indicicated by
    `indices`, with values from `updates`. When multiple values are provided for the same
    index, the result of the update will be to subtract these values respectively. This operation is almost
    equivalent to using ScatterNdSub, except that the updates are applied on `Tensor` instead of `Parameter`.

    The last axis of `indices` is the depth of each index vectors. For each index vector,
    there must be a corresponding value in `updates`. The shape of `updates` should be
    equal to the shape of `input_x[indices]`. For more details, see use cases.

    Note:
        If some values of the `indices` are out of bound, instead of raising an index error,
        the corresponding `updates` will not be updated to `input_x`.

    Inputs:
        - **input_x** (Tensor) - The target tensor. The dimension of input_x must be no less than indices.shape[-1].
        - **indices** (Tensor) - The index of input tensor whose data type is int32 or int64.
          The rank must be at least 2.
        - **updates** (Tensor) - The tensor to update the input tensor, has the same type as input,
          and updates.shape should be equal to indices.shape[:-1] + input_x.shape[indices.shape[-1]:].

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If dtype of `indices` is neither int32 nor int64.
        ValueError: If length of shape of `input_x` is less than the last dimension of shape of `indices`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [0, 0]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> # Next, demonstrate the approximate operation process of this operator:
        >>> # 1, indices[0] = [0, 0], indices[1] = [0, 0]
        >>> # 2, And input_x[0, 0] = -0.1
        >>> # 3, So input_x[indices] = [-0.1, -0.1]
        >>> # 4, Satisfy the above formula: input_x[indices].shape=(2) == updates.shape=(2)
        >>> op = ops.TensorScatterSub()
        >>> # 5, Perform the subtract operation for the first time:
        >>> #      first_input_x = input_x[0][0] - updates[0] = [[-1.1, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> # 6, Perform the subtract operation for the second time:
        >>> #      second_input_x = input_x[0][0] - updates[1] = [[-3.3, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> output = op(input_x, indices, updates)
        >>> print(output)
        [[-3.3000002  0.3        3.6      ]
         [ 0.4        0.5       -3.2      ]]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['input_x', 'indices', 'updates'], outputs=['y'])

    def infer_shape(self, input_x_shape, indices_shape, updates_shape):
        if len(indices_shape) < 2:
            raise ValueError(f"For '{self.name}', the dimension of 'indices' cannot be less than 2,"
                             f" but got {len(indices_shape)}.")

        if indices_shape[-1] > len(input_x_shape):
            raise ValueError(f"For '{self.name}', the last dimension of 'indices' must be less than or equal to "
                             f"the dimension of 'input_x', but got the "
                             f"last dimension of 'indices': {indices_shape[-1]} and the dimension of 'input_x': "
                             f"{len(indices_shape)}.")

        updates_shape_check = indices_shape[:-1] + input_x_shape[indices_shape[-1]:]
        if updates_shape_check != updates_shape:
            raise ValueError(f"For '{self.name}', the shape of 'update' must be equal to updates_shape_check, "
                             f"where updates_shape_check = indices_shape[:-1] + input_x_shape[indices_shape[-1]:] "
                             f"but got the shape of 'update': {updates_shape}, "
                             f"updates_shape_check: {updates_shape_check}, indices_shape: {indices_shape} and "
                             f"input_x_shape: {input_x_shape}. Please check input_x_shape and indices_shape.")

        return input_x_shape

    def infer_dtype(self, input_x_dtype, indices_dtype, updates_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32, mstype.int64], self.name)
        args = {"input_x": input_x_dtype, "updates": updates_dtype}
        valid_input_types = (mstype.float16, mstype.float32, mstype.int8, mstype.uint8, mstype.int32)
        validator.check_tensors_dtypes_same_and_valid(args, valid_input_types, self.name)
        return input_x_dtype


class SplitV(Primitive):
    r"""
    Splits the input tensor into num_split tensors along the given dimension.

    The `input_x` tensor will be split into sub-tensors with individual shapes given by `size_splits` along the split
    dimension. This requires that `input_x.shape(split_dim)` is equal to the sum of `size_splits`.

    The shape of `input_x` is :math:`(x_1, x_2, ..., x_M, ..., x_R)`. The rank of `input_x` is `R`. Set the given
    `split_dim` as M, and :math:`-R \le M < R`. Set the given `num_split` as `N`, the given `size_splits` as
    :math:`(x_{m_1}, x_{m_2}, ..., x_{m_N})`, :math:`x_M=\sum_{i=1}^Nx_{m_i}`. The output is a list of tensor objects,
    for the :math:`i`-th tensor, it has the shape of :math:`(x_1, x_2, ..., x_{m_i}, ..., x_R)`. :math:`x_{m_i}` is the
    :math:`M`-th dimension of the :math:`i`-th tensor. Then, the shape of the output tensor is

    .. math::

        ((x_1, x_2, ..., x_{m_1}, ..., x_R), (x_1, x_2, ..., x_{m_2}, ..., x_R), ...,
         (x_1, x_2, ..., x_{m_N}, ..., x_R))

    Args:
        size_splits (Union[tuple, list]): The list containing the sizes of each output tensor along the split
                                          dimension. Must sum to the dimension of value along `split_dim`.
                                          Can contain one -1 indicating that dimension is to be inferred.
        split_dim (int): The dimension along which to split. Must be in the range [-len(input_x.shape),
                         len(input_x.shape)).
        num_split (int): The number of output tensors. Must be positive int.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ...,x_M ..., x_R)`.

    Outputs:
        Tensor, a list of `num_split` Tensor objects with the shape :math:`((x_1, x_2, ..., x_{m_1}, ..., x_R),
        (x_1, x_2, ..., x_{m_2}, ..., x_R), ..., (x_1, x_2, ..., x_{m_N}, ..., x_R))`, :math:`x_M=\sum_{i=1}^Nx_{m_i}`.
        The data type is the same with `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        TypeError: If `size_splits` is not a tuple or a list.
        TypeError: If element of `size_splits` is not an int.
        TypeError: If `split_dim` or `num_split` is not an int.
        ValueError: If rank of the `size_splits` is not equal to `num_split`.
        ValueError: If sum of the `size_splits` is not equal to the dimension of value along `split_dim`.
        ValueError: If `split_dim` is out of the range [-len(input_x.shape), len(input_x.shape)).
        ValueError: If the `num_split` is less than or equal to 0.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.int32)
        >>> op = ops.SplitV(size_splits=[1, -1], split_dim=1, num_split=2)
        >>> output = op(input_x)
        >>> print(output)
        (Tensor(shape=[3, 1], dtype=Int32, value=
        [[1],
         [4],
         [7]]), Tensor(shape=[3, 2], dtype=Int32, value=
        [[2, 3],
         [5, 6],
         [8, 9]]))
        >>> input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.int32)
        >>> op = ops.SplitV(size_splits=[2, 1], split_dim=0, num_split=2)
        >>> output = op(input_x)
        >>> print(output)
        (Tensor(shape=[2, 3], dtype=Int32, value=
        [[1, 2, 3],
         [4, 5, 6]]), Tensor(shape=[1, 3], dtype=Int32, value=
        [[7, 8, 9]]))
    """

    @prim_attr_register
    def __init__(self, size_splits, split_dim, num_split):
        """Initialize SplitV"""
        validator.check_value_type("size_splits", size_splits, [tuple, list], self.name)
        for elements_of_size_splits in size_splits:
            validator.check_value_type("elements of size_splits", elements_of_size_splits, [int], self.name)
            if elements_of_size_splits != -1 and elements_of_size_splits < 1:
                raise ValueError(f"For \'{self.name}\', all elements of size_splits must be positive (except at most "
                                 f"one default value -1), but got: {elements_of_size_splits}.")
        validator.check_value_type("split_dim", split_dim, [int], self.name)
        validator.check_value_type("num_split", num_split, [int], self.name)
        validator.check_positive_int(num_split, "num_split", self.name)
        self.init_prim_io_names(inputs=['input_x'], outputs=['output'])
