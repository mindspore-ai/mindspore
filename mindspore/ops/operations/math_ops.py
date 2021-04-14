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

"""Operators for math."""

import copy

import numpy as np
from ... import context
from .. import signature as sig
from ..._checkparam import Validator as validator
from ..._checkparam import Rel
from ...common import dtype as mstype
from ...common.tensor import Tensor
from ...common._decorator import deprecated
from .._utils import get_broadcast_shape
from ..primitive import PrimitiveWithInfer, PrimitiveWithCheck, prim_attr_register, _run_op


def _infer_shape_reduce(x, axis, keep_dims, prim_name):
    """Common infer for reduce operator"""

    def reduce_one_axis(one_axis):
        validator.check_int_range(one_axis, -dim, dim, Rel.INC_LEFT, 'axis', prim_name)
        if one_axis < 0:
            one_axis += dim
        axis_reduce.add(one_axis)

    validator.check_value_type('axis', axis, [int, tuple, list], prim_name)
    dim = len(x)
    axis_reduce = set()

    if isinstance(axis, int):
        reduce_one_axis(axis)
    else:
        if not axis:
            if keep_dims:
                return [1] * dim
            return []
        for index, one_axis in enumerate(axis):
            validator.check_value_type('axis[%d]' % index, one_axis, [int], prim_name)
            reduce_one_axis(one_axis)

    out_shape = []
    for i in range(dim):
        if i in axis_reduce:
            if keep_dims:
                out_shape.append(1)
        else:
            out_shape.append(x[i])
    return out_shape


class _BinaryOp(PrimitiveWithInfer):
    """
    Define binary operators.
    """

    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        """Initialize _BinaryOp"""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])

    def infer_shape(self, x_shape, y_shape):
        return get_broadcast_shape(x_shape, y_shape, self.name)


class _MathBinaryOp(_BinaryOp):
    """
    Define math binary operators.
    """

    @staticmethod
    def do_infer_dtype(x_dtype, y_dtype, valid_dtype=mstype.number_type, prim_name=None):
        args_type = {"x": x_dtype, "y": y_dtype}
        validator.check_tensors_dtypes_same_and_valid(args_type, valid_dtype, prim_name)
        return x_dtype

    def infer_dtype(self, x_dtype, y_dtype):
        return _MathBinaryOp.do_infer_dtype(x_dtype, y_dtype, mstype.number_type, self.name)


class _BitwiseBinaryOp(_MathBinaryOp):
    """
    Define bitwise binary operators.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize _BitwiseBinaryOp"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])

    @staticmethod
    def _check_bitwise_op_input_type(x1_type, x2_type, prim):
        args = {'x1': x1_type, 'x2': x2_type}
        valid_dtypes = mstype.int_type + mstype.uint_type
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, prim)
        return x1_type

    def infer_dtype(self, x1_type, x2_type):
        return _BitwiseBinaryOp._check_bitwise_op_input_type(x1_type, x2_type, self.name)


class Add(_MathBinaryOp):
    r"""
    Adds two input tensors element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} = x_{i} + y_{i}

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number, or a bool,
          or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number,  or a bool when the first input
          is a tensor, or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `input_x` and `input_y` is not one of the following: Tensor, Number, bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> add = ops.Add()
        >>> input_x = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> input_y = Tensor(np.array([4, 5, 6]).astype(np.float32))
        >>> output = add(input_x, input_y)
        >>> print(output)
        [5. 7. 9.]
    """

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = x + y
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class TensorAdd(_MathBinaryOp):
    """
    Same as operator Add. TensorAdd will be deprecated in the future.
    Please use Add instead.
    """

    @deprecated("1.1", "Add", True)
    @prim_attr_register
    def __init__(self):
        _MathBinaryOp.__init__(self)

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = x + y
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class AssignAdd(PrimitiveWithInfer):
    """
    Updates a `Parameter` by adding a value to it.

    Inputs of `variable` and `value` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    If `value` is a number, the number is automatically converted to Tensor,
    and the data type is consistent with the Tensor data type involved in the operation.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Inputs:
        - **variable** (Parameter) - The `Parameter`.
        - **value** (Union[numbers.Number, Tensor]) - The value to be added to the `variable`.
          It must have the same shape as `variable` if it is a Tensor.

    Raises:
        TypeError: If `value` is neither Number nor Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.AssignAdd = ops.AssignAdd()
        ...         self.variable = mindspore.Parameter(initializer(1, [1], mindspore.int64), name="global_step")
        ...
        ...     def construct(self, x):
        ...         self.AssignAdd(self.variable, x)
        ...         return self.variable
        ...
        >>> net = Net()
        >>> value = Tensor(np.ones([1]).astype(np.int64)*100)
        >>> output = net(value)
        >>> print(output)
        Parameter (name=global_step, shape=(1,), dtype=Int64, requires_grad=True)
    """
    __mindspore_signature__ = (
        sig.make_sig('x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('value', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize AssignAdd"""
        self.init_prim_io_names(inputs=['ref', 'value'], outputs=['output'])
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, variable, value):
        return value

    def infer_dtype(self, variable, value):
        args = {"variable": variable, "value": value}
        validator.check_scalar_or_tensor_types_same(args, mstype.number_type, self.name)
        return value


class AssignSub(PrimitiveWithInfer):
    """
    Updates a `Parameter` by subtracting a value from it.

    Inputs of `variable` and `value` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    If `value` is a number, the number is automatically converted to Tensor,
    and the data type is consistent with the Tensor data type involved in the operation.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Inputs:
        - **variable** (Parameter) - The `Parameter`.
        - **value** (Union[numbers.Number, Tensor]) - The value to be subtracted from the `variable`.
          It must have the same shape as `variable` if it is a Tensor.

    Raises:
        TypeError: If `value` is neither Number nor Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.AssignSub = ops.AssignSub()
        ...         self.variable = mindspore.Parameter(initializer(1, [1], mindspore.int32), name="global_step")
        ...
        ...     def construct(self, x):
        ...         self.AssignSub(self.variable, x)
        ...         return self.variable
        ...
        >>> net = Net()
        >>> value = Tensor(np.ones([1]).astype(np.int32)*100)
        >>> output = net(value)
        >>> print(output)
        Parameter (name=global_step, shape=(1,), dtype=Int32, requires_grad=True)
    """

    __mindspore_signature__ = (
        sig.make_sig('variable', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('value', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize AssignSub"""
        self.init_prim_io_names(inputs=['ref', 'value'], outputs=['output'])
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, variable, value):
        return value

    def infer_dtype(self, variable, value):
        args = {"variable": variable, "value": value}
        validator.check_scalar_or_tensor_types_same(args, mstype.number_type, self.name)
        return value


class _Reduce(PrimitiveWithInfer):
    """
    Definition of base class of reduction class operators.

    Args:
         keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                           If false, don't keep these dimensions.
    """

    __mindspore_signature__ = (
        sig.make_sig('input_x'),
        sig.make_sig('axis', default=())
    )

    @prim_attr_register
    def __init__(self, keep_dims=False):
        """Initialize Reduce"""
        validator.check_value_type('keep_dims', keep_dims, [bool], self.name)
        self.init_prim_io_names(inputs=['input_x', 'axis'], outputs=['y'])

    def __call__(self, x, axis=()):
        args = [x, axis]
        output = _run_op(self, self.name, args)
        return output

    def do_infer(self, input_x, axis, valid_dtype=mstype.number_type):
        """ return meta infos of input parameters """
        axis_v = axis['value']
        input_shp = input_x['shape']
        args = {'input_x': input_x['dtype']}
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtype, self.name)

        if axis_v is None:
            raise ValueError(f"For {self.name}, axis must be const.")
        input_shp = _infer_shape_reduce(input_shp, axis_v, self.keep_dims, self.name)
        value = None
        if input_x['value'] is not None:
            prim_map = {
                'ReduceSum': np.sum,
                'ReduceMax': np.max,
                'ReduceMin': np.min,
            }
            np_reduce_func = prim_map.get(self.name, None)

            if np_reduce_func is not None:
                value = input_x['value'].asnumpy()
                if not axis_v:
                    axis_v = [i for i in range(len(input_x['shape']))]
                    axis_v = tuple(axis_v)
                value = np_reduce_func(value, axis_v, keepdims=self.keep_dims)
                value = np.array(value)
                value = Tensor(value)
        if 'max_shape' and 'min_shape' in input_x:
            output_max_shape = _infer_shape_reduce(input_x['max_shape'], axis_v, self.keep_dims, self.name)
            output_min_shape = _infer_shape_reduce(input_x['min_shape'], axis_v, self.keep_dims, self.name)
        else:
            output_max_shape = input_shp
            output_min_shape = input_shp

        return {'shape': input_shp,
                'min_shape': output_min_shape,
                'max_shape': output_max_shape,
                'dtype': input_x['dtype'],
                'value': value}

    def __infer__(self, input_x, axis):
        return self.do_infer(input_x, axis)


class ReduceMean(_Reduce):
    """
    Reduces a dimension of a tensor by averaging all elements in the dimension.

    The dtype of the tensor to be reduced is number.

    Args:
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions. Default: False.

    Inputs:
        - **input_x** (Tensor[Number]) - The input tensor.
        - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed. Must be in the range [-rank(input_x), rank(input_x)).

    Outputs:
        Tensor, has the same dtype as the `input_x`.

        - If axis is (), and keep_dims is False,
          the output is a 0-D tensor representing the mean of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `input_x` is not a Tensor.
        ValueError: If `axis` is not one of the following: int, tuple or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = ops.ReduceMean(keep_dims=True)
        >>> output = op(input_x, 1)
        >>> result = output.shape
        >>> print(result)
        (3, 1, 5, 6)
    """


class ReduceSum(_Reduce):
    """
    Reduces a dimension of a tensor by summing all elements in the dimension.

    The dtype of the tensor to be reduced is number.

    Args:
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions. Default: False.

    Inputs:
         - **input_x** (Tensor[Number]) - The input tensor.
         - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
           Only constant value is allowed. Must be in the range [-rank(input_x), rank(input_x)).

    Outputs:
        Tensor, has the same dtype as the `input_x`.

        - If axis is (), and keep_dims is False,
          the output is a 0-D tensor representing the sum of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `input_x` is not a Tensor.
        ValueError: If `axis` is None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = ops.ReduceSum(keep_dims=True)
        >>> output = op(input_x, 1)
        >>> output.shape
        (3, 1, 5, 6)
    """

    @prim_attr_register
    def __init__(self, keep_dims=False):
        """Initialize ReduceSum"""
        super(ReduceSum, self).__init__(keep_dims)
        self.__setattr_flag__ = True


class ReduceAll(_Reduce):
    """
    Reduces a dimension of a tensor by the "logicalAND" of all elements in the dimension.

    The dtype of the tensor to be reduced is bool.

    Args:
       keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                         If false, don't keep these dimensions.
                         Default : False, don't keep these reduced dimensions.

    Inputs:
        - **input_x** (Tensor[bool]) - The input tensor.
        - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed. Must be in the range [-rank(input_x), rank(input_x)).

    Outputs:
        Tensor, the dtype is bool.

        - If axis is (), and keep_dims is False,
          the output is a 0-D tensor representing the "logical and" of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `input_x` is not a Tensor.
        ValueError: If `axis` is not one of the following: int, tuple or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[True, False], [True, True]]))
        >>> op = ops.ReduceAll(keep_dims=True)
        >>> output = op(input_x, 1)
        >>> print(output)
        [[False]
         [ True]]
    """

    def __infer__(self, input_x, axis):
        return self.do_infer(input_x, axis, (mstype.bool_,))


class ReduceAny(_Reduce):
    """
    Reduces a dimension of a tensor by the "logical OR" of all elements in the dimension.

    The dtype of the tensor to be reduced is bool.

    Args:
       keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                         If false, don't keep these dimensions.
                         Default : False, don't keep these reduced dimensions.

    Inputs:
        - **input_x** (Tensor[bool]) - The input tensor.
        - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed. Must be in the range [-rank(input_x), rank(input_x)).

    Outputs:
        Tensor, the dtype is bool.

        - If axis is (), and keep_dims is False,
          the output is a 0-D tensor representing the "logical or" of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `input_x` is not a Tensor.
        ValueError: If `axis` is not one of the following: int, tuple or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[True, False], [True, True]]))
        >>> op = ops.ReduceAny(keep_dims=True)
        >>> output = op(input_x, 1)
        >>> print(output)
        [[ True]
         [ True]]
    """

    def __infer__(self, input_x, axis):
        return self.do_infer(input_x, axis, (mstype.bool_,))


class ReduceMax(_Reduce):
    """
    Reduces a dimension of a tensor by the maximum value in this dimension.

    The dtype of the tensor to be reduced is number.

    Args:
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions.
                          Default : False, don't keep these reduced dimensions.

    Inputs:
         - **input_x** (Tensor[Number]) - The input tensor.
         - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
           Only constant value is allowed. Must be in the range [-rank(input_x), rank(input_x)).

    Outputs:
        Tensor, has the same dtype as the `input_x`.

        - If axis is (), and keep_dims is False,
          the output is a 0-D tensor representing the maximum of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `input_x` is not a Tensor.
        ValueError: If `axis` is not one of the following: int, tuple or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = ops.ReduceMax(keep_dims=True)
        >>> output = op(input_x, 1)
        >>> result = output.shape
        >>> print(result)
        (3, 1, 5, 6)
    """

    @prim_attr_register
    def __init__(self, keep_dims=False):
        """ReduceMax"""
        super(ReduceMax, self).__init__(keep_dims)
        self.__setattr_flag__ = True

    def __infer__(self, input_x, axis):
        return self.do_infer(input_x, axis, mstype.number_type + (mstype.bool_,))


class ReduceMin(_Reduce):
    """
    Reduces a dimension of a tensor by the minimum value in the dimension.

    The dtype of the tensor to be reduced is number.

    Args:
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions.
                          Default : False, don't keep these reduced dimensions.

    Inputs:
        - **input_x** (Tensor[Number]) - The input tensor.
        - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed. Must be in the range [-rank(input_x), rank(input_x)).

    Outputs:
        Tensor, has the same dtype as the `input_x`.

        - If axis is (), and keep_dims is False,
          the output is a 0-D tensor representing the minimum of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `input_x` is not a Tensor.
        ValueError: If `axis` is not one of the following: int, tuple or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = ops.ReduceMin(keep_dims=True)
        >>> output = op(input_x, 1)
        >>> result = output.shape
        >>> print(result)
        (3, 1, 5, 6)
    """


class ReduceProd(_Reduce):
    """
    Reduces a dimension of a tensor by multiplying all elements in the dimension.

    The dtype of the tensor to be reduced is number.

    Args:
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions.
                          Default : False, don't keep these reduced dimensions.

    Inputs:
        - **input_x** (Tensor[Number]) - The input tensor.
        - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed. Must be in the range [-rank(input_x), rank(input_x)).

    Outputs:
        Tensor, has the same dtype as the `input_x`.

        - If axis is (), and keep_dims is False,
          the output is a 0-D tensor representing the product of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `input_x` is not a Tensor.
        ValueError: If `axis` is not one of the following: int, tuple or list.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = ops.ReduceProd(keep_dims=True)
        >>> output = op(input_x, 1)
        >>> result = output.shape
        >>> print(result)
        (3, 1, 5, 6)
    """


class CumProd(PrimitiveWithInfer):
    """
    Computes the cumulative product of the tensor x along axis.

    Args:
        exclusive (bool): If true, perform exclusive cumulative product. Default: False.
        reverse (bool): If true, reverse the result along axis. Default: False

    Inputs:
        - **input_x** (Tensor[Number]) - The input tensor.
        - **axis** (int) - The dimensions to compute the cumulative product.
          Only constant value is allowed.

    Outputs:
        Tensor, has the same shape and dtype as the `input_x`.

    Raises:
        TypeError: If `exclusive` or `reverse` is not a bool.
        ValueError: If `axis` is None.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> a, b, c, = 1, 2, 3
        >>> input_x = Tensor(np.array([a, b, c]).astype(np.float32))
        >>> op0 = ops.CumProd()
        >>> output0 = op0(input_x, 0) # output=[a, a * b, a * b * c]
        >>> op1 = ops.CumProd(exclusive=True)
        >>> output1 = op1(input_x, 0) # output=[1, a, a * b]
        >>> op2 = ops.CumProd(reverse=True)
        >>> output2 = op2(input_x, 0) # output=[a * b * c, b * c, c]
        >>> op3 = ops.CumProd(exclusive=True, reverse=True)
        >>> output3 = op3(input_x, 0) # output=[b * c, c, 1]
        >>> print(output0)
        [1. 2. 6.]
        >>> print(output1)
        [1. 1. 2.]
        >>> print(output2)
        [6. 6. 3.]
        >>> print(output3)
        [6. 3. 1.]
    """

    @prim_attr_register
    def __init__(self, exclusive=False, reverse=False):
        cls_name = self.name
        self.exclusive = validator.check_value_type("exclusive", exclusive, [bool], cls_name)
        self.reverse = validator.check_value_type("reverse", reverse, [bool], cls_name)
        self.init_prim_io_names(inputs=['x', 'axis'], outputs=['y'])

    def infer_shape(self, x_shape, axis_shape):
        return x_shape

    def infer_dtype(self, x_type, axis_type):
        cls_name = self.name
        validator.check_tensor_dtype_valid('x', x_type, mstype.number_type, cls_name)
        validator.check_subclass("axis", axis_type, mstype.int_, cls_name)
        return x_type

    def infer_value(self, x, axis):
        if axis is None:
            raise ValueError(f"For {self.name}, axis must be const.")


class MatMul(PrimitiveWithCheck):
    r"""
    Multiplies matrix `a` and matrix `b`.

    The rank of input tensors must equal to `2`.

    Args:
        transpose_a (bool): If true, `a` is transposed before multiplication. Default: False.
        transpose_b (bool): If true, `b` is transposed before multiplication. Default: False.

    Inputs:
        - **input_x** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(N, C)`. If
          `transpose_a` is True, its shape must be :math:`(N, C)` after transpose.
        - **input_y** (Tensor) - The second tensor to be multiplied. The shape of the tensor is :math:`(C, M)`. If
          `transpose_b` is True, its shape must be :math:`(C, M)` after transpose.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, M)`.

    Raises:
        TypeError: If `transpose_a` or `transpose_b` is not a bool.
        ValueError: If the column of matrix dimensions of `input_x` is not equal to
                    the row of matrix dimensions of `input_y`.
        ValueError: If length of shape of `input_x` or `input_y` is not equal to 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x1 = Tensor(np.ones(shape=[1, 3]), mindspore.float32)
        >>> input_x2 = Tensor(np.ones(shape=[3, 4]), mindspore.float32)
        >>> matmul = ops.MatMul()
        >>> output = matmul(input_x1, input_x2)
        >>> print(output)
        [[3. 3. 3. 3.]]
    """

    @prim_attr_register
    def __init__(self, transpose_a=False, transpose_b=False):
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['output'])
        cls_name = self.name
        validator.check_value_type("transpose_a", transpose_a, [bool], cls_name)
        validator.check_value_type("transpose_b", transpose_b, [bool], cls_name)

    def check_shape_size(self, x1, x2):
        if len(x1) != 2 or len(x2) != 2:
            raise ValueError('P.MatMul inputs x1, x2 should have the same dimension size and '
                             + f'equal to 2, while x1 size is ({len(x1)}) and x2 size is ({len(x2)}).')

    def check_shape(self, x1, x2):
        self.check_shape_size(x1, x2)
        cls_name = self.name
        # expected dimension of x, y, x:[...,a,b] y:[..., c,d], the dim size should be the same except the last two
        for i in range(len(x1) - 2):
            if x1[i] != x2[i]:
                raise ValueError(f'For \'{cls_name}\' shape in dim[{i}] not the same, '
                                 + f'while x1 is {x1[i]}, x2 is {x2[i]}')

        # validate whether last two dims satisfying matrix multiply
        x1_last = x1[-2:]
        x2_last = x2[-2:]
        x1_col = x1_last[not self.transpose_a]
        x2_row = x2_last[self.transpose_b]
        if np.all(np.array(x1) != -1) and np.all(np.array(x2) != -1):
            if x1_col != x2_row:
                raise ValueError(f'For \'{cls_name}\' evaluator shapes of inputs can not do this operator,'
                                 + f' got {x1_col} and {x2_row}, with x1 shape {x1}(transpose_a={self.transpose_a})'
                                 + f', x2 shape {x2}(transpose_b={self.transpose_b}).')
        # set attribute
        self.add_prim_attr('transpose_x1', self.transpose_a)
        self.add_prim_attr('transpose_x2', self.transpose_b)

    def check_dtype(self, x1, x2):
        args = {"x1": x1, "x2": x2}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.float_type + mstype.int_type, self.name)


class BatchMatMul(MatMul):
    r"""
    Computes matrix multiplication between two tensors by batch.

    .. math::

        \text{output}[..., :, :] = \text{matrix}(a[..., :, :]) * \text{matrix}(b[..., :, :])

    The two input tensors must have the same rank and the rank must be not less than `3`.

    Args:
        transpose_a (bool): If true, the last two dimensions of `a` is transposed before multiplication.
            Default: False.
        transpose_b (bool): If true, the last two dimensions of `b` is transposed before multiplication.
            Default: False.

    Inputs:
        - **input_x** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(*B, N, C)`,
          where :math:`*B` represents the batch size which can be multidimensional, :math:`N` and :math:`C` are the
          size of the last two dimensions. If `transpose_a` is True, its shape must be :math:`(*B, C, N)`.
        - **input_y** (Tensor) - The second tensor to be multiplied. The shape of the tensor is :math:`(*B, C, M)`. If
          `transpose_b` is True, its shape must be :math:`(*B, M, C)`.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(*B, N, M)`.

    Raises:
        TypeError: If `transpose_a` or `transpose_b` is not a bool.
        ValueError: If length of shape of `input_x` is not equal to length of shape of `input_y` or
                    length of shape of `input_x` is less than 3.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.ones(shape=[2, 4, 1, 3]), mindspore.float32)
        >>> input_y = Tensor(np.ones(shape=[2, 4, 3, 4]), mindspore.float32)
        >>> batmatmul = ops.BatchMatMul()
        >>> output = batmatmul(input_x, input_y)
        >>> print(output)
        [[[[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]]
         [[[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]]]
        >>> input_x = Tensor(np.ones(shape=[2, 4, 3, 1]), mindspore.float32)
        >>> input_y = Tensor(np.ones(shape=[2, 4, 3, 4]), mindspore.float32)
        >>> batmatmul = ops.BatchMatMul(transpose_a=True)
        >>> output = batmatmul(input_x, input_y)
        >>> print(output)
        [[[[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]]
         [[[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]]]
    """

    @prim_attr_register
    def __init__(self, transpose_a=False, transpose_b=False):
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['output'])
        cls_name = self.name
        validator.check_value_type("transpose_a", transpose_a, [bool], cls_name)
        validator.check_value_type("transpose_b", transpose_b, [bool], cls_name)

    def check_shape_size(self, x, y):
        if len(x) != len(y) or len(x) < 3:
            raise ValueError('For \'BatchMatMul\' input x, y should be the same dimension size and should be '
                             'greater or equal to 3,' + f' while x size = {len(x)}, y size= {len(y)}')


class CumSum(PrimitiveWithInfer):
    """
    Computes the cumulative sum of input tensor along axis.

    .. math::

        y_i = x_1 + x_2 + x_3 + ... + x_i

    Args:
        exclusive (bool): If true, perform exclusive mode. Default: False.
        reverse (bool): If true, perform inverse cumulative sum. Default: False.

    Inputs:
        - **input** (Tensor) - The input tensor to accumulate.
        - **axis**  (int) - The axis to accumulate the tensor's value. Only constant value is allowed.
          Must be in the range [-rank(input), rank(input)).

    Outputs:
        Tensor, the shape of the output tensor is consistent with the input tensor's.

    Raises:
        TypeError: If `exclusive` or `reverse` is not a bool.
        TypeError: If `axis` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input = Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float32))
        >>> cumsum = ops.CumSum()
        >>> output = cumsum(input, 1)
        >>> print(output)
        [[ 3.  7. 13. 23.]
         [ 1.  7. 14. 23.]
         [ 4.  7. 15. 22.]
         [ 1.  4. 11. 20.]]
    """

    @prim_attr_register
    def __init__(self, exclusive=False, reverse=False):
        """Initialize cumsum"""
        cls_name = self.name
        validator.check_value_type('exclusive', exclusive, [bool], cls_name)
        validator.check_value_type('reverse', reverse, [bool], cls_name)
        self.init_prim_io_names(inputs=['x', 'axis'], outputs=['y'])

    def __infer__(self, x, axis):
        cls_name = self.name
        x_shp = x['shape']
        if axis['value'] is None:
            raise ValueError(f"For {self.name}, axis must be const.")
        validator.check_value_type('axis', axis['value'], [int], cls_name)
        valid_dtypes = [mstype.uint8, mstype.int8, mstype.int32, mstype.float16, mstype.float32, mstype.float64]
        validator.check_tensor_dtype_valid('x', x['dtype'], valid_dtypes, cls_name)
        return {'shape': x_shp,
                'dtype': x['dtype'],
                'value': None}


class AddN(PrimitiveWithInfer):
    """
    Computes addition of all input tensors element-wise.

    All input tensors must have the same shape.

    Inputs:
        - **input_x** (Union(tuple[Tensor], list[Tensor])) - The input tuple or list
          is made up of multiple tensors whose dtype is number or bool to be added together.

    Outputs:
        Tensor, has the same shape and dtype as each entry of the `input_x`.

    Raises:
        TypeError: If `input_x` is neither tuple nor list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> class NetAddN(nn.Cell):
        ...     def __init__(self):
        ...         super(NetAddN, self).__init__()
        ...         self.addN = ops.AddN()
        ...
        ...     def construct(self, *z):
        ...         return self.addN(z)
        ...
        >>> net = NetAddN()
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> input_y = Tensor(np.array([4, 5, 6]), mindspore.float32)
        >>> output = net(input_x, input_y, input_x, input_y)
        >>> print(output)
        [10. 14. 18.]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=["inputs"], outputs=["sum"])

    def check_elim(self, inputs):
        if len(inputs) != 1:
            return (False, None)
        if isinstance(inputs[0], Tensor):
            return (True, inputs[0])
        raise TypeError("Expecting Tensor, got : {}".format(type(inputs[0])))

    def infer_shape(self, inputs):
        cls_name = self.name
        validator.check_int(len(inputs), 1, Rel.GE, "inputs", cls_name)
        self.add_prim_attr('n', len(inputs))
        shp0 = inputs[0]
        for i, shp in enumerate(inputs):
            validator.check(f"shape of inputs[{i}]", shp, 'shape of inputs[0]', shp0, Rel.EQ, cls_name)
        return shp0

    def infer_dtype(self, inputs):
        cls_name = self.name
        validator.check_value_type("inputs", inputs, [tuple, list], cls_name)
        validator.check_int(len(inputs), 1, Rel.GE, "inputs", cls_name)
        args = {}
        contains_undetermined = False
        for i, dtype in enumerate(inputs):
            args[f"inputs[{i}]"] = dtype
            if dtype == mstype.undetermined:
                contains_undetermined = True
        if not contains_undetermined:
            validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type + (mstype.bool_,), cls_name)
        return inputs[0]

    def infer_value(self, inputs):
        if inputs is None:
            return None

        for x in inputs:
            if x is None:
                return None

        added = copy.deepcopy(inputs[0].asnumpy())
        for x in inputs[1:]:
            added += x.asnumpy()
        out = np.array(added, inputs[0].asnumpy().dtype)
        return Tensor(out)


class AccumulateNV2(PrimitiveWithInfer):
    """
    Computes accumulation of all input tensors element-wise.

    AccumulateNV2 is similar to AddN, but there is a significant difference
    among them: AccumulateNV2 will not wait for all of its inputs to be ready
    before summing. That is to say, AccumulateNV2 is able to save
    memory when inputs are ready at different time since the minimum temporary
    storage is proportional to the output size rather than the input size.

    Inputs:
        - **input_x** (Union(tuple[Tensor], list[Tensor])) - The input tuple or list
          is made up of multiple tensors whose dtype is number to be added together.

    Outputs:
        Tensor, has the same shape and dtype as each entry of the `input_x`.

    Raises:
        TypeError: If `input_x` is neither tuple nor list.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> class NetAccumulateNV2(nn.Cell):
        ...     def __init__(self):
        ...         super(NetAccumulateNV2, self).__init__()
        ...         self.accumulateNV2 = ops.AccumulateNV2()
        ...
        ...     def construct(self, *z):
        ...         return self.accumulateNV2(z)
        ...
        >>> net = NetAccumulateNV2()
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> input_y = Tensor(np.array([4, 5, 6]), mindspore.float32)
        >>> output = net(input_x, input_y, input_x, input_y)
        >>> print(output)
        [10. 14. 18.]
    """

    @prim_attr_register
    def __init__(self):
        self.__setattr_flag__ = True
        self.init_prim_io_names(inputs=["inputs"], outputs=["sum"])

    def check_elim(self, inputs):
        if len(inputs) != 1:
            return (False, None)
        if isinstance(inputs[0], Tensor):
            return (True, inputs[0])
        raise TypeError("Expecting Tensor, got : {}".format(type(inputs[0])))

    def infer_shape(self, inputs):
        cls_name = self.name
        validator.check_int(len(inputs), 1, Rel.GE, "inputs", cls_name)
        self.add_prim_attr('n', len(inputs))
        shp0 = inputs[0]
        for i, shp in enumerate(inputs):
            validator.check(f"shape of inputs[{i}]", shp, 'shape of inputs[0]', shp0, Rel.EQ, cls_name)
        return shp0

    def infer_dtype(self, inputs):
        cls_name = self.name
        validator.check_value_type("inputs", inputs, [tuple, list], cls_name)
        validator.check_int(len(inputs), 1, Rel.GE, "inputs", cls_name)
        args = {}
        for i, dtype in enumerate(inputs):
            args[f"inputs[{i}]"] = dtype
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type + (mstype.bool_,), cls_name)
        return inputs[0]


class Neg(PrimitiveWithInfer):
    """
    Returns a tensor with negative values of the input tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor whose dtype is number.

    Outputs:
        Tensor, has the same shape and dtype as input.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> neg = ops.Neg()
        >>> input_x = Tensor(np.array([1, 2, -1, 2, 0, -3.5]), mindspore.float32)
        >>> output = neg(input_x)
        >>> print(output)
        [-1.  -2.   1.  -2.   0.   3.5]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Neg"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x", x_dtype, mstype.number_type, self.name)
        return x_dtype

    def infer_value(self, input_x):
        if input_x is not None:
            input_x = input_x.asnumpy()
            out = np.array(-input_x, input_x.dtype)
            return Tensor(out)

        return None


class InplaceAdd(PrimitiveWithInfer):
    """
    Adds v into specified rows of x. Computes y = x; y[i,] += v.

    Args:
        indices (Union[int, tuple]): Indices into the left-most dimension of x, and determines which rows of x
            to add with v. It is an integer or a tuple, whose value is in [0, the first dimension size of x).

    Inputs:
        - **input_x** (Tensor) - The first input is a tensor whose data type is float16, float32 or int32.
        - **input_v** (Tensor) - The second input is a tensor that has the same dimension sizes as x except
          the first dimension, which must be the same as indices's size. It has the same data type with `input_x`.

    Outputs:
        Tensor, has the same shape and dtype as input_x.

    Raises:
        TypeError: If `indices` is neither int nor tuple.
        TypeError: If `indices` is a tuple whose elements are not all int.
        ValueError: If length of shape of `input_x` is not equal to length of shape of `input_v`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> indices = (0, 1)
        >>> input_x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]), mindspore.float32)
        >>> input_v = Tensor(np.array([[0.5, 1.0], [1.0, 1.5]]), mindspore.float32)
        >>> inplaceAdd = ops.InplaceAdd(indices)
        >>> output = inplaceAdd(input_x, input_v)
        >>> print(output)
        [[1.5 3. ]
         [4.  5.5]
         [5.  6. ]]
    """

    @prim_attr_register
    def __init__(self, indices):
        """Initialize InplaceAdd"""
        self.init_prim_io_names(inputs=['x', 'v'], outputs=['y'])
        self.indices = indices
        validator.check_value_type('indices', indices, [tuple, int], self.name)
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
                raise ValueError(f'The value of indices must be in [0, {x_shape[0]}), but got {i}.')
        x_rank = len(x_shape)
        for idx in range(x_rank)[1:]:
            validator.check('v dim %d' % idx, v_shape[idx], "x dim %d" % idx, x_shape[idx], Rel.EQ, self.name)

        return x_shape


class InplaceSub(PrimitiveWithInfer):
    """
    Subtracts v into specified rows of x. Computes y = x; y[i, :] -= v.

    Args:
        indices (Union[int, tuple]): Indices into the left-most dimension of x, and determines which rows of x
            to subtract with v. It is a int or tuple, whose value is in [0, the first dimension size of x).

    Inputs:
        - **input_x** (Tensor) - The first input is a tensor whose data type is float16, float32 or int32.
        - **input_v** (Tensor) - The second input is a tensor who has the same dimension sizes as x except
          the first dimension, which must be the same as indices's size. It has the same data type with `input_x`.

    Outputs:
        Tensor, has the same shape and dtype as input_x.

    Raises:
        TypeError: If `indices` is neither int nor tuple.
        TypeError: If `indices` is a tuple whose elements are not all int.
        ValueError: If length of shape of `input_x` is not equal to length of shape of `input_v`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> indices = (0, 1)
        >>> input_x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]), mindspore.float32)
        >>> input_v = Tensor(np.array([[0.5, 1.0], [1.0, 1.5]]), mindspore.float32)
        >>> inplaceSub = ops.InplaceSub(indices)
        >>> output = inplaceSub(input_x, input_v)
        >>> print(output)
        [[0.5 1. ]
         [2.  2.5]
         [5.  6. ]]
    """

    @prim_attr_register
    def __init__(self, indices):
        """Initialize InplaceSub"""
        self.init_prim_io_names(inputs=['x', 'v'], outputs=['y'])
        self.indices = indices
        validator.check_value_type('indices', indices, [tuple, int], self.name)
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
                raise ValueError(f'The value of indices must be in [0, {x_shape[0]}), but got {i}.')
        x_rank = len(x_shape)
        for idx in range(x_rank)[1:]:
            validator.check('v dim %d' % idx, v_shape[idx], "x dim %d" % idx, x_shape[idx], Rel.EQ, self.name)

        return x_shape


class Sub(_MathBinaryOp):
    """
    Subtracts the second input tensor from the first input tensor element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number, or a bool,
          or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number, or a bool when the first input
          is a tensor, or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `input_x` and `input_y` is not a Number or a bool or a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> input_y = Tensor(np.array([4, 5, 6]), mindspore.int32)
        >>> sub = ops.Sub()
        >>> output = sub(input_x, input_y)
        >>> print(output)
        [-3 -3 -3]
    """

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = x - y
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class Mul(_MathBinaryOp):
    """
    Multiplies two tensors element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} = x_{i} * y_{i}

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `input_x` and `input_y` is not one of the following: Tensor, Number, bool.
        ValueError: If `input_x` and `input_y` are not the same shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> input_y = Tensor(np.array([4.0, 5.0, 6.0]), mindspore.float32)
        >>> mul = ops.Mul()
        >>> output = mul(input_x, input_y)
        >>> print(output)
        [ 4. 10. 18.]
    """

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = x * y
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class SquaredDifference(_MathBinaryOp):
    """
    Subtracts the second input tensor from the first input tensor element-wise and returns square of it.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number, or a bool,
          or a tensor whose data type is float16, float32, int32 or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number, or a bool when the first input
          is a tensor or a tensor whose data type is float16, float32, int32 or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: if `input_x` and `input_y` is not a Number or a bool or a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> input_y = Tensor(np.array([2.0, 4.0, 6.0]), mindspore.float32)
        >>> squared_difference = ops.SquaredDifference()
        >>> output = squared_difference(input_x, input_y)
        >>> print(output)
        [1. 4. 9.]
    """

    def infer_dtype(self, x_dtype, y_dtype):
        valid_type = [mstype.float16, mstype.float32, mstype.int32]
        return _MathBinaryOp.do_infer_dtype(x_dtype, y_dtype, valid_type, self.name)


class Square(PrimitiveWithCheck):
    """
    Returns square of a tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor whose dtype is number.

    Outputs:
        Tensor, has the same shape and dtype as the `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> square = ops.Square()
        >>> output = square(input_x)
        >>> print(output)
        [1. 4. 9.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Square"""
        self.init_prim_io_names(inputs=['input_x'], outputs=['output'])

    def __check__(self, x):
        x_dtype = x["dtype"]
        validator.check_tensor_dtype_valid("x", x_dtype, mstype.number_type, self.name)

    def infer_value(self, x):
        if x is not None:
            x = x.asnumpy()
            out = x * x
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class Rsqrt(PrimitiveWithInfer):
    """
    Computes reciprocal of square root of input tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input of Rsqrt. Each element must be a non-negative number.

    Outputs:
        Tensor, has the same type and shape as `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> input_tensor = Tensor([[4, 4], [9, 9]], mindspore.float32)
        >>> rsqrt = ops.Rsqrt()
        >>> output = rsqrt(input_tensor)
        >>> print(output)
        [[0.5        0.5       ]
         [0.33333334 0.33333334]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Rsqrt"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x", x_dtype, mstype.number_type, self.name)
        return x_dtype

    def infer_value(self, x):
        if x is not None:
            x = x.asnumpy()
            out = 1.0 / np.sqrt(x)
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class Sqrt(PrimitiveWithCheck):
    """
    Returns square root of a tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor whose dtype is number.

    Outputs:
        Tensor, has the same shape as the `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1.0, 4.0, 9.0]), mindspore.float32)
        >>> sqrt = ops.Sqrt()
        >>> output = sqrt(input_x)
        >>> print(output)
        [1. 2. 3.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Sqrt"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def check_dtype(self, x_type):
        validator.check_tensor_dtype_valid("x", x_type, mstype.number_type, self.name)

    def infer_value(self, x):
        if x is not None:
            x = x.asnumpy()
            out = np.sqrt(x)
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class Reciprocal(PrimitiveWithInfer):
    """
    Returns reciprocal of a tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape as the `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> reciprocal = ops.Reciprocal()
        >>> output = reciprocal(input_x)
        >>> print(output)
        [1.   0.5  0.25]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Reciprocal"""
        if context.get_context("device_target") == "GPU":
            self.target = "GPU"
        else:
            self.target = "OTHER"
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x):
        return x

    def infer_dtype(self, x):
        validator.check_subclass("x", x, mstype.tensor, self.name)
        return x

    def infer_value(self, x):
        if x is not None:
            x = x.asnumpy()
            out = 1.0 / x
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class Pow(_MathBinaryOp):
    """
    Computes a tensor to the power of the second input.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `input_x` and `input_y` is not one of the following: Tensor, Number, bool.
        ValueError: If `input_x` and `input_y` are not the same shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> input_y = 3.0
        >>> pow = ops.Pow()
        >>> output = pow(input_x, input_y)
        >>> print(output)
        [ 1.  8. 64.]
        >>>
        >>> input_x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> input_y = Tensor(np.array([2.0, 4.0, 3.0]), mindspore.float32)
        >>> pow = ops.Pow()
        >>> output = pow(input_x, input_y)
        >>> print(output)
        [ 1. 16. 64.]
    """

    def infer_value(self, x, power):
        if x is not None and power is not None:
            x = x.asnumpy()
            power = power.asnumpy()
            out = np.power(x, power)
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class Exp(PrimitiveWithInfer):
    r"""
    Returns exponential of a tensor element-wise.

    .. math::

        out_i = e^{x_i}

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape and dtype as the `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> exp = ops.Exp()
        >>> output = exp(input_x)
        >>> print(output)
        [ 2.718282  7.389056 54.598152]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Exp"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        validator.check_subclass("x", x_type, mstype.tensor, self.name)
        return x_type

    def infer_value(self, x):
        if x is not None:
            x = x.asnumpy()
            out = np.exp(x)
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class Expm1(PrimitiveWithInfer):
    r"""
    Returns exponential then minus 1 of a tensor element-wise.

    .. math::

        out_i = e^{x_i} - 1

    Inputs:
        - **input_x** (Tensor) - The input tensor. With float16 or float32 data type.

    Outputs:
        Tensor, has the same shape as the `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([0.0, 1.0, 2.0, 4.0]), mindspore.float32)
        >>> expm1 = ops.Expm1()
        >>> output = expm1(input_x)
        >>> print(output)
        [ 0.        1.718282  6.389056 53.598152]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Exp"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        validator.check_tensor_dtype_valid("x", x_type, [mstype.float16, mstype.float32], self.name)
        return x_type


class HistogramFixedWidth(PrimitiveWithInfer):
    """
    Returns a rank 1 histogram counting the number of entries in values that fall into every bin. The bins are equal
    width and determined by the arguments range and nbins.

    Args:
        dtype (str): An optional attribute. Must be one of the following types: "int32", "int64". Default: "int32".
        nbins (int): The number of histogram bins, the type is a positive integer.

    Inputs:
        - **x** (Tensor) - Numeric Tensor. Must be one of the following types: int32, float32, float16.
        - **range** (Tensor) - Must has the same data type as `x`, and the shape is [2].
          x <= range[0] will be mapped to hist[0], x >= range[1] will be mapped to hist[-1].

    Outputs:
        Tensor, the type is int32.

    Raises:
        TypeError: If `dtype` is not a str or `nbins` is not an int.
        ValueError: If `nbins` is less than 1.
        ValueError: If `dtype` is neither 'int32' nor 'int64'.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor([-1.0, 0.0, 1.5, 2.0, 5.0, 15], mindspore.float16)
        >>> range = Tensor([0.0, 5.0], mindspore.float16)
        >>> hist = ops.HistogramFixedWidth(5)
        >>> output = hist(x, range)
        >>> print(output)
        [2 1 1 0 2]
    """

    @prim_attr_register
    def __init__(self, nbins, dtype='int32'):
        self.nbins = validator.check_value_type("nbins", nbins, [int], self.name)
        validator.check_int(nbins, 1, Rel.GE, "nbins", self.name)
        valid_values = ['int32', 'int64']
        self.dtype = validator.check_string(dtype, valid_values, "dtype", self.name)
        self.init_prim_io_names(inputs=['x', 'range'], outputs=['y'])

    def infer_shape(self, x_shape, range_shape):
        return (self.nbins,)

    def infer_dtype(self, x_dtype, range_dtype):
        valid_dtypes = (mstype.float16, mstype.float32, mstype.int32)
        validator.check_tensor_dtype_valid("x", x_dtype, valid_dtypes, self.name)
        validator.check_tensor_dtype_valid("range", range_dtype, valid_dtypes, self.name)
        y_dtype = mstype.int32
        return y_dtype


class Log(PrimitiveWithInfer):
    """
    Returns the natural logarithm of a tensor element-wise.

    .. math::
        y_i = log_e(x_i)

    Inputs:
        - **input_x** (Tensor) - The input tensor. The value must be greater than 0.

    Outputs:
        Tensor, has the same shape as the `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> log = ops.Log()
        >>> output = log(input_x)
        >>> print(output)
        [0.        0.6931472 1.3862944]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x):
        return x

    def infer_dtype(self, x):
        validator.check_subclass("x", x, mstype.tensor, self.name)
        return x

    def infer_value(self, x):
        if x is not None:
            x = x.asnumpy()
            out = np.log(x)
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class Log1p(PrimitiveWithInfer):
    """
    Returns the natural logarithm of one plus the input tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor. With float16 or float32 data type. The value must be greater than -1.

    Outputs:
        Tensor, has the same shape as the `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> input_x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> log1p = ops.Log1p()
        >>> output = log1p(input_x)
        >>> print(output)
        [0.6931472 1.0986123 1.609438 ]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("x", x_dtype, mstype.tensor, self.name)
        validator.check_tensor_dtype_valid("x", x_dtype, [mstype.float16, mstype.float32], self.name)
        return x_dtype


class Erf(PrimitiveWithInfer):
    r"""
    Computes the Gauss error function of `input_x` element-wise.

    .. math::

        erf(x)=\frac{2} {\sqrt{\pi}} \int\limits_0^{x} e^{-t^{2}} dt

    Inputs:
        - **input_x** (Tensor) - The input tensor. The data type must be float16 or float32.

    Outputs:
        Tensor, has the same shape and dtype as the `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.array([-1, 0, 1, 2, 3]), mindspore.float32)
        >>> erf = ops.Erf()
        >>> output = erf(input_x)
        >>> print(output)
        [-0.8427168   0.          0.8427168   0.99530876  0.99997765]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Erf"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x", x_dtype, [mstype.float16, mstype.float32], self.name)
        return x_dtype


class Erfc(PrimitiveWithInfer):
    r"""
    Computes the complementary error function of `input_x` element-wise.

    .. math::

        erfc(x) = 1 - \frac{2} {\sqrt{\pi}} \int\limits_0^{x} e^{-t^{2}} dt

    Inputs:
        - **input_x** (Tensor) - The input tensor. The data type must be float16 or float32.

    Outputs:
        Tensor, has the same shape and dtype as the `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> input_x = Tensor(np.array([-1, 0, 1, 2, 3]), mindspore.float32)
        >>> erfc = ops.Erfc()
        >>> output = erfc(input_x)
        >>> print(output)
        [1.8427168e+00 1.0000000e+00 1.5728319e-01 4.6912432e-03 2.2351742e-05]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Erfc"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        validator.check_tensor_dtype_valid("x", x_type, [mstype.float16, mstype.float32], self.name)
        return x_type


class Minimum(_MathBinaryOp):
    """
    Computes the minimum of input tensors element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `input_x` and `input_y` is not one of the following: Tensor, Number, bool.
        ValueError: If `input_x` and `input_y` are not the same shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.float32)
        >>> input_y = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float32)
        >>> minimum = ops.Minimum()
        >>> output = minimum(input_x, input_y)
        >>> print(output)
        [1. 2. 3.]
    """

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = np.minimum(x, y)
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class Maximum(_MathBinaryOp):
    """
    Computes the maximum of input tensors element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `input_x` and `input_y` is not one of the following: Tensor, Number, bool.
        ValueError: If `input_x` and `input_y` are not the same shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.float32)
        >>> input_y = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float32)
        >>> maximum = ops.Maximum()
        >>> output = maximum(input_x, input_y)
        >>> print(output)
        [4. 5. 6.]
    """

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = np.maximum(x, y)
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class RealDiv(_MathBinaryOp):
    """
    Divides the first input tensor by the second input tensor in floating-point type element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `input_x` and `input_y` is not one of the following: Tensor, Number, bool.
        ValueError: If `input_x` and `input_y` are not the same shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> input_y = Tensor(np.array([4.0, 5.0, 6.0]), mindspore.float32)
        >>> realdiv = ops.RealDiv()
        >>> output = realdiv(input_x, input_y)
        >>> print(output)
        [0.25 0.4  0.5 ]
    """

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = x / y
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class Div(_MathBinaryOp):
    r"""
    Computes the quotient of dividing the first input tensor by the second input tensor element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} = \frac{x_i}{y_i}

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - When the first input is a tensor, The second input
          could be a number, a bool, or a tensor whose data type is number or bool. When the first input
          is a number or a bool, the second input must be a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If neither `input_x` nor `input_y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([-4.0, 5.0, 6.0]), mindspore.float32)
        >>> input_y = Tensor(np.array([3.0, 2.0, 3.0]), mindspore.float32)
        >>> div = ops.Div()
        >>> output = div(input_x, input_y)
        >>> print(output)
        [-1.3333334  2.5        2.        ]
    """

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = np.array(x / y, x.dtype)
            return Tensor(out)
        return None


class DivNoNan(_MathBinaryOp):
    """
    Computes a safe divide and returns 0 if the y is zero.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If neither `input_x` nor `input_y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> input_x = Tensor(np.array([-1.0, 0., 1.0, 5.0, 6.0]), mindspore.float32)
        >>> input_y = Tensor(np.array([0., 0., 0., 2.0, 3.0]), mindspore.float32)
        >>> div_no_nan = ops.DivNoNan()
        >>> output = div_no_nan(input_x, input_y)
        >>> print(output)
        [0.  0.  0.  2.5 2. ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize _BinaryOp"""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            with np.errstate(divide='ignore', invalid='ignore'):
                out = np.true_divide(x, y)
                out[~np.isfinite(out)] = 0
            return out
        return None


class MulNoNan(_MathBinaryOp):
    r"""
    Computes `input_x` * `input_y` element-wise. If `input_y` is zero, no matter what `input_x` is, it will return 0.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcasted.
    When the inputs are one tensor and one scalar, the scalar could only be a constant.

    Note:
        The shapes of `input_x` and `input_y` should be the same or can be broadcasted.

    Inputs:
        - **input_x** (Union[Tensor]) - The first input is a tensor whose data type is one of
          flota16, float32, int32, int64 currently or scalar.
        - **input_y** (Union[Tensor]) - The second input is a tensor whose data type is one of
          flota16, float32, int32, int64 currently or scalar.

    Outputs:
        Tensor, the shape is the same as the shape after broadcasting,
        and the data type is the one with higher precision among the two inputs.


    Supported Platforms:
        ``Ascend``

    Raises:
        TypeError: If neither `input_x` nor `input_y` is a bool Tensor.

    Examples:
        >>> x = Tensor(np.array([[-1.0, 6.0, np.inf], [np.nan, -7.0, 4.0]]), ms.float32)
        >>> y = Tensor(np.array([[-1.0, 4.0, 0], [0, -3.0, 1.0]]), ms.float32)
        >>> mul_no_nan = ops.MulNoNan()
        >>> output = mul_no_nan(x, y)
        >>> print(output)
        [[ 1. 24. 0.]
        [ 0. 21. 4.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize _BinaryOp"""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            with np.errstate(divide='ignore', invalid='ignore'):
                out = np.multiply(x, y)
                out[y == 0] = 0
            return out
        return None


class FloorDiv(_MathBinaryOp):
    """
    Divides the first input tensor by the second input tensor element-wise and round down to the closest integer.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If neither `input_x` nor `input_y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([2, 4, -1]), mindspore.int32)
        >>> input_y = Tensor(np.array([3, 3, 3]), mindspore.int32)
        >>> floor_div = ops.FloorDiv()
        >>> output = floor_div(input_x, input_y)
        >>> print(output)
        [ 0  1 -1]
    """


class TruncateDiv(_MathBinaryOp):
    """
    Divides the first input tensor by the second input tensor element-wise for integer types, negative numbers will
    round fractional quantities towards zero.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number, or a bool,
          or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number, or a bool when the first input
          is a tensor, or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `input_x` and `input_y` is not one of the following: Tensor, Number, bool.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.array([2, 4, -1]), mindspore.int32)
        >>> input_y = Tensor(np.array([3, 3, 3]), mindspore.int32)
        >>> truncate_div = ops.TruncateDiv()
        >>> output = truncate_div(input_x, input_y)
        >>> print(output)
        [0 1 0]
    """


class TruncateMod(_MathBinaryOp):
    """
    Returns the remainder of division element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number, or a bool,
          or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number, or a bool when the first input
          is a tensor, or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If neither `input_x` nor `input_y` is one of the following: Tensor, Number, bool.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.array([2, 4, -1]), mindspore.int32)
        >>> input_y = Tensor(np.array([3, 3, 3]), mindspore.int32)
        >>> truncate_mod = ops.TruncateMod()
        >>> output = truncate_mod(input_x, input_y)
        >>> print(output)
        [ 2  1 -1]
    """


class Mod(_MathBinaryOp):
    """
    Computes the remainder of dividing the first input tensor by the second input tensor element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar. When the inputs are two tensors,
    both dtypes cannot be bool, and the shapes of them could be broadcast. When the inputs are one tensor
    and one scalar, the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The first input is a number or a tensor whose data type is number.
        - **input_y** (Union[Tensor, Number]) - When the first input is a tensor, The second input
          could be a number or a tensor whose data type is number. When the first input is a number,
          the second input must be a tensor whose data type is number.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        ValueError: When `input_x` and `input_y` are not the same dtype.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([-4.0, 5.0, 6.0]), mindspore.float32)
        >>> input_y = Tensor(np.array([3.0, 2.0, 3.0]), mindspore.float32)
        >>> mod = ops.Mod()
        >>> output = mod(input_x, input_y)
        >>> print(output)
        [-1.  1.  0.]
    """

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            return Tensor(np.fmod(x, y))
        return None


class Floor(PrimitiveWithInfer):
    r"""
    Rounds a tensor down to the closest integer element-wise.

    .. math::

        out_i = \lfloor x_i \rfloor

    Inputs:
        - **input_x** (Tensor) - The input tensor. Its element data type must be float.

    Outputs:
        Tensor, has the same shape as `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is not float.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1.1, 2.5, -1.5]), mindspore.float32)
        >>> floor = ops.Floor()
        >>> output = floor(input_x)
        >>> print(output)
        [ 1.  2. -2.]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x", x_dtype, mstype.float_type, self.name)
        return x_dtype


class FloorMod(_MathBinaryOp):
    """
    Computes the remainder of division element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool , and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If neither `input_x` nor `input_y` is a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.array([2, 4, -1]), mindspore.int32)
        >>> input_y = Tensor(np.array([3, 3, 3]), mindspore.int32)
        >>> floor_mod = ops.FloorMod()
        >>> output = floor_mod(input_x, input_y)
        >>> print(output)
        [2 1 2]
    """


class Ceil(PrimitiveWithInfer):
    r"""
    Rounds a tensor up to the closest integer element-wise.

    .. math::

        out_i = \lceil x_i \rceil = \lfloor x_i \rfloor + 1

    Inputs:
        - **input_x** (Tensor) - The input tensor. It's element data type must be float16 or float32.

    Outputs:
        Tensor, has the same shape as `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.array([1.1, 2.5, -1.5]), mindspore.float32)
        >>> ceil_op = ops.Ceil()
        >>> output = ceil_op(input_x)
        >>> print(output)
        [ 2.  3. -1.]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x", x_dtype, [mstype.float16, mstype.float32], self.name)
        return x_dtype


class Xdivy(_MathBinaryOp):
    """
    Divides the first input tensor by the second input tensor element-wise. Returns zero when `x` is zero.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number, or a bool,
          or a tensor whose data type is float16, float32 or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number,
          or a bool when the first input is a tensor, or a tensor whose data type is float16, float32 or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `input_x` and `input_y` is not one of the following: Tensor, Number, bool.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.array([2, 4, -1]), mindspore.float32)
        >>> input_y = Tensor(np.array([2, 2, 2]), mindspore.float32)
        >>> xdivy = ops.Xdivy()
        >>> output = xdivy(input_x, input_y)
        >>> print(output)
        [ 1.   2.  -0.5]
    """

    def infer_dtype(self, x_dtype, y_dtype):
        return _MathBinaryOp.do_infer_dtype(x_dtype, y_dtype, [mstype.float16, mstype.float32], self.name)


class Xlogy(_MathBinaryOp):
    """
    Computes the first input tensor multiplied by the logarithm of second input tensor element-wise.
    Returns zero when `x` is zero.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is float16, float32 or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is float16, float32 or bool.
          The value must be positive.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `input_x` and `input_y` is not one of the following: Tensor, Number, bool.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.array([-5, 0, 4]), mindspore.float32)
        >>> input_y = Tensor(np.array([2, 2, 2]), mindspore.float32)
        >>> xlogy = ops.Xlogy()
        >>> output = xlogy(input_x, input_y)
        >>> print(output)
        [-3.465736   0.        2.7725887]
    """

    def infer_dtype(self, x_dtype, y_dtype):
        return _MathBinaryOp.do_infer_dtype(x_dtype, y_dtype, [mstype.float16, mstype.float32], self.name)


class Acosh(PrimitiveWithInfer):
    """
    Computes inverse hyperbolic cosine of the input element-wise.

    .. math::

        out_i = cosh^{-1}(input_i)

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The data type should be one of
          the following types: float16, float32.

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> acosh = ops.Acosh()
        >>> input_x = Tensor(np.array([1.0, 1.5, 3.0, 100.0]), mindspore.float32)
        >>> output = acosh(input_x)
        >>> print(output)
        [0. 0.9624236 1.7627472 5.298292]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Acosh"""

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, mstype.number_type, self.name)
        return x_dtype


class Cosh(PrimitiveWithInfer):
    """
    Computes hyperbolic cosine of input element-wise.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape as `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> cosh = ops.Cosh()
        >>> input_x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = cosh(input_x)
        >>> print(output)
        [1.0289385 1.364684 1.048436 1.0040528]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Cosh"""

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, mstype.number_type, self.name)
        return x_dtype


class Asinh(PrimitiveWithInfer):
    r"""
    Computes inverse hyperbolic sine of the input element-wise.

    .. math::

        out_i = sinh^{-1}(input_i)

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The data type should be one of
          the following types: float16, float32.

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> asinh = ops.Asinh()
        >>> input_x = Tensor(np.array([-5.0, 1.5, 3.0, 100.0]), mindspore.float32)
        >>> output = asinh(input_x)
        >>> print(output)
        [-2.3124385  1.1947632  1.8184465  5.298342 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Asinh"""

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, mstype.number_type, self.name)
        return x_dtype


class Sinh(PrimitiveWithInfer):
    """
    Computes hyperbolic sine of the input element-wise.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape as `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> sinh = ops.Sinh()
        >>> input_x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float32)
        >>> output = sinh(input_x)
        >>> print(output)
        [0.6604918  0.28367308 0.44337422 0.6604918 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Sinh"""

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, mstype.number_type, self.name)
        return x_dtype


class _LogicBinaryOp(_BinaryOp):
    """
    Define logic binary operators.
    """

    @staticmethod
    def do_infer_dtype(x_dtype, y_dtype, valid_type=mstype.number_type, prim_name=None):
        args_dtype = {"x": x_dtype, "y": y_dtype}
        validator.check_tensors_dtypes_same_and_valid(args_dtype, valid_type, prim_name)
        return mstype.tensor_type(mstype.bool_)

    def infer_dtype(self, x_dtype, y_dtype):
        return _LogicBinaryOp.do_infer_dtype(x_dtype, y_dtype, prim_name=self.name)


class Equal(_LogicBinaryOp):
    """
    Computes the equivalence between two tensors element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar, the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The first input is a number or
          a tensor whose data type is number.
        - **input_y** (Union[Tensor, Number]) - The second input is a number
          when the first input is a tensor or a tensor whose data type is number.
          The data type is the same as the first input.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,and the data type is bool.

    Raises:
        TypeError: If neither `input_x` nor `input_y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> equal = ops.Equal()
        >>> output = equal(input_x, 2.0)
        >>> print(output)
        [False True False]
        >>>
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> input_y = Tensor(np.array([1, 2, 4]), mindspore.int32)
        >>> equal = ops.Equal()
        >>> output = equal(input_x, input_y)
        >>> print(output)
        [ True  True False]
    """

    def infer_dtype(self, x_dtype, y_dtype):
        return _LogicBinaryOp.do_infer_dtype(x_dtype, y_dtype, mstype.number_type + (mstype.bool_,), self.name)

    def infer_value(self, x, y):
        if x is None or y is None:
            return None
        if isinstance(x, Tensor) and x.has_init:
            x = x.init_data()
        if isinstance(y, Tensor) and y.has_init:
            y = y.init_data()
        return Tensor(x.asnumpy() == y.asnumpy())


class ApproximateEqual(_LogicBinaryOp):
    """
    Returns True if abs(x1-x2) is smaller than tolerance element-wise, otherwise False.

    Inputs of `x1` and `x2` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        tolerance (float): The maximum deviation that two elements can be considered equal. Default: 1e-05.

    Inputs:
        - **x1** (Tensor) - A tensor. Must be one of the following types: float32, float16.
        - **x2** (Tensor) - A tensor of the same type and shape as 'x1'.

    Outputs:
        Tensor, the shape is the same as the shape of 'x1', and the data type is bool.

    Raises:
        TypeError: If `tolerance` is not a float.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x1 = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> x2 = Tensor(np.array([2, 4, 6]), mindspore.float32)
        >>> approximate_equal = ops.ApproximateEqual(2.)
        >>> output = approximate_equal(x1, x2)
        >>> print(output)
        [ True  True  False]
    """

    @prim_attr_register
    def __init__(self, tolerance=1e-05):
        """Initialize ApproximateEqual"""
        validator.check_value_type("tolerance", tolerance, [float], self.name)

    def infer_shape(self, x_shape, y_shape):
        validator.check("x_shape", x_shape, "y_shape", y_shape, Rel.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_dtype, y_dtype):
        args_dtype = {"x": x_dtype, "y": y_dtype}
        valid_type = [mstype.float32, mstype.float16]
        validator.check_tensors_dtypes_same_and_valid(args_dtype, valid_type, prim_name=self.name)
        return mstype.tensor_type(mstype.bool_)


class EqualCount(PrimitiveWithInfer):
    """
    Computes the number of the same elements of two tensors.

    The two input tensors must have the same data type and shape.

    Inputs:
        - **input_x** (Tensor) - The first input tensor.
        - **input_y** (Tensor) - The second input tensor.

    Outputs:
        Tensor, with the type same as input tensor and size as (1,).

    Raises:
        TypeError: If `input_x` or `input_y` is not a Tensor.
        ValueError: If shape of `input_x` is not equal to shape of `input_y`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> input_y = Tensor(np.array([1, 2, 4]), mindspore.int32)
        >>> equal_count = ops.EqualCount()
        >>> output = equal_count(input_x, input_y)
        >>> print(output)
        [2]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize EqualCount"""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])

    def infer_shape(self, x_shape, y_shape):
        validator.check("x_shape", x_shape, "y_shape", y_shape, Rel.EQ, self.name)
        output_shape = (1,)
        return output_shape

    def infer_dtype(self, x_dtype, y_dtype):
        args = {'x': x_dtype, 'y': y_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type + (mstype.bool_,), self.name)
        return x_dtype


class NotEqual(_LogicBinaryOp):
    """
    Computes the non-equivalence of two tensors element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar, the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,and the data type is bool.

    Raises:
        TypeError: If `input_x` and `input_y` is not one of the following: Tensor, Number, bool.
        TypeError: If neither `input_x` nor `input_y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> not_equal = ops.NotEqual()
        >>> output = not_equal(input_x, 2.0)
        >>> print(output)
        [ True False  True]
        >>>
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> input_y = Tensor(np.array([1, 2, 4]), mindspore.int32)
        >>> not_equal = ops.NotEqual()
        >>> output = not_equal(input_x, input_y)
        >>> print(output)
        [False False  True]
    """

    def infer_dtype(self, x_dtype, y_dtype):
        return _LogicBinaryOp.do_infer_dtype(x_dtype, y_dtype, mstype.number_type + (mstype.bool_,), self.name)


class Greater(_LogicBinaryOp):
    """
    Computes the boolean value of :math:`x > y` element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Raises:
        TypeError: If neither `input_x` nor `input_y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> input_y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> greater = ops.Greater()
        >>> output = greater(input_x, input_y)
        >>> print(output)
        [False  True False]
    """

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = np.array(np.greater(x, y))
            return Tensor(out)
        return None


class GreaterEqual(_LogicBinaryOp):
    """
    Computes the boolean value of :math:`x >= y` element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Raises:
        TypeError: If neither `input_x` nor `input_y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> input_y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> greater_equal = ops.GreaterEqual()
        >>> output = greater_equal(input_x, input_y)
        >>> print(output)
        [ True  True False]
    """

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = np.array(np.greater_equal(x, y))
            return Tensor(out)
        return None


class Less(_LogicBinaryOp):
    """
    Computes the boolean value of :math:`x < y` element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,and the data type is bool.

    Raises:
        TypeError: If `input_x` and `input_y` is not one of the following: Tensor, Number, bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> input_y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> less = ops.Less()
        >>> output = less(input_x, input_y)
        >>> print(output)
        [False False  True]
    """

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = np.array(np.less(x, y))
            return Tensor(out)
        return None


class LessEqual(_LogicBinaryOp):
    """
    Computes the boolean value of :math:`x <= y` element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool , and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,and the data type is bool.

    Raises:
        TypeError: If neither `input_x` nor `input_y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> input_y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> less_equal = ops.LessEqual()
        >>> output = less_equal(input_x, input_y)
        >>> print(output)
        [ True False  True]
    """

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = np.array(np.less_equal(x, y))
            return Tensor(out)
        return None


class LogicalNot(PrimitiveWithInfer):
    """
    Computes the "logical NOT" of a tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor whose dtype is bool.

    Outputs:
        Tensor, the shape is the same as the `input_x`, and the dtype is bool.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([True, False, True]), mindspore.bool_)
        >>> logical_not = ops.LogicalNot()
        >>> output = logical_not(input_x)
        >>> print(output)
        [False  True False]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize LogicalNot"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x", x_dtype, [mstype.bool_], self.name + " or '~' operator")
        return mstype.tensor_type(mstype.bool_)

    def infer_value(self, x):
        if x is not None:
            x = x.asnumpy()
            return Tensor(np.logical_not(x))
        return None


class LogicalAnd(_LogicBinaryOp):
    """
    Computes the "logical AND" of two tensors element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one bool.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them must be bool.
    When the inputs are one tensor and one bool, the bool object could only be a constant,
    and the data type of the tensor must be bool.

    Inputs:
        - **input_x** (Union[Tensor, bool]) - The first input is a bool or a tensor whose data type is bool.
        - **input_y** (Union[Tensor, bool]) - The second input is a bool when the first input is a tensor or
          a tensor whose data type is bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Raises:
        TypeError: If neither `input_x` nor `input_y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([True, False, True]), mindspore.bool_)
        >>> input_y = Tensor(np.array([True, True, False]), mindspore.bool_)
        >>> logical_and = ops.LogicalAnd()
        >>> output = logical_and(input_x, input_y)
        >>> print(output)
        [ True False False]
    """

    def infer_dtype(self, x_dtype, y_dtype):
        return _LogicBinaryOp.do_infer_dtype(x_dtype, y_dtype, (mstype.bool_,), self.name)

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = np.array(np.logical_and(x, y))
            return Tensor(out)
        return None


class LogicalOr(_LogicBinaryOp):
    """
    Computes the "logical OR" of two tensors element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one bool.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them must be bool.
    When the inputs are one tensor and one bool, the bool object could only be a constant,
    and the data type of the tensor must be bool.

    Inputs:
        - **input_x** (Union[Tensor, bool]) - The first input is a bool or a tensor whose data type is bool.
        - **input_y** (Union[Tensor, bool]) - The second input is a bool when the first input is a tensor or
          a tensor whose data type is bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,and the data type is bool.

    Raises:
        TypeError: If neither `input_x` nor `input_y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([True, False, True]), mindspore.bool_)
        >>> input_y = Tensor(np.array([True, True, False]), mindspore.bool_)
        >>> logical_or = ops.LogicalOr()
        >>> output = logical_or(input_x, input_y)
        >>> print(output)
        [ True  True  True]
    """

    def infer_dtype(self, x_dtype, y_dtype):
        return _LogicBinaryOp.do_infer_dtype(x_dtype, y_dtype, (mstype.bool_,), self.name)

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = np.array(np.logical_or(x, y))
            return Tensor(out)
        return None


class IsNan(PrimitiveWithInfer):
    """
    Determines which elements are NaN for each position.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape of input, and the dtype is bool.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> is_nan = ops.IsNan()
        >>> input_x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> output = is_nan(input_x)
        >>> print(output)
        [True False False]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize IsNan"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        return mstype.tensor_type(mstype.bool_)


class IsInf(PrimitiveWithInfer):
    """
    Determines which elements are inf or -inf for each position

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape of input, and the dtype is bool.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> is_inf = ops.IsInf()
        >>> input_x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> output = is_inf(input_x)
        >>> print(output)
        [False False True]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize IsInf"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        return mstype.tensor_type(mstype.bool_)


class IsFinite(PrimitiveWithInfer):
    """
    Determines which elements are finite for each position.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape of input, and the dtype is bool.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> is_finite = ops.IsFinite()
        >>> input_x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> output = is_finite(input_x)
        >>> print(output)
        [False  True False]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize IsFinite"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, mstype.number_type + (mstype.bool_,), self.name)
        return mstype.tensor_type(mstype.bool_)


class FloatStatus(PrimitiveWithInfer):
    """
    Determines if the elements contain Not a Number(NaN), infinite or negative infinite. 0 for normal, 1 for overflow.

    Inputs:
        - **input_x** (Tensor) - The input tensor. The data type must be float16 or float32.

    Outputs:
        Tensor, has the shape of `(1,)`, and the dtype is `mindspore.dtype.float32`.

    Raises:
        TypeError: If dtype of `input_x` is neither float16 nor float32.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> float_status = ops.FloatStatus()
        >>> input_x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> result = float_status(input_x)
        >>> print(result)
        [1.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize FloatStatus"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return [1]

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, [mstype.float32, mstype.float16], self.name)
        return mstype.float32


class NPUAllocFloatStatus(PrimitiveWithInfer):
    """
    Allocates a flag to store the overflow status.

    The flag is a tensor whose shape is `(8,)` and data type is `mindspore.dtype.float32`.

    Note:
        Examples: see `NPUGetFloatStatus`.

    Outputs:
        Tensor, has the shape of `(8,)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> alloc_status = ops.NPUAllocFloatStatus()
        >>> output = alloc_status()
        >>> print(output)
        [0. 0. 0. 0. 0. 0. 0. 0.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize NPUAllocFloatStatus"""
        self.add_prim_attr("_side_effect_flag", True)

    def infer_shape(self):
        return [8]

    def infer_dtype(self):
        return mstype.float32


class NPUGetFloatStatus(PrimitiveWithInfer):
    """
    Updates the flag which is the output tensor of `NPUAllocFloatStatus` with the latest overflow status.

    The flag is a tensor whose shape is `(8,)` and data type is `mindspore.dtype.float32`.
    If the sum of the flag equals to 0, there is no overflow happened. If the sum of the flag is bigger than 0, there
    is overflow happened.

    Inputs:
        - **input_x** (Tensor) - The output tensor of `NPUAllocFloatStatus`.
          The data type must be float16 or float32.

    Outputs:
        Tensor, has the same shape as `input_x`. All the elements in the tensor will be zero.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        TypeError: If dtype of `input_x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> alloc_status = ops.NPUAllocFloatStatus()
        >>> get_status = ops.NPUGetFloatStatus()
        >>> init = alloc_status()
        >>> get_status(init)
        Tensor(shape=[8], dtype=Float32, value= [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
        >>> print(init)
        [1. 1. 1. 1. 1. 1. 1. 1.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize NPUGetFloatStatus"""
        self.add_prim_attr("_side_effect_flag", True)

    def infer_shape(self, x_shape):
        cls_name = self.name
        validator.check_equal_int(len(x_shape), 1, "len(x_shape)", cls_name)
        validator.check_equal_int(x_shape[0], 8, "x_shape[0]", cls_name)
        return [8]

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, [mstype.float16, mstype.float32], self.name)
        return mstype.float32


class NPUClearFloatStatus(PrimitiveWithInfer):
    """
    Clears the flag which stores the overflow status.

    Note:
        The flag is in the register on the `Ascend` device. It will be reset and can not be reused again after the
        `NPUClearFloatStatus` is called.

        Examples: see `NPUGetFloatStatus`.

    Inputs:
        - **input_x** (Tensor) - The output tensor of `NPUAllocFloatStatus`.
          The data type must be float16 or float32.

    Outputs:
        Tensor, has the same shape as `input_x`. All the elements in the tensor will be zero.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> alloc_status = ops.NPUAllocFloatStatus()
        >>> get_status = ops.NPUGetFloatStatus()
        >>> clear_status = ops.NPUClearFloatStatus()
        >>> init = alloc_status()
        >>> flag = get_status(init)
        >>> clear_status(init)
        Tensor(shape=[8], dtype=Float32, value= [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
        >>> print(init)
        [1. 1. 1. 1. 1. 1. 1. 1.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize NPUClearFloatStatus"""
        self.add_prim_attr("_side_effect_flag", True)

    def infer_shape(self, x_shape):
        cls_name = self.name
        validator.check_equal_int(len(x_shape), 1, "len(x_shape)", cls_name)
        validator.check_equal_int(x_shape[0], 8, "x_shape[0]", cls_name)
        return [8]

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, [mstype.float16, mstype.float32], self.name)
        return mstype.float32


class Cos(PrimitiveWithInfer):
    """
    Computes cosine of input element-wise.

    .. math::
        out_i = cos(x_i)

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape as `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> cos = ops.Cos()
        >>> input_x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = cos(input_x)
        >>> print(output)
        [0.971338 0.67487574 0.95233357 0.9959527 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Cos"""

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, mstype.number_type, self.name)
        return x_dtype


class ACos(PrimitiveWithInfer):
    r"""
    Computes arccosine of input tensors element-wise.

    .. math::

        out_i = cos^{-1}(x_i)

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape as `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> acos = ops.ACos()
        >>> input_x = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float32)
        >>> output = acos(input_x)
        >>> print(output)
        [0.7377037 1.5307858 1.2661037 0.97641146]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ACos"""

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, mstype.number_type, self.name)
        return x_dtype


class Sin(PrimitiveWithInfer):
    """
    Computes sine of the input element-wise.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape as `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> sin = ops.Sin()
        >>> input_x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float32)
        >>> output = sin(input_x)
        >>> print(output)
        [0.5810352  0.27635565 0.41687083 0.5810352 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Sin."""

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, mstype.number_type, self.name)
        return x_dtype


class Asin(PrimitiveWithInfer):
    r"""
    Computes arcsine of input tensors element-wise.

    .. math::

        out_i = sin^{-1}(x_i)

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape as `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> asin = ops.Asin()
        >>> input_x = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float32)
        >>> output = asin(input_x)
        >>> print(output)
        [0.8330927  0.04001068  0.30469266  0.59438497]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Asin"""

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, mstype.number_type, self.name)
        return x_dtype


class NMSWithMask(PrimitiveWithInfer):
    r"""
    When object detection problem is performed in the computer vision field, object detection algorithm generates
    a plurality of bounding boxes. Selects some bounding boxes in descending order of score. Use the box with the
    highest score calculate the overlap between other boxes and the current box, and delete the box based on a
    certain threshold(IOU). The IOU is as follows,

    .. math::
        \text{IOU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}

    Args:
        iou_threshold (float): Specifies the threshold of overlap boxes with respect to
            IOU. Default: 0.5.

    Inputs:
        - **bboxes** (Tensor) - The shape of tensor is :math:`(N, 5)`. Input bounding boxes.
          `N` is the number of input bounding boxes. Every bounding box
          contains 5 values, the first 4 values are the coordinates(x0, y0, x1, y1) of bounding box which
          represents the point of top-left and bottom-right, and the last value is the score of this bounding box.
          The data type must be float16 or float32.

    Outputs:
        tuple[Tensor], tuple of three tensors, they are selected_boxes, selected_idx and selected_mask.

        - **selected_boxes** (Tensor) - The shape of tensor is :math:`(N, 5)`. The list of bounding boxes
          after non-max suppression calculation.
        - **selected_idx** (Tensor) - The shape of tensor is :math:`(N,)`. The indexes list of
          valid input bounding boxes.
        - **selected_mask** (Tensor) - The shape of tensor is :math:`(N,)`. A mask list of
          valid output bounding boxes.

    Raises:
        ValueError: If the `iou_threshold` is not a float number, or if the first dimension
            of input Tensor is less than or equal to 0, or if the data type of the input
            Tensor is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> bbox = np.array([[100.0, 100.0, 50.0, 68.0, 0.63], [150.0, 75.0, 165.0, 115.0, 0.55],
        ...                  [12.0, 190.0, 288.0, 200.0, 0.9], [28.0, 130.0, 106.0, 172.0, 0.3]])
        >>> bbox[:, 2] += bbox[:, 0]
        >>> bbox[:, 3] += bbox[:, 1]
        >>> inputs = Tensor(bbox, mindspore.float32)
        >>> nms = ops.NMSWithMask(0.1)
        >>> output_boxes, indices, mask = nms(inputs)
        >>> indices_np = indices.asnumpy()
        >>> print(indices_np[mask.asnumpy()])
        [0 1 2]
    """

    @prim_attr_register
    def __init__(self, iou_threshold=0.5):
        """Initialize NMSWithMask"""
        validator.check_value_type("iou_threshold", iou_threshold, [float], self.name)
        self.init_prim_io_names(inputs=['bboxes'], outputs=['selected_boxes', 'selected_idx', 'selected_mask'])
        self.is_ge = context.get_context("enable_ge")

    def infer_shape(self, bboxes_shape):
        cls_name = self.name
        validator.check_equal_int(len(bboxes_shape), 2, "bboxes rank", cls_name)
        validator.check_positive_int(bboxes_shape[0], "bboxes.shape[0]", cls_name)
        validator.check_equal_int(bboxes_shape[1], 5, "bboxes.shape[1]", cls_name)
        num = bboxes_shape[0]
        return (bboxes_shape, (num,), (num,))

    def infer_dtype(self, bboxes_dtype):
        validator.check_tensor_dtype_valid("bboxes", bboxes_dtype, [mstype.float16, mstype.float32], self.name)
        return (bboxes_dtype, mstype.int32, mstype.bool_)


class Abs(PrimitiveWithInfer):
    r"""
    Returns absolute value of a tensor element-wise.

    .. math::

        out_i = |x_i|

    Inputs:
        - **input_x** (Tensor) - The input tensor. The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape as the `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([-1.0, 1.0, 0.0]), mindspore.float32)
        >>> abs = ops.Abs()
        >>> output = abs(input_x)
        >>> print(output)
        [1. 1. 0.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Abs"""
        self.init_prim_io_names(inputs=['input_x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        validator.check_tensor_dtype_valid('x', x_type, mstype.number_type, self.name)
        return x_type

    def infer_value(self, x):
        if x is not None:
            x = x.asnumpy()
            out = np.array(np.abs(x, dtype=x.dtype))
            return Tensor(out)
        return None


class Sign(PrimitiveWithInfer):
    r"""
    Performs sign on the tensor element-wise.

    .. math::
        sign(x) = \begin{cases} -1, &if\ x < 0 \cr
        0, &if\ x = 0 \cr
        1, &if\ x > 0\end{cases}

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape and type as the `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
         >>> input_x = Tensor(np.array([[2.0, 0.0, -1.0]]), mindspore.float32)
         >>> sign = ops.Sign()
         >>> output = sign(input_x)
         >>> print(output)
         [[ 1.  0. -1.]]
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, mstype.number_type, self.name)
        return x_dtype


class Round(PrimitiveWithInfer):
    """
    Returns half to even of a tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape and type as the `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
         >>> input_x = Tensor(np.array([0.8, 1.5, 2.3, 2.5, -4.5]), mindspore.float32)
         >>> round = ops.Round()
         >>> output = round(input_x)
         >>> print(output)
         [ 1.  2.  2.  2. -4.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Round"""
        self.init_prim_io_names(inputs=['input_x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, mstype.number_type, self.name)
        return x_dtype


class Tan(PrimitiveWithInfer):
    """
    Computes tangent of `input_x` element-wise.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. Data type must be
          float16, float32 or int32.

    Outputs:
        Tensor, has the same shape as `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is not one of the following: float16, float32, int32.
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> tan = ops.Tan()
        >>> input_x = Tensor(np.array([-1.0, 0.0, 1.0]), mindspore.float32)
        >>> output = tan(input_x)
        >>> print(output)
        [-1.5574081 0. 1.5574081]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Tan"""

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        valid_dtypes = [mstype.float16, mstype.float32, mstype.int32]
        validator.check_tensor_dtype_valid('x', x_type, valid_dtypes, self.name)
        return x_type


class Atan(PrimitiveWithInfer):
    r"""
    Computes the trigonometric inverse tangent of the input element-wise.

    .. math::

        out_i = tan^{-1}(x_i)

    Inputs:
        - **input_x** (Tensor): The input tensor. The data type should be one of the following types: float16, float32.

    Outputs:
        A Tensor, has the same type as the input.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1.0, 0.0]), mindspore.float32)
        >>> atan = ops.Atan()
        >>> output = atan(input_x)
        >>> print(output)
        [0.7853982 0.       ]
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        validator.check_tensor_dtype_valid('x', x_type, mstype.number_type, self.name)
        return x_type


class Atanh(PrimitiveWithInfer):
    """
    Computes inverse hyperbolic tangent of the input element-wise.

    Inputs:
        - **input_x** (Tensor): The input tensor.

    Outputs:
        A Tensor, has the same type as the input.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1.047, 0.785]), mindspore.float32)
        >>> atanh = ops.Atanh()
        >>> output = atanh(input_x)
        >>> print(output)
        [1.8869909 1.058268 ]
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        validator.check_tensor_dtype_valid('x', x_type, mstype.number_type, self.name)
        return x_type


class Atan2(_MathBinaryOp):
    r"""
    Returns arctangent of input_x/input_y element-wise.

    It returns :math:`\theta\ \in\ [-\pi, \pi]`
    such that :math:`x = r*\sin(\theta), y = r*\cos(\theta)`, where :math:`r = \sqrt{x^2 + y^2}`.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Inputs:
        - **input_x** (Tensor) - The input tensor.
        - **input_y** (Tensor) - The input tensor.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,and the data type is same as `input_x`.

    Raises:
        TypeError: If `input_x` or `input_y` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([0, 1]), mindspore.float32)
        >>> input_y = Tensor(np.array([1, 1]), mindspore.float32)
        >>> atan2 = ops.Atan2()
        >>> output = atan2(input_x, input_y)
        >>> print(output)
        [0.        0.7853982]
    """


class SquareSumAll(PrimitiveWithInfer):
    """
    Returns the square sum of a tensor element-wise

    Inputs:
        - **input_x1** (Tensor) - The input tensor. The data type must be float16 or float32.
        - **input_x2** (Tensor) - The input tensor has the same type and shape as the `input_x1`.

    Note:
        SquareSumAll only supports float16 and float32 data type.

    Outputs:
        - **output_y1** (Tensor) - The same type as the `input_x1`.
        - **output_y2** (Tensor) - The same type as the `input_x1`.

    Raises:
        TypeError: If neither `input_x1` nor `input_x2` is a Tensor.
        ValueError: If `input_x1` and `input_x2` are not the same shape.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> input_x1 = Tensor(np.array([0, 0, 2, 0]), mindspore.float32)
        >>> input_x2 = Tensor(np.array([0, 0, 2, 4]), mindspore.float32)
        >>> square_sum_all = ops.SquareSumAll()
        >>> output = square_sum_all(input_x1, input_x2)
        >>> print(output)
        (Tensor(shape=[], dtype=Float32, value= 4),
         Tensor(shape=[], dtype=Float32, value= 20))
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SquareSumAll"""

    def infer_shape(self, x_shape, y_shape):
        validator.check("x1_shape", x_shape, "x2_shape", y_shape, Rel.EQ, self.name)
        return [], []

    def infer_dtype(self, x_type, y_type):
        valid_types = (mstype.float16, mstype.float32)
        args = {"x1_type": x_type, "x2_type": y_type}
        validator.check_tensors_dtypes_same_and_valid(args, valid_types, self.name)
        return x_type, y_type


class BitwiseAnd(_BitwiseBinaryOp):
    """
    Returns bitwise `and` of two tensors element-wise.

    Inputs of `input_x1` and `input_x2` comply with the implicit type conversion rules to
    make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Inputs:
        - **input_x1** (Tensor) - The input tensor with int16, int32 or uint16 data type.
        - **input_x2** (Tensor) - The input tensor with same type as the `input_x1`.

    Outputs:
        Tensor, has the same type as the `input_x1`.

    Raises:
        TypeError: If `input_x1` or `input_x2` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x1 = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int16)
        >>> input_x2 = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int16)
        >>> bitwise_and = ops.BitwiseAnd()
        >>> output = bitwise_and(input_x1, input_x2)
        >>> print(output)
        [ 0  0  1 -1  1  0  1]
    """


class BitwiseOr(_BitwiseBinaryOp):
    """
    Returns bitwise `or` of two tensors element-wise.

    Inputs of `input_x1` and `input_x2` comply with the implicit type conversion rules to
    make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Inputs:
        - **input_x1** (Tensor) - The input tensor with int16, int32 or uint16 data type.
        - **input_x2** (Tensor) - The input tensor with same type as the `input_x1`.

    Outputs:
        Tensor, has the same type as the `input_x1`.

    Raises:
        TypeError: If `input_x1` or `input_x2` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x1 = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int16)
        >>> input_x2 = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int16)
        >>> bitwise_or = ops.BitwiseOr()
        >>> output = bitwise_or(input_x1, input_x2)
        >>> print(output)
        [ 0  1  1 -1 -1  3  3]
    """


class BitwiseXor(_BitwiseBinaryOp):
    """
    Returns bitwise `xor` of two tensors element-wise.

    Inputs of `input_x1` and `input_x2` comply with the implicit type conversion rules to
    make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Inputs:
        - **input_x1** (Tensor) - The input tensor with int16, int32 or uint16 data type.
        - **input_x2** (Tensor) - The input tensor with same type as the `input_x1`.

    Outputs:
        Tensor, has the same type as the `input_x1`.

    Raises:
        TypeError: If `input_x1` or `input_x2` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x1 = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int16)
        >>> input_x2 = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int16)
        >>> bitwise_xor = ops.BitwiseXor()
        >>> output = bitwise_xor(input_x1, input_x2)
        >>> print(output)
        [ 0  1  0  0 -2  3  2]
    """


class BesselI0e(PrimitiveWithInfer):
    """
    Computes BesselI0e of input element-wise.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. Data type must be float16 or
          float32.

    Outputs:
        Tensor, has the same shape as `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> bessel_i0e = ops.BesselI0e()
        >>> input_x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = bessel_i0e(input_x)
        >>> print(output)
        [0.7979961  0.5144438  0.75117415  0.9157829 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize BesselI0e"""

    def infer_shape(self, x):
        return x

    def infer_dtype(self, x):
        validator.check_tensor_dtype_valid('x', x, mstype.number_type, self.name)
        return x


class BesselI1e(PrimitiveWithInfer):
    """
    Computes BesselI1e of input element-wise.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. Data type must be float16 or
          float32.

    Outputs:
        Tensor, has the same shape as `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> bessel_i1e = ops.BesselI1e()
        >>> input_x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = bessel_i1e(input_x)
        >>> print(output)
        [0.09507662 0.19699717 0.11505538 0.04116856]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize BesselI1e"""

    def infer_shape(self, x):
        return x

    def infer_dtype(self, x):
        validator.check_tensor_dtype_valid('x', x, mstype.number_type, self.name)
        return x


class Inv(PrimitiveWithInfer):
    """
    Computes Inv(Reciprocal) of input tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
          Must be one of the following types: float16, float32, int32.

    Outputs:
        Tensor, has the same shape and data type as `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is not one of float16, float32, int32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> inv = ops.Inv()
        >>> input_x = Tensor(np.array([0.25, 0.4, 0.31, 0.52]), mindspore.float32)
        >>> output = inv(input_x)
        >>> print(output)
        [4.        2.5       3.2258065 1.923077 ]
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x_dtype', x_dtype, [mstype.float16, mstype.float32,
                                                                mstype.int32], self.name)
        return x_dtype


class Invert(PrimitiveWithInfer):
    """
    Flips all bits of input tensor element-wise.

    Inputs:
        - **input_x** (Tensor[int16], Tensor[uint16]) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape as `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is neither int16 nor uint16.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> invert = ops.Invert()
        >>> input_x = Tensor(np.array([25, 4, 13, 9]), mindspore.int16)
        >>> output = invert(input_x)
        >>> print(output)
        [-26 -5 -14 -10]
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x_dtype', x_dtype, [mstype.int16, mstype.uint16], self.name)
        return x_dtype


class Eps(PrimitiveWithInfer):
    """
    Creates a tensor filled with `input_x` dtype minimum value.

    Inputs:
        - **input_x** (Tensor) - Input tensor. The data type must be float16 or float32.

    Outputs:
        Tensor, has the same type and shape as `input_x`, but filled with `input_x` dtype minimum val.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> input_x = Tensor([4, 1, 2, 3], mindspore.float32)
        >>> output = ops.Eps()(input_x)
        >>> print(output)
        [1.5258789e-05 1.5258789e-05 1.5258789e-05 1.5258789e-05]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Eps"""
        self.init_prim_io_names(inputs=['input_x'], outputs=['y'])

    def __infer__(self, input_x):
        valid_dtypes = [mstype.float16, mstype.float32]
        validator.check_tensor_dtype_valid('input_x', input_x['dtype'], valid_dtypes, self.name)

        x_nptype = mstype.dtype_to_nptype(input_x['dtype'].element_type())
        if x_nptype == np.float16:
            min_val = 2 ** (-14)
        else:
            min_val = 2 ** (-16)

        res = np.full(input_x['shape'], min_val, x_nptype)
        out = {
            'value': Tensor(res),
            'shape': input_x['shape'],
            'dtype': input_x['dtype'],
        }
        return out


class LinSpace(PrimitiveWithInfer):
    r"""
    Generates values in an interval (inclusive of start and stop) and returns the corresponding
    interpolated array with **num** number of ticks.

    Inputs:
        - **start** (Tensor[float32]) - Start value of interval, With shape of 0-D.
        - **stop** (Tensor[float32]) - Last value of interval, With shape of 0-D.
        - **num** (int) - Number of ticks in the interval, inclusive of start and stop.

    Outputs:
        Tensor, has the same shape as `start`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> linspace = P.LinSpace()
        >>> start = Tensor(1, mindspore.float32)
        >>> stop = Tensor(10, mindspore.float32)
        >>> num = 5
        >>> output = linspace(start, stop, num)
        >>> print(output)
        [ 1.    3.25  5.5   7.75 10.  ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize LinSpace"""

    def __infer__(self, start, stop, num):
        args = {"start": start['dtype'], "stop": start['dtype']}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float32,), self.name)
        start_shape = start['shape']
        stop_shape = stop['shape']
        validator.check_equal_int(len(start_shape), 0, "rank of start_shape", self.name)
        validator.check_equal_int(len(stop_shape), 0, "rank of stop_shape", self.name)
        num_v = num['value']
        validator.check_value_type('num', num_v, [int], self.name)
        validator.check_positive_int(num_v, "num", self.name)
        out_shape = [num_v]
        out = {'shape': out_shape,
               'dtype': start['dtype'],
               'value': None}
        return out


class MatrixInverse(PrimitiveWithInfer):
    """
    Returns the inverse of the input matrix. If the matrix is irreversible, an error may be reported or an unknown
    result may be returned.

    Note:
        The parameter 'adjoint' is only supporting False right now. Because complex number is not supported at present.

    Args:
        adjoint (bool) : An optional bool. Default: False.

    Inputs:
        - **x** (Tensor) - A matrix to be calculated. The matrix must be at least two dimensions, and the last two
          dimensions must be the same size. types: float32, float64.

    Outputs:
        Tensor, has the same type and shape as input `x`.

    Raises:
        TypeError: If `adjoint` is not a bool.
        TypeError: If dtype of `x` is neither float32 nor float64.
        ValueError: If the last two dimensions of `x` is not same size.
        ValueError: If the dimension of `x` is less than 2.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> x = Tensor(np.array([[[-0.710504  , -1.1207525],
        ...                       [-1.7651395 , -1.7576632]],
        ...                      [[ 0.52412605,  1.9070215],
        ...                       [ 1.3384849 ,  1.4274558]]]), mindspore.float32)
        >>> matrix_inverse = ops.MatrixInverse(adjoint=False)
        >>> output = matrix_inverse(x)
        >>> print(output)
        [[[ 2.4095483  -1.536419  ]
          [-2.4197974   0.97401696]]
         [[-0.79111797  1.0569006 ]
          [ 0.74180895 -0.2904787 ]]]
    """

    @prim_attr_register
    def __init__(self, adjoint=False):
        """Initialize MatrixInverse"""
        validator.check_type_name("adjoint", adjoint, False, self.name)
        self.adjoint = adjoint

    def infer_dtype(self, x_dtype):
        valid_type = [mstype.float32, mstype.double]
        validator.check_tensor_dtype_valid("x_dtype", x_dtype, valid_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape):
        validator.check_int(len(x_shape), 2, Rel.GE, self.name, None)
        validator.check_equal_int(x_shape[-1], x_shape[-2], self.name, None)
        return x_shape


class IndexAdd(PrimitiveWithInfer):
    """
    Adds tensor y to specified axis and indices of tensor x. The axis should be in the range from 0 to len(x.dim) - 1,
    and indices should be in the range from 0 to the size of x at the axis dimension.

    Args:
        axis (int): The dimension along which to index.

    Inputs:
        - **input_x** (Parameter) - The input tensor to add to, with data type float64, float32, float16, int32, int16,
          int8, uint8.
        - **indices** (Tensor) - The index of `input_x` on the `axis` th dimension to add to, with data type int32.
          The `indices` must be 1D with the same size as the size of the `axis` th dimension of `input_y`. The values
          of `indices` should be in the range of 0 to the size of the `axis` th dimension of `input_x`.
        - **input_y** (Tensor) - The input tensor with the value to add. Must have same data type as `input_x`.
          The shape must be the same as `input_x` except the `axis` th dimension.

    Outputs:
        Tensor, has the same shape and dtype as input_x.

    Raises:
        TypeError: If dtype of `input_x` is not one of: float64, float32, float16, int32, int16, int8, uint8.
        TypeError: If neither `indices` nor `input_y` is a Tensor.
        TypeError: If shape of `input_y` is not same as the `input_x`.
        ValueError: If axis is out of `input_x` rank's range.
        ValueError: If `input_x` rank is not the same as `input_y` rank.
        ValueError: If size of `indices` is not equal to dimension of y[axis].
        ValueError: If `input_y`'s shape is not the same as `input_x` except the `axis` th dimension.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.index_add = ops.IndexAdd(axis=1)
        ...         self.input_x = Parameter(Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32))
        ...         self.indices = Tensor(np.array([0, 2]), mindspore.int32)
        ...
        ...     def construct(self, input_y):
        ...         return self.index_add(self.input_x, self.indices, input_y)
        ...
        >>> input_y = Tensor(np.array([[0.5, 1.0], [1.0, 1.5], [2.0, 2.5]]), mindspore.float32)
        >>> net = Net()
        >>> output = net(input_y)
        >>> print(output)
        [[ 1.5  2.   4. ]
         [ 5.   5.   7.5]
         [ 9.   8.  11.5]]
    """
    __mindspore_signature__ = (
        sig.make_sig('input_x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('input_y', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self, axis, use_lock=True, check_index_bound=True):
        """Initialize InplaceAdd"""
        self.init_prim_io_names(inputs=['input_x', 'indices', 'input_y'], outputs=['output'])
        self.axis = axis
        validator.check_value_type('axis', axis, [int], self.name)

    def infer_dtype(self, x_dtype, idx_type, y_dtype):
        args = {'input_x': x_dtype, 'input_y': y_dtype}
        valid_type = [mstype.float64, mstype.float32, mstype.float16, mstype.int32, mstype.int16, mstype.int8,
                      mstype.uint8]
        validator.check_tensors_dtypes_same_and_valid(args, valid_type, self.name)
        valid_idx_type = [mstype.int32]
        validator.check_tensor_dtype_valid('indices', idx_type, valid_idx_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape, idx_shape, y_shape):
        validator.check("x rank", len(x_shape), "y rank", len(y_shape), Rel.EQ, self.name)
        x_rank = len(x_shape)
        validator.check_int_range(self.axis, -x_rank - 1, x_rank, Rel.INC_NEITHER, 'axis', self.name)
        validator.check_equal_int(len(idx_shape), 1, "rank of idx_shape", self.name)
        validator.check("size of indices", idx_shape[0], "dimension of y[axis]", y_shape[self.axis],
                        Rel.EQ, self.name)
        axis = self.axis if self.axis >= 0 else x_rank + self.axis
        for dim in range(x_rank):
            if dim != axis:
                validator.check('x dim %d' % dim, x_shape[dim], "y dim %d" % dim, y_shape[dim], Rel.EQ, self.name)
        return x_shape
