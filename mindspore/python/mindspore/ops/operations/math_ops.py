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

import numpy as np
from ... import context
from .. import signature as sig
from ..._checkparam import Validator as validator
from ..._checkparam import Rel
from ...common import dtype as mstype
from ...common.tensor import Tensor
from ...common._decorator import deprecated
from .._utils import get_broadcast_shape
from ..primitive import Primitive, PrimitiveWithInfer, PrimitiveWithCheck, prim_attr_register, _run_op


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

    def infer_min_shape(self, x_shape, y_shape):
        return get_broadcast_shape(x_shape, y_shape, self.name, "min_shape")

    def infer_max_shape(self, x_shape, y_shape):
        return get_broadcast_shape(x_shape, y_shape, self.name, "max_shape")


class _MathBinaryOp(_BinaryOp):
    """
    Define math binary operators.
    """

    @staticmethod
    def do_infer_dtype(x_dtype, y_dtype, valid_dtype=mstype.number_type, prim_name=None):
        """Staticmethod of infer dtype for _MathBinaryOp."""
        args_type = {"x": x_dtype, "y": y_dtype}
        complex_types = [mstype.tensor_type(mstype.complex64), mstype.tensor_type(mstype.complex128)]
        if x_dtype in complex_types or y_dtype in complex_types:
            type_infer_dict = {
                (mstype.complex64, mstype.complex64): mstype.tensor_type(mstype.complex64),
                (mstype.complex64, mstype.float32): mstype.tensor_type(mstype.complex64),
                (mstype.float32, mstype.complex64): mstype.tensor_type(mstype.complex64),
                (mstype.complex128, mstype.complex128): mstype.tensor_type(mstype.complex128),
                (mstype.complex128, mstype.float64): mstype.tensor_type(mstype.complex128),
                (mstype.float64, mstype.complex128): mstype.tensor_type(mstype.complex128),
            }
            if (x_dtype.element_type(), y_dtype.element_type()) not in type_infer_dict.keys():
                raise TypeError('Complex math binary op expecting Tensor [complex64, complex64],'
                                + '[complex64, float32], [float32, complex64], [complex128, complex128],'
                                + '[complex128, float64], [float64, complex128],'
                                + f'but got : [{format(x_dtype)},{format(y_dtype)}].')
            return type_infer_dict.get((x_dtype.element_type(), y_dtype.element_type()))

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


class Ger(Primitive):
    r"""
    Ger product of `x1` and `x2`. Calculate the outer product of two one-dimensional arrays.If `x1` is a 1D Tensor of
    shape :math:`(m,)` and `x2` is a 1D Tensor of shape :math:`(n,)`,then `output` must be a Tensor of shape
    :math:`(m * n)`.


    Inputs:
        - **x1** - (Tensor) - 1-D input Tensor, with dtype of float16 or float32.
        - **x2** - (Tensor) - 1-D input Tensor, with dtype of float16 or float32.

    Outputs:
        Tensor, output matrix with the same dtype as inputs.With `x1` shape :math:`(m,)` and
        `x2` shape of :math:`(n,)`,the `output` has shape :math:`(m * n)`.

    Raises:
        TypeError: If `x1` or `x2` is not a Tensor.
        TypeError: If the dtype of `x1` and `x2` is neither float16 nor float32.
        ValueError: If `x1` or `x2` is not a 1D Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x1 = Tensor([1., 2., 3., 4.], mindspore.float32)
        >>> x2 = Tensor([1., 2., 3.], mindspore.float32)
        >>> ger = ops.Ger()
        >>> output = ger(x1, x2)
        >>> print(output)
        [[ 1.  2.  3.]
         [ 2.  4.  6.]
         [ 3.  6.  9.]
         [ 4.  8. 12.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Ger"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])


class Add(_MathBinaryOp):
    r"""
    Adds two input tensors element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} = x_{i} + y_{i}

    Inputs:
        - **x** (Union[Tensor, Number, bool]) - The first input is a number, or a bool,
          or a tensor whose data type is number or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number, or a bool when the first input
          is a tensor, or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, Number, bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case 1: x and y are both Tensor.
        >>> add = ops.Add()
        >>> x = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> y = Tensor(np.array([4, 5, 6]).astype(np.float32))
        >>> output = add(x, y)
        >>> print(output)
        [5. 7. 9.]
        >>> # case 2: x is a scalar and y is a Tensor
        >>> add = ops.Add()
        >>> x = Tensor(1, mindspore.int32)
        >>> y = Tensor(np.array([4, 5, 6]).astype(np.float32))
        >>> output = add(x, y)
        >>> print(output)
        [5. 6. 7.]
        >>> # the data type of x is int32, the data type of y is float32,
        >>> # and the output is the data format of higher precision float32.
        >>> print(output.dtype)
        Float32
    """

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = x + y
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class Addcdiv(Primitive):
    """
    Performs the element-wise division of tensor x1 by tensor x2,
    multiply the result by the scalar value and add it to input_data.

    .. math::
        y[i] = input_data[i] + value[i] * (x1[i] / x2[i])

    Inputs:
        - **input_data**(Tensor) - The tensor to be added, with data type float16 and float32.
        - **x1** (Tensor) - The numerator tensor, with data type float16 and float32.
        - **x2** (Tensor) - The denominator tensor, with data type float16 and float32.
        - **value** (Tensor) - The multiplier for tensor x1/x2, with data type float16, float32.

    Outputs:
        Tensor y, has the same shape and dtype as x1/x2.

    Raises:
        TypeError: If dtype of `x1`, `x2`, `value`, `input_data`is not tensor.
        TypeError: If dtype of `input_data` is not one of: float32, float16.
        TypeError: If dtype of `x1` or 'x2' is not one of: float32, float16.
        TypeError: If dtype of `value` is not one of: float32, float16.
        ValueError: If `x1` could not be broadcast to a tensor with shape of `x2`.
        ValueError: If `value` could not be broadcast to tensors with shapes of `x1/x2`.
        ValueError: If `input_data` could not be broadcast to tensors with shapes of `value*(x1/x2)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_data = Tensor(np.array([1, 1, 1, 1]), mindspore.float32)
        >>> x1 = Tensor(np.array([1, 2, 3, 4]), mindspore.float32)
        >>> x2 = Tensor(np.array([4, 3, 2, 1]), mindspore.float32)
        >>> value = Tensor([1], mindspore.float32)
        >>> addcdiv = ops.Addcdiv()
        >>> y = addcdiv(input_data, x1, x2, value)
        >>> print(y)
        [1.25      1.6666667 2.5       5.       ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Addcdiv """
        self.init_prim_io_names(inputs=['input_data', 'x1', 'x2', 'value'], outputs=['y'])


class Addcmul(Primitive):
    """
    Performs the element-wise product of tensor x1 and tensor x2,
    multiply the result by the scalar value and add it to input_data.

    .. math::
        output[i] = input_data[i] + value[i] * (x1[i] * x2[i])

    Inputs:
        - **input_data**(Tensor) - The tensor to be added, with data type float16, float32 and int32.
        - **x1** (Tensor) - The tensor to be multiplied, with data type float16, float32 and int32.
        - **x2** (Tensor) - The tensor to be multiplied, with data type float16, float32 and int32.
        - **value** (Tensor) - The multiplier for tensor x1*x2, with data type float16, float32 and int32.

    Outputs:
        Tensor, has the same shape and dtype as x1*x2.

    Raises:
        TypeError: If dtype of `x1`, `x2`, `value`, `input_data`is not tensor.
        TypeError: If dtype of `input_data` is not one of: float32, float16, int32.
        TypeError: If dtype of `x1` or 'x2' is not one of: float32, float16, int32.
        TypeError: If dtype of `value` is not one of: float32, float16, int32.
        ValueError: If `x1` could not be broadcast to a tensor with shape of `x2`.
        ValueError: If `value` could not be broadcast to tensors with shapes of `x1` * `x2`.
        ValueError: If `input_data` could not be broadcast to tensors with shapes of `value*(x1*x2)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_data = Tensor(np.array([1, 1, 1]), mindspore.float32)
        >>> x1 = Tensor(np.array([[1], [2], [3]]), mindspore.float32)
        >>> x2 = Tensor(np.array([[1, 2, 3]]), mindspore.float32)
        >>> value = Tensor([1], mindspore.float32)
        >>> addcmul = ops.Addcmul()
        >>> y = addcmul(input_data, x1, x2, value)
        >>> print(y)
        [[ 2.  3.  4.]
         [ 3.  5.  7.]
         [ 4.  7. 10.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Addcmul """
        self.init_prim_io_names(inputs=['input_data', 'x1', 'x2', 'value'], outputs=['y'])


class TensorAdd(_MathBinaryOp):
    """
    Same as operator Add. TensorAdd will be deprecated in the future.
    Please use Add instead.
    """

    @deprecated("1.1", "Add", True)
    @prim_attr_register
    def __init__(self):
        """Initialize TensorAdd."""
        _MathBinaryOp.__init__(self)

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = x + y
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class AssignAdd(Primitive):
    """
    Updates a `Parameter` by adding a value to it.

    Inputs of `variable` and `value` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.
    If `value` is a number, the number is automatically converted to Tensor,
    and the data type is consistent with the Tensor data type involved in the operation.

    Note:
        Since `variable` is a data type Parameter, the data type cannot be changed,
        so only the type of `value` is allowed to be promoted to the type of `variable`.
        And the conversion type supported by different devices will be different,
        it is recommended to use the same data type when using this operator.

    Inputs:
        - **variable** (Parameter) - The `Parameter`.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        - **value** (Union[numbers.Number, Tensor]) - The value to be added to the `variable`.
          It must have the same shape as `variable` if it is a Tensor.
          it is recommended to use the same data type when using this operator.

    Outputs:
        Tensor, has the same data type and shape as original `variable`.

    Raises:
        TypeError: If `value` is neither Number nor Tensor.
        RuntimeError: If the data type of `variable` and `value` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.

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
        >>> print(net.variable.asnumpy())
        [101]
    """
    __mindspore_signature__ = (
        sig.make_sig('ref', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('value', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize AssignAdd"""
        self.init_prim_io_names(inputs=['ref', 'value'], outputs=['ref'])
        self.add_prim_attr('side_effect_mem', True)


class AssignSub(Primitive):
    """
    Updates a `Parameter` by subtracting a value from it.

    Inputs of `variable` and `value` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.
    If `value` is a number, the number is automatically converted to Tensor,
    and the data type is consistent with the Tensor data type involved in the operation.

    Note:
        Since `variable` is a data type Parameter, the data type cannot be changed,
        so only the type of `value` is allowed to be promoted to the type of `variable`.
        And the conversion type supported by different devices will be different,
        it is recommended to use the same data type when using this operator.

    Inputs:
        - **variable** (Parameter) - The `Parameter`.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank be should be less than 8.
        - **value** (Union[numbers.Number, Tensor]) - The value to be subtracted from the `variable`.
          It must have the same shape as `variable` if it is a Tensor.
          it is recommended to use the same data type when using this operator.

    Outputs:
        Tensor, has the same data type and shape as original `variable`.

    Raises:
        TypeError: If `value` is neither Number nor Tensor.
        RuntimeError: If the data type of `x`, `y` conversion of Parameter is required
                      when data type conversion of Parameter is not supported.

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
        [-99]
    """

    __mindspore_signature__ = (
        sig.make_sig('val', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('value', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize AssignSub"""
        self.init_prim_io_names(inputs=['val', 'value'], outputs=['val'])
        self.add_prim_attr('side_effect_mem', True)


class _Reduce(PrimitiveWithInfer):
    """
    Definition of base class of reduction class operators.

    Args:
         keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                           If false, don't keep these dimensions. Default: False.
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
        if not isinstance(axis['dtype'], mstype.tensor_type) and axis_v is None:
            raise ValueError(f"For '{self.name}', the 'axis' cannot be None, but got {axis}.")
        if -1 in input_shp:
            if axis_v is None:
                max_v = max(input_shp)
                if 'max_shape' and 'min_shape' in input_x:
                    input_max_shp = input_x['max_shape']
                    max_v = max(input_max_shp)
                axis_shape_list = axis['shape']
                if len(axis_shape_list) != 1:
                    raise ValueError(f"For '{self.name}', the shape of 'axis' must be 1-D, but "
                                     f"got {len(axis_shape_list)}.")
                axis_shape = axis_shape_list[0]
                if axis_shape == -1 and not self.keep_dims:
                    out_shape = np.array([-2]).tolist()
                    output_min_shape = input_x['min_shape']
                    output_max_shape = input_x['max_shape']
                elif not self.keep_dims:
                    out_shape = -1 * np.ones_like(input_shp[:-axis_shape])
                    out_shape = out_shape.tolist()
                    output_min_shape = np.ones_like(out_shape).tolist()
                    output_max_shape = max_v * np.ones_like(out_shape)
                    output_max_shape = output_max_shape.tolist()
                else:
                    out_shape = -1 * np.ones_like(input_shp)
                    out_shape = out_shape.tolist()
                    output_min_shape = np.ones_like(input_shp).tolist()
                    output_max_shape = max_v * np.ones_like(input_shp)
                    output_max_shape = output_max_shape.tolist()
            else:
                output_max_shape = _infer_shape_reduce(input_x['max_shape'], axis_v, self.keep_dims, self.name)
                output_min_shape = _infer_shape_reduce(input_x['min_shape'], axis_v, self.keep_dims, self.name)
                out_shape = _infer_shape_reduce(input_shp, axis_v, self.keep_dims, self.name)
        else:
            if axis_v is None:
                raise ValueError(f"For {self.name}, axis must be const, its value cannot be None.")
            out_shape = _infer_shape_reduce(input_shp, axis_v, self.keep_dims, self.name)
            output_max_shape = out_shape
            output_min_shape = out_shape

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
                if isinstance(axis_v, int):
                    pass
                elif axis_v:
                    axis_v = tuple(set(axis_v))
                else:
                    axis_v = tuple(range(len(input_x['shape'])))
                value = np_reduce_func(value, axis_v, keepdims=self.keep_dims)
                value = np.array(value)
                value = Tensor(value)
        return {'shape': out_shape,
                'min_shape': output_min_shape,
                'max_shape': output_max_shape,
                'dtype': input_x['dtype'],
                'value': value}

    def __infer__(self, input_x, axis):
        return self.do_infer(input_x, axis)


class ReduceMean(_Reduce):
    """
    Reduces a dimension of a tensor by averaging all elements in the dimension, by default. And also can reduce
    a dimension of `x` along the axis. Determine whether the dimensions of the output and input are the same by
    controlling `keep_dims`.

    Args:
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions. Default: False.

    Inputs:
        - **x** (Tensor[Number]) - The input tensor. The dtype of the tensor to be reduced is number.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed. Must be in the range [-rank(`x`), rank(`x`)).

    Outputs:
        Tensor, has the same dtype as the `x`.

        - If axis is (), and keep_dims is False,
          the output is a 0-D tensor representing the mean of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `x` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = ops.ReduceMean(keep_dims=True)
        >>> output = op(x, 1)
        >>> result = output.shape
        >>> print(result)
        (3, 1, 5, 6)
        >>> # case 1: Reduces a dimension by averaging all elements in the dimension.
        >>> x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mindspore.float32)
        >>> output = op(x)
        >>> print(output)
        [[[5.]]]
        >>> print(output.shape)
        (1, 1, 1)
        >>> # case 2: Reduces a dimension along the axis 0
        >>> output = op(x, 0)
        >>> print(output)
        [[[4. 4. 4. 4. 4. 4.]
          [5. 5. 5. 5. 5. 5.]
          [6. 6. 6. 6. 6. 6.]]]
        >>> # case 3: Reduces a dimension along the axis 1
        >>> output = op(x, 1)
        >>> print(output)
        [[[2. 2. 2. 2. 2. 2.]]
         [[5. 5. 5. 5. 5. 5.]]
         [[8. 8. 8. 8. 8. 8.]]]
        >>> # case 4: Reduces a dimension along the axis 2
        >>> output = op(x, 2)
        >>> print(output)
        [[[1.       ]
          [2.       ]
          [3.       ]]
         [[4.       ]
          [5.       ]
          [6.       ]]
         [[7.0000005]
          [8.       ]
          [9.       ]]]
    """


class ReduceSum(_Reduce):
    """
    Reduces a dimension of a tensor by summing all elements in the dimension, by default. And also can reduce a
    dimension of `x` along the axis. Determine whether the dimensions of the output and input are the same by
    controlling `keep_dims`.

    Args:
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions. Default: False.

    Inputs:
         - **x** (Tensor[Number]) - The input tensor. The dtype of the tensor to be reduced is number.
           :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
         - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
           Only constant value is allowed. Must be in the range [-rank(`x`), rank(`x`)).

    Outputs:
        Tensor, has the same dtype as the `x`.

        - If axis is (), and keep_dims is False,
          the output is a 0-D tensor representing the sum of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `x` is not a Tensor.
        ValueError: If `axis` is None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = ops.ReduceSum(keep_dims=True)
        >>> output = op(x, 1)
        >>> output.shape
        (3, 1, 5, 6)
        >>> # case 1: Reduces a dimension by summing all elements in the dimension.
        >>> x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mindspore.float32)
        >>> output = op(x)
        >>> print(output)
        [[[270.]]]
        >>> print(output.shape)
        (1, 1, 1)
        >>> # case 2: Reduces a dimension along axis 0.
        >>> output = op(x, 0)
        >>> print(output)
        [[[12. 12. 12. 12. 12. 12.]
          [15. 15. 15. 15. 15. 15.]
          [18. 18. 18. 18. 18. 18.]]]
        >>> # case 3: Reduces a dimension along axis 1.
        >>> output = op(x, 1)
        >>> print(output)
        [[[ 6.  6.  6.  6.  6.  6.]]
         [[15. 15. 15. 15. 15. 15.]]
         [[24. 24. 24. 24. 24. 24.]]]
        >>> # case 4: Reduces a dimension along axis 2.
        >>> output = op(x, 2)
        >>> print(output)
        [[[ 6.]
          [12.]
          [18.]]
         [[24.]
          [30.]
          [36.]]
         [[42.]
          [48.]
          [54.]]]
    """

    @prim_attr_register
    def __init__(self, keep_dims=False):
        """Initialize ReduceSum"""
        super(ReduceSum, self).__init__(keep_dims)
        self.__setattr_flag__ = True

    def __infer__(self, input_x, axis):
        return self.do_infer(input_x, axis, mstype.number_type + (mstype.bool_,))


class ReduceAll(_Reduce):
    """
    Reduces a dimension of a tensor by the "logicalAND" of all elements in the dimension, by default. And also can
    reduce a dimension of `x` along the axis. Determine whether the dimensions of the output and input are the same by
    controlling `keep_dims`.

    Args:
       keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                         If false, don't keep these dimensions. Default : False.

    Inputs:
        - **x** (Tensor[bool]) - The input tensor. The dtype of the tensor to be reduced is bool.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed. Must be in the range [-rank(x), rank(x)).

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
        TypeError: If `x` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[True, False], [True, True]]))
        >>> op = ops.ReduceAll(keep_dims=True)
        >>> # case 1: Reduces a dimension by the "logicalAND" of all elements in the dimension.
        >>> output = op(x)
        >>> print(output)
        [[False]]
        >>> print(output.shape)
        (1, 1)
        >>> # case 2: Reduces a dimension along axis 0.
        >>> output = op(x, 0)
        >>> print(output)
        [[ True False]]
        >>> # case 3: Reduces a dimension along axis 1.
        >>> output = op(x, 1)
        >>> print(output)
        [[False]
        [ True]]
    """

    def __infer__(self, input_x, axis):
        return self.do_infer(input_x, axis, (mstype.bool_,))


class ReduceAny(_Reduce):
    """
    Reduces a dimension of a tensor by the "logical OR" of all elements in the dimension, by default. And also can
    reduce a dimension of `x` along the axis. Determine whether the dimensions of the output and input are the same by
    controlling `keep_dims`.

    Args:
       keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                         If false, don't keep these dimensions. Default : False.

    Inputs:
        - **x** (Tensor[bool]) - The input tensor. The dtype of the tensor to be reduced is bool.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed. Must be in the range [-rank(x), rank(x)).

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
        TypeError: If `x` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[True, False], [True, True]]))
        >>> op = ops.ReduceAny(keep_dims=True)
        >>> # case 1: Reduces a dimension by the "logical OR" of all elements in the dimension.
        >>> output = op(x)
        >>> print(output)
        [[ True]]
        >>> print(output.shape)
        (1, 1)
        >>> # case 2: Reduces a dimension along axis 0.
        >>> output = op(x, 0)
        >>> print(output)
        [[ True True]]
        >>> # case 3: Reduces a dimension along axis 1.
        >>> output = op(x, 1)
        >>> print(output)
        [[True]
        [ True]]
    """

    def __infer__(self, input_x, axis):
        return self.do_infer(input_x, axis, (mstype.bool_,))


class ReduceMax(_Reduce):
    """
    Reduces a dimension of a tensor by the maximum value in this dimension, by default. And also can
    reduce a dimension of `x` along the axis. Determine whether the dimensions of the output and input are the same by
    controlling `keep_dims`.

    Args:
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions. Default : False.

    Inputs:
         - **x** (Tensor[Number]) - The input tensor. The dtype of the tensor to be reduced is number.
           :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
         - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
           Only constant value is allowed. Must be in the range [-rank(x), rank(x)).

    Outputs:
        Tensor, has the same dtype as the `x`.

        - If axis is (), and keep_dims is False,
          the output is a 0-D tensor representing the maximum of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `x` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = ops.ReduceMax(keep_dims=True)
        >>> output = op(x, 1)
        >>> result = output.shape
        >>> print(result)
        (3, 1, 5, 6)
        >>> # case 1: Reduces a dimension by the maximum value of all elements in the dimension.
        >>> x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mindspore.float32)
        >>> output = op(x)
        >>> print(output)
        [[[9.]]]
        >>> print(output.shape)
        (1, 1, 1)
        >>> # case 2: Reduces a dimension along axis 0.
        >>> output = op(x, 0)
        >>> print(output)
        [[[7. 7. 7. 7. 7. 7.]
          [8. 8. 8. 8. 8. 8.]
          [9. 9. 9. 9. 9. 9.]]]
        >>> # case 3: Reduces a dimension along axis 1.
        >>> output = op(x, 1)
        >>> print(output)
        [[[3. 3. 3. 3. 3. 3.]]
         [[6. 6. 6. 6. 6. 6.]]
         [[9. 9. 9. 9. 9. 9.]]]
        >>> # case 4: Reduces a dimension along axis 2.
        >>> output = op(x, 2)
        >>> print(output)
        [[[1.]
          [2.]
          [3.]]
         [[4.]
          [5.]
          [6.]]
         [[7.]
          [8.]
          [9.]]]
    """

    @prim_attr_register
    def __init__(self, keep_dims=False):
        """Initialize ReduceMax."""
        super(ReduceMax, self).__init__(keep_dims)
        self.__setattr_flag__ = True

    def __infer__(self, input_x, axis):
        return self.do_infer(input_x, axis, mstype.number_type + (mstype.bool_,))


class ReduceMin(_Reduce):
    """
    Reduces a dimension of a tensor by the minimum value in the dimension, by default. And also can
    reduce a dimension of `x` along the axis. Determine whether the dimensions of the output and input are the same by
    controlling `keep_dims`.

    Args:
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions. Default : False.

    Inputs:
        - **x** (Tensor[Number]) - The input tensor. The dtype of the tensor to be reduced is number.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed. Must be in the range [-rank(x), rank(x)).

    Outputs:
        Tensor, has the same dtype as the `x`.

        - If axis is (), and keep_dims is False,
          the output is a 0-D tensor representing the minimum of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `x` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = ops.ReduceMin(keep_dims=True)
        >>> output = op(x, 1)
        >>> result = output.shape
        >>> print(result)
        (3, 1, 5, 6)
        >>> # case 1: Reduces a dimension by the minimum value of all elements in the dimension.
        >>> x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mindspore.float32)
        >>> output = op(x)
        >>> print(output)
        [[[1.]]]
        >>> print(output.shape)
        (1, 1, 1)
        >>> # case 2: Reduces a dimension along axis 0.
        >>> output = op(x, 0)
        >>> print(output)
        [[[1. 1. 1. 1. 1. 1.]
          [2. 2. 2. 2. 2. 2.]
          [3. 3. 3. 3. 3. 3.]]]
        >>> # case 3: Reduces a dimension along axis 1.
        >>> output = op(x, 1)
        >>> print(output)
        [[[1. 1. 1. 1. 1. 1.]]
         [[4. 4. 4. 4. 4. 4.]]
         [[7. 7. 7. 7. 7. 7.]]]
        >>> # case 4: Reduces a dimension along axis 2.
        >>> output = op(x, 2)
        >>> print(output)
        [[[1.]
          [2.]
          [3.]]
         [[4.]
          [5.]
          [6.]]
         [[7.]
          [8.]
          [9.]]]
    """


class ReduceProd(_Reduce):
    """
    Reduces a dimension of a tensor by multiplying all elements in the dimension, by default. And also can
    reduce a dimension of `x` along the axis. Determine whether the dimensions of the output and input are the same by
    controlling `keep_dims`.

    Args:
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions. Default : False.

    Inputs:
        - **x** (Tensor[Number]) - The input tensor. The dtype of the tensor to be reduced is number.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed. Must be in the range [-rank(x), rank(x)).

    Outputs:
        Tensor, has the same dtype as the `x`.

        - If axis is (), and keep_dims is False,
          the output is a 0-D tensor representing the product of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `x` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple or list.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = ops.ReduceProd(keep_dims=True)
        >>> output = op(x, 1)
        >>> result = output.shape
        >>> print(result)
        (3, 1, 5, 6)
        >>> # case 1: Reduces a dimension by multiplying all elements in the dimension.
        >>> x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mindspore.float32)
        >>> output = op(x)
        >>> print(output)
        [[[2.2833798e+33]]]
        >>> print(output.shape)
        (1, 1, 1)
        >>> # case 2: Reduces a dimension along axis 0.
        >>> output = op(x, 0)
        >>> print(output)
        [[[ 28.  28.  28.  28.  28.  28.]
          [ 80.  80.  80.  80.  80.  80.]
          [162. 162. 162. 162. 162. 162.]]]
        >>> # case 3: Reduces a dimension along axis 1.
        >>> output = op(x, 1)
        >>> print(output)
        [[[  6.   6.   6.   6.   6.   6.]]
         [[120. 120. 120. 120. 120. 120.]]
         [[504. 504. 504. 504. 504. 504.]]]
        >>> # case 4: Reduces a dimension along axis 2.
        >>> output = op(x, 2)
        >>> print(output)
        [[[1.00000e+00]
          [6.40000e+01]
          [7.29000e+02]]
         [[4.09600e+03]
          [1.56250e+04]
          [4.66560e+04]]
         [[1.17649e+05]
          [2.62144e+05]
          [5.31441e+05]]]
    """


class CumProd(PrimitiveWithInfer):
    """
    Computes the cumulative product of the tensor x along axis.
    For example, if input is a vector of size N, the result will also be a vector of size N, with elements.

    .. math::
        y_i = x_1 * x_2 * x_3 * ... * x_i

    Args:
        exclusive (bool): If true, perform exclusive cumulative product. Default: False.
        reverse (bool): If true, reverse the result along axis. Default: False

    Inputs:
        - **x** (Tensor[Number]) - The input tensor.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        - **axis** (int) - The dimensions to compute the cumulative product.
          Only constant value is allowed.

    Outputs:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `exclusive` or `reverse` is not a bool.
        TypeError: If `axis` is not an int.
        ValueError: If `axis` is None.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> a, b, c, = 1, 2, 3
        >>> x = Tensor(np.array([a, b, c]).astype(np.float32))
        >>> op0 = ops.CumProd()
        >>> output0 = op0(x, 0) # output=[a, a * b, a * b * c]
        >>> op1 = ops.CumProd(exclusive=True)
        >>> output1 = op1(x, 0) # output=[1, a, a * b]
        >>> op2 = ops.CumProd(reverse=True)
        >>> output2 = op2(x, 0) # output=[a * b * c, b * c, c]
        >>> op3 = ops.CumProd(exclusive=True, reverse=True)
        >>> output3 = op3(x, 0) # output=[b * c, c, 1]
        >>> print(output0)
        [1. 2. 6.]
        >>> print(output1)
        [1. 1. 2.]
        >>> print(output2)
        [6. 6. 3.]
        >>> print(output3)
        [6. 3. 1.]
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [5, 3, 5]]).astype(np.float32))
        >>> output4 = op0(x, 0)
        >>> output5 = op0(x, 1)
        >>> print(output4)
        [[ 1.  2.  3.]
         [ 4. 10. 18.]
         [20. 30. 90.]]
        >>> print(output5)
        [[1.  2.   6.]
         [4. 20. 120.]
         [5. 15.  75.]]
    """

    @prim_attr_register
    def __init__(self, exclusive=False, reverse=False):
        """Initialize CumProd."""
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
            raise ValueError(f"For '{self.name}', the 'axis' cannot be None, but got {axis}.")


class Cdist(Primitive):
    """
    Computes batched the p-norm distance between each pair of the two collections of row vectors.

    Args:
        p (float): P value for the p-norm distance to calculate between each vector pair, P ∈ [0,∞]. Default: 2.0.

    Inputs:
        - **input_x** (Tensor) - Input tensor of shape :math:`(B, P, M)`.
          Letter :math:`B` represents 0 or positive int number.
          When :math:`B` is equal to 0, it means this dimension can be ignored,
          i.e. shape of the tensor is :math:`(P, M)`.
        - **input_y** (Tensor) - Input tensor of shape :math:`(B, R, M)`.

    Outputs:
        Tensor, has the same dtype as `input_x`, which shape is :math:`(B, P, R)`.

    Raises:
        TypeError: If `input_x` or `input_y` is not a Tensor.
        TypeError: If dtype of `input_x` or `input_y` is neither float16 nor float32.
        TypeError: If `p` is not a float.
        ValueError: If `p` is a negative float.
        ValueError: If dimension of `input_x` is not the same as `input_y`.
        ValueError: If dimension of `input_x` or `input_y` is neither 2 nor 3.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
        >>> input_y = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
        >>> op = ops.Cdist(p=2.0)
        >>> output = op(input_x, input_y)
        >>> print(output)
        [[[2.8284273 2.8284273]
          [1.4142137 1.4142137]]]
    """

    @prim_attr_register
    def __init__(self, p=2.0):
        """Initialize Cdist"""
        validator.check_value_type("p", p, [float], self.name)
        validator.check_non_negative_float(p, "p", self.name)
        self.init_prim_io_names(inputs=['input_x', 'input_y'], outputs=['output'])


class LpNorm(Primitive):
    """
    Returns the matrix norm or vector norm of a given tensor.

    .. math::
        output = sum(abs(input)**p)**(1/p)

    Args:
        axis(int,list,tuple): Specifies which dimension or dimensions of input to calculate the norm across.
        p(int): The order of norm. Default: 2.
        keep_dims(bool): Whether the output tensors have dim retained or not. Default: False.
        epsilon(float): A value added to the denominator for numerical stability. Default: 1e-12.

    Inputs:
        - **input** (Tensor) - Input tensor.

    Outputs:
        Tensor, has the same dtype as `input`, which shape depends on the args axis.For example, if the size of input
        is (2, 3, 4), axis is [0, 1], Outputs' shape will be (4,).

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If dtype of `input` is not one of: float16, float32.
        TypeError: If `p` is not an int.
        TypeError: If `axis` is not an int, a tuple or a list.
        TypeError: If `axis` is a tuple or a list, but the element of `axis` is not an int.
        TypeError: If `keep_dims` is not a bool.
        ValueError: If the element of `axis` is out of the range [-len(input.shape), len(input.shape)).
        ValueError: If the length of shape of `axis` is bigger than the length of shape of `input`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
        >>> op = ops.LpNorm(axis=[0, 1], p=2, keep_dims=False)
        >>> output = op(input_x)
        >>> print(output)
        [ 9.165152 10.954452]
    """

    @prim_attr_register
    def __init__(self, axis, p=2, keep_dims=False, epsilon=1e-12):
        """Initialize LpNorm"""
        validator.check_value_type("p", p, [int], self.name)
        validator.check_value_type("axis", axis, [int, tuple, list], self.name)
        validator.check_value_type("keep_dims", keep_dims, [bool], self.name)
        validator.check_value_type("epsilon", epsilon, [float], self.name)
        validator.check_non_negative_int(p, "p", self.name)
        validator.check_non_negative_float(epsilon, "epsilon", self.name)
        if isinstance(axis, int):
            self.add_prim_attr('axis', [self.axis])
        else:
            for element_of_axis in axis:
                validator.check_value_type("element_of_axis", element_of_axis, [int], self.name)
        self.init_prim_io_names(inputs=['input'], outputs=['output'])


class MatMul(PrimitiveWithCheck):
    r"""
    Multiplies matrix `a` and matrix `b`.

    .. math::

        (Output)_{i j}=\sum_{k=1}^{p} a_{i k} b_{k j}=a_{i 1} b_{1 j}+a_{i 2} b_{2 j}+\cdots+a_{i p} b_{p j}, p\in N

    where the :math:`i,j` indicates the output of the i-th row and j-th column element.

    Args:
        transpose_a (bool): If true, `a` is transposed before multiplication. Default: False.
        transpose_b (bool): If true, `b` is transposed before multiplication. Default: False.

    Inputs:
        - **a** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(N, C)`. If
          `transpose_a` is True, its shape must be :math:`(N, C)` after transpose.
        - **b** (Tensor) - The second tensor to be multiplied. The shape of the tensor is :math:`(C, M)`. If
          `transpose_b` is True, its shape must be :math:`(C, M)` after transpose.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, M)`.

    Raises:
        TypeError: If `transpose_a` or `transpose_b` is not a bool.
        ValueError: If the column of matrix dimensions of `a` is not equal to
                    the row of matrix dimensions of `b`.
        ValueError: If length of shape of `a` or `b` is not equal to 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = Tensor(np.ones(shape=[1, 3]), mindspore.float32)
        >>> b = Tensor(np.ones(shape=[3, 4]), mindspore.float32)
        >>> matmul = ops.MatMul()
        >>> output = matmul(a, b)
        >>> print(output)
        [[3. 3. 3. 3.]]
    """

    @prim_attr_register
    def __init__(self, transpose_a=False, transpose_b=False):
        """Initialize MatMul."""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['output'])
        cls_name = self.name
        validator.check_value_type("transpose_a", transpose_a, [bool], cls_name)
        validator.check_value_type("transpose_b", transpose_b, [bool], cls_name)

    def check_shape_size(self, x1, x2):
        if len(x1) != 2 or len(x2) != 2:
            raise ValueError(f"For '{self.name}', inputs 'x', 'y' should have the same dimension size and "
                             f"be equal to 2, but got the size of 'x': ({len(x1)}) and the size of 'y': ({len(x2)}).")

    def check_shape(self, x1, x2):
        self.check_shape_size(x1, x2)
        cls_name = self.name

        # validate whether last two dims satisfying matrix multiply
        x1_last = x1[-2:]
        x2_last = x2[-2:]
        x1_col = x1_last[not self.transpose_a]
        x2_row = x2_last[self.transpose_b]
        if np.all(np.array(x1) != -1) and np.all(np.array(x2) != -1):
            if x1_col != x2_row:
                raise ValueError(f"For '{cls_name}', the input dimensions must be equal, but got 'x1_col': {x1_col} "
                                 f"and 'x2_row': {x2_row}. And 'x' shape {x1}(transpose_a={self.transpose_a}), "
                                 f"'y' shape {x2}(transpose_b={self.transpose_b}).")
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

        \text{output}[..., :, :] = \text{matrix}(x[..., :, :]) * \text{matrix}(y[..., :, :])

    The first input tensor must be not less than `3` and the second input must be not less than `2`.

    Args:
        transpose_a (bool): If true, the last two dimensions of `x` is transposed before multiplication.
            Default: False.
        transpose_b (bool): If true, the last two dimensions of `y` is transposed before multiplication.
            Default: False.

    Inputs:
        - **x** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(*B, N, C)`,
          where :math:`*B` represents the batch size which can be multidimensional, :math:`N` and :math:`C` are the
          size of the last two dimensions. If `transpose_a` is True, its shape must be :math:`(*B, C, N)`.
        - **y** (Tensor) - The second tensor to be multiplied. The shape of the tensor is :math:`(*B, C, M)`. If
          `transpose_b` is True, its shape must be :math:`(*B, M, C)`.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(*B, N, M)`.

    Raises:
        TypeError: If `transpose_a` or `transpose_b` is not a bool.
        ValueError: If length of shape of `x` is not equal to length of shape of `y` or
                    length of shape of `x` is less than 3.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.ones(shape=[2, 4, 1, 3]), mindspore.float32)
        >>> y = Tensor(np.ones(shape=[2, 4, 3, 4]), mindspore.float32)
        >>> batmatmul = ops.BatchMatMul()
        >>> output = batmatmul(x, y)
        >>> print(output)
        [[[[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]]
         [[[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]]]
        >>> x = Tensor(np.ones(shape=[2, 4, 3, 1]), mindspore.float32)
        >>> y = Tensor(np.ones(shape=[2, 4, 3, 4]), mindspore.float32)
        >>> batmatmul = ops.BatchMatMul(transpose_a=True)
        >>> output = batmatmul(x, y)
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
        """Initialize BatchMatMul."""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['output'])
        cls_name = self.name
        validator.check_value_type("transpose_a", transpose_a, [bool], cls_name)
        validator.check_value_type("transpose_b", transpose_b, [bool], cls_name)

    def check_shape_size(self, x, y):
        if len(x) < 3 or len(y) < 2:
            raise ValueError(f"For '{self.name}', input 'x' should be greater than or equal to 3, input 'y' should "
                             f"be greater than or equal to 2, but got 'x' size: {len(x)}, 'y' size: {len(y)}.")


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
        >>> x = Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float32))
        >>> cumsum = ops.CumSum()
        >>> # case 1: along the axis 0
        >>> y = cumsum(x, 0)
        >>> print(y)
        [[ 3.  4.  6. 10.]
         [ 4. 10. 13. 19.]
         [ 8. 13. 21. 26.]
         [ 9. 16. 28. 35.]]
        >>> # case 2: along the axis 1
        >>> y = cumsum(x, 1)
        >>> print(y)
        [[ 3.  7. 13. 23.]
         [ 1.  7. 14. 23.]
         [ 4.  7. 15. 22.]
         [ 1.  4. 11. 20.]]
        >>> # Next demonstrate exclusive and reverse, along axis 1
        >>> # case 3: exclusive = True
        >>> cumsum = ops.CumSum(exclusive=True)
        >>> y = cumsum(x, 1)
        >>> print(y)
        [[ 0.  3.  7. 13.]
         [ 0.  1.  7. 14.]
         [ 0.  4.  7. 15.]
         [ 0.  1.  4. 11.]]
        >>> # case 4: reverse = True
        >>> cumsum = ops.CumSum(reverse=True)
        >>> y = cumsum(x, 1)
        >>> print(y)
        [[23. 20. 16. 10.]
         [23. 22. 16.  9.]
         [22. 18. 15.  7.]
         [20. 19. 16.  9.]]
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
            raise ValueError(f"For '{self.name}', the 'axis' cannot be None, but got {axis}.")
        validator.check_value_type('axis', axis['value'], [int], cls_name)
        valid_dtypes = [mstype.uint8, mstype.int8, mstype.int32, mstype.float16, mstype.float32, mstype.float64]
        validator.check_tensor_dtype_valid('x', x['dtype'], valid_dtypes, cls_name)
        return {'shape': x_shp,
                'dtype': x['dtype'],
                'value': None}


class AddN(Primitive):
    """
    Computes addition of all input tensors element-wise.

    All input tensors must have the same shape.

    Inputs:
        - **x** (Union(tuple[Tensor], list[Tensor])) - The input tuple or list
          is made up of multiple tensors whose dtype is number to be added together.

    Outputs:
        Tensor, has the same shape and dtype as each entry of the `x`.

    Raises:
        TypeError: If `x` is neither tuple nor list.

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
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> y = Tensor(np.array([4, 5, 6]), mindspore.float32)
        >>> output = net(x, y, x, y)
        >>> print(output)
        [10. 14. 18.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize AddN."""
        self.init_prim_io_names(inputs=["inputs"], outputs=["sum"])

    def check_elim(self, inputs):
        if len(inputs) != 1:
            return False, None
        if isinstance(inputs[0], Tensor):
            return True, inputs[0]
        raise TypeError(f"For '{self.name}', the type of 'inputs[0]' should be a tensor, but "
                        f"got {type(inputs[0]).__name__}, "
                        f"or the length of 'inputs' should not be equal to 1, but got ({len(inputs)}).")


class AccumulateNV2(Primitive):
    """
    Computes accumulation of all input tensors element-wise.

    AccumulateNV2 is similar to AddN, but there is a significant difference
    among them: AccumulateNV2 will not wait for all of its inputs to be ready
    before summing. That is to say, AccumulateNV2 is able to save
    memory when inputs are ready at different time since the minimum temporary
    storage is proportional to the output size rather than the input size.

    Inputs:
        - **x** (Union(tuple[Tensor], list[Tensor])) - The input tuple or list
          is made up of multiple tensors whose dtype is number to be added together.
          Each element of tuple or list should have the same shape.

    Outputs:
        Tensor, has the same shape and dtype as each entry of the `x`.

    Raises:
        TypeError: If `x` is neither tuple nor list.
        ValueError: If there is an input element with a different shape.

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
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> y = Tensor(np.array([4, 5, 6]), mindspore.float32)
        >>> output = net(x, y, x, y)
        >>> print(output)
        [10. 14. 18.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize AccumulateNV2."""
        self.__setattr_flag__ = True
        self.init_prim_io_names(inputs=["inputs"], outputs=["sum"])

    def check_elim(self, inputs):
        if len(inputs) != 1:
            return False, None
        if isinstance(inputs[0], Tensor):
            return True, inputs[0]
        raise TypeError(f"For '{self.name}', the type of 'inputs[0]' should be a tensor, "
                        f"but got {type(inputs[0]).__name__}, "
                        f"or the length of 'inputs' should not be equal to 1, but got ({len(inputs)}).")


class Neg(PrimitiveWithInfer):
    """
    Returns a tensor with negative values of the input tensor element-wise.

    .. math::

        out_{i} = - x_{i}

    Inputs:
        - **x** (Tensor) - The input tensor whose dtype is number.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Outputs:
        Tensor, has the same shape and dtype as input.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> neg = ops.Neg()
        >>> x = Tensor(np.array([1, 2, -1, 2, 0, -3.5]), mindspore.float32)
        >>> output = neg(x)
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
    Adds `v` into specified rows of `x`. Computes `y` = `x`; y[i,] += `v`.

    Args:
        indices (Union[int, tuple]): Indices into the left-most dimension of `x`, and determines which rows of `x`
            to add with `v`. It is an integer or a tuple, whose value is in [0, the first dimension size of `x`).

    Inputs:
        - **x** (Tensor) - The first input is a tensor whose data type is float16, float32 or int32.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        - **input_v** (Tensor) - The second input is a tensor that has the same dimension sizes as `x` except
          the first dimension, which must be the same as indices' size. It has the same data type with `x`.

    Outputs:
        Tensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `indices` is neither int nor tuple.
        TypeError: If `indices` is a tuple whose elements are not all int.
        ValueError: If length of shape of `x` is not equal to length of shape of `input_v`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> indices = (0, 1)
        >>> x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]), mindspore.float32)
        >>> input_v = Tensor(np.array([[0.5, 1.0], [1.0, 1.5]]), mindspore.float32)
        >>> inplaceAdd = ops.InplaceAdd(indices)
        >>> output = inplaceAdd(x, input_v)
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
                raise ValueError(f"For '{self.name}', the value of 'indices' must be "
                                 f"in [0, {x_shape[0]}), but got {i}.")
        x_rank = len(x_shape)
        for idx in range(x_rank)[1:]:
            validator.check('v dim %d' % idx, v_shape[idx], "x dim %d" % idx, x_shape[idx], Rel.EQ, self.name)

        return x_shape


class InplaceSub(PrimitiveWithInfer):
    """
    Subtracts `v` into specified rows of `x`. Computes `y` = `x`; y[i,] -= `v`.

    Args:
        indices (Union[int, tuple]): Indices into the left-most dimension of `x`, and determines which rows of `x`
            to subtract with `v`. It is an int or tuple, whose value is in [0, the first dimension size of `x`).

    Inputs:
        - **x** (Tensor) - The first input is a tensor whose data type is float16, float32 or int32.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        - **input_v** (Tensor) - The second input is a tensor who has the same dimension sizes as `x` except
          the first dimension, which must be the same as indices' size. It has the same data type with `x`.

    Outputs:
        Tensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `indices` is neither int nor tuple.
        TypeError: If `indices` is a tuple whose elements are not all int.
        ValueError: If length of shape of `x` is not equal to length of shape of `input_v`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> indices = (0, 1)
        >>> x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]), mindspore.float32)
        >>> input_v = Tensor(np.array([[0.5, 1.0], [1.0, 1.5]]), mindspore.float32)
        >>> inplaceSub = ops.InplaceSub(indices)
        >>> output = inplaceSub(x, input_v)
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
                raise ValueError(f"For '{self.name}', the value of 'indices' must be "
                                 f"in [0, {x_shape[0]}), but got {i}.")
        x_rank = len(x_shape)
        for idx in range(x_rank)[1:]:
            validator.check('v dim %d' % idx, v_shape[idx], "x dim %d" % idx, x_shape[idx], Rel.EQ, self.name)

        return x_shape


class Sub(_MathBinaryOp):
    """
    Subtracts the second input tensor from the first input tensor element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} = x_{i} - y_{i}

    Inputs:
        - **x** (Union[Tensor, Number, bool]) - The first input is a number, or a bool,
          or a tensor whose data type is number or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number, or a bool when the first input
          is a tensor, or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not a Number or a bool or a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([4, 5, 6]), mindspore.int32)
        >>> sub = ops.Sub()
        >>> output = sub(x, y)
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

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} = x_{i} * y_{i}

    Inputs:
        - **x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, Number, bool.
        ValueError: If `x` and `y` are not the same shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> y = Tensor(np.array([4.0, 5.0, 6.0]), mindspore.float32)
        >>> mul = ops.Mul()
        >>> output = mul(x, y)
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

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} = (x_{i} - y_{i}) * (x_{i} - y_{i}) = (x_{i} - y_{i})^2

    Inputs:
        - **x** (Union[Tensor, Number, bool]) - The first input is a number, or a bool,
          or a tensor whose data type is float16, float32, int32 or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number, or a bool when the first input
          is a tensor or a tensor whose data type is float16, float32, int32 or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: if `x` and `y` is not a Number or a bool or a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> y = Tensor(np.array([2.0, 4.0, 6.0]), mindspore.float32)
        >>> squared_difference = ops.SquaredDifference()
        >>> output = squared_difference(x, y)
        >>> print(output)
        [1. 4. 9.]
    """


class Square(Primitive):
    """
    Returns square of a tensor element-wise.

    .. math::

        out_{i} = (x_{i})^2

    Inputs:
        - **x** (Tensor) - The input tensor whose dtype is number.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Outputs:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> square = ops.Square()
        >>> output = square(x)
        >>> print(output)
        [1. 4. 9.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Square"""
        self.init_prim_io_names(inputs=['input_x'], outputs=['output'])


class Rsqrt(Primitive):
    r"""
    Computes reciprocal of square root of input tensor element-wise.

    .. math::

        out_{i} =  \frac{1}{\sqrt{x_{i}}}

    Inputs:
        - **x** (Tensor) - The input of Rsqrt. Each element must be a non-negative number.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Outputs:
        Tensor, has the same type and shape as `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.

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


class Sqrt(PrimitiveWithCheck):
    r"""
    Returns square root of a tensor element-wise.

    Note:
        When there are some negative number, it will return a Tensor whose specific position is nan.

    .. math::

        out_{i} =  \sqrt{x_{i}}


    Inputs:
        - **x** (Tensor) - The input tensor whose dtype is number.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Outputs:
        Tensor, has the same shape and data type as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        ValueError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 4.0, 9.0]), mindspore.float32)
        >>> sqrt = ops.Sqrt()
        >>> output = sqrt(x)
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
        """Infer the value of input for Sqrt."""
        if x is not None:
            x = x.asnumpy()
            out = np.sqrt(x)
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class Reciprocal(PrimitiveWithInfer):
    r"""
    Returns reciprocal of a tensor element-wise.

    .. math::

        out_{i} =  \frac{1}{x_{i}}

    Inputs:
        - **x** (Tensor) - The input tensor.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Outputs:
        Tensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> reciprocal = ops.Reciprocal()
        >>> output = reciprocal(x)
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


class Pow(Primitive):
    """
    Computes a tensor to the power of the second input.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} = x_{i} ^{ y_{i}}

    Inputs:
        - **x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, Number, bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> y = 3.0
        >>> pow = ops.Pow()
        >>> output = pow(x, y)
        >>> print(output)
        [ 1.  8. 64.]
        >>>
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> y = Tensor(np.array([2.0, 4.0, 3.0]), mindspore.float32)
        >>> pow = ops.Pow()
        >>> output = pow(x, y)
        >>> print(output)
        [ 1. 16. 64.]
    """

    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        """Initialize _BinaryOp"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])

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
        - **x** (Tensor) - The input tensor.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Outputs:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> exp = ops.Exp()
        >>> output = exp(x)
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


class Expm1(Primitive):
    r"""
    Returns exponential then minus 1 of a tensor element-wise.

    .. math::

        out_i = e^{x_i} - 1

    Inputs:
        - **x** (Tensor) - The input tensor. With float16 or float32 data type.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Outputs:
        Tensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.0, 1.0, 2.0, 4.0]), mindspore.float32)
        >>> expm1 = ops.Expm1()
        >>> output = expm1(x)
        >>> print(output)
        [ 0.        1.718282  6.389056 53.598152]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Expm1."""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])



class HistogramFixedWidth(PrimitiveWithInfer):
    """
    Returns a rank 1 histogram counting the number of entries in values that fall into every bin. The bins are equal
    width and determined by the inputs `range` and the arguments `nbins`.

    Args:
        dtype (str): An optional attribute. The dtype must be "int32". Default: "int32".
        nbins (int): The number of histogram bins, the type is a positive integer.

    Inputs:
        - **x** (Tensor) - Numeric Tensor. Must be one of the following types: int32, float32, float16.
        - **range** (Tensor) - Must have the same data type as `x`, and the shape is (2,).
          x <= range[0] will be mapped to histogram[0], x >= range[1] will be mapped to histogram[-1].

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
        >>> range_op = Tensor([0.0, 5.0], mindspore.float16)
        >>> hist = ops.HistogramFixedWidth(5)
        >>> output = hist(x, range_op)
        >>> print(output)
        [2 1 1 0 2]
    """

    @prim_attr_register
    def __init__(self, nbins, dtype='int32'):
        """Initialize HistogramFixedWidth."""
        self.nbins = validator.check_value_type("nbins", nbins, [int], self.name)
        validator.check_int(nbins, 1, Rel.GE, "nbins", self.name)
        valid_values = ['int32']
        self.dtype = validator.check_string(dtype, valid_values, "dtype", self.name)
        self.init_prim_io_names(inputs=['x', 'range'], outputs=['y'])
        self.add_prim_attr('dtype', 3)

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

    .. warning::
        If the input value of operator Log is within the range (0, 0.01] or [0.95, 1.05], the output accuracy
        is subject to change.

    Inputs:
        - **x** (Tensor) - The input tensor. The data type must be float16, float32 or float64. The value must be
          greater than 0. :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should
          be less than 8.

    Outputs:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> log = ops.Log()
        >>> output = log(x)
        >>> print(output)
        [0.        0.6931472 1.3862944]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Log."""
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


class Log1p(Primitive):
    """
    Returns the natural logarithm of one plus the input tensor element-wise.

    .. math::
        out_i = {log_e}(x_i + 1)

    Inputs:
        - **x** (Tensor) - The input tensor. With float16 or float32 data type.
          The value must be greater than -1.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Outputs:
        Tensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> log1p = ops.Log1p()
        >>> output = log1p(x)
        >>> print(output)
        [0.6931472 1.0986123 1.609438 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Log1p."""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class Erf(Primitive):
    r"""
    Computes the Gauss error function of `x` element-wise.

    .. math::

        erf(x)=\frac{2} {\sqrt{\pi}} \int\limits_0^{x} e^{-t^{2}} dt

    Inputs:
        - **x** (Tensor) - The input tensor. The data type must be float16 or float32.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Outputs:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([-1, 0, 1, 2, 3]), mindspore.float32)
        >>> erf = ops.Erf()
        >>> output = erf(x)
        >>> print(output)
        [-0.8427168   0.          0.8427168   0.99530876  0.99997765]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Erf"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class Erfc(Primitive):
    r"""
    Computes the complementary error function of `x` element-wise.

    .. math::

        erfc(x) = 1 - \frac{2} {\sqrt{\pi}} \int\limits_0^{x} e^{-t^{2}} dt

    Inputs:
        - **x** (Tensor) - The input tensor. The data type must be float16 or float32.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Outputs:
        Tensor, has the same shap dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([-1, 0, 1, 2, 3]), mindspore.float32)
        >>> erfc = ops.Erfc()
        >>> output = erfc(x)
        >>> print(output)
        [1.8427168e+00 1.0000000e+00 1.5728319e-01 4.6912432e-03 2.2351742e-05]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Erfc"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

class Minimum(_MathBinaryOp):
    r"""
    Computes the minimum of input tensors element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.
    If one of the elements being compared is a NaN, then that element is returned.

    .. math::
        output_i = min(x_i, y_i)

    Inputs:
        - **x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, Number, bool.
        ValueError: If `x` and `y` are not the same shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case 1 : same data type
        >>> x = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.float32)
        >>> y = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float32)
        >>> minimum = ops.Minimum()
        >>> output = minimum(x, y)
        >>> print(output)
        [1. 2. 3.]
        >>> # case 2 : different data type
        >>> x = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.int32)
        >>> y = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float32)
        >>> output = minimum(x, y)
        >>> print(output.dtype)
        Float32
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

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.
    If one of the elements being compared is a NaN, then that element is returned.

    .. math::
        output_i = max(x_i, y_i)

    Inputs:
        - **x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, Number, bool.
        ValueError: If `x` and `y` are not the same shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case 1 : same data type
        >>> x = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.float32)
        >>> y = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float32)
        >>> maximum = ops.Maximum()
        >>> output = maximum(x, y)
        >>> print(output)
        [4. 5. 6.]
        >>> # case 2 : different data type
        >>> x = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.int32)
        >>> y = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float32)
        >>> output = maximum(x, y)
        >>> print(output.dtype)
        Float32
    """


class RealDiv(_MathBinaryOp):
    """
    Divides the first input tensor by the second input tensor in floating-point type element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} = x_{i} / y_{i}

    Inputs:
        - **x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, Number, bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> y = Tensor(np.array([4.0, 5.0, 6.0]), mindspore.float32)
        >>> realdiv = ops.RealDiv()
        >>> output = realdiv(x, y)
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

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} = \frac{x_i}{y_i}

    Inputs:
        - **x** (Union[Tensor, number.Number, bool]) - The first input is a number.Number or
          a bool or a tensor whose data type is
          `number <https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
          `bool_ <https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        - **y** (Union[Tensor, number.Number, bool]) - The second input is a number.Number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool\_.
          When the first input is Scalar, the second input must be a Tensor whose data type is number or bool\_.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not a number.Number or a bool or a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case 1 :has same data type and shape of the two inputs
        >>> x = Tensor(np.array([-4.0, 5.0, 6.0]), mindspore.float32)
        >>> y = Tensor(np.array([3.0, 2.0, 3.0]), mindspore.float32)
        >>> div = ops.Div()
        >>> output = div(x, y)
        >>> print(output)
        [-1.3333334  2.5        2.        ]
        >>> # case 2 : different data type and shape of the two inputs
        >>> x = Tensor(np.array([-4.0, 5.0, 6.0]), mindspore.int32)
        >>> y = Tensor(2, mindspore.float32)
        >>> output = div(x, y)
        >>> print(output)
        [-2.  2.5  3.]
        >>> print(output.dtype)
        Float32
    """

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = np.array(x / y, x.dtype)
            return Tensor(out)
        return None


class DivNoNan(_MathBinaryOp):
    r"""
    Computes a safe divide and returns 0 if the y is zero.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::
        output_{i} = \begin{cases}
        0, & \text{ if } y_{i} = 0\\
        x_{i} / y_{i}, & \text{ if } y_{i} \ne 0
        \end{cases}

    Inputs:
        - **x** (Union[Tensor, number.Number, bool]) - The first input is a number.Number or
          a bool or a tensor whose data type is
          `number <https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
          `bool_ <https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        - **y** (Union[Tensor, number.Number, bool]) - The second input is a number.Number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool\_.
          When the first input is Scalar, the second input must be a Tensor whose data type is number or bool\_.


    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not a number.Number or a bool or a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([-1.0, 0., 1.0, 5.0, 6.0]), mindspore.float32)
        >>> y = Tensor(np.array([0., 0., 0., 2.0, 3.0]), mindspore.float32)
        >>> div_no_nan = ops.DivNoNan()
        >>> output = div_no_nan(x, y)
        >>> print(output)
        [0.  0.  0.  2.5 2. ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize _BinaryOp"""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])



class MulNoNan(_MathBinaryOp):
    r"""
    Computes `x` * `y` element-wise. If `y` is zero, no matter what `x` is, it will return 0.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcasted.
    When the inputs are one tensor and one scalar, the scalar could only be a constant.

    .. math::
        output_{ij} = \begin{cases}
        0, & if\ x_{ij} = 0\ or\ y_{ij} = 0;\\
        x_{ij} * y_{ij}, & otherwise.
        \end{cases}

    Note:
        The shapes of `x` and `y` should be the same or can be broadcasted.
        This is noncommutative: if `y` is NaN or infinite and `x` is 0, the result will be NaN.

    Inputs:
        - **x** (Union[Tensor]) - The first input is a tensor whose data type is one of
          float16, float32, int32, int64 currently or scalar.
        - **y** (Union[Tensor]) - The second input is a tensor whose data type is one of
          float16, float32, int32, int64 currently or scalar.

    Outputs:
        Tensor, the shape is the same as the shape after broadcasting,
        and the data type is the one with higher precision among the two inputs.


    Supported Platforms:
        ``Ascend``

    Raises:
        TypeError: If neither `x` nor `y` is a bool Tensor.

    Examples:
        >>> # case 1 : same data type and shape of two inputs, there are some 0 in y.
        >>> x = Tensor(np.array([[-1.0, 6.0, np.inf], [np.nan, -7.0, 4.0]]), mindspore.float32)
        >>> y = Tensor(np.array([[-1.0, 4.0, 0], [0, -3.0, 1.0]]), mindspore.float32)
        >>> mul_no_nan = ops.MulNoNan()
        >>> output = mul_no_nan(x, y)
        >>> print(output)
        [[ 1. 24. 0.]
        [ 0. 21. 4.]]
        >>> # case 2 : the shape of two inputs is same, there are some 0 in x, y.
        >>> x = Tensor(np.array([[-1.0, 6.0, 0], [0, np.nan, 4.0]]), mindspore.int32)
        >>> y = Tensor(np.array([[-1.0, 4.0, np.inf], [np.nan, 0, 1.0]]), mindspore.float32)
        >>> output = mul_no_nan(x, y)
        >>> print(output)
        [[ 1. 24. nan]
         [ nan  0. 4.]]
        >>> print(output.dtype)
        Float32
        >>> # case 3 : the y is a scalar.
        >>> x = Tensor(np.array([[-1.0, 6.0, 0], [0, np.nan, 4.0]]), mindspore.float32)
        >>> y = Tensor(0, mindspore.float32)
        >>> output = mul_no_nan(x, y)
        >>> print(output)
        [[ 0. 0. 0.]
         [ 0. 0. 0.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize _BinaryOp"""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])


class FloorDiv(Primitive):
    """
    Divides the first input tensor by the second input tensor element-wise and round down to the closest integer.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} = \\text{floor}( \\frac{x_i}{y_i})

    where the :math:`floor` indicates the Floor operator, for more details, please refer to the Floor operator.

    Inputs:
        - **x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([2, 4, -1]), mindspore.int32)
        >>> y = Tensor(np.array([3, 3, 3]), mindspore.int32)
        >>> floor_div = ops.FloorDiv()
        >>> output = floor_div(x, y)
        >>> print(output)
        [ 0  1 -1]
    """
    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        """Initialize FloorDiv."""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])

class TruncateDiv(Primitive):
    """
    Divides the first input tensor by the second input tensor element-wise for integer types, negative numbers will
    round fractional quantities towards zero.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Note:
        Broadcasting is supported.

    Inputs:
        - **x** (Union[Tensor, Number, bool]) - The first input is a number, or a bool,
          or a tensor whose data type is number or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number, or a bool when the first input
          is a tensor, or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, Number, bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([2, 4, -1]), mindspore.int32)
        >>> y = Tensor(np.array([3, 3, 3]), mindspore.int32)
        >>> truncate_div = ops.TruncateDiv()
        >>> output = truncate_div(x, y)
        >>> print(output)
        [0 1 0]
    """

    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        """Initialize TruncateDiv."""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])


class TruncateMod(Primitive):
    r"""
    Returns the remainder of division element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. warning::
        - The input data does not support 0.
        - When the elements of input exceed 2048 , the accuracy of operator cannot guarantee the requirement of
          double thousandths in the mini form.
        - Due to different architectures, the calculation results of this operator on NPU and CPU may be inconsistent.
        - If shape is expressed as (D1,D2... ,Dn), then D1\*D2... \*DN<=1000000,n<=8.

    Inputs:
        - **x** (Union[Tensor, numbers.Number, bool]) - The first input is a number, or a bool,
          or a tensor whose data type is number or bool.
        - **y** (Union[Tensor, numbers.Number, bool]) - The second input is a number, or a bool when the first input
          is a tensor, or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision among the two inputs.

    Raises:
        TypeError: If neither `x` nor `y` is one of the following: Tensor, number, bool.
        TypeError: If neither `x` nor `y` is a Tensor.
        ValueError: If the shape `x` and `y` cannot be broadcasted to each other.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([2, 4, -1]), mindspore.int32)
        >>> y = Tensor(np.array([3, 3, 3]), mindspore.int32)
        >>> truncate_mod = ops.TruncateMod()
        >>> output = truncate_mod(x, y)
        >>> print(output)
        [ 2  1 -1]
    """
    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        """Initialize TruncateMod."""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])

class Mod(_MathBinaryOp):
    r"""
    Computes the remainder of dividing the first input tensor by the second input tensor element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar. When the inputs are two tensors,
    both dtypes cannot be bool, and the shapes of them could be broadcast. When the inputs are one tensor
    and one scalar, the scalar could only be a constant.

    .. math::

        out_{i} = x_{i} \text{ % } y_{i}

    .. warning::
        - The input data does not support 0.
        - When the elements of input exceed 2048, the accuracy of operator cannot guarantee the requirement of
          double thousandths in the mini form.
        - Due to different architectures, the calculation results of this operator on NPU and CPU may be inconsistent.
        - If shape is expressed as (D1,D2... ,Dn), then D1\*D2... \*DN<=1000000,n<=8.

    Inputs:
        - **x** (Union[Tensor, numbers.Number, bool]) - The first input is a number, a bool
          or a tensor whose data type is number.
        - **y** (Union[Tensor, numbers.Number, bool]) - When the first input is a tensor, The second input
          could be a number, a bool or a tensor whose data type is number. When the first input is a number or a bool
          the second input must be a tensor whose data type is number.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If neither `x` nor `y` is one of the following: Tensor, number, bool.
        TypeError: If neither `x` nor `y` is a Tensor.
        ValueError: If the shape `x` and `y` cannot be broadcasted to each other.


    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-4.0, 5.0, 6.0]), mindspore.float32)
        >>> y = Tensor(np.array([3.0, 2.0, 3.0]), mindspore.float32)
        >>> mod = ops.Mod()
        >>> output = mod(x, y)
        >>> print(output)
        [-1.  1.  0.]
    """

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            return Tensor(np.fmod(x, y))
        return None


class Floor(Primitive):
    r"""
    Rounds a tensor down to the closest integer element-wise.

    .. math::

        out_i = \lfloor x_i \rfloor

    Inputs:
        - **x** (Tensor) - The input tensor. Its element data type must be float16 or float32.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not in [float16, float32, float64].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.1, 2.5, -1.5]), mindspore.float32)
        >>> floor = ops.Floor()
        >>> output = floor(x)
        >>> print(output)
        [ 1.  2. -2.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Floor."""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class FloorMod(_MathBinaryOp):
    r"""
    Computes the remainder of division element-wise. It's a flooring divide.
    E.g. :math:`floor(x / y) * y + mod(x, y) = x`.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} =\text{floor}(x_{i} // y_{i})

    where the :math:`floor` indicates the Floor operator, for more details, please refer to the Floor operator.

    .. warning::
        - The input data does not support 0.
        - When the elements of input exceeds 2048 , the accuracy of operator cannot guarantee the requirement of
          double thousandths in the mini form.
        - Due to different architectures, the calculation results of this operator on NPU and CPU may be inconsistent.
        - If shape is expressed as (D1,D2... ,Dn), then D1\*D2... \*DN<=1000000,n<=8.

    Inputs:
        - **x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision of the two inputs.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([2, 4, -1]), mindspore.int32)
        >>> y = Tensor(np.array([3, 3, 3]), mindspore.int32)
        >>> floor_mod = ops.FloorMod()
        >>> output = floor_mod(x, y)
        >>> print(output)
        [2 1 2]
    """


class Ceil(PrimitiveWithInfer):
    r"""
    Rounds a tensor up to the closest integer element-wise.

    .. math::

        out_i = \lceil x_i \rceil = \lfloor x_i \rfloor + 1

    Inputs:
        - **x** (Tensor) - The input tensor. It's element data type must be float16 or float32.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If x is not a Tensor.
        TypeError: If dtype of x is not float16 or float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.array([1.1, 2.5, -1.5]), mindspore.float32)
        >>> ceil_op = ops.Ceil()
        >>> output = ceil_op(x)
        >>> print(output)
        [ 2.  3. -1.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Ceil."""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class Xdivy(Primitive):
    """
    Divides the first input tensor by the second input tensor element-wise. Returns zero when `x` is zero.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **x** (Union[Tensor, Number, bool]) - The first input is a number, or a bool,
          or a tensor whose data type is float16, float32 or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number,
          or a bool when the first input is a tensor, or a tensor whose data type is float16, float32 or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, Number, bool.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.array([2, 4, -1]), mindspore.float32)
        >>> y = Tensor(np.array([2, 2, 2]), mindspore.float32)
        >>> xdivy = ops.Xdivy()
        >>> output = xdivy(x, y)
        >>> print(output)
        [ 1.   2.  -0.5]
    """

    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        """Initialize Xdivy."""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])

class Xlogy(Primitive):
    r"""
    Computes the first input tensor multiplied by the logarithm of second input tensor element-wise.
    Returns zero when `x` is zero.

    .. math::

        out_i = x_{i}\ln{y_{i}}

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **x** (Union[Tensor, number.Number, bool]) - The first input is a number.Number or
          a bool or a tensor whose data type is
          `number <https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
          `bool_ <https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        - **y** (Union[Tensor, number.Number, bool]) - The second input is a number.Number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool\_.
          When the first input is Scalar, the second input must be a Tensor whose data type is number or bool\_.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not a number.Number or a bool or a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.array([-5, 0, 4]), mindspore.float32)
        >>> y = Tensor(np.array([2, 2, 2]), mindspore.float32)
        >>> xlogy = ops.Xlogy()
        >>> output = xlogy(x, y)
        >>> print(output)
        [-3.465736   0.        2.7725887]
    """
    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        """Initialize Xlogy."""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])


class Acosh(Primitive):
    r"""
    Computes inverse hyperbolic cosine of the inputs element-wise.

    .. math::

        out_i = \cosh^{-1}(input_i)

    .. warning::
        Given an input tensor x, the function computes inverse hyperbolic cosine of every element.
        Input range is [1, inf].

    Inputs:
        - **x** (Tensor) - The data type should be one of the following types: float16, float32.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Outputs:
        Tensor, has the same shape and type as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor, dtype
        >>> acosh = ops.Acosh()
        >>> x = Tensor(np.array([1.0, 1.5, 3.0, 100.0]), dtype.float32)
        >>> output = acosh(x)
        >>> print(output)
        [0. 0.9624236 1.7627472 5.298292]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Acosh"""
        self.init_prim_io_names(inputs=['x'], outputs='output')

class Cosh(Primitive):
    r"""
    Computes hyperbolic cosine of input element-wise.

    .. math::

        out_i = \cosh(x_i)

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> cosh = ops.Cosh()
        >>> x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = cosh(x)
        >>> print(output)
        [1.0289385 1.364684 1.048436 1.0040528]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Cosh"""


class Asinh(Primitive):
    r"""
    Computes inverse hyperbolic sine of the input element-wise.

    .. math::

        out_i = \sinh^{-1}(input_i)

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
          The data type should be one of the following types: float16, float32.

    Outputs:
        Tensor, has the same shape and type as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> asinh = ops.Asinh()
        >>> x = Tensor(np.array([-5.0, 1.5, 3.0, 100.0]), mindspore.float32)
        >>> output = asinh(x)
        >>> print(output)
        [-2.3124385  1.1947632  1.8184465  5.298342 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Asinh"""


class Sinh(Primitive):
    r"""
    Computes hyperbolic sine of the input element-wise.

    .. math::

        out_i = \sinh(x_i)

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> sinh = ops.Sinh()
        >>> x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float32)
        >>> output = sinh(x)
        >>> print(output)
        [0.6604918  0.28367308 0.44337422 0.6604918 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Sinh"""



class _LogicBinaryOp(_BinaryOp):
    """
    Define logic binary operators.
    """

    @staticmethod
    def do_infer_dtype(x_dtype, y_dtype, valid_type=mstype.number_type, prim_name=None):
        """Staticmethod of infer dtype for _LogicBinaryOp."""
        args_dtype = {"x": x_dtype, "y": y_dtype}
        validator.check_tensors_dtypes_same_and_valid(args_dtype, valid_type, prim_name)
        return mstype.tensor_type(mstype.bool_)

    def infer_dtype(self, x_dtype, y_dtype):
        return _LogicBinaryOp.do_infer_dtype(x_dtype, y_dtype, prim_name=self.name)


class Equal(_LogicBinaryOp):
    r"""
    Computes the equivalence between two tensors element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar, the scalar could only be a constant.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } x_{i} = y_{i} \\
            & \text{False,   if } x_{i} \ne y_{i}
            \end{cases}

    Inputs:
        - **x** (Union[Tensor, Number]) - The first input is a number or
          a tensor whose data type is number.
        - **y** (Union[Tensor, Number]) - The second input is a number
          when the first input is a tensor or a tensor whose data type is number.
          The data type is the same as the first input.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,and the data type is bool.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case 1: The shape of two inputs are different
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> equal = ops.Equal()
        >>> output = equal(x, 2.0)
        >>> print(output)
        [False True False]
        >>> # case 2: The shape of two inputs are the same
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([1, 2, 4]), mindspore.int32)
        >>> equal = ops.Equal()
        >>> output = equal(x, y)
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
    r"""
    Returns True if abs(x-y) is smaller than tolerance element-wise, otherwise False.

    .. math::

        out_i = \begin{cases}
        & \text{ if } \left | x_{i} - y_{i} \right | < \text{tolerance},\ \ True  \\
        & \text{ if } \left | x_{i} - y_{i} \right | \ge \text{tolerance},\ \  False
        \end{cases}

    where `tolerance` indicates Acceptable maximum tolerance.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower precision data type will be converted to
    the relatively highest precision data type.

    Args:
        tolerance (float): The maximum deviation that two elements can be considered equal. Default: 1e-05.

    Inputs:
        - **x** (Tensor) - A tensor. Must be one of the following types: float32, float16.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        - **y** (Tensor) - A tensor of the same type and shape as `x`.

    Outputs:
        Tensor, the shape is the same as the shape of `x`, and the data type is bool.

    Raises:
        TypeError: If `tolerance` is not a float.
        RuntimeError: If the data type of `x`, `y` conversion of Parameter is given
                      but data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> y = Tensor(np.array([2, 4, 6]), mindspore.float32)
        >>> approximate_equal = ops.ApproximateEqual(2.)
        >>> output = approximate_equal(x, y)
        >>> print(output)
        [ True  True  False]
    """

    @prim_attr_register
    def __init__(self, tolerance=1e-05):
        """Initialize ApproximateEqual"""
        validator.check_value_type("tolerance", tolerance, [float], self.name)


class EqualCount(PrimitiveWithInfer):
    """
    Computes the number of the same elements of two tensors.

    The two input tensors must have the same data type and shape.

    Inputs:
        - **x** (Tensor) - The first input tensor. If the data type and shape of `y` are determined, then `x`
          must be the same as `y`, and vice versa.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        - **y** (Tensor) - The second input tensor. If the data type and shape of `x` are determined, then `y`
          must be the same as `x`, and vice versa.

    Outputs:
        Tensor, with the type same as input tensor and shape as (1,).

    Raises:
        TypeError: If `x` or `y` is not a Tensor.
        ValueError: If shape of `x` is not equal to shape of `y`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([1, 2, 4]), mindspore.int32)
        >>> equal_count = ops.EqualCount()
        >>> output = equal_count(x, y)
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
    r"""
    Computes the non-equivalence of two tensors element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar, the scalar could only be a constant.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } x_{i} \ne y_{i} \\
            & \text{False,   if } x_{i} = y_{i}
            \end{cases}

    Inputs:
        - **x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,and the data type is bool.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, Number, bool.
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> not_equal = ops.NotEqual()
        >>> output = not_equal(x, 2.0)
        >>> print(output)
        [ True False  True]
        >>>
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([1, 2, 4]), mindspore.int32)
        >>> not_equal = ops.NotEqual()
        >>> output = not_equal(x, y)
        >>> print(output)
        [False False  True]
    """

    def infer_dtype(self, x_dtype, y_dtype):
        return _LogicBinaryOp.do_infer_dtype(x_dtype, y_dtype, mstype.number_type + (mstype.bool_,), self.name)


class Greater(_LogicBinaryOp):
    r"""
    Computes the boolean value of :math:`x > y` element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } x_{i}>y_{i} \\
            & \text{False,   if } x_{i}<=y_{i}
            \end{cases}

    Note:
        Broadcasting is supported.

    Inputs:
        - **x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> greater = ops.Greater()
        >>> output = greater(x, y)
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
    r"""
    Computes the boolean value of :math:`x >= y` element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } x_{i}>=y_{i} \\
            & \text{False,   if } x_{i}<y_{i}
            \end{cases}

    Inputs:
        - **x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> greater_equal = ops.GreaterEqual()
        >>> output = greater_equal(x, y)
        >>> print(output)
        [True True False]
    """

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = np.array(np.greater_equal(x, y))
            return Tensor(out)
        return None


class Lerp(Primitive):
    """
    Does a linear interpolation of two tensors start and end based on a float or tensor weight.

    If `weight` is a tensor, the shapes of three inputs need to be broadcast;
    If `weight` is a float, the shapes of `start` and `end` need to be broadcast.

    .. math::

        output_{i} = start_{i} + weight_{i} * (end_{i} - start_{i})

    Inputs:
        - **start** (Tensor) - The tensor with the starting points. Data type must be float16 or float32.
        - **end** (Tensor) - The tensor with the ending points. Data type must be float16 or float32.
        - **weight** (Union[float, Tensor]) – The weight for the interpolation formula. Must be a float
          or a scalar tensor with float16 or float32 data type.

    Outputs:
        Tensor, has the same type and shape as input `start`.

    Raises:
        TypeError: If `start` or `end` is not a tensor.
        TypeError: If `weight` is neither float nor tensor.
        TypeError: If dtype of `start` or `end` is neither float16 nor float32.
        TypeError: If dtype of `weight` is neither float16 nor float32 when it is a tensor.
        TypeError: If `start` and `end` have different data types.
        TypeError: If `start`, `end` and `weight` have different data types when `weight` is a tensor.
        ValueError: If `end` could not be broadcast to a tensor with shape of `start`.
        ValueError: If `weight` could not be broadcast to tensors with shapes of `start` and `end` when it is a tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> start = Tensor(np.array([1., 2., 3., 4.]), mindspore.float32)
        >>> end = Tensor(np.array([10., 10., 10., 10.]), mindspore.float32)
        >>> lerp = ops.Lerp()
        >>> output = lerp(start, end, 0.5)
        >>> print(output)
        [5.5 6. 6.5 7. ]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['start', 'end', 'weight'], outputs=['output'])


class Less(_LogicBinaryOp):
    r"""
    Computes the boolean value of :math:`x < y` element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } x_{i}<y_{i} \\
            & \text{False,   if } x_{i}>=y_{i}
            \end{cases}

    Inputs:
        - **x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,and the data type is bool.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, Number, bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> less = ops.Less()
        >>> output = less(x, y)
        >>> print(output)
        [False False True]
    """


class LessEqual(_LogicBinaryOp):
    r"""
    Computes the boolean value of :math:`x <= y` element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool , and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } x_{i}<=y_{i} \\
            & \text{False,   if } x_{i}>y_{i}
            \end{cases}

    Inputs:
        - **x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,and the data type is bool.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> less_equal = ops.LessEqual()
        >>> output = less_equal(x, y)
        >>> print(output)
        [ True False  True]
    """


class LogicalNot(Primitive):
    """
    Computes the "logical NOT" of a tensor element-wise.

    .. math::

        out_{i} = \\neg x_{i}

    Inputs:
        - **x** (Tensor) - The input tensor whose dtype is bool.
          :math:`(N,*)` where :math:`*` means,any number of additional dimensions.

    Outputs:
        Tensor, the shape is the same as the `x`, and the dtype is bool.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([True, False, True]), mindspore.bool_)
        >>> logical_not = ops.LogicalNot()
        >>> output = logical_not(x)
        >>> print(output)
        [False  True False]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize LogicalNot"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])



class LogicalAnd(_LogicBinaryOp):
    r"""
    Computes the "logical AND" of two tensors element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one bool.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them must be bool.
    When the inputs are one tensor and one bool, the bool object could only be a constant,
    and the data type of the tensor must be bool.

    .. math::

        out_{i} = x_{i} \wedge y_{i}

    Note:
        LogicalAnd supports broadcasting.

    Inputs:
        - **x** (Union[Tensor, bool]) - The first input is a bool or a tensor whose data type is bool.
        - **y** (Union[Tensor, bool]) - The second input is a bool when the first input is a tensor or
          a tensor whose data type is bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([True, False, True]), mindspore.bool_)
        >>> y = Tensor(np.array([True, True, False]), mindspore.bool_)
        >>> logical_and = ops.LogicalAnd()
        >>> output = logical_and(x, y)
        >>> print(output)
        [ True False False]
    """


class LogicalOr(_LogicBinaryOp):
    """
    Computes the "logical OR" of two tensors element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one bool.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them must be bool.
    When the inputs are one tensor and one bool, the bool object could only be a constant,
    and the data type of the tensor must be bool.

    .. math::

        out_{i} = x_{i} \\vee y_{i}

    Note:
        LogicalOr supports broadcasting.

    Inputs:
        - **x** (Union[Tensor, bool]) - The first input is a bool or a tensor whose data type is bool.
        - **y** (Union[Tensor, bool]) - The second input is a bool when the first input is a tensor or
          a tensor whose data type is bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([True, False, True]), mindspore.bool_)
        >>> y = Tensor(np.array([True, True, False]), mindspore.bool_)
        >>> logical_or = ops.LogicalOr()
        >>> output = logical_or(x, y)
        >>> print(output)
        [ True  True  True]
    """



class IsNan(Primitive):
    r"""
    Determines which elements are NaN for each position.

    .. math::

        out_i = \begin{cases}
          & \ True,\ \text{ if } x_{i} = \text{Nan} \\
          & \ False,\ \text{ if } x_{i} \ne  \text{Nan}
        \end{cases}

    where :math:`Nan` means not a number.

    Inputs:
        - **x** (Tensor) - The input tensor.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, has the same shape of input, and the dtype is bool.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> is_nan = ops.IsNan()
        >>> x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> output = is_nan(x)
        >>> print(output)
        [ True False False]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize IsNan"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


class IsInf(Primitive):
    r"""
    Determines which elements are inf or -inf for each position

    .. math::

        out_i = \begin{cases}
        & \text{ if } x_{i} = \text{Inf},\ \ True \\
        & \text{ if } x_{i} \ne \text{Inf},\ \ False
        \end{cases}

    where :math:`Inf` means not a number.

    Inputs:
        - **x** (Tensor) - The input tensor.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, has the same shape of input, and the dtype is bool.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> is_inf = ops.IsInf()
        >>> x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> output = is_inf(x)
        >>> print(output)
        [False False True]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize IsInf"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


class IsFinite(PrimitiveWithInfer):
    r"""
    Determines which elements are finite for each position.

    .. math::

        out_i = \begin{cases}
          & \text{ if } x_{i} = \text{Finite},\ \ True\  \\
          & \text{ if } x_{i} \ne \text{Finite},\ \ False
        \end{cases}

    Inputs:
        - **x** (Tensor) - The input tensor.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, has the same shape of input, and the dtype is bool.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> is_finite = ops.IsFinite()
        >>> x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> output = is_finite(x)
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
        - **x** (Tensor) - The input tensor. The data type must be float16 or float32.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, has the shape of `(1,)`, and the dtype is `mindspore.dtype.float32`.

    Raises:
        TypeError: If dtype of `x` is not in [float16, float32, float64].

    Supported Platforms:
        ``GPU``

    Examples:
        >>> float_status = ops.FloatStatus()
        >>> x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> result = float_status(x)
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
        validator.check_tensor_dtype_valid('x', x_dtype, [mstype.float32, mstype.float16, mstype.float64], self.name)
        return mstype.float32


class NPUAllocFloatStatus(PrimitiveWithInfer):
    """
    Allocates a flag to store the overflow status.

    The flag is a tensor whose shape is `(8,)` and data type is `mindspore.dtype.float32`.

    Note:
        Please refer to the Examples of :class:`mindspore.ops.NPUGetFloatStatus`.

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
    In addition, there are strict sequencing requirements for use, i.e., before using the NPUGetFloatStatus operator,
    need to ensure that the NPUClearFlotStatus and your compute has been executed.
    We use Depend on ensure the execution order.

    Inputs:
        - **x** (Tensor) - The output tensor of `NPUAllocFloatStatus`.
          The data type must be float16 or float32.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Outputs:
        Tensor, has the same shape as `x`. All the elements in the tensor will be zero.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> import mindspore.ops.functional as F
        >>> from mindspore.common import dtype as mstype
        >>> from mindspore.common.tensor import Tensor
        >>> from mindspore.ops import operations as P
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.alloc_status = P.NPUAllocFloatStatus()
        ...         self.get_status = P.NPUGetFloatStatus()
        ...         self.clear_status = P.NPUClearFloatStatus()
        ...         self.sub = P.Sub()
        ...         self.neg = P.Neg()
        ...
        ...     def construct(self, x):
        ...         init = self.alloc_status()
        ...         clear_status = self.clear_status(init)
        ...         x = F.depend(x, clear_status)
        ...         res = self.sub(x, self.neg(x))
        ...         init = F.depend(init, res)
        ...         get_status = self.get_status(init)
        ...         res = F.depend(res, get_status)
        ...         return res
        >>>
        >>> value = 5
        >>> data = np.full((2, 3), value, dtype=np.float16)
        >>> x = Tensor(data, dtype=mstype.float16)
        >>> net = Net()
        >>> res = net(x)
        >>> print(res)
        [[10. 10. 10.]
         [10. 10. 10.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize NPUGetFloatStatus"""

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
        In addition, there are strict sequencing requirements for use, i.e., before using the NPUGetFloatStatus
        operator, need to ensure that the NPUClearFlotStatus and your compute has been executed.
        We use :class:`mindspore.ops.Depend` on ensure the execution order.

        Please refer to the Examples of :class:`mindspore.ops.NPUGetFloatStatus`.

    Inputs:
        - **x** (Tensor) - The output tensor of `NPUAllocFloatStatus`.
          The data type must be float16 or float32.

    Outputs:
        Tensor, has the same shape as `x`. All the elements in the tensor will be zero.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> import mindspore.ops.functional as F
        >>> from mindspore.common import dtype as mstype
        >>> from mindspore.common.tensor import Tensor
        >>> from mindspore.ops import operations as P
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.alloc_status = P.NPUAllocFloatStatus()
        ...         self.get_status = P.NPUGetFloatStatus()
        ...         self.clear_status = P.NPUClearFloatStatus()
        ...         self.sub = P.Sub()
        ...         self.neg = P.Neg()
        ...
        ...     def construct(self, x):
        ...         init = self.alloc_status()
        ...         clear_status = self.clear_status(init)
        ...         x = F.depend(x, clear_status)
        ...         res = self.sub(x, self.neg(x))
        ...         init = F.depend(init, res)
        ...         get_status = self.get_status(init)
        ...         res = F.depend(res, get_status)
        ...         return res
        >>>
        >>> value = 5
        >>> data = np.full((2, 3), value, dtype=np.float16)
        >>> x = Tensor(data, dtype=mstype.float16)
        >>> net = Net()
        >>> res = net(x)
        >>> print(res)
        [[10. 10. 10.]
         [10. 10. 10.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize NPUClearFloatStatus"""

    def infer_shape(self, x_shape):
        cls_name = self.name
        validator.check_equal_int(len(x_shape), 1, "len(x_shape)", cls_name)
        validator.check_equal_int(x_shape[0], 8, "x_shape[0]", cls_name)
        return [8]

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, [mstype.float16, mstype.float32], self.name)
        return mstype.float32


class Cos(Primitive):
    r"""
    Computes cosine of input element-wise.

    .. warning::
        Currently support Float16, Float32 data type. If use Float64, there may
        be a problem of missing precision.

    .. math::
        out_i = cos(x_i)

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> cos = ops.Cos()
        >>> x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = cos(x)
        >>> print(output)
        [0.971338 0.67487574 0.95233357 0.9959527 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Cos"""


class ACos(Primitive):
    r"""
    Computes arccosine of input tensors element-wise.

    .. math::

        out_i = cos^{-1}(x_i)

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should less than 8.
          The data type should be one of the following types: float16, float32.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> acos = ops.ACos()
        >>> x = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float32)
        >>> output = acos(x)
        >>> print(output)
        [0.7377037 1.5307858 1.2661037 0.97641146]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ACos"""
        self.init_prim_io_names(inputs=['x'], outputs='output')


class Sin(Primitive):
    r"""
    Computes sine of the input element-wise.

    .. math::

        out_i = sin(x_i)

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> sin = ops.Sin()
        >>> x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float32)
        >>> output = sin(x)
        >>> print(output)
        [0.5810352  0.27635565 0.41687083 0.5810352 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Sin."""



class Asin(Primitive):
    r"""
    Computes arcsine of input tensors element-wise.

    .. math::

        out_i = sin^{-1}(x_i)

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          The data type should be one of the following types: float16, float32.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of x is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> asin = ops.Asin()
        >>> x = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float32)
        >>> output = asin(x)
        >>> print(output)
        [0.8330927  0.04001068  0.30469266  0.59438497]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Asin"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


class NMSWithMask(PrimitiveWithInfer):
    r"""
    When object detection problem is performed in the computer vision field, object detection algorithm generates
    a plurality of bounding boxes. Selects some bounding boxes in descending order of score(Descending order is not
    supported in Ascend platform currently). Use the box with the highest score calculate the overlap between other
    boxes and the current box, and delete the box based on a certain threshold(IOU). The IOU is as follows,

    .. math::
        \text{IOU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}

    .. warning::
        Only supports up to 2864 input boxes at one time.

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
        ValueError: If the `iou_threshold` is not a float number.
        ValueError:  if the first dimension of input Tensor is less than or equal to 0.
        TypeError: if the dtype of the `bboxes` is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
        return bboxes_shape, (num,), (num,)

    def infer_dtype(self, bboxes_dtype):
        validator.check_tensor_dtype_valid("bboxes", bboxes_dtype, [mstype.float16, mstype.float32], self.name)
        return bboxes_dtype, mstype.int32, mstype.bool_


class Abs(Primitive):
    r"""
    Returns absolute value of a tensor element-wise.

    .. math::

        out_i = |x_i|

    Inputs:
        - **x** (Tensor) - The input tensor. The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1.0, 1.0, 0.0]), mindspore.float32)
        >>> abs = ops.Abs()
        >>> output = abs(x)
        >>> print(output)
        [1. 1. 0.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Abs"""
        self.init_prim_io_names(inputs=['input_x'], outputs=['output'])


class Sign(Primitive):
    r"""
    Performs sign on the tensor element-wise.

    .. math::
        sign(x) = \begin{cases} -1, &if\ x < 0 \cr
        0, &if\ x = 0 \cr
        1, &if\ x > 0\end{cases}

    Inputs:
        - **x** (Tensor) - The input tensor.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
         >>> x = Tensor(np.array([[2.0, 0.0, -1.0]]), mindspore.float32)
         >>> sign = ops.Sign()
         >>> output = sign(x)
         >>> print(output)
         [[ 1.  0. -1.]]
    """

    @prim_attr_register
    def __init__(self):
        pass


class Round(Primitive):
    r"""
    Returns half to even of a tensor element-wise.

    .. math::

        out_i \approx x_i

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape and type as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
         >>> x = Tensor(np.array([0.8, 1.5, 2.3, 2.5, -4.5]), mindspore.float32)
         >>> round = ops.Round()
         >>> output = round(x)
         >>> print(output)
         [ 1.  2.  2.  2. -4.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Round"""
        self.init_prim_io_names(inputs=['input_x'], outputs=['output'])


class Tan(Primitive):
    r"""
    Computes tangent of `x` element-wise.

    .. math::

        out_i = tan(x_i)

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Data type must be float16 or float32.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> tan = ops.Tan()
        >>> x = Tensor(np.array([-1.0, 0.0, 1.0]), mindspore.float32)
        >>> output = tan(x)
        >>> print(output)
        [-1.5574081 0. 1.5574081]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Tan"""



class Atan(Primitive):
    r"""
    Computes the trigonometric inverse tangent of the input element-wise.

    .. math::

        out_i = tan^{-1}(x_i)

    Inputs:
        - **x** (Tensor): The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          The data type should be one of the following types: float16, float32.

    Outputs:
        A Tensor, has the same type as the input.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 0.0]), mindspore.float32)
        >>> atan = ops.Atan()
        >>> output = atan(x)
        >>> print(output)
        [0.7853982 0.       ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Atan"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


class Atanh(Primitive):
    r"""
    Computes inverse hyperbolic tangent of the input element-wise.

    .. math::

        out_i = \tanh^{-1}(x_{i})

    Inputs:
        - **x** (Tensor): The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          The data type should be one of the following types: float16, float32.

    Outputs:
        A Tensor, has the same type as the input.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0, -0.5]), mindspore.float32)
        >>> atanh = ops.Atanh()
        >>> output = atanh(x)
        >>> print(output)
        [ 0.         -0.54930615]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Atanh"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])



class Atan2(_MathBinaryOp):
    r"""
    Returns arctangent of x/y element-wise.

    It returns :math:`\theta\ \in\ [-\pi, \pi]`
    such that :math:`x = r*\sin(\theta), y = r*\cos(\theta)`, where :math:`r = \sqrt{x^2 + y^2}`.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower precision data type will be converted to
    the relatively highest precision data type.

    Inputs:
        - **x** (Tensor) - The input tensor.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        - **y** (Tensor) - The input tensor. It has the same shape with `x`.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,and the data type is same as `x`.

    Raises:
        TypeError: If `x` or `y` is not a Tensor.
        RuntimeError: If the data type of `x` and `y` conversion of Parameter is required
                      when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([0, 1]), mindspore.float32)
        >>> y = Tensor(np.array([1, 1]), mindspore.float32)
        >>> atan2 = ops.Atan2()
        >>> output = atan2(x, y)
        >>> print(output)
        [0.        0.7853982]
    """


class SquareSumAll(PrimitiveWithInfer):
    r"""
    Returns the square sum of a tensor element-wise

    .. math::

        \left\{\begin{matrix}out_{x} = {\textstyle \sum_{0}^{N}} (x_{i})^2
        \\out_{y} = {\textstyle \sum_{0}^{N}} (y_{i})^2
        \end{matrix}\right.

    Inputs:
        - **x** (Tensor) - The input tensor. The data type must be float16 or float32.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        - **y** (Tensor) - The input tensor has the same type and shape as the `x`.

    Note:
        SquareSumAll only supports float16 and float32 data type.

    Outputs:
        - **output_x** (Tensor) - The same type as the `x`.
        - **output_y** (Tensor) - The same type as the `x`.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.
        ValueError: If `x` and `y` are not the same shape.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([0, 0, 2, 0]), mindspore.float32)
        >>> y = Tensor(np.array([0, 0, 2, 4]), mindspore.float32)
        >>> square_sum_all = ops.SquareSumAll()
        >>> output = square_sum_all(x, y)
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
    r"""
    Returns bitwise `and` of two tensors element-wise.

    .. math::

        out_i = x_{i} \wedge y_{i}

    Inputs of `x` and `y` comply with the implicit type conversion rules to
    make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.

    Inputs:
        - **x** (Tensor) - The input tensor with int16, int32 or uint16 data type.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        - **y** (Tensor) - The input tensor with same type as the `x`.

    Outputs:
        Tensor, has the same type as the `x`.

    Raises:
        TypeError: If `x` or `y` is not a Tensor.
        RuntimeError: If the data type of `x` and `y` conversion of Parameter is required
                      when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int16)
        >>> y = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int16)
        >>> bitwise_and = ops.BitwiseAnd()
        >>> output = bitwise_and(x, y)
        >>> print(output)
        [ 0  0  1 -1  1  0  1]
    """


class BitwiseOr(_BitwiseBinaryOp):
    r"""
    Returns bitwise `or` of two tensors element-wise.

    .. math::

        out_i = x_{i} \mid y_{i}

    Inputs of `x` and `y` comply with the implicit type conversion rules to
    make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.

    Inputs:
        - **x** (Tensor) - The input tensor with int16, int32 or uint16 data type.
        - **y** (Tensor) - The input tensor with same type as the `x`.

    Outputs:
        Tensor, has the same type as the `x`.

    Raises:
        TypeError: If `x` or `y` is not a Tensor.
        RuntimeError: If the data type of `x`, `y` conversion of Parameter is required
                      when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int16)
        >>> y = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int16)
        >>> bitwise_or = ops.BitwiseOr()
        >>> output = bitwise_or(x, y)
        >>> print(output)
        [ 0  1  1 -1 -1  3  3]
    """


class BitwiseXor(_BitwiseBinaryOp):
    r"""
    Returns bitwise `xor` of two tensors element-wise.

    .. math::

        out_i = x_{i} \oplus y_{i}

    Inputs of `x` and `y` comply with the implicit type conversion rules to
    make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.

    Inputs:
        - **x** (Tensor) - The input tensor with int16, int32 or uint16 data type.
        - **y** (Tensor) - The input tensor with same type as the `x`.

    Outputs:
        Tensor, has the same type as the `x`.

    Raises:
        TypeError: If `x` or `y` is not a Tensor.
        RuntimeError: If the data type of `x`, `y` conversion of Parameter is required
                      when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int16)
        >>> y = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int16)
        >>> bitwise_xor = ops.BitwiseXor()
        >>> output = bitwise_xor(x, y)
        >>> print(output)
        [ 0  1  0  0 -2  3  2]
    """


class BesselI0(Primitive):
    """
    Computes BesselI0 of input element-wise.

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Data type must be float16, float32 or float64.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor of float16, float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> bessel_i0 = ops.BesselI0()
        >>> x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = bessel_i0(x)
        >>> print(output)
        [1.014452  1.179784  1.0241697 1.0020261]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize BesselI0"""


class BesselI1(Primitive):
    """
    Computes BesselI1 of input element-wise.

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Data type must be float16, float32 or float64.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor of float16, float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> bessel_i1 = ops.BesselI1()
        >>> x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = bessel_i1(x)
        >>> print(output)
        [0.1208661  0.45177728 0.1568694  0.04504559]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize BesselI1"""


class BesselI0e(Primitive):
    r"""
    Computes BesselI0e of input element-wise.

    The formula is defined as:

    .. math::
        BesselI0e(x) = \exp(|x|) * bessel\_i0(x)

    where bessel_i0 is Bessel function of the first kind with 0 order.

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Data type must be float16 or float32.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> bessel_i0e = ops.BesselI0e()
        >>> x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = bessel_i0e(x)
        >>> print(output)
        [0.7979961  0.5144438  0.75117415  0.9157829 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize BesselI0e"""
        self.init_prim_io_names(inputs=['x'], outputs='output')


class BesselI1e(Primitive):
    r"""
    Computes BesselI1e of input element-wise.

    The formula is defined as:

    .. math::
        BesselI1e(x) = \exp(|x|) * bessel\_i1(x)

    where bessel_i1 is Bessel function of the first kind with 1 order.

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Data type must be float16 or float32.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> bessel_i1e = ops.BesselI1e()
        >>> x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = bessel_i1e(x)
        >>> print(output)
        [0.09507662 0.19699717 0.11505538 0.04116856]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize BesselI1e"""
        self.init_prim_io_names(inputs=['x'], outputs='output')



class Inv(Primitive):
    r"""
    Computes Reciprocal of input tensor element-wise.

    .. math::

        out_i = \frac{1}{x_{i} }

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Must be one of the following types: float16, float32, int32.

    Outputs:
        Tensor, has the same shape and data type as `x`.

    Raises:
        TypeError: If dtype of `x` is not one of float16, float32, int32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> inv = ops.Inv()
        >>> x = Tensor(np.array([0.25, 0.4, 0.31, 0.52]), mindspore.float32)
        >>> output = inv(x)
        >>> print(output)
        [4.        2.5       3.2258065 1.923077 ]
    """

    @prim_attr_register
    def __init__(self):
        pass



class Invert(Primitive):
    r"""
    Flips all bits of input tensor element-wise.

    .. math::

        out_i = -x_{i}

    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
          The data type should be one of the following types: int16, uint16.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If dtype of `x` is neither int16 nor uint16.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> invert = ops.Invert()
        >>> x = Tensor(np.array([25, 4, 13, 9]), mindspore.int16)
        >>> output = invert(x)
        >>> print(output)
        [-26 -5 -14 -10]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Invert"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class Eps(PrimitiveWithInfer):
    """
    Creates a tensor filled with minimum value in `x` dtype.

    Inputs:
        - **x** (Tensor) - Input tensor. The data type must be float16, float32 or float64.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, has the same type and shape as `x`, but filled with `x` dtype minimum val.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([4, 1, 2, 3], mindspore.float32)
        >>> output = ops.Eps()(x)
        >>> print(output)
        [1.5258789e-05 1.5258789e-05 1.5258789e-05 1.5258789e-05]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Eps"""
        self.init_prim_io_names(inputs=['input_x'], outputs=['y'])

    def __infer__(self, input_x):
        valid_dtypes = [mstype.float16, mstype.float32, mstype.float64]
        validator.check_tensor_dtype_valid('input_x', input_x['dtype'], valid_dtypes, self.name)

        x_nptype = mstype.dtype_to_nptype(input_x['dtype'].element_type())
        if x_nptype == np.float16:
            min_val = 2 ** (-14)
        elif x_nptype == np.float32:
            min_val = 2 ** (-16)
        else:
            min_val = 2 ** (-52)

        res = np.full(input_x['shape'], min_val, x_nptype)
        out = {
            'value': Tensor(res),
            'shape': input_x['shape'],
            'dtype': input_x['dtype'],
        }
        return out


class LinSpace(PrimitiveWithInfer):
    r"""
    Returns a Tensor whose value is `num` evenly spaced in the interval `start` and `stop` (including `start` and
    `stop`), and the length of the output Tensor is `num`.

    .. math::
        \begin{aligned}
        &step = (stop - start)/(num - 1)\\
        &output = [start, start+step, start+2*step, ... , stop]
        \end{aligned}

    Inputs:
        - **start** (Tensor) - The data type must be float32. Start value of interval, With shape of 0-D.
        - **stop** (Tensor) - The data type must be float32. Last value of interval, With shape of 0-D.
        - **num** (int) - Number of ticks in the interval, inclusive of start and stop.

    Outputs:
        Tensor, has the same shape and dtype as `start`.

    Raises:
        TypeError: If `start` or `stop` is not a Tensor.
        TypeError: If dtype of `start` or dtype of `stop` is not float32.
        TypeError: If `num` is not a int.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> linspace = ops.LinSpace()
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


class MatrixInverse(Primitive):
    """
    Returns the inverse of the input matrix. If the matrix is irreversible, an error may be reported or an unknown
    result may be returned.

    Note:
        The parameter 'adjoint' is only supporting False right now. Because complex number is not supported at present.

    Args:
        adjoint (bool) : An optional bool. Default: False.

    Inputs:
        - **x** (Tensor) - A matrix to be calculated. The matrix must be at least two dimensions, and the last two
          dimensions must be the same size.

    Outputs:
        Tensor, has the same type and shape as input `x`.

    Raises:
        TypeError: If `adjoint` is not a bool.
        TypeError: If `x` is not a Tensor.
        ValueError: If the last two dimensions of `x` is not same size.
        ValueError: If the dimension of `x` is less than 2.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[[-0.710504  , -1.1207525],
        ...                       [-1.7651395 , -1.7576632]],
        ...                      [[ 0.52412605,  1.9070215],
        ...                       [ 1.3384849 ,  1.4274558]]]), mindspore.float32)
        >>> matrix_inverse = ops.MatrixInverse(adjoint=False)
        >>> output = matrix_inverse(x)
        >>> print(output)
        [[[ 2.4095478  -1.5364188 ]
          [-2.419797    0.9740167 ]]
         [[-0.79111797  1.0569006 ]
          [ 0.74180895 -0.2904787 ]]]
    """

    @prim_attr_register
    def __init__(self, adjoint=False):
        """Initialize MatrixInverse"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        validator.check_value_type('adjoint', adjoint, [bool], self.name)


class MatrixDeterminant(Primitive):
    """
    Computes the determinant of one or more square matrices.

    Inputs:
        - **x** (Tensor) - A matrix to be calculated. The matrix must be at least two dimensions, and the last two
          dimensions must be the same size.

    Outputs:
        Tensor, the shape is `x_shape[:-2]`, the dtype is same as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        ValueError: If the last two dimensions of `x` is not same size.
        ValueError: If the dimension of `x` is less than 2.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[[-4.5, -1.5], [7.0, 6.0]], [[2.5, 0.5], [3.0, 9.0]]]), mindspore.float32)
        >>> op = P.MatrixDeterminant()
        >>> output = op(input_x)
        >>> print(output)
        [-16.5 21. ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize MatrixDeterminant."""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class LogMatrixDeterminant(Primitive):
    """
    Computes the sign and the log of the absolute value of the determinant of one or more square matrices.

    Inputs:
        - **x** (Tensor) - A matrix to be calculated. The matrix must be at least two dimensions, and the last two
          dimensions must be the same size.

    Outputs:
        - **sign** (Tensor) - The signs of the log determinants. The shape is `x_shape[:-2]`, the dtype is same as `x`.
        - **y** (Tensor) - The absolute values of the log determinants. The shape is `x_shape[:-2]`, the dtype is same
          as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        ValueError: If the last two dimensions of `x` is not same size.
        ValueError: If the dimension of `x` is less than 2.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[[-4.5, -1.5], [7.0, 6.0]], [[2.5, 0.5], [3.0, 9.0]]]), mindspore.float32)
        >>> op = P.LogMatrixDeterminant()
        >>> output = op(input_x)
        >>> print(output)
        (Tensor(shape=[2], dtype=Float32, value= [-1.00000000e+00,  1.00000000e+00]), Tensor(shape=[2], dtype=Float32,
        value= [ 2.80336046e+00,  3.04452229e+00]))
    """

    @prim_attr_register
    def __init__(self):
        """Initialize LogMatrixDeterminant."""
        self.init_prim_io_names(inputs=['x'], outputs=['sign', 'y'])


class IndexAdd(Primitive):
    """
    Adds tensor `y` to specified axis and indices of tensor `x`. The axis should be in [0,  len(x.dim) - 1],
    and indices should be in [0, the size of `x`] at the axis dimension.

    Args:
        axis (int): The dimension along which to index.
        use_lock (bool): If true, use lock mode. If false, don't use lock mode. Default: True.
        check_index_bound (bool): If true, check index boundary. If false, don't check index boundary. Default: True.

    Inputs:
        - **x** (Parameter) - The input Parameter to add to.
        - **indices** (Tensor) - Add the  value of `x` and `y` along the dimension of the `axis` according to the
          specified index value, with data type int32.
          The `indices` must be 1D with the same size as the size of `y` in the `axis` dimension. The values
          of `indices` should be in [0, b), where the b is the size of `x` in the `axis` dimension.
        - **y** (Tensor) - The input tensor with the value to add. Must have same data type as `x`.
          The shape must be the same as `x` except the `axis` th dimension.

    Outputs:
        Tensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `x` is not a Parameter.
        TypeError: If neither `indices` nor `y` is a Tensor.
        ValueError: If axis is out of `x` rank's range.
        ValueError: If `x` rank is not the same as `y` rank.
        ValueError: If size of `indices` is not equal to dimension of y[axis].
        ValueError: If `y`'s shape is not the same as `x` except the `axis` th dimension.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.index_add = ops.IndexAdd(axis=1)
        ...         self.x = Parameter(Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32),
        ...                 name="name_x")
        ...         self.indices = Tensor(np.array([0, 2]), mindspore.int32)
        ...
        ...     def construct(self, y):
        ...         return self.index_add(self.x, self.indices, y)
        ...
        >>> y = Tensor(np.array([[0.5, 1.0], [1.0, 1.5], [2.0, 2.5]]), mindspore.float32)
        >>> net = Net()
        >>> output = net(y)
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


class Erfinv(Primitive):
    r"""
    Computes the inverse error function of input. The inverse error function is defined in the range (-1, 1) as:

    .. math::
                                erfinv(erf(x)) = x

    Inputs:
        - **input_x** (Tensor) - The input tensor to compute to, with data type float32 or float16.

    Outputs:
        Tensor, has the same shape and dtype as `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is neither float32 nor float16.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.array([0, 0.5, -0.9]), mindspore.float32)
        >>> erfinv = ops.Erfinv()
        >>> output = erfinv(x)
        >>> print(output)
        [ 0.          0.47695306 -1.1630805 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Erfinv"""
        self.init_prim_io_names(inputs=['input_x'], outputs=['output'])


class Conj(PrimitiveWithInfer):
    """
    Returns a tensor of complex numbers that are the complex conjugate of each element in input.
    The complex numbers in input must be of the form a + bj, where a is the real part and b is the imaginary part.

    The complex conjugate returned by this operation is of the form a - bj.

    If input is real, it is returned unchanged.

    Inputs:
        - **input** (Tensor) - The input tensor to compute to. Must have numeric type.

    Outputs:
        Tensor, has the same dtype as the input.

    Raises:
       TypeError: If the dtype of input is not a numeric type.
       TypeError: If the input is not a Tensor.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> x = Tensor(np.asarray(np.complex(1.3+0.4j)), mindspore.complex64)
        >>> conj = ops.Conj()
        >>> output = conj(x)
        >>> print(output)
        (1.3-0.4j)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Conj"""
        self.init_prim_io_names(inputs=['input'], outputs=['output'])


class Real(PrimitiveWithInfer):
    """
    Returns a Tensor that is the real part of the input.
    If input is real, it is returned unchanged.

    Inputs:
        -**input** (Tensor) - The input tensor to compute to.

    Outputs:
        Tensor, the shape is the same as the input.

    Raises:
       TypeError: If the input is not a Tensor.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> x = Tensor(np.asarray(np.complex(1.3+0.4j)), mindspore.complex64)
        >>> real = ops.Real()
        >>> output = real(x)
        >>> print(output)
        1.3
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Real"""
        self.init_prim_io_names(inputs=['input'], outputs=['output'])


class Complex(Primitive):
    """
    Returns a complex Tensor from the real part and the imag part.

    Inputs:
        - **real** (Tensor) - The real input tensor. types: float32, float64.
        - **imag** (Tensor) - The imag input tensor. types: float32, float64.

    Outputs:
        Tensor, has the complex type.

    Raises:
       TypeError: If the dtype of input is not one of: float32, float64.
                  If the dtypes of two inputs are not same.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> real = Tensor(np.asarray(1, mindspore.complex64)
        >>> imag = Tensor(np.asarray(2, mindspore.complex64)
        >>> complex = ops.Complex()
        >>> output = complex(real, imag)
        >>> print(output)
        (1 + 2j)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Complex"""
        self.init_prim_io_names(inputs=['input_real', 'input_imag'], outputs=['output'])


class Imag(PrimitiveWithInfer):
    """
    Returns a new tensor containing imaginary value of the input.
    If input is real, it is returned zeros.

    Inputs:
        - **input** (Tensor) - The input tensor to compute to.

    Outputs:
        Tensor, the shape is the same as the input.

    Raises:
       TypeError: If the input is not a Tensor.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> x = Tensor(np.asarray(np.complex(1.3+0.4j)), mindspore.complex64)
        >>> imag = ops.Imag()
        >>> output = imag(x)
        >>> print(output)
        0.4
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Imag"""
        self.init_prim_io_names(inputs=['input'], outputs=['output'])


class Trunc(Primitive):
    """
    Returns a new tensor with the truncated integer values of the elements of input.

    Inputs:
        - **input_x** (Tensor) - Input_x is a tensor.

    Outputs:
        Tensor, the same shape and data type as the input.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> trunc = ops.Trunc()
        >>> output = trunc(Tensor(np.array([3.4742, 0.5466, -0.8008, -3.9079]),mindspore.float32))
        >>> print(output)
        [ 3. 0. 0. -3.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Trunc"""

class IsClose(Primitive):
    r"""
    Returns a boolean tensor where two tensors are element-wise equal within a tolerance.

    Note:
        Returns a new tensor with boolean elements representing if each element of input
        is “close” to the corresponding element of other. Closeness is defined as:
            ∣input−other∣  ≤  atol + rtol × ∣other∣

    .. warning::
        When the input is nan or inf, the result is uncertain.

    Args:
        rtol(float): Relative tolerance. Default: 1e-05.
        atol(float): Absolute tolerance. Default: 1e-08.
        equal_nan(bool): If True, then two NaNs will be considered equal. At present, `equal_nan` must be True,
                         we will support False in future version. Default: True.

    Inputs:
        -**input**(Tensor) – First tensor to compare, with data type belongs to float32, float16, int32.
        -**other**(Tensor) – Second tensor to compare, with data type belongs to float32, float16, int32.

    Outputs:
        Tensor, with same shape as input and other. When the input is close to the other, it is true,
        otherwise it is false.

    Raises:
        TypeError: If either of `input` and `other` is not tensor.
        TypeError: If either of `input` and `other` is not float16, float32 or int32.
        TypeError: If either of `atol` and `rtol` is not float.
        TypeError: If `equal_nan` is not bool.
        TypeError: If the dtype of `input` is not same as the `other`.
        ValueError: If shape of `input` is not same as the `other`.
        ValueError: If either of `atol` and `rtol` is less than zero.
        ValueError: If `equal_nan` is False.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input = Tensor(np.array([1.3, 2.1, 3.2, 4.1, 5.1]), mindspore.float16)
        >>> other = Tensor(np.array([1.3, 3.3, 2.3, 3.1, 5.1]), mindspore.float16)
        >>> output = ops.IsClose()(input, other)
        >>> print(output)
            [true false false false true]
    """
    @prim_attr_register
    def __init__(self, rtol=1e-05, atol=1e-08, equal_nan=True):
        """Initialize IsClose"""
        validator.check_value_type('rtol', rtol, [float], self.name)
        validator.check_value_type('atol', atol, [float], self.name)
        validator.check_value_type('equal_nan', equal_nan, [bool], self.name)
        if not equal_nan:
            raise ValueError("For IsClose, the `equal_nan` must be True, but got False.")
        validator.check_non_negative_float(rtol, 'rtol', self.name)
        validator.check_non_negative_float(atol, 'atol', self.name)
