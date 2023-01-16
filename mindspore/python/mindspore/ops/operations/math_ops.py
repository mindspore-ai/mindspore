# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import
from __future__ import division

import numpy as np

from mindspore import context
from mindspore.ops import signature as sig
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common._decorator import deprecated
from mindspore.ops._utils import get_broadcast_shape
from mindspore.ops.primitive import Primitive, PrimitiveWithInfer, PrimitiveWithCheck, prim_attr_register, _run_op
from mindspore._c_expression import Tensor as Tensor_
from mindspore.common._utils import is_shape_unknown


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
                raise TypeError('Complex math binary op expecting Tensor [Complex64, Complex64],'
                                + '[Complex64, Float32], [Float32, Complex64], [Complex128, Complex128],'
                                + '[Complex128, Float64], [Float64, Complex128],'
                                + f'but got : [{format(x_dtype)},{format(y_dtype)}].')
            return type_infer_dict.get((x_dtype.element_type(), y_dtype.element_type()))

        validator.check_tensors_dtypes_same_and_valid(args_type, valid_dtype, prim_name)
        return x_dtype

    def infer_dtype(self, x_dtype, y_dtype):
        return _MathBinaryOp.do_infer_dtype(x_dtype, y_dtype, mstype.number_type, self.name)

    def _convert_back_shape(self, shape_value, cmp_shape):
        if isinstance(cmp_shape, (Tensor, Tensor_)):
            cmp_shape = cmp_shape.asnumpy()
        if not isinstance(cmp_shape, tuple):
            return shape_value
        real_shape = [dim if cmp_dim > 0 else cmp_dim for dim, cmp_dim in zip(shape_value, cmp_shape)]
        return tuple(real_shape)


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
    Ger product of `x1` and `x2`. Calculate the outer product of two arrays. If `x1` is a 1D Tensor of
    shape :math:`(m,)` and `x2` is a 1D Tensor of shape :math:`(n,)`, then `output` must be a 2D Tensor of shape
    :math:`(m, n)`.

    Refer to :func:`mindspore.ops.ger` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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

    Refer to :func:`mindspore.ops.add` for more details.

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

    @staticmethod
    def _infer_specified_add_value(a, b):
        """Calculate min/max value for output for Add op"""
        if a is not None and b is not None:
            if isinstance(a, (Tensor, Tensor_)):
                a = a.asnumpy()
            if isinstance(b, (Tensor, Tensor_)):
                b = b.asnumpy()
            a = np.array(a)
            b = np.array(b)
            out = a + b
            out = tuple(out.tolist())
            return out
        return None

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = x + y
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None

    def _infer_min_value(self, x, y):
        """Calculate min value for output for Add op"""
        return self._infer_specified_add_value(x, y)

    def _infer_max_value(self, x, y):
        """Calculate max value for output for Add op"""
        return self._infer_specified_add_value(x, y)

    def _infer_shape_value(self, x, y):
        shape_value = self._infer_specified_add_value(x, y)
        shape_value = self._convert_back_shape(shape_value, x)
        return self._convert_back_shape(shape_value, y)


class Addcdiv(Primitive):
    r"""
    Performs the element-wise division of tensor `x1` by tensor `x2`,
    multiply the result by the scalar `value` and add it to `input_data`.

    .. math::
        y[i] = input\_data[i] + value[i] * (x1[i] / x2[i])

    Inputs:
        - **input_data** (Tensor) - The tensor to be added.
        - **x1** (Tensor) - The numerator tensor.
        - **x2** (Tensor) - The denominator tensor.
        - **value** (Tensor) - The multiplier for tensor x1/x2.

    Outputs:
        Tensor, has the same shape and dtype as x1/x2.

    Raises:
        TypeError: If dtype of `x1`, `x2`, `value`, `input_data` is not tensor.
        ValueError: If `x1` could not be broadcast to `x2`.
        ValueError: If `value` could not be broadcast to `x1/x2`.
        ValueError: If `input_data` could not be broadcast to `value*(x1/x2)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
    r"""
    Performs the element-wise product of tensor `x1` and tensor `x2`,
    multiply the result by the scalar `value` and add it to `input_data`.

    .. math::
        output[i] = input\_data[i] + value[i] * (x1[i] * x2[i])

    Inputs:
        - **input_data** (Tensor) - The tensor to be added.
        - **x1** (Tensor) - The tensor to be multiplied.
        - **x2** (Tensor) - The tensor to be multiplied.
        - **value** (Tensor) - The multiplier for tensor x1*x2.

    Outputs:
        Tensor, has the same shape and dtype as x1*x2.

    Raises:
        TypeError: If dtype of `x1`, `x2`, `value`, `input_data` is not tensor.
        TypeError: If dtype of `input_data` is not one of: float32, float16, int32.
        TypeError: If dtype of `x1` or `x2` is not one of: float32, float16, int32.
        TypeError: If dtype of `value` is not one of: float32, float16, int32.
        ValueError: If `x1` could not be broadcast to `x2`.
        ValueError: If `value` could not be broadcast to `x1` * `x2`.
        ValueError: If `input_data` could not be broadcast to `value*(x1*x2)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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


class AddV2(Primitive):
    r"""
    Adds two input tensors element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, and the shapes of them can be broadcast.
    When the inputs are one tensor and one scalar, the scalar could only be a constant.
    CPU/Ascend does not support broadcast for now.

    .. math::

        out_{i} = x_{i} + y_{i}

    Inputs:
        - **x** (Union[Tensor]) - The first input is a tensor whose data type is one of
          uint8, int8, int16, int32, int64, float16, float32, float64,
          complex64, complex128 currently or scalar.
        - **y** (Union[Tensor]) - The second input is a tensor whose data type is one of
          uint8, int8, int16, int32, int64, float16, float32, float64,
          complex64, complex128 currently or scalar.

    Outputs:
        Tensor, the shape is the same as the input tensor,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.
        TypeError: If dtype of `x` or `y` is not in [float16, float32, float64,
        uint8, int8, int16, int32, int64, complex64, complex128].
        ValueError: If the shape of 'x' and 'y' is not the same for CPU and Ascend.


    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.ops.operations.math_ops import AddV2
        >>> addv2 = AddV2()
        >>> x = Tensor(np.array([1, 2, 3]).astype(np.int32))
        >>> y = Tensor(np.array([4, 5, 6]).astype(np.int32))
        >>> output = addv2(x, y)
        >>> print(output)
        [5 7 9]
    """
    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        """Initialize AddV2"""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])


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

    Refer to :func:`mindspore.ops.assign_add` for more details.

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

    Refer to :func:`mindspore.ops.assign_sub` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
        >>> print(net.variable.asnumpy())
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


class _Reduce(PrimitiveWithCheck):
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

    def infer_value(self, input_x, axis):
        """ return reduce op value"""
        value = None
        if input_x is not None and axis is not None:
            prim_map = {
                'ReduceMax': np.max,
                'ReduceMin': np.min,
                'ReduceProd': np.prod,
                'ReduceMean': np.mean,
                'ReduceAll': np.all,
                'ReduceAny': np.any,
            }
            np_reduce_func = prim_map.get(self.name, None)

            if np_reduce_func is not None:
                value = input_x.asnumpy()
                if isinstance(axis, int):
                    pass
                elif axis:
                    axis = tuple(set(axis))
                else:
                    axis = tuple(range(len(value.shape)))
                value = np_reduce_func(value, axis, keepdims=self.keep_dims)
                value = np.array(value)
                value = Tensor(value)
        return value


class EuclideanNorm(Primitive):
    """
    Computes the euclidean norm of elements across dimensions of a tensor.
    Reduces input along the dimensions given in axis.

    Args:
        keep_dims (bool, optional): If true, the reduceed dimensions are retained with length 1.
                          If false, don't keep these dimensions. Default: False.

    Inputs:
        - **x** (Tensor) - The input tensor. Must be one of the following types :float16, float32, float64, int8, int16,
          int32, int64, complex64, complex128, uint8, uint16, uint32, uint64. The tensor to reduce.
        - **axes** (Tensor) - The dimensions to reduce. Must be one of the following types: int32, int64.
          Must be in the range [-rank(x), rank(x)).

    Outputs:
        Tensor, has the same type as the 'x'.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `x` is not a Tensor.
        ValueError: If `axes` is out of range.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> x = Tensor(np.array([[3, 5], [4, 12]])).astype(np.int32)
        >>> axes = Tensor([0])
        >>> op = ops.EuclideanNorm(keep_dims=True)
        >>> output = op(x, axes)
        >>> print(output)
        [[5 13]]
    """

    @prim_attr_register
    def __init__(self, keep_dims=False):
        """Initialize"""
        self.init_prim_io_names(inputs=['x', 'axes'], outputs=['y'])
        validator.check_value_type("keep_dims", keep_dims, [bool], self.name)


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
          Only constant value is allowed. Must be in the range [-r, r).

    Outputs:
        Tensor, has the same dtype as the `x`.

        - If `axis` is (), and `keep_dims` is False,
          the output is a 0-D tensor representing the mean of all elements in the input tensor.
        - If `axis` is int, set as 1, and `keep_dims` is False,
          the shape of output is :math:`(x_0, x_2, ..., x_R)`.
        - If `axis` is tuple(int) or list(int), set as (1, 2), and `keep_dims` is False,
          the shape of output is :math:`(x_0, x_3, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `x` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple or list.
        ValueError: If `axis` is out of range.

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
        >>> x = Tensor(np.array([[[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
        ... [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ... [[6, 6, 6, 6, 6, 6], [8, 8, 8, 8, 8, 8], [10, 10, 10, 10, 10, 10]]]),
        ... mindspore.float32)
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
        [[[ 2.]
          [ 2.]
          [ 2.]]
         [[ 4.]
          [ 5.]
          [ 6.]]
         [[ 6.]
          [ 8.]
          [10.]]]
    """

    @prim_attr_register
    def __init__(self, keep_dims=False):
        """Initialize ReduceMean"""
        super(ReduceMean, self).__init__(keep_dims)


class CumulativeLogsumexp(Primitive):
    """
    Compute the cumulative log-sum-exp of the input tensor `x` along `axis` . For example, with all parameters at
    default values, if the input `x` is a tensor [a, b, c], the output will be [a, log(exp(a) + exp(b)),
    log(exp(a) + exp(b) + exp(c))].

    Args:
        exclusive (bool, optional): If true, the last element will be skipped during the calculation and thus an
                                    exclusive cumulative log-sum-exp will be performed. For example, this operation
                                    will output [-inf, a, log(exp(a) * exp(b))] with tensor [a, b, c] as the input.
                                    Note that the minimal value -inf, for performance reasons, is representable by the
                                    floating point type. Default: False.
        reverse (bool, optional): If true, the function accumulation values will be calculated after the elements of
                                  `x` on `axis` are flipped, and the calculation result will be flipped afterwards. For
                                  example, this operation will output [log(exp(c) + exp(b) + exp(a)), log(exp(c) +
                                  exp(b)), c] with tensor [a, b, c] as the input. Default: False.

    Inputs:
        - **x** (Tensor) - The input tensor. Must be one of the following types: float16, float32, float64. The
          dimension of `x` must greater than 0.
        - **axis** (Tensor) - A 0-D tensor describing the dimension to compute the cumulative product. Must be one of
          the following types: int64, int32, int16. Must be in the range [-rank(x), rank(x)). Default: 0.

    Outputs:
        Tensor, has the same dtype and shape as the `x`.

    Raises:
        TypeError: If `x` or `axis` not a Tensor.
        TypeError: If dtype of `x` is not in [float16, float32, float64].
        TypeError: If dtype of `axis` is not in [int16, int32, int64].
        TypeError: If `exclusive` or `reverse` is not a bool.
        ValueError: If the dimension of `x` is not greater than 0.
        RuntimeError: If `axis` is out of range [-rank(x), rank(x)).

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
        >>> op = ops.CumulativeLogsumexp(exclusive=False, reverse=False)
        >>> output = op(x, Tensor(0))
        >>> print(output)
        [1.        2.3132617 3.407606 ]
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
        >>> op = ops.CumulativeLogsumexp(exclusive=True, reverse=False)
        >>> output = op(x, Tensor(0))
        >>> print(output)
        [-3.4028235e+38  1.0000000e+00  2.3132617e+00]
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
        >>> op = ops.CumulativeLogsumexp(exclusive=False, reverse=True)
        >>> output = op(x, Tensor(0))
        >>> print(output)
        [3.407606  3.3132617 3.       ]
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
        >>> op = ops.CumulativeLogsumexp(exclusive=True, reverse=True)
        >>> output = op(x, Tensor(0))
        >>> print(output)
        [ 3.3132617e+00  3.0000000e+00 -3.4028235e+38]
    """

    @prim_attr_register
    def __init__(self, exclusive=False, reverse=False):
        """Initialize  CumulativeLogsumexp"""
        self.init_prim_io_names(inputs=['x', 'axis'], outputs=['y'])
        validator.check_bool(exclusive, "exclusive", self.name)
        validator.check_bool(reverse, "reverse", self.name)


class ReduceSum(PrimitiveWithCheck):
    """
    Reduces a dimension of a tensor by summing all elements in the dimension, by default. And also can reduce a
    dimension of `x` along the axis. Determine whether the dimensions of the output and input are the same by
    controlling `keep_dims`.

    Args:
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions. Default: False.
        skip_mode (bool): If true and axis is empty tuple or empty list, the ReduceSum operation isn't performed,
                          skip it.
                          If true and axis is other values, the ReduceSum calculation is performed normally.
                          If false, do reduce. Default: False.

    Inputs:
         - **x** (Tensor[Number]) - The input tensor. The dtype of the tensor to be reduced is number.
           :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
         - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions
           when skip_mode is false. Only constant value is allowed. Must be in the range [-rank(`x`), rank(`x`)).

    Outputs:
        Tensor, has the same dtype as the `x`.

        - If axis is (), keep_dims is False, and skip_mode is False,
          the output is a 0-D tensor representing the sum of all elements in the input tensor.
        - If axis is (), and skip_mode is True,
          the ReduceSum operation is not performed, output tensor is equal to the input tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int) or list(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `skip_mode` is not a bool.
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

    __mindspore_signature__ = (
        sig.make_sig('input_x'),
        sig.make_sig('axis', default=())
    )

    @prim_attr_register
    def __init__(self, keep_dims=False, skip_mode=False):
        """Initialize Reduce"""
        validator.check_value_type('keep_dims', keep_dims, [bool], self.name)
        validator.check_value_type('skip_mode', skip_mode, [bool], self.name)
        self.init_prim_io_names(inputs=['input_x', 'axis'], outputs=['y'])
        self.keep_dims = keep_dims
        self.skip_mode = skip_mode
        self.__setattr_flag__ = True

    def __call__(self, x, axis=()):
        args = [x, axis]
        output = _run_op(self, self.name, args)
        return output

    def infer_value(self, input_x, axis):
        """ return reduce op value"""
        value = None
        if input_x is not None and axis is not None:
            value = input_x.asnumpy()
            if isinstance(axis, int):
                pass
            elif axis:
                axis = tuple(set(axis))
            elif axis in ((), []) and self.skip_mode:
                return input_x
            else:
                axis = tuple(range(len(value.shape)))
            value = np.sum(value, axis, keepdims=self.keep_dims)
            value = np.array(value)
            value = Tensor(value)
        return value


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
           Only constant value is allowed. Must be in the range [-r, r).

    Outputs:
        Tensor, has the same dtype as the `x`.

        - If `axis` is (), and `keep_dims` is False,
          the output is a 0-D tensor representing the maximum of all elements in the input tensor.
        - If `axis` is int, set as 1, and `keep_dims` is False,
          the shape of output is :math:`(x_0, x_2, ..., x_R)`.
        - If `axis` is tuple(int) or list(int), set as (1, 2), and `keep_dims` is False,
          the shape of output is :math:`(x_0, x_3, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `x` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple or list.
        ValueError: If `axis` is out of range.

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
          Only constant value is allowed. Must be in the range [-r, r).

    Outputs:
        Tensor, has the same dtype as the `x`.

        - If `axis` is (), and `keep_dims` is False,
          the output is a 0-D tensor representing the minimum of all elements in the input tensor.
        - If `axis` is int, set as 1, and `keep_dims` is False,
          the shape of output is :math:`(x_0, x_2, ..., x_R)`.
        - If `axis` is tuple(int), set as (1, 2), and `keep_dims` is False,
          the shape of output is :math:`(x_0, x_3, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `x` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple or list.
        ValueError: If `axis` is out of range.

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


class Bucketize(Primitive):
    """
    Bucketizes 'input' based on 'boundaries'.

    Args:
        boundaries (list[float]): A sorted list of floats gives the boundary of the buckets, and no default value.

    Inputs:
        - **input** (Tensor) - A tensor containing the search value(s).

    Outputs:
        Tensor, with the same shape as the input, and data type is int32.

    Raises:
        TypeError: If `boundaries` is not a listFloat.
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> class Bucketize(nn.Cell):
        ...     def __init__(self, boundaries):
        ...         super().__init__()
        ...         self.bucketize = ops.Bucketize(boundaries=boundaries)
        ...     def construct(self, input):
        ...         return self.bucketize(input)
        >>> input = Tensor(np.array([[3, 6, 9], [3, 6, 9]]).astype(np.int32))
        >>> boundaries = list(np.array([1., 3., 5., 7., 9.]))
        >>> net = Bucketize(boundaries)
        >>> output = net(input)
        >>> print(output)
        [[2 3 5]
         [2 3 5]]
    """

    @prim_attr_register
    def __init__(self, boundaries):
        """Initialize Bucketize"""
        validator.check_value_type("boundaries", boundaries, [list], self.name)
        for index, one_boundaries in enumerate(boundaries):
            validator.check_value_type('boundaries[%d]' % index, one_boundaries, [float], self.name)
        self.init_prim_io_names(inputs=['input'], outputs=['output'])


class ReduceProd(_Reduce):
    """
    Reduces a dimension of a tensor by multiplying all elements in the dimension, by default. And also can
    reduce a dimension of `x` along the axis. Determine whether the dimensions of the output and input are the same by
    controlling `keep_dims`.

    Args:
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions. Default: False.

    Inputs:
        - **x** (Tensor[Number]) - The input tensor. The dtype of the tensor to be reduced is number.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed. Must be in the range [-r, r).

    Outputs:
        Tensor, has the same dtype as the `x`.

        - If `axis` is (), and `keep_dims` is False,
          the output is a 0-D tensor representing the product of all elements in the input tensor.
        - If `axis` is int, set as 1, and `keep_dims` is False,
          the shape of output is :math:`(x_0, x_2, ..., x_R)`.
        - If `axis` is tuple(int), set as (1, 2), and `keep_dims` is False,
          the shape of output is :math:`(x_0, x_3, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `x` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple or list.
        ValueError: If `axis` is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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

    @prim_attr_register
    def __init__(self, keep_dims=False):
        """Initialize ReduceProd"""
        super(ReduceProd, self).__init__(keep_dims)


class CumProd(Primitive):
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
        ``Ascend`` ``GPU`` ``CPU``

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
        [[  1.   2.   6.]
         [  4.  20. 120.]
         [  5.  15.  75.]]
    """

    @prim_attr_register
    def __init__(self, exclusive=False, reverse=False):
        """Initialize CumProd."""
        cls_name = self.name
        self.exclusive = validator.check_value_type("exclusive", exclusive, [bool], cls_name)
        self.reverse = validator.check_value_type("reverse", reverse, [bool], cls_name)
        self.init_prim_io_names(inputs=['x', 'axis'], outputs=['y'])


class Lcm(Primitive):
    """
    Computes least common multiplier of input tensors element-wise.
    The shape of two inputs should be broadcastable, and data type of them should be
    one of: int32, int64

    Inputs:
        - **x1** (Tensor) - The first input tensor.
        - **x2** (Tensor) - The second input tensor.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting, and the data type is one
        with higher digits in the two inputs.

    Raises:
        TypeError: If data type `x1` or `x2` is not int32 or int64.
        ValueError: If shape of two inputs are not broadcastable.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor(np.array([7, 8, 9]))
        >>> x2 = Tensor(np.array([14, 6, 12]))
        >>> lcm_ = ops.Lcm()
        >>> y = lcm_(x1, x2)
        >>> print(y)
        [14 24 36]
    """

    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])


class Cdist(Primitive):
    """
    Computes batched the p-norm distance between each pair of the two collections of row vectors.

    Refer to :func:`mindspore.ops.cdist` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor, ops
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
        if (p < 0 or np.isnan(p)):
            raise ValueError('Cdist p must be a non-negative value, but got `{p}`.')
        self.init_prim_io_names(inputs=['input_x', 'input_y'], outputs=['output'])


class LpNorm(Primitive):
    """
    Returns the matrix norm or vector norm of a given tensor.

    .. math::
        output = sum(abs(input)**p)**(1/p)

    Refer to :func:`mindspore.ops.norm` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
        super().__init__("LpNorm")
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

    Note:
        If :math:`N * M` cannot be divided by 16, the performance will be poor in ascend environment.

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
        self.add_prim_attr('transpose_x1', self.transpose_a)
        self.add_prim_attr('transpose_x2', self.transpose_b)

    def check_shape_size(self, x1, x2):
        if len(x1) != 2 or len(x2) != 2:
            raise ValueError(f"For '{self.name}', inputs 'x', 'y' should have the same dimension size and "
                             f"be equal to 2, but got the size of 'x': ({len(x1)}) and the size of 'y': ({len(x2)}).")

    def check_shape(self, x1, x2):
        is_dyn_shape = is_shape_unknown(x1) or is_shape_unknown(x2)
        if not is_dyn_shape:
            self.check_shape_size(x1, x2)
        cls_name = self.name

        # set attribute
        self.add_prim_attr('transpose_x1', self.transpose_a)
        self.add_prim_attr('transpose_x2', self.transpose_b)

        if is_dyn_shape:
            return

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

    def check_dtype(self, x1, x2):
        args = {"x1": x1, "x2": x2}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.float_type + mstype.int_type
                                                      + (mstype.complex64, mstype.complex128), self.name)


class BatchMatMul(Primitive):
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
        >>> print(output.shape)
        (2, 4, 1, 4)
        >>> x = Tensor(np.ones(shape=[2, 4, 3, 1]), mindspore.float32)
        >>> y = Tensor(np.ones(shape=[2, 4, 3, 4]), mindspore.float32)
        >>> batmatmul = ops.BatchMatMul(transpose_a=True)
        >>> output = batmatmul(x, y)
        >>> print(output.shape)
        (2, 4, 1, 4)
    """

    @prim_attr_register
    def __init__(self, transpose_a=False, transpose_b=False):
        """Initialize BatchMatMul."""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['output'])
        cls_name = self.name
        validator.check_value_type("transpose_a", transpose_a, [bool], cls_name)
        validator.check_value_type("transpose_b", transpose_b, [bool], cls_name)
        self.add_prim_attr('adj_x1', self.transpose_a)
        self.add_prim_attr('adj_x2', self.transpose_b)


class Betainc(Primitive):
    r"""
    Computes the regularized incomplete beta integral
    :math:`I_{x}(a, b)`.

    The regularized incomplete beta integral is defined as:

    .. math::

        I_{x}(a, b)=\frac{B(x ; a, b)}{B(a, b)}

    where

    .. math::

        B(x ; a, b)=\int_{0}^{x} t^{a-1}(1-t)^{b-1} d t

    is the incomplete beta function and B(a, b) is the complete beta function

    Inputs:
        - **a** (Tensor) - A Tensor of types: float32, float64.
        - **b** (Tensor) - A Tensor, must have the same dtype and shape as `a` .
        - **x** (Tensor) - A Tensor, must have the same dtype and shape as `a` .

    Outputs:
        A Tensor, has the same dtype and shape as `a` .

    Raises:
        TypeError: If dtype of `a` is not float32 nor float64.
        TypeError: If either dtype of `b` and `x` is not the same as the `a`.
        ValueError: If either shape of `b` and `x` is not the same as the `a`.


    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = Tensor(np.array([1, 1, 1]), mindspore.float32)
        >>> b = Tensor(np.array([1, 1, 1]), mindspore.float32)
        >>> x = Tensor(np.array([1, 1,1 ]), mindspore.float32)
        >>> betainc = ops.Betainc()
        >>> print(betainc(a, b, x))
        [1. 1. 1.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Betainc"""
        self.init_prim_io_names(inputs=['a', 'b', 'x'], outputs=['output'])


class CumSum(Primitive):
    """
    Computes the cumulative sum of input tensor along axis.

    .. math::

        y_i = x_1 + x_2 + x_3 + ... + x_i

    Args:
        exclusive (bool): By default, this op performs an inclusive cumsum, which means that the first
            element of the input is identical to the first element of the output. Default: False.
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


class AddN(Primitive):
    """
    Computes addition of all input tensors element-wise.

    Refer to :func:`mindspore.ops.addn` for more details.

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
        raise TypeError(f"For '{self.name}', the type of 'inputs[0]' must be a tensor, but "
                        f"got {type(inputs[0]).__name__}, "
                        f"or the length of 'inputs' should not be equal to 1, but got ({len(inputs)}).")


class AccumulateNV2(Primitive):
    """
    Computes accumulation of all input tensors element-wise.

    Refer to :func:`mindspore.ops.accumulate_n` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU``

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
        raise TypeError(f"For '{self.name}', the type of 'inputs[0]' must be a tensor, "
                        f"but got {type(inputs[0]).__name__}, "
                        f"or the length of 'inputs' should not be equal to 1, but got ({len(inputs)}).")


class Neg(Primitive):
    """
    Returns a tensor with negative values of the input tensor element-wise.

    Refer to :func:`mindspore.ops.neg` for more details.

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


class InplaceUpdateV2(Primitive):
    r"""
    Updates specified rows with values in `v`.

    Note:
        This operator only supports dynamic shape. As for static shape, please use operator 'InplaceUpdate' instead.

    Args:

    Inputs:
        - **x** (Tensor) - A tensor which to be inplace updated. It can be one of the following data types:
          float32, float16 and int32.
        - **indices** (Union[int, tuple]): Indices into the left-most dimension of `x`, and determines which rows of x
            to update with v. It is an int or tuple, whose value is in [0, the first dimension size of x).
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
    def __init__(self):
        """Initialize InplaceUpdateV2"""
        self.init_prim_io_names(inputs=['x', 'indices', 'v'], outputs=['y'])

    def __call__(self, x, indices, v):
        args = [x, indices, v]
        output = _run_op(self, self.name, args)
        return output


class InplaceUpdate(Primitive):
    r"""
    Updates specified rows with values in `v`.

    Args:
        indices (Union[int, tuple]): Indices into the left-most dimension of `x`, and determines which rows of x
            to update with v. It is an int or tuple, whose value is in [0, the first dimension size of x).

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
        ``Ascend`` ``GPU`` ``CPU``

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


class InplaceAdd(Primitive):
    """
    Adds `v` into specified rows of `x`. Computes `y` = `x`; y[i,] += `v`.

    Refer to :func:`mindspore.ops.inplace_add` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor, ops
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


class InplaceIndexAdd(Primitive):
    """
    Adds tensor `updates` to specified axis and indices of tensor `var`. The axis should be in [0,  len(var.dim) - 1],
    and indices should be in [0, the size of `var` - 1] at the axis dimension.

    Refer to :func:`mindspore.ops.inplace_index_add` for more details.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> var = Parameter(np.array([[1, 2], [3, 4], [5, 6]]), mindspore.float32)
        >>> indices = Tensor(np.array([0, 1]), mindspore.int32)
        >>> updates = Tensor(np.array([[0.5, 1.0], [1.0, 1.5]]), mindspore.float32)
        >>> inplaceIndexAdd = ops.InplaceIndexAdd(axis=0)
        >>> var = inplaceIndexAdd(var, indices, updates)
        >>> print(var)
        [[1.5 3. ]
         [4.  5.5]
         [5.  6. ]]
    """

    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self, axis):
        """Initialize InplaceIndexAdd"""
        self.init_prim_io_names(inputs=['var', 'indices', 'updates'], outputs=['var'])
        self.axis = axis
        validator.check_value_type('axis', axis, [int], self.name)


class InplaceSub(Primitive):
    r"""
    Subtracts `v` into specified rows of `x`. Computes :math:`y = x`; :math:`y[i,] -= input\_v`.

    Refer to :func:`mindspore.ops.inplace_sub` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor, ops
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
        self.add_prim_attr("indices", self.indices)


class Sub(_MathBinaryOp):
    r"""
    Subtracts the second input tensor from the first input tensor element-wise.

    Refer to :func:`mindspore.ops.sub` for more details.

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
    r"""
    Multiplies two tensors element-wise.

    Refer to :func:`mindspore.ops.mul` for more details.

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

    # Let x/y using same sig_dtype to enable implicit conversion for compatibility
    __mindspore_signature__ = (
        sig.make_sig('x', rw=sig.sig_rw.RW_READ, dtype=sig.sig_dtype.T),
        sig.make_sig('y', rw=sig.sig_rw.RW_READ, dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize Xdivy."""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])

    @staticmethod
    def _infer_specified_mul_value(x, y):
        """Calculate min/max value for output of Mul op"""
        if x is not None and y is not None:
            if isinstance(x, (Tensor, Tensor_)):
                x = x.asnumpy()
            if isinstance(y, (Tensor, Tensor_)):
                y = y.asnumpy()
            x = np.array(x)
            y = np.array(y)
            out = x * y
            out = tuple(out.tolist())
            return out
        return None

    def _infer_min_value(self, x, y):
        """Calculate min value for output for Mul op"""
        return self._infer_specified_mul_value(x, y)

    def _infer_max_value(self, x, y):
        """Calculate max value for output for Mul op"""
        return self._infer_specified_mul_value(x, y)

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = x * y
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None

    def _infer_shape_value(self, x, y):
        shape_value = self._infer_specified_mul_value(x, y)
        shape_value = self._convert_back_shape(shape_value, x)
        return self._convert_back_shape(shape_value, y)


class SquaredDifference(Primitive):
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
        - **x** (Union[Tensor, Number, bool]) - The first input is a number, or a bool, or a tensor.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number, or a bool when the first input
          is a tensor, or a tensor.

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
    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        """Initialize _BinaryOp"""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])


class Square(Primitive):
    """
    Returns square of a tensor element-wise.

    .. math::

        out_{i} = (x_{i})^2

    Inputs:
        - **x** (Tensor) - The input tensor with a dtype of Number, its rank must be in [0, 7] inclusive.

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
        - **x** (Tensor) - The input of Rsqrt. Its rank must be in [0, 7] inclusive and
          each element must be a non-negative number.

    Outputs:
        Tensor, has the same type and shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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


class Sqrt(Primitive):
    r"""
    Returns square root of a tensor element-wise.

    Note:
        When there are some negative number, it will return a Tensor whose specific position is nan.

    .. math::

        out_{i} =  \sqrt{x_{i}}

    Inputs:
        - **x** (Tensor) - The input tensor with a dtype of Number, its rank must be in [0, 7] inclusive.

    Outputs:
        Tensor, has the same shape and data type as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

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


class Reciprocal(PrimitiveWithCheck):
    r"""
    Returns reciprocal of a tensor element-wise.

    .. math::

        out_{i} =  \frac{1}{x_{i}}

    Inputs:
        - **x** (Tensor) - The input tensor.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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

    def infer_value(self, x):
        if x is not None:
            x = x.asnumpy()
            out = 1.0 / x
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class Pow(Primitive):
    r"""
    Calculates the `y` power of each element in `x`.

    Refer to :func:`mindspore.ops.pow` for more details.

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


class Exp(Primitive):
    r"""
    Returns exponential of a tensor element-wise.

    Refer to :func:`mindspore.ops.exp` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.0, 1.0, 3.0]), mindspore.float32)
        >>> exp = ops.Exp()
        >>> output = exp(x)
        >>> print(output)
        [ 1.        2.718282 20.085537]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Exp"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        self.add_prim_attr("base", -1.0)
        self.add_prim_attr("scale", 1.0)
        self.add_prim_attr("shift", 0.0)


class Logit(Primitive):
    r"""
    Calculate the logit of a tensor element-wise. Element in `x` is clamped to [eps, 1-eps].

    .. math::
        \begin{align}
        y_{i} & = \ln(\frac{z_{i}}{1 - z_{i}}) \\
        z_{i} & = \begin{cases}
        x_{i} & \text{if eps is None} \\
        \text{eps} & \text{if } x_{i} \lt \text{eps} \\
        x_{i} & \text{if } \text{eps} \leq x_{i} \leq 1 - \text{eps} \\
        1 - \text{eps} & \text{if } x_{i} \gt 1 - \text{eps}
        \end{cases}
        \end{align}

    Refer to :func:`mindspore.ops.logit` for more details.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.1, 0.2, 0.3]).astype(np.float32))
        >>> op = ops.Logit(eps=1e-5)
        >>> output = op(x)
        >>> print(output)
        [-2.1972246 -1.3862944 -0.8472978]
    """

    @prim_attr_register
    def __init__(self, eps=-1.0):
        """Initialize Exp"""
        self.add_prim_attr("eps", eps)
        validator.check_value_type("eps", eps, [float], self.name)
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class ReduceStd(Primitive):
    """
    Returns the standard-deviation and mean of each row of the input tensor in the dimension `axis`.
    If `axis` is a list of dimensions, reduce over all of them.

    Refer to :func:`mindspore.ops.std` for more details.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[1, 2, 3], [-1, 1, 4]]).astype(np.float32))
        >>> op = ops.ReduceStd(axis=1, unbiased=True, keep_dims=False)
        >>> output = op(input_x)
        >>> output_std, output_mean = output[0], output[1]
        >>> print(output_std)
        [1.        2.5166113]
        >>> print(output_mean)
        [2.        1.3333334]
    """

    @prim_attr_register
    def __init__(self, axis=(), unbiased=True, keep_dims=False):
        """Initialize ReduceStd """
        validator.check_value_type("axis", axis, [int, tuple, list], self.name)
        validator.check_value_type("unbiased", unbiased, [bool], self.name)
        validator.check_value_type("keep_dims", keep_dims, [bool], self.name)
        if isinstance(axis, int):
            self.add_prim_attr('axis', [self.axis])
        else:
            for element_of_axis in axis:
                validator.check_value_type("element_of_axis", element_of_axis, [int], self.name)
        self.init_prim_io_names(inputs=['input_x'], outputs=['output_std', 'output_mean'])


class Einsum(Primitive):
    """
    Sums the product of the elements of the input Tensor along
    dimensions specified notation based on the Einstein summation convention(Einsum).
    You can use this operator to perform diagonal/reducesum/transpose/matmul/mul/inner product operations, etc.

    The inputs must be a tuple of tensors.
    When the inputs are only one tensor, you can input (tensor, )
    dtypes of them should be float16/float32/float64.

    Args:
        equation (str): An attribute, represent the operation you want to do.
            the value can contain only letters([a-z][A-Z]), commas(,), ellipsis(...),
            and arrow(->). the letters represent inputs's tensor dimension,
            commas(,)represent separate tensors, ellipsis(...) indicates
            the tensor dimension that you do not care about, the left of the
            arrow(->) indicates the input tensors,
            and the right of it indicates the desired output dimension.

    Inputs:
        - **x** (Tuple) - input tensor used for calculation. the data type of the tensor must be the same.

    Outputs:
        Tensor, the shape of it can be obtained from the equation,
        and the data type is the same as input tensors.

    Raises:
        TypeError: If equation itself is invalid, or the equation does not match the input tensor.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> equation = "i->"
        >>> einsum = ops.Einsum(equation)
        >>> output = einsum([x])
        >>> print(output)
        [7.]
        >>>
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> y = Tensor(np.array([2.0, 4.0, 3.0]), mindspore.float32)
        >>> equation = "i,i->i"
        >>> einsum = ops.Einsum(equation)
        >>> output = einsum((x, y))
        >>> print(output)
        [ 2. 8. 12.]
        >>>
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> y = Tensor(np.array([[2.0, 3.0], [1.0, 2.0], [4.0, 5.0]]), mindspore.float32)
        >>> equation = "ij,jk->ik"
        >>> einsum = ops.Einsum(equation)
        >>> output = einsum((x, y))
        >>> print(output)
        [[16. 22.]
        [37. 52.]]
        >>>
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "ij->ji"
        >>> einsum = ops.Einsum(equation)
        >>> output = einsum((x,))
        >>> print(output)
        [[1. 4.]
        [2. 5.]
        [3. 6.]]
        >>>
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "ij->j"
        >>> einsum = ops.Einsum(equation)
        >>> output = einsum((x,))
        >>> print(output)
        [5. 7. 9.]
        >>>
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "...->"
        >>> einsum = ops.Einsum(equation)
        >>> output = einsum((x,))
        >>> print(output)
        [21.]
        >>>
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> y = Tensor(np.array([2.0, 4.0, 1.0]), mindspore.float32)
        >>> equation = "j,i->ji"
        >>> einsum = ops.Einsum(equation)
        >>> output = einsum((x, y))
        >>> print(output)
        [[ 2. 4. 1.]
        [ 4. 8. 2.]
        [ 6. 12. 3.]]
    """

    @prim_attr_register
    def __init__(self, equation):
        if not isinstance(equation, str):
            raise TypeError("the equation must be str!")
        seg_equation = equation.split("->")
        if len(seg_equation) > 2:
            raise TypeError("the equation can contain only one arrow !")
        self.add_prim_attr('equation', equation)
        self.init_prim_io_names(inputs=['inputs'], outputs=['output'])


class Diagonal(Primitive):
    """
    Create a tensor with specific diagonal elements of input. This operator extracts the diagonal elements with
    offset from the 2-D sub-tensors which specified by dim1 and dim2.
    The shape of output tensor can be dertermined by removing dim1 and dim2 form the shape of input and appending
    a dimension at the end. The size of the last dimension is the length of diagonal.

    Args:
        offset (int): The offset of main diagonal, which controls which diagonal to consider. If :math:`offset=0`,
            return the main diagonal elements with respect to dim1 and dim2. If :math:`offset>0`, return the
            diagonal elements that are `offset` units upward from the main diagonal. If :math:`offset<0`, return the
            diagonal elements that are `offset` units downward from the main diagonal. Default: 0.
        dim1 (int): The first dimension with respect to which to take diagonal. Default: 0.
        dim2 (int): The second dimension with respect to which to take diagonal. Default: 1.

    Inputs:
        - **x** (Tensor) - The input to take diagonal, with float32 or double data type.
          The input must be at least 2-dimensional.
          The shape is :math:`(N_{0}, N_{1}, *)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        - **y** (Tensor) - A tensor whose values are diagonal of input, with the same data type as input.
          The shape of the output is one dimension lower than the input.
          If the shape of `x` is :math:`(d_{0}, d_{1}, ..., d_{n-1})`, the size of the `dim1` dimension
          is :math:`d_{i}` and the size of the `dim2` dimension is :math:`d_{j}`, the shape of `y` is the same
          as the input shape with `dim1` and `dim2` dimension removed and the diagonal dimension appended.
          If the `offset` is nonnegative, the size of output's last dimension is
          :math:`max(min(d_{i}, d_{j}-offset), 0)`. But if the `offset` is negative, the size of output's
          last dimension is :math:`max(min(d_{i} + offset, d_{j}), 0)`.

    Raises:
        TypeError: If dtype of `x` is neither float32 nor double.
        TypeError: If `offset` is not an int.
        TypeError: If `dim1` is not an int.
        TypeError: If `dim2` is not an int.
        ValueError: If the dimension of input is less than 2 dimensions.
        ValueError: If `dim1` is not in range of [-len(x.shape), len(x.shape)).
        ValueError: If `dim2` is not in range of [-len(x.shape), len(x.shape)).
        ValueError: If `dim1` and `dim2` are identical.

    Supported Platforms:
        ``Ascend``  ``CPU``

    Examples:
        >>> x = Tensor(np.array([[[ 0.,  1.,  2.,  3.], [ 4.,  5.,  6.,  7.], [ 8.,  9., 10., 11.]],
        ... [[12., 13., 14., 15.], [16., 17., 18., 19.], [20., 21., 22., 23.]]]), mindspore.float32)
        >>> diagonal_ops = ops.Diagonal(offset=1, dim1=-1, dim2=1)
        >>> y = diagonal_ops(x)
        >>> print(y)
        [[ 4.  9.]
         [16. 21.]]
    """

    @prim_attr_register
    def __init__(self, offset=0, dim1=0, dim2=1):
        """Initialize Diagonal"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        validator.check_is_int(offset, "offset", self.name)
        validator.check_is_int(dim1, "dim1", self.name)
        validator.check_is_int(dim2, "dim2", self.name)


class Expm1(Primitive):
    r"""
    Returns exponential then minus 1 of a tensor element-wise.

    Refer to :func:`mindspore.ops.expm1` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.0, 2.0, 3.0, 5.0]), mindspore.float32)
        >>> expm1 = ops.Expm1()
        >>> output = expm1(x)
        >>> print(output)
        [  0.         6.389056  19.085537 147.41316 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Expm1."""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class Histogram(Primitive):
    """
    Computes the histogram of a tensor.

    The elements are sorted into equal width bins between `min` and `max`.
    If `min` and `max` are both zero, the minimum and maximum values of the data are used.

    Elements lower than min and higher than max are ignored.

    Args:
        bins (int, optional): Number of histogram bins, optional. Default 100. If specified, must be positive.
        min (float, optional): An optional float of the lower end of the range (inclusive). Default value is 0.0.
        max (float, optional): An optional float of the upper end of the range (inclusive). Default value is 0.0.

    Inputs:
        - **x** (Tensor) - the input tensor, type support list [float16, float32, int32]

    Outputs:
        Tensor, 1-D Tensor with type int32.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `x` datetype not in support list.
        TypeError: If attr `min` or `max` is not float.
        TypeError: If attr `bins` is not int.
        ValueError: If attr value `min` > `max`.
        ValueError: If attr `bins` <= 0.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor([1., 2, 1])
        >>> op = ops.Histogram(bins=4, min=0.0, max=3.0)
        >>> y = op(x)
        >>> print(y)
        [0 2 1 0]
    """

    @prim_attr_register
    def __init__(self, bins=100, min=0.0, max=0.0):  # pylint: disable=W0622
        """Initialize Histogram."""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        validator.check_value_type("bins", bins, [int], self.name)
        validator.check_value_type("min", min, [float], self.name)
        validator.check_value_type("max", max, [float], self.name)
        validator.check_positive_int(bins, 'bins', self.name)
        validator.check('min', min, 'max', max, Rel.LE, self.name)


class HistogramFixedWidth(PrimitiveWithInfer):
    """
    Returns a rank 1 histogram counting the number of entries in values that fall into every bin. The bins are equal
    width and determined by the inputs `range` and the arguments `nbins`.

    Args:
        nbins (int): The number of histogram bins, the type is a positive integer.
        dtype (str, optional): An optional attribute. The dtype must be str. Default: "int32".

    Inputs:
        - **x** (Tensor) - Numeric Tensor. Must be one of the following types: int32, float32, float16.
        - **range** (Tensor) - Must have the same data type as `x`, and the shape is (2,).
          x <= range[0] will be mapped to histogram[0], x >= range[1] will be mapped to histogram[-1].

    Outputs:
        1-D Tensor, whose length is the type is `nbins` with dtype of int32.

    Raises:
        TypeError: If `dtype` is not a str or `nbins` is not an int.
        ValueError: If `nbins` is less than 1.
        ValueError: If `dtype` is not 'int32'.

    Supported Platforms:
        ``Ascend`` ``GPU``

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


class Log(Primitive):
    """
    Returns the natural logarithm of a tensor element-wise.

    Refer to :func:`mindspore.ops.log` for more details.

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
        self.add_prim_attr("cust_aicpu", self.name)
        self.add_prim_attr('base', -1.0)
        self.add_prim_attr('scale', 1.0)
        self.add_prim_attr('shift', 0.0)


class Log1p(Primitive):
    r"""
    Returns the natural logarithm of one plus the input tensor element-wise.

    Refer to :func:`mindspore.ops.log1p` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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


class Hypot(Primitive):
    """
    Computes hypotenuse of input tensors element-wise as legs of a right triangle.
    The shape of two inputs should be broadcastable, and data type of them should be
    one of: float32, float64

    Inputs:
        - **x1** (Tensor) - The first input tensor.
        - **x2** (Tensor) - The second input tensor.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting, and the data type is one
        with higher precision in the two inputs.

    Raises:
        TypeError: If data type `x1` or `x2` is not float32 or float64.
        ValueError: If shape of two inputs are not broadcastable.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor(np.array([3., 5., 7.]))
        >>> x2 = Tensor(np.array([4., 12., 24.]))
        >>> hypot_ = ops.Hypot()
        >>> y = hypot_(x1, x2)
        >>> print(y)
        [ 5. 13. 25.]
    """

    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])


class Heaviside(Primitive):
    r"""
    Computes the Heaviside step function for each element in input.

    .. math::
            \text { heaviside }(\text { x, values })=\left\{\begin{array}{ll}
            0, & \text { if x }<0 \\
            \text { values, } & \text { if x }==0 \\
            1, & \text { if x }>0
            \end{array}\right.

    Inputs:
        - **x** (Tensor) - The input tensor. With real number data type.
        - **values** (Tensor) - The values to use where `x` is zero.
          Values can be broadcast with `x` . 'x' should have the same
          dtype with 'values'.

    Outputs:
        Tensor, has the same type as 'x' and 'values'.

    Raises:
        TypeError: If `x` or `values` is not Tensor.
        TypeError: If data type `x` and `values` is different.
        ValueError: If shape of two inputs are not broadcastable.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1.5, 0., 2.]))
        >>> values = Tensor(np.array([0.5]))
        >>> heaviside = ops.Heaviside()
        >>> y = heaviside(x, values)
        >>> print(y)
        [0.  0.5 1. ]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x', 'values'], outputs=['y'])


class Erf(Primitive):
    r"""
    Computes the Gauss error function of `x` element-wise.

    Refer to :func:`mindspore.ops.erf` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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

    Refer to :func:`mindspore.ops.erfc` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU``  ``CPU``

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

    Refer to :func:`mindspore.ops.minimum` for more details.

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

    Refer to :func:`mindspore.ops.maximum` for more details.

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

    Refer to :func:`mindspore.ops.div` for more details.

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

    .. math::

        out_{i} = \frac{x_i}{y_i}

    Note:
        - Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
        - The inputs must be two tensors or one tensor and one scalar.
        - When the inputs are two tensors,
          dtypes of them cannot be bool at the same time, and the shapes of them can be broadcast.
        - When the inputs are one tensor and one scalar, the scalar could only be a constant.

    Inputs:
        - **x** (Union[Tensor, number.Number, bool]) - The first input is a number.Number or
          a bool or a tensor whose data type is
          `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
          `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        - **y** (Union[Tensor, number.Number, bool]) - The second input, when the first input is a Tensor,
          the second input should be a number.Number or bool value, or a Tensor whose data type is number or bool\_.
          When the first input is Scalar, the second input must be a Tensor whose data type is number or bool\_.

    Outputs:
        Tensor, the shape is the same as the one of the input `x` , `y` after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.
        TypeError: If data types of `x` and `y` are both Tensor with bool\_.

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
        >>> x = Tensor(np.array([-4.0, 5.0, 6.0]), mindspore.float32)
        >>> y = Tensor(2, mindspore.int32)
        >>> output = div(x, y)
        >>> print(output)
        [-2.  2.5  3.]
        >>> print(output.dtype)
        Float32
    """

    @staticmethod
    def _infer_specified_div_value(x, y):
        """Calculate min/max value for output of Div op"""
        if x is not None and y is not None:
            if isinstance(x, (Tensor, Tensor_)):
                x = x.asnumpy()
            if isinstance(y, (Tensor, Tensor_)):
                y = y.asnumpy()
            x = np.array(x)
            y = np.array(y)
            out = x / y
            out = tuple(out.tolist())
            return out
        return None

    def _infer_min_value(self, x, y):
        """Calculate min value for output for Div op"""
        return self._infer_specified_div_value(x, y)

    def _infer_max_value(self, x, y):
        """Calculate max value for output for Div op"""
        return self._infer_specified_div_value(x, y)

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = np.array(x / y, x.dtype)
            return Tensor(out)
        return None

    def _infer_shape_value(self, x, y):
        shape_value = self._infer_specified_div_value(x, y)
        shape_value = self._convert_back_shape(shape_value, x)
        return self._convert_back_shape(shape_value, y)


class DivNoNan(Primitive):
    r"""
    Operates a safe division between `x1` and `x2` element-wise. Returns 0 if element of `x2` is zero.

    Inputs of `x1` and `x2` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::
        output_{i} = \begin{cases}
        0, & \text{ if } x2_{i} = 0\\
        x1_{i} / x2_{i}, & \text{ if } x2_{i} \ne 0
        \end{cases}

    Inputs:
        - **x1** (Union[Tensor, number.Number, bool]) - The first input is a number.Number or
          a bool or a tensor whose data type is
          `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
          `bool_ <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_.
        - **x2** (Union[Tensor, number.Number, bool]) - The second input is a number.Number or
          a bool when the first input is a bool or a tensor whose data type is number or bool\_.
          When the first input is Scalar, the second input must be a Tensor whose data type is number or bool\_.


    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x1` and `x2` is not a number.Number or a bool or a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor(np.array([-1.0, 0., 1.0, 5.0, 6.0]), mindspore.float32)
        >>> x2 = Tensor(np.array([0., 0., 0., 2.0, 3.0]), mindspore.float32)
        >>> div_no_nan = ops.DivNoNan()
        >>> output = div_no_nan(x1, x2)
        >>> print(output)
        [0.  0.  0.  2.5 2. ]
    """

    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        """Initialize DivNoNan"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])


class MulNoNan(_MathBinaryOp):
    r"""
    Computes `x` * `y` element-wise. If `y` is zero, no matter what `x` is, it will return 0.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcasted.
    When the inputs are one tensor and one scalar, the scalar could only be a constant.

    .. math::
        output_{ij} = \begin{cases}
        0, & y_{ij} = 0;\\
        x_{ij} * y_{ij}, & otherwise.
        \end{cases}

    Note:
        The shapes of `x` and `y` should be the same or can be broadcasted.
        This is noncommutative: if `y` is NaN or infinite and `x` is 0, the result will be NaN.

    Inputs:
        - **x** (Union[Tensor]) - The first input is a tensor whose data type is one of
          int32, int64, float16, float32, float64, complex64, complex128 currently or scalar.
        - **y** (Union[Tensor]) - The second input is a tensor whose data type is one of
          int32, int64, float16, float32, float64, complex64, complex128 currently or scalar.

    Outputs:
        Tensor, the shape is the same as the shape after broadcasting,
        and the data type is the one with higher precision among the two inputs.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
        >>> x = Tensor(np.array([[-1.0, 6.0, 0], [0, np.nan, 4.0]]), mindspore.float32)
        >>> y = Tensor(np.array([[-1.0, 4.0, np.inf], [np.nan, 0, 1.0]]), mindspore.float32)
        >>> output = mul_no_nan(x, y)
        >>> print(output)
        [[ 1. 24. nan]
         [nan  0. 4.]]
        >>> print(output.dtype)
        Float32
        >>> # case 3 : the y is a scalar.
        >>> x = Tensor(np.array([[-1.0, 6.0, 0], [0, np.nan, 4.0]]), mindspore.float32)
        >>> y = Tensor(0, mindspore.float32)
        >>> output = mul_no_nan(x, y)
        >>> print(output)
        [[0. 0. 0.]
         [0. 0. 0.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize _BinaryOp"""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])


class FloorDiv(Primitive):
    """
    Divides the first input tensor by the second input tensor element-wise and round down to the closest integer.

    Refer to :func:`mindspore.ops.floor_div` for more details.

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
    Divides the first input tensor by the second input tensor element-wise and rounds the results
    of division towards zero. Equivalent to C-style integer division.

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
        ``Ascend``  ``GPU`` ``CPU``

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
        - When the elements of input exceed 2048, the accuracy of operator cannot guarantee the requirement of
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
        ``Ascend``  ``GPU`` ``CPU``

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
        - If shape is expressed as :math:`(D1,D2... ,Dn)`, then :math:`D1*D2... *DN<=1000000,n<=8`.

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

    Refer to :func:`mindspore.ops.floor` for more details.

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


class FloorMod(Primitive):
    r"""
    Computes the remainder of division element-wise, and it's a flooring divide.

    Refer to :func:`mindspore.ops.floor_mod` for more details.

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

    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        """Initialize FloorMod."""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])


class Ceil(PrimitiveWithInfer):
    r"""
    Rounds a tensor up to the closest integer element-wise.

    Refer to :func:`mindspore.ops.ceil` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
          or a tensor whose data type is float16, float32, float64, complex64, complex128 or bool.
        - **y** (Union[Tensor, Number, bool]) - The second input is a number,
          or a bool when the first input is a tensor, or a tensor whose data type is float16,
          float32, float64, complex64, complex128 or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, Number, bool.
        TypeError: If dtype of `x` and 'y' is not in [float16, float32, float64, complex64, complex128, bool].
        ValueError: If `x` could not be broadcast to a tensor with shape of `y`.
        RuntimeError: If the data type of `x`, `y` conversion of Parameter is given
                      but data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([2, 4, -1]), mindspore.float32)
        >>> y = Tensor(np.array([2, 2, 2]), mindspore.float32)
        >>> xdivy = ops.Xdivy()
        >>> output = xdivy(x, y)
        >>> print(output)
        [ 1.   2.  -0.5]
    """

    # Let x/y using same sig_dtype to enable implicit conversion for compatibility
    __mindspore_signature__ = (
        sig.make_sig('x', rw=sig.sig_rw.RW_READ, dtype=sig.sig_dtype.T),
        sig.make_sig('y', rw=sig.sig_rw.RW_READ, dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize Xdivy."""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])

    def infer_shape(self, x_shape, y_shape):
        """
        Infer shape for output of Xdivy
        :param x_shape: input shape of x
        :param y_shape: input shape of y
        :return:
        """
        output_shape = get_broadcast_shape(x_shape, y_shape, self.name)
        return output_shape

    def infer_dtype(self, x_dtype, y_dtype):
        """
        Infer type for output of Xdivy
        :param x_dtype: input type of x
        :param y_dtype: input type of y
        :return:
        """
        args = {'x': x_dtype, 'y': y_dtype}
        validator.check_scalar_or_tensor_types_same(args,
                                                    [mstype.float16, mstype.float32, mstype.float64, mstype.complex64,
                                                     mstype.complex128], self.name, True)
        return x_dtype

    def infer_value(self, x, y):
        """
        Infer value for constant folding
        :param x:
        :param y:
        :return:
        """
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = x / y
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class Xlogy(Primitive):
    r"""
    Computes the first input tensor multiplied by the logarithm of second input tensor element-wise.
    Returns zero when `x` is zero.

    Refer to :func:`mindspore.ops.xlogy` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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

    Refer to :func:`mindspore.ops.acosh` for more details.

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
        [0.        0.9624237 1.7627472 5.298292 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Acosh"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class Cosh(Primitive):
    r"""
    Computes hyperbolic cosine of input element-wise.

    Refer to :func:`mindspore.ops.cosh` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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

    Refer to :func:`mindspore.ops.asinh` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> asinh = ops.Asinh()
        >>> x = Tensor(np.array([-5.0, 1.5, 3.0, 100.0]), mindspore.float32)
        >>> output = asinh(x)
        >>> print(output)
        [-2.3124382  1.1947632  1.8184465  5.298342 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Asinh"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class Sinc(Primitive):
    r"""
    Computes the normalized sinc of input.

    Refer to :func:`mindspore.ops.sinc` for more details.

    .. math::

        y_i = \begin{cases}1 & \text{ if } x_i= 0\\ \frac{sin(\pi x_i)}{x_i} &
        \text{ otherwise } \end{cases}

    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape as the `x`. The dtype of output is float32 when dtype of `x` is in
        [uint8, int8, uint16, int16, uint32, int32, uint64, int64, bool]. Otherwise output has the
        same dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> import mindspore.ops.operations.math_ops as ops
        >>> from mindspore import Tensor, dtype
        >>> sinc = ops.Sinc()
        >>> x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float32)
        >>> output = sinc(x)
        >>> print(output)
        [0.47735003 0.8759357  0.7224278  0.47735003]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Sinc"""


class Sinh(Primitive):
    r"""
    Computes hyperbolic sine of the input element-wise.

    Refer to :func:`mindspore.ops.sinh` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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


class Quantile(Primitive):
    r"""
    Computes the q-th quantiles of all elements in the input tensor, doing a linear interpolation when the
    q-th quantile lies between two data points.

    Refer to :func:`mindspore.ops.quantile` and :func:`mindspore.ops.nanquantile` for more detail.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> quantile = ops.Quantile()
        >>> input = Tensor(np.array([0.0700, -0.5446,  0.9214]), mindspore.float32)
        >>> q = Tensor(np.array([0, 0.5, 1]), mindspore.float32)
        >>> output = quantile(input, q)
        >>> print(output)
        [-0.5446  0.07  0.9214]
    """

    @prim_attr_register
    def __init__(self, dim=None, keep_dims=False, ignore_nan=False):
        """Initialize Quantile"""
        if dim is not None:
            validator.check_value_type("dim", dim, [int], self.name)
        else:
            self.add_prim_attr("dim", 10000)
        if keep_dims is not None:
            validator.check_value_type("keep_dims", keep_dims, [bool], self.name)
        else:
            self.add_prim_attr("keep_dims", False)
        if ignore_nan is not None:
            validator.check_value_type("ignore_nan", ignore_nan, [bool], self.name)
        else:
            self.add_prim_attr("ignore_nan", False)


class Equal(Primitive):
    r"""
    Computes the equivalence between two tensors element-wise.

    Refer to :func:`mindspore.ops.equal` for more details.

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

    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        """Initialize Equal"""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])


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
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> y = Tensor(np.array([2, 3, 6]), mindspore.float32)
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
        ``Ascend`` ``GPU`` ``CPU``

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


class NotEqual(Primitive):
    """
    Computes the non-equivalence of two tensors element-wise.

    Refer to :func:`mindspore.ops.ne` for more details.

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
    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        """Initialize NotEqual"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['output'])


class Greater(PrimitiveWithCheck):
    r"""
    Compare the value of the input parameters :math:`x,y` element-wise, and the output result is a bool value.

    Refer to :func:`mindspore.ops.gt` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> greater = ops.Greater()
        >>> output = greater(x, y)
        >>> print(output)
        [False True False]
    """
    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])

    def infer_value(self, x, y):
        """
        Infer value for Greater.
        """
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = np.array(np.greater(x, y))
            return Tensor(out)
        return None


class GreaterEqual(PrimitiveWithCheck):
    r"""
    Computes the boolean value of :math:`x >= y` element-wise.

    Refer to :func:`mindspore.ops.ge` for more details.

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
    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = np.array(np.greater_equal(x, y))
            return Tensor(out)
        return None


class Lerp(Primitive):
    """
    Calculate the linear interpolation between two tensors based on the weight parameter.

    Refer to :func:`mindspore.ops.lerp` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> start = Tensor(np.array([1., 2., 3., 4.]), mindspore.float32)
        >>> end = Tensor(np.array([10., 10., 10., 10.]), mindspore.float32)
        >>> lerp = ops.Lerp()
        >>> output = lerp(start, end, 0.5)
        >>> print(output)
        [5.5 6. 6.5 7. ]
    """

    @prim_attr_register
    def __init__(self, name="Lerp"):
        super().__init__(name)
        self.init_prim_io_names(inputs=['start', 'end', 'weight'], outputs=['output'])


class Gcd(Primitive):
    """
    Computes greatest common divisor of input tensors element-wise.
    The shape of two inputs should be broadcastable, and data type of them should be
    one of: int32, int64

    Inputs:
        - **x1** (Tensor) - The first input tensor.
        - **x2** (Tensor) - The second input tensor.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting, and the data type is one
        with higher digits in the two inputs.

    Raises:
        TypeError: If data type `x1` or `x2` is not int32 or int64.
        ValueError: If shape of two inputs are not broadcastable.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor(np.array([7, 8, 9]))
        >>> x2 = Tensor(np.array([14, 6, 12]))
        >>> gcd_ = ops.Gcd()
        >>> y = gcd_(x1, x2)
        >>> print(y)
        [7 2 3]
    """

    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])


class Less(PrimitiveWithCheck):
    r"""
    Computes the boolean value of :math:`x < y` element-wise.

    Refer to :func:`mindspore.ops.less` for more details.

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
    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = np.array(np.less(x, y))
            return Tensor(out)
        return None


class LessEqual(PrimitiveWithCheck):
    r"""
    Computes the boolean value of :math:`x <= y` element-wise.

    Refer to :func:`mindspore.ops.le` for more details.

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
    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = np.array(np.less_equal(x, y))
            return Tensor(out)
        return None


class LogicalNot(Primitive):
    """
    Computes the "logical NOT" of a tensor element-wise.

    Refer to :func:`mindspore.ops.logical_not` for more details.

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

    Refer to :func:`mindspore.ops.logical_and` for more details.

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

    Refer to :func:`mindspore.ops.logical_or` for more details.

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


class LogicalXor(Primitive):
    r"""
    Computes the "logical XOR" of two tensors element-wise.

    .. math::

        out_{i} = x_{i} \oplus y_{i}

    Inputs:
        - **x** (Tensor) - The first input is a tensor whose data type is bool.
        - **y** (Tensor) - The second input is a the tensor to compute XOR with the first input.
          Datatype must be bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor whose data type is bool.
        ValueError: If the shape of two inputs cannot be broadcast.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([True, False, True]), mindspore.bool_)
        >>> y = Tensor(np.array([True, True, False]), mindspore.bool_)
        >>> logical_xor = ops.LogicalXor()
        >>> output = logical_xor(x, y)
        >>> print(output)
        [ False True True]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize LogicalXor"""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])


class IsNan(Primitive):
    r"""
    Determines which elements are NaN for each position.

    Refer to :func:`mindspore.ops.isnan` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
    Determines which elements are inf or -inf for each position.

    Refer to :func:`mindspore.ops.isinf` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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


class IsFinite(Primitive):
    r"""
    Determines which elements are finite for each position.

    Refer to :func:`mindspore.ops.isfinite` for more details.

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


class FloatStatus(Primitive):
    """
    Determines if the elements contain Not a Number(NaN), infinite or negative infinite. 0 for normal, 1 for overflow.

    Inputs:
        - **x** (Tensor) - The input tensor. The data type must be float16, float32 or float64.
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


class NPUAllocFloatStatus(Primitive):
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


class NPUGetFloatStatus(Primitive):
    """
    `mindspore.ops.NPUGetFloatStatus` updates the flag which is
    the output tensor of :class:`mindspore.ops.NPUAllocFloatStatus` with the latest overflow status.


    Note:
        The flag is a tensor whose shape is `(8,)` and data type is `mindspore.dtype.float32`.
        If the sum of the flag equals to 0, there is no overflow happened. If the sum of the
        flag is bigger than 0, there is overflow happened.
        In addition, there are strict sequencing requirements for use, i.e., before
        using the NPUGetFloatStatus operator, need to ensure that the NPUClearFlotStatus
        and your compute has been executed. We use :class:`mindspore.ops.Depend` to ensure the execution order.

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
        >>> from mindspore import ops
        >>> from mindspore.common import dtype as mstype
        >>> from mindspore.common.tensor import Tensor
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.alloc_status = ops.NPUAllocFloatStatus()
        ...         self.get_status = ops.NPUGetFloatStatus()
        ...         self.clear_status = ops.NPUClearFloatStatus()
        ...         self.sub = ops.Sub()
        ...         self.neg = ops.Neg()
        ...
        ...     def construct(self, x):
        ...         init = self.alloc_status()
        ...         clear_status = self.clear_status(init)
        ...         x = ops.depend(x, clear_status)
        ...         res = self.sub(x, self.neg(x))
        ...         init = ops.depend(init, res)
        ...         get_status = self.get_status(init)
        ...         res = ops.depend(res, get_status)
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


class NPUClearFloatStatus(Primitive):
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
        >>> from mindspore import ops
        >>> from mindspore.common import dtype as mstype
        >>> from mindspore.common.tensor import Tensor
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.alloc_status = ops.NPUAllocFloatStatus()
        ...         self.get_status = ops.NPUGetFloatStatus()
        ...         self.clear_status = ops.NPUClearFloatStatus()
        ...         self.sub = ops.Sub()
        ...         self.neg = ops.Neg()
        ...
        ...     def construct(self, x):
        ...         init = self.alloc_status()
        ...         clear_status = self.clear_status(init)
        ...         x = ops.depend(x, clear_status)
        ...         res = self.sub(x, self.neg(x))
        ...         init = ops.depend(init, res)
        ...         get_status = self.get_status(init)
        ...         res = ops.depend(res, get_status)
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


class Cos(Primitive):
    r"""
    Computes cosine of input element-wise.

    Refer to :func:`mindspore.ops.cos` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> cos = ops.Cos()
        >>> x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = cos(x)
        >>> print(output)
        [0.971338 0.6748758 0.95233357 0.9959527]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Cos"""


class ACos(Primitive):
    r"""
    Computes arccosine of input tensors element-wise.

    Refer to :func:`mindspore.ops.acos` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> acos = ops.ACos()
        >>> x = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float32)
        >>> output = acos(x)
        >>> print(output)
        [0.737726  1.5307857 1.2661036 0.9764105]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ACos"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class Sin(Primitive):
    r"""
    Computes sine of the input element-wise.

    Refer to :func:`mindspore.ops.sin` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> sin = ops.Sin()
        >>> x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float32)
        >>> output = sin(x)
        >>> print(output)
        [0.5810352 0.27635565 0.41687083 0.5810352]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Sin."""


class Asin(Primitive):
    r"""
    Computes arcsine of input tensors element-wise.

    Refer to :func:`mindspore.ops.asin` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> asin = ops.Asin()
        >>> x = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float32)
        >>> output = asin(x)
        >>> print(output)
        [0.8330704  0.04001067 0.30469266 0.5943858 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Asin"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class NMSWithMask(PrimitiveWithInfer):
    r"""
    Non-maximum Suppression. When object detection problem is performed in the computer vision field,
    object detection algorithm generates
    a plurality of bounding boxes. Use the box with the highest score, calculate the overlap between other boxes and
    the current box, and delete the box based on a certain threshold(IOU). On Ascend platform, the input box score is
    ignored, which only selects boexs based on the IOU between boxes, which means if you want to remove boxes that has
    lower scores, you need to sort the input boxes by score in descending order in advance. The IOU is as follows:

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
        tuple[Tensor], tuple of three tensors, they are output_boxes, output_idx and selected_mask.

        - **output_boxes** (Tensor) - The shape of tensor is :math:`(N, 5)`. On GPU and CPU platform, it is a sorted
          list of bounding boxes by sorting the input `bboxes` in descending order of score. On Ascend platform,
          it is same as input `bboxes`.
        - **output_idx** (Tensor) - The shape of tensor is :math:`(N,)`. The indexes list of `output_boxes`.
        - **selected_mask** (Tensor) - The shape of tensor is :math:`(N,)`. A mask list of
          valid output bounding boxes. Apply this mask on `output_boxes` to get the list of bounding boxes after
          non-max suppression calculation, or apply this mask on `output_idx` to get the indexes list of bounding boxes
          after non-max suppression calculation.

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
        if bboxes_shape[0] != -1:
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

    Refer to :func:`mindspore.ops.abs` for more details.

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
        ``Ascend`` ``GPU`` ``CPU``

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

    Refer to :func:`mindspore.ops.round` for more detailsed.

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

    Refer to :func:`mindspore.ops.tan` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class Atan(Primitive):
    r"""
    Computes the trigonometric inverse tangent of the input element-wise.

    Refer to :func:`mindspore.ops.atan` for more details.

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

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Refer to :func:`mindspore.ops.atanh` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class Atan2(_MathBinaryOp):
    r"""
    Returns arctangent of x/y element-wise.

    Refer to :func:`mindspore.ops.atan2` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0, 1]), mindspore.float32)
        >>> y = Tensor(np.array([1, 1]), mindspore.float32)
        >>> atan2 = ops.Atan2()
        >>> output = atan2(x, y)
        >>> print(output)
        [0.        0.7853982]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Atan2"""
        _MathBinaryOp.__init__(self)


class SquareSumAll(Primitive):
    r"""
    Returns the square sum of a tensor element-wise.

    .. math::

        \left\{\begin{matrix}out_{x} = {\textstyle \sum_{0}^{N}} (x_{i})^2
        \\out_{y} = {\textstyle \sum_{0}^{N}} (y_{i})^2
        \end{matrix}\right.

    Note:
        SquareSumAll only supports float16 and float32 data type.

    Inputs:
        - **x** (Tensor) - The input tensor. The data type must be float16 or float32.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        - **y** (Tensor) - The input tensor has the same type and shape as the `x`.

    Outputs:
        - **output_x** (Tensor) - The same type as the `x`.
        - **output_y** (Tensor) - The same type as the `x`.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.
        ValueError: If `x` and `y` are not the same shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
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
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output_x', 'output_y'])


class BitwiseAnd(_BitwiseBinaryOp):
    r"""
    Returns bitwise `and` of two tensors element-wise.

    Refer to :func:`mindspore.ops.bitwise_and` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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

    Refer to :func:`mindspore.ops.bitwise_or` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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

    Refer to :func:`mindspore.ops.bitwise_xor` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
        TypeError: If `x` is not a Tensor of float16, float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> bessel_i0 = ops.BesselI0()
        >>> x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = bessel_i0(x)
        >>> print(output)
        [1.0144521 1.1797839 1.0241698 1.0020262]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs='y')


class BesselI1(Primitive):
    """
    Computes BesselI1 of input element-wise.

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Data type must be float16 or float32.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor of float16, float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

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
          Data type must be float16, float32 or float64.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
          Data type must be float16 or float32, float64.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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


class BesselK0(Primitive):
    r"""
    Computes BesselK0 of input element-wise.

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Data type must be float16, float32, float64.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor of float16, float32, float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> bessel_k0 = ops.BesselK0()
        >>> x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = bessel_k0(x)
        >>> print(output)
        [1.579826  0.5402144 1.3424659 2.5310173]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize BesselK0"""


class BesselK1(Primitive):
    r"""
    Computes BesselK1 of input element-wise.

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Data type must be float16, float32, float64.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor of float16, float32, float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> bessel_k1 = ops.BesselK1()
        >>> x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = bessel_k1(x)
        >>> print(output)
        [3.9190812  0.8143549  2.9440577 10.974864]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize BesselK1"""


class BesselK0e(Primitive):
    """
    Computes BesselK0e of input element-wise.

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Data type must be float16, float32, float64.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor of float16, float32, float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> bessel_k0e = ops.BesselK0e()
        >>> x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = bessel_k0e(x)
        >>> print(output)
        [2.0083523 1.2388839 1.8303517 2.769374 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize BesselK0e"""


class BesselK1e(Primitive):
    """
    Computes BesselK1e of input element-wise.

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Data type must be float16, float32, float64.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor of float16, float32, float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> bessel_k1e = ops.BesselK1e()
        >>> x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = bessel_k1e(x)
        >>> print(output)
        [ 4.9821286  1.8675754  4.0140023 12.008413 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize BesselK1e"""


class BesselJ0(Primitive):
    """
    Computes BesselJ0 of input element-wise.

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Data type must be float16, float32 or float64.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor of float16, float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> bessel_j0 = ops.BesselJ0()
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = bessel_j0(x)
        >>> print(output)
        [0.93846981  0.76519769  0.22389078  -0.39714981]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize BesselJ0"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


class BesselJ1(Primitive):
    """
    Computes BesselJ1 of input element-wise.

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Data type must be float16, float32 or float64.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor of float16, float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> bessel_j1 = ops.BesselJ1()
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = bessel_j1(x)
        >>> print(output)
        [0.24226846,  0.44005059,  0.57672481, -0.06604333]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize BesselJ1"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


class BesselY0(Primitive):
    """
    Computes BesselY0 of input element-wise.

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Data type must be float16, float32 or float64.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor of float16, float32, float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> bessel_y0 = ops.BesselY0()
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = bessel_y0(x)
        >>> print(output)
        [-0.44451873  0.08825696  0.51037567  -0.01694074]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize BesselY0"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


class BesselY1(Primitive):
    """
    Computes BesselY1 of input element-wise.

    Inputs:
        - **x** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Data type must be float16, float32 or float64.

    Outputs:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor of float16, float32, float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> bessel_y1 = ops.BesselY1()
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = bessel_y1(x)
        >>> print(output)
        [-1.47147239  -0.78121282  -0.10703243  0.39792571]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize BesselY1"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


class Inv(Primitive):
    r"""
    Computes Reciprocal of input tensor element-wise.

    Refer to :func:`mindspore.ops.inv` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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

    Refer to :func:`mindspore.ops.invert` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
    Create a Tensor with the same data type and shape as input, and the element value is the minimum value that the
    corresponding data type can be expressed.

    Inputs:
        - **x** (Tensor) - Tensor of any dimension used to obtain the minimum value that its data type can be expressed.
          The data type must be float16, float32 or float64.

    Outputs:
        Tensor, has the same type and shape as `x`, but filled with `x` dtype minimum val.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If data type of `x` is neither float16 nor float32.

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


class LinSpace(Primitive):
    r"""
    Returns a Tensor whose value is `num` evenly spaced in the interval `start` and `stop` (including `start` and
    `stop`), and the length of the output Tensor is `num`.

    Refer to :func:`mindspore.ops.linspace` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
        self.init_prim_io_names(inputs=['start', 'stop', 'num'], outputs=['output'])


class MatrixInverse(Primitive):
    """
    Returns the inverse of the input matrix. If the matrix is irreversible, an error may be reported or an unknown
    result may be returned.

    Note:
        The parameter 'adjoint' is only supporting False right now, because complex number is not supported at present.

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


class MatrixPower(Primitive):
    """
    Computes the n-th power of a batch of square matrices.
    If n = 0, it returns a batch of identity matrices. If n is negative, it
    returns the inverse of each matrix (if invertible) raised to the power of abs(n).

    Args:
        n (int) : The exponent, a required int.

    Inputs:
        - **x** (Tensor) - A 3-D Tensor. Supported data types are float16 and float32.
          The shape is :math:`(b, m, m)`, represents b m-D square matrices.

    Outputs:
        - **y** (Tensor) - A 3-D Tensor. Data type and shape are the same as `x`'s.

    Raises:
        TypeError: If the data type of `n` is not int.
        TypeError: If the data type of `x` is neither float32 nor float16.
        TypeError: If x is not a Tensor.
        ValueError: If `x` is not a 3-D tensor.
        ValueError: If shape[1] and shape[2] of `x` are not the same.
        ValueError: If n is negative but got input x has singular matrices.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor([[[0, 1], [-1, 0]], [[1, 0], [0, -1]]], dtype=ms.float32)
        >>> matrix_power = ops.MatrixPower(n=2)
        >>> y = matrix_power(x)
        >>> print(y)
        [[[-1.  0.]
          [-0. -1.]]
         [[ 1.  0.]
          [ 0.  1.]]]
    """

    @prim_attr_register
    def __init__(self, n):
        super().__init__(name="MatrixPower")
        self.n = validator.check_value_type("n", n, [int], self.name)


class MatrixDeterminant(Primitive):
    """
    Computes the determinant of one or more square matrices.

    Refer to :func:`mindspore.ops.matrix_determinant` for more details.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[[-4.5, -1.5], [7.0, 6.0]], [[2.5, 0.5], [3.0, 9.0]]]), mindspore.float32)
        >>> op = ops.MatrixDeterminant()
        >>> output = op(input_x)
        >>> print(output)
        [-16.5 21. ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize MatrixDeterminant."""
        super().__init__("MatrixDeterminant")
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class LogMatrixDeterminant(Primitive):
    """
    Computes the sign and the log of the absolute value of the determinant of one or more square matrices.

    Refer to :func:`mindspore.ops.log_matrix_determinant` for more details.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[[-4.5, -1.5], [7.0, 6.0]], [[2.5, 0.5], [3.0, 9.0]]]), mindspore.float32)
        >>> op = ops.LogMatrixDeterminant()
        >>> sign, output = op(input_x)
        >>> print(sign)
        [-1.   1.]
        >>> print(output)
        [2.80336046e+00    3.04452229e+00]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize LogMatrixDeterminant."""
        super().__init__("LogMatrixDeterminant")
        self.init_prim_io_names(inputs=['x'], outputs=['sign', 'y'])


class MatrixLogarithm(Primitive):
    """
    Return the matrix logarithm of one or more square matrices.

    Inputs:
        - **x** (Tensor) - x is a tensor. The shape of tensor is :math:`[..., M, M]`.
          Must be one of the following types:complex64, complex128. And shape must be 2D-7D.

    Outputs:
        - **y** (Tensor) - has the same shape and type as input.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not one of: complex64, complex128.
        ValueError: If the dimension of `x` is less to 2.
        ValueError: If the inner two dimension is not equal.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor([[1 + 2j, 2 + 1j], [4 + 1j, 5 + 2j]])
        >>> matrix_logarithm = ops.MatrixLogarithm()
        >>> y = matrix_logarithm(x)
        >>> print(y)
        [[0.69155775+1.71618359j 0.64665196-0.34928196j]
         [1.02426074-0.88736831j 1.44677531+0.6400109j ]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize MatrixLogarithm"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class IndexAdd(Primitive):
    """
    Adds tensor `y` to specified axis and indices of tensor `x`. The axis should be in [0,  len(x.dim) - 1],
    and indices should be in [0, the size of `x` - 1] at the axis dimension.

    Args:
        axis (int): The dimension along which to index.
        use_lock (bool): Whether to enable a lock to protect the updating process of variable tensors.
            If true, when updating the value of `x`, this process will be protected by a lock by using atomic operation.
            If false, the result may be unpredictable. Default: True.
        check_index_bound (bool): If true, check index boundary. If false, don't check index boundary. Default: True.

    Inputs:
        - **x** (Parameter) - The input Parameter to add to.
        - **indices** (Tensor) - Add the value of `x` and `y` along the dimension of the `axis` according to the
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
        ValueError: If shape of `indices` is not 1D or size of `indices` is not equal to dimension of y[axis].
        ValueError: If `y`'s shape is not the same as `x` except the `axis` th dimension.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
        self.add_prim_attr('side_effect_mem', True)


class Erfinv(Primitive):
    r"""
    Computes the inverse error function of input. The inverse error function is defined in the range (-1, 1).

    The formula is defined as:

    .. math::
                                erfinv(erf(x)) = x

    Inputs:
        - **input_x** (Tensor) - The input tensor to compute to, with data type float32, float16 or float64.

    Outputs:
        Tensor, has the same shape and dtype as `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is not float16, float32 or float64.

    Supported Platforms:
        ``Ascend`` ``GPU``

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


class Conj(Primitive):
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
        ``Ascend`` ``GPU`` ``CPU``

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


class ComplexAbs(Primitive):
    r"""
    Returns a Tensor that contains the magnitudes of the input.

    The complex numbers in input must be of the form a + bj, where a is the real part and b is the imaginary part.

    .. math::

        y = \sqrt{a^2+b^2}.

    Inputs:
        - **x** (Tensor) - A Tensor, types: complex64, complex128.

    Outputs:
        Tensor, has the same shape as x. If the type of x is complex64, the type of output is float32.
        If the type of x is complex128, the type of output is float64.

    Raises:
       TypeError: If the input is not a Tensor.
       TypeError: If the input type is not complex64 or complex128.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.asarray(np.complex(3+4j)), mindspore.complex64)
        >>> complex_abs = ops.ComplexAbs()
        >>> output = complex_abs(x)
        >>> print(output)
        5.0
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ComplexAbs"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class Real(Primitive):
    """
    Returns a Tensor that is the real part of the input.
    If input is real, it is returned unchanged.

    Inputs:
        - **input** (Tensor) - The input tensor to compute to.

    Outputs:
        Tensor, the shape is the same as the input.

    Raises:
       TypeError: If the input is not a Tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

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
       TypeError: If the dtypes of two inputs are not same.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> real = Tensor(np.array([1]), mindspore.float32)
        >>> imag = Tensor(np.array([2]), mindspore.float32)
        >>> complex = ops.Complex()
        >>> output = complex(real, imag)
        >>> print(output)
        [1.+2.j]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Complex"""
        self.init_prim_io_names(inputs=['real', 'imag'], outputs=['output'])


class Imag(Primitive):
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
        ``GPU`` ``CPU``

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


class Angle(Primitive):
    """
    Returns the element-wise argument of a complex tensor.
    The elements in input are considered to be complex numbers of the form a+bj, where a is the real part and b
    is the imaginary part. The argument returned by this function is of the form atan2(b,a).

    Inputs:
        - **input** (Tensor) - The input tensor. types: complex64, complex128.

    Outputs:
        Tensor, has the float32 or float64 type and the same shape as input.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If the dtype of input is not one of: complex64, complex128.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input = Tensor([-1.5 + 7.8j, 3 + 5.75j], mindspore.complex64)
        >>> angle = ops.Angle()
        >>> output = angle(input)
        >>> print(output)
        [1.7607845 1.0899091]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Angle"""
        self.init_prim_io_names(inputs=['input'], outputs=['output'])


class Trunc(Primitive):
    """
    Returns a new tensor with the truncated integer values of the elements of input.

    Refer to :func:`mindspore.ops.trunc` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([3.4742, 0.5466, -0.8008, -3.9079]), mindspore.float32)
        >>> output = ops.Trunc()(x)
        >>> print(output)
        [ 3.  0. -0. -3.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Trunc"""
        self.init_prim_io_names(inputs=['input'], outputs=['output'])


class TridiagonalMatMul(Primitive):
    """
    Return the result of a multiplication of two matrices, where the left one is a Tridiagonal Matrix.

    Inputs:
        - **superdiag** (Tensor) - Superdiagonals of Tridiagonal Matrices to the left of multiplication.
          Data types must be: float16, float32, double, complex64, complex128.
          The shape is [..., 1, M].
          Last element is ignored.
        - **maindiag** (Tensor) - Maindiagonals of Tridiagonal Matrices to the left of multiplication.
          Data types must be: float16, float32, double, complex64, complex128.
          The shape is [..., 1, M].
        - **subdiag** (Tensor) - Subdiagonals of Tridiagonal Matrices to the left of multiplication.
          Data types must be: float16, float32, double, complex64, complex128.
          The shape is [..., 1, M].
          First element is ignored.
        - **rhs** (Tensor) - MxN Matrices to the right of multiplication.
          Data types must be: float16, float32, double, complex64, complex128.
          The shape is [..., M, N].

    Outputs:
        Tensor, with the same shape and data type as the `rhs`.

    Raises:
        TypeError: If dtypes of `superdiag`, `maindiag`, `subdiag` and `rhs`
                   are not float16, float32, double, complex64, complex128.
        ValueError: If the col of input `superdiag`, the col of input `maindiag`,
                    the col of input `subdiag` and the row of input `rhs` are not equal.
        ValueError: If the row of input `superdiag`, the row of input `maindiag` and
                    the row of input `subdiag` are not 1.
        ValueError: If the rank of input `superdiag`, the rank of input `maindiag`,
                    the rank of input `subdiag` and rank row of input `rhs`
                    are not equal to or greater than 2.
        ValueError: If the shape of input `superdiag`, the shape of input `maindiag` and
                    the shape of input `subdiag` are not same.
        ValueError: If the shape of input `superdiag` ignoring the last two elements,
                    the shape of input `maindiag` ignoring the last two elements,
                    the shape of input `subdiag` ignoring the last two elements and
                    the shape of input `rhs` ignoring the last two elements
                    are not same.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> tridiagonalmatmul = ops.TridiagonalMatMul()
        >>> superdiag = Tensor(np.array([[1, 2, 3]]).astype(np.float32))
        >>> maindiag = Tensor(np.array([[1, 2, 3]]).astype(np.float32))
        >>> subdiag = Tensor(np.array([[1, 2, 3]]).astype(np.float32))
        >>> rhs = Tensor(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.float32))
        >>> output = tridiagonalmatmul(superdiag,maindiag,subdiag,rhs)
        >>> print(output)
        [[ 2.  2.  2. ]
         [ 6.  6.  6.]
         [ 6.  6.  6.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize TridiagonalMatMul"""
        self.init_prim_io_names(
            inputs=['superdiag', 'maindiag', 'subdiag', 'rhs'],
            outputs=['y'])


class Igamma(Primitive):
    r"""
    Calculates lower regularized incomplete Gamma function.
    The lower regularized incomplete Gamma function is defined as:

    .. math::
        P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)

    where

    .. math::
        gamma(a, x) = \int_0^x t^{a-1} \exp^{-t} dt

    is the lower incomplete Gamma function.

    Above :math:`Q(a, x)` is the upper regularized complete Gamma function.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Inputs:
        - **a** (Tensor) - The input tensor. With type of float32 or float64.
        - **x** (Tensor) - The input tensor. With float32 or float64 type. `x` should have
          the same dtype with `a`.

    Outputs:
        Tensor, has the same dtype as `a` and `x`.

    Raises:
        TypeError: If a or x is not a Tensor.
        TypeError: If dtype of input x and a is not float32 nor float64.
        TypeError: If x has different dtype with a.
        ValueError: If `a` could not be broadcast to a tensor with shape of `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = Tensor(np.array([2.0, 4.0, 6.0, 8.0]).astype(np.float32))
        >>> x = Tensor(np.array([2.0, 3.0, 4.0, 5.0]).astype(np.float32))
        >>> igamma = P.Igamma()
        >>> output = igamma(a, x)
        >>> print (output)
        [0.593994  0.35276785  0.21486944  0.13337152]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Igamma"""
        self.init_prim_io_names(inputs=['a', 'x'], outputs=['z'])


class Igammac(Primitive):
    r"""
    Compute the upper regularized incomplete Gamma function Q(a, x).

    The upper regularized incomplete Gamma function is defined as:
    \(Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)\)
    where
    \(Gamma(a, x) = int_{x}^{\infty} t^{a-1} exp(-t) dt\)

    is the upper incomplete Gama function.

    Note, above P(a, x) (Igamma) is the lower regularized complete Gamma function.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Inputs:
        - **a** (Tensor) - The input tensor of igammac. With float32 or float64 data type.
        - **x** (Tensor) - The input tensor of igammac. With float32 or float64 type. `x` should have
          the same type with `a`.

    Outputs:
        A Tensor, has the same dtype as `a` and `x`.

    Raises:
        TypeError: If dtype of input x and a is not float32 nor float64.
        TypeError: If a or x is not a Tensor.
        TypeError: If x has different dtype with a.
        ValueError: If `a` could not be broadcast to a tensor with shape of `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = Tensor(np.array([2.0, 4.0, 6.0, 8.0]).astype(np.float32))
        >>> x = Tensor(np.array([2.0, 3.0, 4.0, 5.0]).astype(np.float32))
        >>> igammac = P.Igammac()
        >>> output = igammac(a, x)
        >>> print (output)
        [0.40600586 0.6472318  0.7851304  0.8666283 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Igammac"""
        self.init_prim_io_names(inputs=['a', 'x'], outputs=['z'])


class IsClose(Primitive):
    r"""
    Returns a boolean Tensor where two tensors are element-wise equal within a tolerance.

    Refer to :func:`mindspore.ops.isclose` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.ops import IsClose
        >>> input = Tensor(np.array([1.3, 2.1, 3.2, 4.1, 5.1]), mindspore.float16)
        >>> other = Tensor(np.array([1.3, 3.3, 2.3, 3.1, 5.1]), mindspore.float16)
        >>> isclose = IsClose()
        >>> output = isclose(input, other)
        >>> print(output)
        [ True False False False  True]
    """

    @prim_attr_register
    def __init__(self, rtol=1e-05, atol=1e-08, equal_nan=True):
        """Initialize IsClose"""
        validator.check_value_type('rtol', rtol, [float], self.name)
        validator.check_value_type('atol', atol, [float], self.name)
        validator.check_value_type('equal_nan', equal_nan, [bool], self.name)
        if context.get_context("device_target") == "Ascend" and not equal_nan:
            raise ValueError("For IsClose, the `equal_nan` must be True on Ascend, but got False.")
        validator.check_non_negative_float(rtol, 'rtol', self.name)
        validator.check_non_negative_float(atol, 'atol', self.name)


class MatrixExp(Primitive):
    r"""
    Computes the matrix exponential of a square matrix. Supports batched inputs.

    Refer to :func:`mindspore.ops.matrix_exp` for more details.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> matrix_exp = ops.MatrixExp()
        >>> x = Tensor(np.array([[1, 2], [0, 1]]), mindspore.float32)
        >>> output = matrix_exp(x)
        >>> print(output)
        [[2.7182817 5.436563 ]
        [0.        2.7182817]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize MatrixExp"""


class MatrixSolve(Primitive):
    """
    Solves systems of linear equations.

    Args:
        adjoint (bool, optional): Indicating whether to solve with matrix or
            its (block-wise) adjoint. Default: False.

    Inputs:
        - **matrix** (Tensor) - A tensor of shape :math:`[..., M, M]`,
          is a matrix of coefficients for a system of linear equations.
        - **rhs** (Tensor) - A tensor of shape :math:`[..., M, K]`,
          is a matrix of the resulting values of a system of linear equations.
          'rhs' must have the same type as `matrix`.

    Outputs:
        Tensor, a matrix composed of solutions to a system of linear equations,
        which has the same type and shape as 'rhs'.

    Raises:
        TypeError: If `adjoint` is not the type of bool.
        TypeError: If the type of `matrix` is not one of the following dtype:
                   mstype.float16, mstype.float32, mstype.float64, mstype.complex64,
                   mstype.complex128.
        TypeError: If the type of `matrix` is not the same as that of `rhs`.
        ValueError: If the rank of `matrix` less than 2.
        ValueError: If the dimension of `matrix` is not the same as `rhs` .
        ValueError: If the inner-most 2 dimension of `matrix` is not the same.
        ValueError: If the inner-most 2 dimension of `rhs` does not match `matrix` .

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> matrix = Tensor(np.array([[1.0  , 4.0],
        ...                       [2.0 , 7.0]]), mindspore.float32)
        >>> rhs = Tensor(np.array([[1.0]  , [3.0]]), mindspore.float32)
        >>> matrix_solve = ops.MatrixSolve(adjoint = False)
        >>> output = matrix_solve(matrix, rhs)
        >>> print(output)
        [[5.0], [-1.0]]
    """

    @prim_attr_register
    def __init__(self, adjoint=False):
        super().__init__(name="MatrixSolve")
        self.adjoint = validator.check_value_type("adjoint", adjoint, [bool], self.name)


class MatrixSolveLs(Primitive):
    r"""
    Solves one or more linear least-squares problems.

    If `fast` is `True`,then the solution is computed by solving the normal equations using Cholesky decomposition.
    If `fast` is `False` an algorithm based on the numerically robust complete orthogonal decomposition is used. This
    path is typically 6-7 times slower than the fast path. If `fast` is `False` then `l2_regularizer` is ignored.

    Args:
        fast (bool): An optional bool. Defaults to True.

    Inputs:
        - **matrix** (Tensor) -  A Tensor. Must be one of the following data types: float64, float32, complex64,
          complex128. Shape is :math:`(*, M, N)`.
        - **rhs** (Tensor) -  A Tensor. Must have the same data type as matrix. Shape is :math:`(*, M, K)`.
          `matrix` and `rhs` should have the same dimensions except the last one.
        - **l2_regularizer** (Tensor) - A Tensor of type float64. Scalar tensor.

    Outputs:
        Tensor of shape :math:`(*, N, K)` with the same data type as `matrix`.

    Raises:
        TypeError: If `matrix`, `rhs` or `l2_regularizer` is not tensor.
        TypeError: If either of `matrix` and `rhs` is not float32, float64, complex64 or complex128.
        TypeError: If `l2_regularizer` is not float64.
        TypeError: If `fast` is not bool.
        ValueError: If dimensions of `matrix` or `rhs` is less than 2.
        ValueError: If shape of `matrix` dose not match the shape of `rhs`.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> matrix_solve_ls = ops.MatrixSolveLs(fast=True)
        >>> matrix = Tensor([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], mstype.float32)
        >>> rhs = Tensor(np.array([[4], [2], [4], [2]]), mstype.float32)
        >>> l2 = Tensor(0.0, mstype.float64)
        >>> output = matrix_solve_ls(matrix, rhs, l2)
        >>> print(output)
        [[ 1.3333334]
        [-0.6666667]
        [ 2.6666665]
        [-1.3333333]]
    """

    @prim_attr_register
    def __init__(self, fast=True):
        """Initialize MatrixSolveLs"""
        validator.check_value_type('fast', fast, [bool], self.name)


class Lu(Primitive):
    """
    Computes the LU decomposition of one or more square matrices.

    Args:
        output_idx_type (:class:`mindspore.dtype`): An optional data type of `mindspore.dtype.int32`.
            Default: `mindspore.dtype.int32`.

    Inputs:
        - **input** (Tensor) - A tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
          matrices of size `[M, M]`, with data type float32, float64, complex64, complex128.

    Outputs:
        - **lu** (Tensor) - A tensor of shape `[..., M, M]` whose strictly lower triangular part denotes the lower
          triangular factor `L` with unit diagonal. Upper triangular part denotes the upper triangular factor `U`.
        - **p** (Tensor) - Permutation of the rows encoded as a list of indices in `0..M-1`, shape is `[..., M]`.

    Raises:
        TypeError: If the dtype of `input` is not one of the following dtype:
            float32, float64, complex64, complex128.
        TypeError: If `output_idx_type` is neither int32 nor int64.
        ValueError: If `input` rank is less than 2.
        ValueError: If input[-1] is not equal to input[-2].

    Supported Platforms:
        ``GPU``

    Examples:
        >>> input = Tensor(np.array([[2.5,3.1,3.5], [4.7,1.9,0.2], [1.1,3.6,2.0]]), mindspore.float32)
        >>> lu, p = ops.Lu(output_idx_type=mindspore.int32)(input)
        >>> print(lu)
        [[4.7        1.9        0.2       ]
         [0.23404257 3.155319   1.9531915 ]
         [0.5319149  0.6621713  2.1002696 ]]
        >>> print(p)
        [1 2 0]
    """

    @prim_attr_register
    def __init__(self, output_idx_type):
        super().__init__(name="Lu")
        self.init_prim_io_names(inputs=['input'], outputs=['lu', 'p'])
        validator.check_type_name("output_idx_type", output_idx_type, [mstype.int32, mstype.int64], self.name)
        self.add_prim_attr('output_idx_type', output_idx_type)


class LuSolve(Primitive):
    r"""
    Return the solution of the linear equation :math:`Ax = b` .

    Note:
        The batch dimensions of lu_pivots must match the batch dimensions of lu_data, the size of the dimension and the
        number of each dimension must be the same. For example, lu_data is (3, 3, 2, 2) lu_pivots is (3, 3, 2),
        lu_data's batch dimensions is (3, 3), lu_pivots's batch dimensions is (3, 3).

        The batch dimensions of lu_data must match the batch dimensions of x, the batch dimensions may have
        different sizes, from right to left, the corresponding dimensions must be equal. For example, lu_data
        is (3, 3, 2, 2) x is (2, 3, 3, 2, 1), lu_data's batch dimensions is (3, 3), x's batch dimensions is (2, 3, 3).

    Inputs:
        - **x** (Tensor) - The input is a tensor of size `(*, m, k)`, where * is batch dimensions, with data type
          float32, float16.
        - **lu_data** (Tensor) - The input is a tensor of size `(*, m, m)`, where * is batch dimensions, that can
          be decomposed into an upper
          triangular matrix U and a lower triangular matrix L, with data type float32, float16.
        - **lu_pivots** (Tensor) - The input is a tensor of size `(*, m)`, where * is batch dimensions, that can
          be converted to a permutation matrix P, with data type int32.

    Outputs:
        Tensor, the same data type as the x and lu_data.

    Raises:
        TypeError: If dtype of `x` or `lu_data` is not one of: float32, float16.
        TypeError: If dtype of `lu_pivots` is not: int32.
        TypeError: If `x`, `lu_data` or `lu_pivots` is not Tensor.
        TypeError: If dtype of `x` is not same as dtype of `lu_data`.
        ValueError: If the batch dimensions of lu_pivots does not match the batch dimensions of lu_data.
        ValueError: If `x` dimension less than 2, `lu_data` dimension less than 2 or `lu_pivots` dimension less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1], [3], [3]]), mindspore.float32)
        >>> lu_data = Tensor(np.array([[2, 1, 1], [0.5, 1, 1.5], [0.5, 0, 2.5]]), mindspore.float32)
        >>> lu_pivots = Tensor(np.array([2, 2, 3]), mindspore.int32)
        >>> net = ops.LuSolve()
        >>> y = net(x, lu_data, lu_pivots)
        >>> print(y)
        [[ 1.9000002]
         [-1.4000001]
         [ 0.6      ]]
    """

    @prim_attr_register
    def __init__(self):
        pass


class LuUnpack(Primitive):
    """
    Unpack the LU_data and LU_pivots from a LU factorization of a tensor.

    Args:
        unpack_data (bool, optional): A flag indicating if the LU_data should be unpacked.
            If False, then the returned L and U are None. Default: True.
        unpack_pivots (bool, optional): A flag indicating if the LU_pivots should be unpacked
            into a permutation matrix P. If False, then the returned P is None. Default: True.

    Inputs:
        - **LU_data** (Tensor) - The packed LU factorization data. A tensor of size `[*, M, N]`,
          where * is batch dimensions, with data type int8, uint8, int16, int32, int64, float16,
          float32, float64. The dims of LU_data must be equal to or greater than 2.
        - **LU_pivots** (Tensor) - The packed LU factorization pivots. A tensor of size `[*, min(M, N)]`,
          where * is batch dimensions, with data type int8, uint8, int16, int32, int64.

    Outputs:
        - **pivots** (Tensor) - The permutation matrix of LU factorization. The shape is `[*, M, M]`,
          the dtype is same as `LU_data`.
        - **L** (Tensor) - The L matrix  of LU factorization. The dtype is the same as `LU_data`.
        - **U** (Tensor) - The U matrix  of LU factorization. The dtype is the same as `LU_data`.

    Raises:
        TypeError: If the dtype of `LU_data` is not one of the following: int8, uint8, int16, int32,
                   int64, float16, float32, float64.
        TypeError: If the dtype of `LU_pivots` is not one of the following: int8, uint8, int16, int32, int64.
        ValueError: If the dimension of `LU_data` is less than 2.
        ValueError: If the dimension of `LU_pivots` is less than 1.
        ValueError: If the size of the last dimension of LU_pivots is not equal to the minimum of the sizes of
                    the last two dimensions of LU_data.
        ValueError: If the batch dimensions of LU_data's does not match LU_pivots's batch dimensions.
        ValueError: On the CPU platform, if the value of `LU_pivots` are out of range[1, LU_data.shape[-2]).
        RuntimeError: On the Ascend platform, if the value of `LU_pivots` are out of range[1, LU_data.shape[-2]).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> LU_data = Tensor(np.array([[[-0.3806, -0.4872,  0.5536],
        ...                             [-0.1287,  0.6508, -0.2396],
        ...                             [ 0.2583,  0.5239,  0.6902]],
        ...                             [[ 0.6706, -1.1782,  0.4574],
        ...                             [-0.6401, -0.4779,  0.6701],
        ...                             [ 0.1015, -0.5363,  0.6165]]]), mstype.float32)
        >>> LU_pivots = Tensor(np.array([[1, 3, 3],
        ...                              [2, 3, 3]]), mstype.int32)
        >>> lu_unpack = ops.LuUnpack()
        >>> pivots, L, U = lu_unpack(LU_data, LU_pivots)
        >>> print(pivots)
        [[[1. 0. 0.]
          [0. 0. 1.]
          [0. 1. 0.]]
        <BLANKLINE>
         [[0. 0. 1.]
          [1. 0. 0.]
          [0. 1. 0.]]]
        >>> print(L)
        [[[ 1.      0.      0.    ]
          [-0.1287  1.      0.    ]
          [ 0.2583  0.5239  1.    ]]
        <BLANKLINE>
         [[ 1.      0.      0.    ]
          [-0.6401  1.      0.    ]
          [ 0.1015 -0.5363  1.    ]]]
        >>> print(U)
        [[[-0.3806 -0.4872  0.5536]
          [ 0.      0.6508 -0.2396]
          [ 0.      0.      0.6902]]
        <BLANKLINE>
         [[ 0.6706 -1.1782  0.4574]
          [ 0.     -0.4779  0.6701]
          [ 0.      0.      0.6165]]]
    """

    @prim_attr_register
    def __init__(self, unpack_data=True, unpack_pivots=True):
        """Initialize LuUnpack"""
        validator.check_value_type("unpack_data", unpack_data, [bool], self.name)
        validator.check_value_type("unpack_pivots", unpack_pivots, [bool], self.name)


class Lgamma(Primitive):
    r"""
    Computes the natural logarithm of the gamma function on input `x`.

    .. math::
        \text{out}_{i} = \ln \Gamma(\text{input}_{i})

    Inputs:
        - **x** (Tensor) - The input tensor. The dtype can be float16, float32 or float64.

    Outputs:
        Tensor, has the same dtype as `x`.

    Raises:
        TypeError: If x is not a Tensor.
        TypeError: If dtype of input x is not one of: float16, float32, float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.5, 3.2, 8.5]), mindspore.float32)
        >>> lgamma = ops.Lgamma()
        >>> output = lgamma(x)
        >>> print(output)
        [0.5723649 0.8854049 9.549267 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Lgamma"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class Digamma(Primitive):
    r"""
    Computes the grad of the lgamma function on input.

    .. math::
        P(x) = grad(ln(gamma(x)))

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Inputs:
        - **x** (Tensor) - The input tensor. With type of float16 or float32 or float64.

    Outputs:
        Tensor, has the same dtype as `x`.

    Raises:
        TypeError: If x is not a Tensor.
        TypeError: If dtype of input x is not float16 or float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.5, 0.5, 9]).astype(np.float16))
        >>> digamma = ops.Digamma()
        >>> output = digamma(x)
        >>> print(output)
        [ 0.0365 -1.964   2.14  ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Digamma"""
        self.init_prim_io_names(inputs=['input'], outputs=['output'])


class Polygamma(Primitive):
    r"""
    Computes the :math:`a^{th}` derivative of the polygamma function on `x`.

    .. math::
        \psi^{(a)}(x) = \frac{d^{(a)}}{dx^{(a)}} \psi(x)

    Inputs:
        - **a** (Tensor) - The order of the polygamma function, types: int32, int64.
        - **x** (Tensor) - The input tensor, types: float16, float32, float64.

    Outputs:
        Tensor, has the same dtype as `x`.

    Raises:
        TypeError: If x is not a Tensor.
        TypeError: If dtype of input x is not one of: float16, float32, float64.
        TypeError: If dtype of input a is not one of: int32, int64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, -0.5]), mindspore.float32)
        >>> a = Tensor(np.array(1), mindspore.int64)
        >>> polygamma = ops.Polygamma()
        >>> output = polygamma(a, x)
        >>> print(output)
        [1.644934 8.934802]
        >>> a = Tensor(np.array(2), mindspore.int64)
        >>> output = polygamma(a, x)
        >>> print(output)
        [-2.404114  -0.8287967]
        >>> a = Tensor(np.array(3), mindspore.int64)
        >>> output = polygamma(a, x)
        >>> print(output)
        [  6.4939404 193.40909  ]
        >>> a = Tensor(np.array(4), mindspore.int64)
        >>> output = polygamma(a, x)
        >>> print(output)
        [-24.886265   -3.4742498]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Polygamma"""
        self.init_prim_io_names(inputs=['a', 'x'], outputs=['y'])


class CholeskyInverse(Primitive):
    """
    Returns the inverse of the positive definite matrix using cholesky matrix factorization.

    Refer to :func:`mindspore.ops.cholesky_inverse` for more details.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[2,0,0], [4,1,0], [-1,1,2]]), mindspore.float32)
        >>> net = ops.CholeskyInverse()
        >>> y = net(x)
        >>> print(y)
        [[ 5.8125 -2.625   0.625 ]
         [-2.625   1.25   -0.25  ]
         [ 0.625  -0.25    0.25  ]]
    """

    @prim_attr_register
    def __init__(self, upper=False):
        """Initialize CholeskyInverse"""
        validator.check_value_type("upper", upper, [bool], self.name)
        self.upper = upper


class Cross(Primitive):
    """
    Returns the cross product of vectors in dimension `dim` of x1 and x2.

    Refer to :func:`mindspore.ops.cross` for more details.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.common import dtype as mstype
        >>> import mindspore.ops as ops
        >>> cross = ops.Cross(dim = 0)
        >>> x1 = Tensor([1, 2, 3], mstype.int8)
        >>> x2 = Tensor([1, 2, 3], mstype.int8)
        >>> output = cross(x1, x2)
        >>> print(output)
        [0 0 0]
    """

    @prim_attr_register
    def __init__(self, dim=-65530):
        validator.check_value_type('dim', dim, [int], self.name)
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])


class RaggedRange(Primitive):
    """
    Returns a `RaggedTensor` containing the specified sequences of numbers.

    Args:
        Tsplits (mindspore.dtype): An mindspore.dtype from: mindspore.int32, mindspore.int64.

    Inputs:
        - **starts** (Tensor) - The starts of each range, whose type is int32, int64, float32 or float64,
          and shape is 0D or 1D.
        - **limits** (Tensor) - The limits of each range, whose type and shape should be same as input `starts`.
        - **deltas** (Tensor) - The deltas of each range, whose type and shape should be same as input `starts`,
          and each element in the tensor should not be equal to 0.

    Outputs:
        - **rt_nested_splits** (Tensor) - The nested splits of the return `RaggedTensor`,
          and type of the tensor is `Tsplits`,
          shape of the tensor is equal to shape of input `starts` plus 1.
        - **rt_dense_values**  (Tensor) - The dense values of the return `RaggedTensor`,
          and type of the tensor should be same as input `starts`.
          Let size of input `starts`, input `limits` and input `deltas` are i,

          - if type of the input `starts`, input `limits` and input `deltas`
            are int32 or int64, shape of the output `rt_dense_values` is equal to
            sum(abs(limits[i] - starts[i]) + abs(deltas[i]) - 1) / abs(deltas[i])),
          - if type of the input `starts`, input `limits` and input `deltas`
            are float32 or float64, shape of the output `rt_dense_values` is equal to
            sum(ceil(abs((limits[i] - starts[i]) / deltas[i]))).

    Raises:
        TypeError: If any input is not Tensor.
        TypeError: If the type of `starts` is not one of the following dtype: int32, int64, float32, float64.
        TypeError: If the type of `starts`, `limits` and `deltas` are not same.
        TypeError: If the type of `Tsplits` is not one of the following dtype: mstype.int32, mstype.int64.
        ValueError: If the inputs `starts`, `limits`, and `deltas` are not 0D or 1D.
        ValueError: If the input `deltas` is equal to 0.
        ValueError: If the shape of `starts`, `limits` and `deltas` are not same.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> raggedrange = ops.RaggedRange(Tsplits=mstype.int64)
        >>> starts = Tensor(np.array([2, 5, 8]).astype(np.int32))
        >>> limits = Tensor(np.array([3, 5, 12]).astype(np.int32))
        >>> deltas = Tensor(np.array([1, 1, 1]).astype(np.int32))
        >>> (rt_nested_splits, rt_dense_values) = raggedrange(starts, limits, deltas)
        >>> print(rt_nested_splits)
        [0 1 1 5]
        >>> print(rt_dense_values)
        [ 2  8  9 10 11]
    """

    @prim_attr_register
    def __init__(self, Tsplits):
        """Initialize RaggedRange."""
        self.add_prim_attr("max_length", 1000000)
        self.init_prim_io_names(inputs=['starts', 'limits', 'deltas'], outputs=['rt_nested_splits', 'rt_dense_values'])
        validator.check_value_type("Tsplits", Tsplits, [mstype.Type], self.name)
        valid_values = (mstype.int64, mstype.int32)
        validator.check_type_name("Tsplits", Tsplits, valid_values, self.name)


class Trace(Primitive):
    """
    Returns a new tensor that is the sum of the input trace.

    Note:
        Input must be matrix, and complex number is not supported at present.

    Inputs:
        - **x** (Tensor) - A matrix to be calculated. The matrix must be two dimensional.

    Outputs:
        Tensor, with the same data type as input `x`, and size equals to 1.

    Raises:
        TypeError: If `x` is not a Tensor.
        ValueError: If the dimension of `x` is not equal to 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> trace = ops.Trace()
        >>> output = trace(x)
        >>> print(output)
        15.
    """

    @prim_attr_register
    def __init__(self):
        pass


class Median(Primitive):
    """
    Computes the median of elements of input tensor in the `axis` dimension. If `global_median` is True, computes the
    median of all elements of tensor.

    .. warning::
        When attr `global_median` is True, the value of the second output tensor `indices` is meaningless.

    Args:
        global_median (bool): Whether the output tensor is the median of all input tensor elements or not.
        axis (int): The dimension need to reduce. Default: 0.
        keep_dims (bool): Whether the output tensor need to retain `axis` dimension or not. Default: False.

    Inputs:
        - **x** (Tensor) - A Tensor, whose dtype is int16, int32, int64, float32 or float64.

    Outputs:
        - **y** (Tensor) - A Tensor, Has the same dtype as the `x`. If `global_median` is true, the `y` has only one
          element. If `keep_dims` is true, the `y` has the same shape as the `x` except the shape of `y` in dimension
          `axis` is size 1. Otherwise, the `y` lacks `axis` dimension than input.
        - **indices** (Tensor) - A Tensor, Has the same shape as the `y`, but dtype is int64.

    Raises:
        TypeError: If dtype of `x` is not one of the following: int16, int32, int64, float32, double.
        TypeError: If input `x` is not a Tensor.
        TypeError: If `global_median` is not a bool.
        TypeError: If `axis` is not a int.
        TypeError: If `keep_dims` is not a bool.
        ValueError: If `axis` is not in range of [-x.dim, x.dim-1].

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> # case 1 : common median compute
        >>> from mindspore import Tensor
        >>> from mindspore.ops.operations.math_ops import Median
        >>> import numpy as np
        >>> x = Tensor(np.array([[5, 1, 2],[3, 5, 7], [1, 6, 4]]).astype(np.int64))
        >>> median = Median(global_median=False, axis=0, keep_dims=False)
        >>> y = median(x)
        >>> print(y)
        (Tensor(shape=[3], dtype=Int64, value= [3, 5, 4]), Tensor(shape=[3], dtype=Int64, value= [1, 1, 2]))
        >>> # case 2 : global median compute
        >>> from mindspore import Tensor
        >>> from mindspore.ops.operations.math_ops import Median
        >>> import numpy as np
        >>> x = Tensor(np.array([[1, 7, 6],[5, 1, 3],[9, 17, 1]]).astype(np.int32))
        >>> median = Median(global_median=True)
        >>> y = median(x)
        >>> print(y)
        (Tensor(shape=[], dtype=Int32, value= 5), Tensor(shape=[], dtype=Int64, value= 0))
    """

    @prim_attr_register
    def __init__(self, global_median=False, axis=0, keep_dims=False):
        validator.check_value_type("global_median", global_median, [bool], self.name)
        self.global_median = global_median
        if global_median is False:
            validator.check_value_type("axis", axis, [int], self.name)
            validator.check_value_type("keep_dims", keep_dims, [bool], self.name)
        self.init_prim_io_names(inputs=['x'], outputs=['y', 'indices'])


class SparseSegmentMean(Primitive):
    """
    Computes the mean along sparse segments of a Tensor.

    Refer to :func:`mindspore.ops.sparse_segment_mean` for more details.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore.ops.operations.math_ops import SparseSegmentMean
        >>> x = Tensor([[0, 1, 2], [1, 2, 3], [3, 6, 7]], dtype=mindspore.float32)
        >>> indices = Tensor([0, 1, 2], dtype=mindspore.int32)
        >>> segment_ids = Tensor([1,2,2], dtype=mindspore.int32)
        >>> sparse_segment_mean = SparseSegmentMean()
        >>> out = sparse_segment_mean(x, indices, segment_ids)
        >>> print(out)
        [[0. 0. 0.]
         [0. 1. 2.]
         [2. 4. 5.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseSegmentMean"""
        self.init_prim_io_names(inputs=['x', 'indices', 'segment_ids'], outputs=['y'])


class Zeta(Primitive):
    """
    Compute the Hurwitz zeta function ζ(x,q) of input Tensor.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    .. math::

        \\zeta \\left ( x,q \\right )=  \\textstyle \\sum_{n=0} ^ {\\infty} \\left (  q+n\\right )^{-x}

    Inputs:
        - **x** (Tensor) - A Tensor, types: float32, float64.
        - **q** (Tensor) - A Tensor, must have the same shape and type as `x`.

    Outputs:
        Tensor, has the same dtype and shape as the x.

    Raises:
        TypeError: If either of `x` and `q` is not tensor.
        TypeError: If dtype of `x` is neither float32 nor float64.
        TypeError: If dtype of `q` is neither float32 nor float64.
        ValueError: If shape of `x` is not same as the `q`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([10.]), mindspore.float32)
        >>> q = Tensor(np.array([1.]), mindspore.float32)
        >>> zeta = ops.Zeta()
        >>> z = zeta(x, q)
        >>> print(z)
        [1.0009946]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Zeta"""


class Bernoulli(Primitive):
    """
    Randomly set the elements of output to 0 or 1 with the probability of P which follows the Bernoulli distribution.

    Refer to :func:`mindspore.ops.bernoulli` for more details.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor([0.1, 0.2, 0.3], mindspore.float32)
        >>> bernoulli = Bernoulli()
        >>> output = bernoulli(input_x, Tensor([1.0]))
        >>> print(output)
        [1. 1. 1.]
        >>> input_p = Tensor([0.0, 1.0, 1.0], mindspore.float32)
        >>> output = bernoulli(input_x, input_p)
        >>> print(output)
        [0. 1. 1.]
    """

    @prim_attr_register
    def __init__(self, seed=-1, offset=0):
        """Initialize Bernoulli"""
        self.init_prim_io_names(inputs=['x', 'p'], outputs=['y'])
        validator.check_value_type("seed", seed, [int], self.name)
        if seed != -1 and seed < 0:
            raise ValueError(f"Seed must be -1 or a non-negative integer, but got {seed}.")


class TridiagonalSolve(Primitive):
    """
    Return the results of tridiagonal systems of equations.

    Solve the tridiagonal systems of equations like:AX = B.
    and only the main diagonal, superdiagonal and subdiagonal has values.
    The type of diagonals and rhs should be the same.
    The penultimate dimension of diagonals must be 3.

    Args:
        partial_pivoting (bool): decide if use the method of partial_pivoting. Default: True.

    Inputs:
        - **diagonals** [Tensor] - The input tensor A of the equation AX = B, with data type of float32,
          float64, complex64, complex128.
          The penultimate dimension of diagonals must be 3.
          Diagonals and rhs must have the same rank and the same type.
        - **rhs** [Tensor] - The input tensor B of the equation AX = B, with data type of float32,
          float64, complex64, complex128.
          The penultimate dimension of rhs should be the same to the last dimension of diagonals.
          Diagonals and rhs must have the same rank and the same type.

    Outputs:
        Tensor, has the same type and shape as the input "rhs".

    Raises:
        TypeError: If `diagonals` and "rhs" are not a float32, float64, complex64 or complex128.
        TypeError: If the args `partial_pivoting` is not bool.
        ValueError: If the last second value of the "diagonals" is not "3".
        ValueError: If the last value of the "diagonals" is not equal to the last second value of the "rhs".
        ValueError: If diagonals and rhs have different rank of shape.

    Supported Platforms:
        ``CPU``
    Examples:
        >>> diagonals = Tensor(np.array([[1.0,2.0,3.0],[2.0,3.0,4.0],[3.0,4.0,5.0]]).astype(np.float32))
        >>> rhs = Tensor(np.array([[1.0],[2.0],[3.0]]).astype(np.float32))
        >>> y = P.TridiagonalSolve()(diagonals,rhs)
        >>> print(output)
        [[ 0. ]
         [ 1. ]
         [-0.5]]
    """

    @prim_attr_register
    def __init__(self, partial_pivoting=True):
        self.init_prim_io_names(inputs=['diagonals', 'rhs'], outputs=['y'])
        self.partial_pivoting = validator.check_value_type(
            "partial_pivoting", partial_pivoting, [bool], self.name)


class Renorm(Primitive):
    """
    Renormalizes the sub-tensors along dimension `dim`, and each sub-tensor's p-norm should not exceed the
    'maxnorm'. The values of current sub-tensor don't need change if the p-norm of the sub-tensor is less than
    `maxnorm`. Otherwise the sub-tensor needs to be modified to the original value of the corresponding position
    divided by the p-norm of the substensor and then multiplied by `maxnorm`.

    Refer to :func:`mindspore.ops.renorm` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), mindspore.float32)
        >>> y = ops.Renorm(p=1, dim=0, maxnorm=5.)(x)
        >>> print(y)
        [[1.       1.        1.        ]
        [1.6666666 1.6666666 1.6666666 ]
        [1.6666667 1.6666667 1.6666667 ]]
    """

    @prim_attr_register
    def __init__(self, p, dim, maxnorm):
        """Initialize Renorm."""
        if int(p) <= 0:
            raise ValueError(f"Renorm op don't support non-positive-norm, but got{p}")
        validator.check_value_type("p", p, [int], self.name)
        validator.check_value_type("dim", dim, [int], self.name)
        validator.check_value_type("maxnorm", maxnorm, [float], self.name)
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        self.add_prim_attr("p", float(p))


class Cholesky(Primitive):
    """
    Computes the Cholesky decomposition of a symmetric positive-definite matrix `A`
    or for batches of symmetric positive-definite matrices.

    Refer to :func:`mindspore.ops.cholesky` for more details.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[1.0, 1.0], [1.0, 2.0]]), mindspore.float32)
        >>> cholesky = ops.Cholesky(upper=False)
        >>> output = cholesky(input_x)
        >>> print(output)
        [[1. 0.]
         [1. 1.]]
    """

    @prim_attr_register
    def __init__(self, upper=False):
        """Initialize Cholesky"""
        self.init_prim_io_names(inputs=['input_x'], outputs=['output'])
        validator.check_value_type('upper', upper, [bool], self.name)


class STFT(Primitive):
    """
    STFTs can be used as a way of quantifying the change of a nonstationary signal’s
    frequency and phase content over time.

    Args:
        n_fft (int): The size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): the size of window frame and STFT filter.
        normalized (bool): controls whether to return the normalized STFT results.
        onesided (bool): controls whether to return half of results to
            avoid redundancy for real inputs.
        return_complex (bool): If True, return a complex tensor. If False, return
            a real tensor with an extra last dimension for the real and imaginary components.

    Inputs:
        - **x** (Tensor) - Time sequence of stft, must be either a 1-D time tensor or a 2-D tensor.
        - **window** (Tensor) - the optional window function.

    Outputs:
        - **y** (Tensor) - A tensor containing the STFT result with shape described above.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore.ops import STFT
        >>> import numpy as np
        >>> x = ms.Tensor(np.random.rand(2,7192), ms.float32)
        >>> window = ms.Tensor(np.random.rand(64), ms.float32)
        >>> stft = STFT(64, 16, 64, False, True, True)
        >>> output = stft(x, window)
        >>> print(output.shape)
        (2, 33, 446)
    """

    @prim_attr_register
    def __init__(self, n_fft, hop_length, win_length, normalized, onesided, return_complex):
        """Initialize STFT."""
        self.init_prim_io_names(inputs=['x', 'window'], outputs=['y'])
        validator.check_value_type('n_fft', n_fft, [int], self.name)
        validator.check_value_type('hop_length', hop_length, [int], self.name)
        validator.check_value_type('win_length', win_length, [int], self.name)
        validator.check_value_type('normalized', normalized, [bool], self.name)
        validator.check_value_type('onesided', onesided, [bool], self.name)
        validator.check_value_type('return_complex', return_complex, [bool], self.name)


class CholeskySolve(Primitive):
    """
    Given its Cholesky factor `u`, solves a linear system of equations with a positive definite matrix.

    If `upper` is `True`, `u` is upper triangular and `c` is returned such that:

    .. math::
        c = (u^{T}u)^{{-1}}b

    If `upper` is `False`, `u` is lower triangular and `c` is returned such that:

    .. math::
        c = (uu^{T})^{{-1}}b

    Args:
        upper (bool, optional): Flag which indicates whether to consider the Cholesky factor
            as a lower or upper triangular matrix. Default: False.

    Inputs:
        - **x1** (Tensor) - Tensor of shape :math:`(*, N, M)`, indicating 2D or 3D matrices,
          with float32 or float64 data type.
        - **x2** (Tensor) - Tensor of shape :math:`(*, N, N)`, indicating 2D or 3D square matrices composed of
          upper or lower triangular Cholesky factor, with float32 or float64 data type.
          x1 and x2 must have the same type.

    Outputs:
        Tensor, has the same shape and data type as `x1`.

    Raises:
        TypeError: If `upper` is not a bool.
        TypeError: If dtype of `x1` and `x2` is not one of: float64, float32.
        TypeError: If `x1` is not a Tensor.
        TypeError: If `x2` is not a Tensor.
        ValueError: If `x1` and `x2` have different batch size.
        ValueError: If `x1` and `x2` have different row numbers.
        ValueError: If `x1` is not 2D or 3D matrices.
        ValueError: If `x2` is not 2D or 3D square matrices.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), mindspore.float32)
        >>> x2 = Tensor(np.array([[2, 0, 0], [4, 1, 0], [-1, 1, 2]]), mindspore.float32)
        >>> net = ops.CholeskySolve()
        >>> y = net(x1, x2)
        >>> print(y)
        [[ 5.8125 -2.625   0.625 ]
         [-2.625   1.25   -0.25  ]
         [ 0.625  -0.25    0.25  ]]
    """

    @prim_attr_register
    def __init__(self, upper=False):
        """Initialize CholeskySolve"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])
        validator.check_value_type('upper', upper, [bool], self.name)


class FFTWithSize(Primitive):
    r"""
    Fourier transform, can be adjusted by parameters to achieve FFT/IFFT/RFFT/IRFFT.

    For fft, it computes the following expression:

    .. math::
        X[\omega_1, \dots, \omega_d] =
            \sum_{n_1=0}^{N_1-1} \dots \sum_{n_d=0}^{N_d-1} x[n_1, \dots, n_d]
             e^{-j\ 2 \pi \sum_{i=0}^d \frac{\omega_i n_i}{N_i}},

    where :math:`d` = `signal_ndim` is number of dimensions for the
    signal, and :math:`N_i` is the size of signal dimension :math:`i`.

    For ifft, it computes the following expression:

    .. math::
        X[\omega_1, \dots, \omega_d] =
            \frac{1}{\prod_{i=1}^d N_i} \sum_{n_1=0}^{N_1-1} \dots \sum_{n_d=0}^{N_d-1} x[n_1, \dots, n_d]
             e^{\ j\ 2 \pi \sum_{i=0}^d \frac{\omega_i n_i}{N_i}},

    where :math:`d` = `signal_ndim` is number of dimensions for the
    signal, and :math:`N_i` is the size of signal dimension :math:`i`.

    Note:
        - FFT/IFFT requires complex64 or complex128 inputs, return complex64 or complex128 outputs.
        - RFFT requires float32 or float64 inputs, return complex64 or complex128 outputs.
        - IRFFT requires complex64 or complex128 inputs, return float32 or float64 outputs.

    Args:
        signal_ndim (int): The number of dimensions in each signal, this controls how many dimensions
            of the fourier transform are realized, can only be 1, 2 or 3.
        inverse (bool): Whether it is the inverse transformation.
        real (bool): Whether it is the real transformation.

            - "inverse:False real:False" corresponds to FFT.
            - "inverse:True real:False" corresponds to IFFT.
            - "inverse:False real:True" corresponds to RFFT.
            - "inverse:True real:True" corresponds to IRFFT.

        norm (str, optional): The normalization, optional values: ["backward", "forward", "ortho"].
            Default value: "backward".

            - "backward" has the direct transforms unscaled and the inverse transforms scaled by 1/n,
              where n is the input x's element numbers.
            - "ortho" has both direct and inverse transforms are scaled by :math:`1/\sqrt(n)`.
            - "forward" has the direct transforms scaled by 1/n and the inverse transforms unscaled.

        onesided (bool, optional): Controls whether the input is halved to avoid redundancy. Default: True.
        signal_sizes (list, optional): Size of the original signal (the signal before rfft, no batch dimension),
            only in irfft mode and set onesided=true requires the parameter, signal_sizes satisfies the following
            three rules. Default: [].

            - len(signal_sizes)==signal_ndim, the length of signal_sizes is equal to the signal_ndim of the IRFFT.
            - signal_size[-1]/2+1==x.shape[-1], the last dimension of signal_sizes divided by 2 is equal to
              the last dimension of the IRFFT input.
            - signal_sizes[:-1]==x.shape[:-1], signal_sizes has exactly the same dimensions as the input shape
              except for the last dimension.

    Inputs:
        - **x** (Tensor) - The dimension of the input tensor must be greater than or equal to signal_ndim.

    Outputs:
        A tensor containing the complex-to-complex, real-to-complex or complex-to-real Fourier transform result.

    Raises:
        TypeError: If the input type of FFT/IFFT/IRFF is not one of: complex64, complex128.
        TypeError: If the input type of RFFT is not one of: float32, float64.
        TypeError: If the input type is not Tensor.
        ValueError: If `x` dimension is less than signal_ndim.
        ValueError: If signal_ndim is greater than 3 or less than 1.
        ValueError: If norm is none of "backward", "forward" or "ortho".

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> # case FFT: signal_ndim: 1, inverse: False, real: False.
        >>> fft_in = Tensor(np.array([2, 1, 2]), mindspore.complex64)
        >>> fft_net = ops.FFTWithSize(signal_ndim=1, inverse=False, real=False)
        >>> fft_output = fft_net(fft_in)
        >>> print(fft_output)
        [5.        +0.j         0.5       +0.86602545j 0.50000006-0.8660255j ]
        >>> # case IFFT: signal_ndim: 1, inverse: True, real: False.
        >>> ifft_in = fft_output
        >>> ifft_net = ops.FFTWithSize(signal_ndim=1, inverse=True, real=False)
        >>> ifft_output = ifft_net(ifft_in)
        >>> print(ifft_output)
        [2.        -1.9868216e-08j 0.99999994+0.0000000e+00j
         1.9999999 +7.9472862e-08j]
        >>> # case RFFT2D: signal_ndim: 2, inverse: False, real: True.
        >>> rfft_in = Tensor(np.array([[2, 1, 2], [3, 1, 6]]), mindspore.float32)
        >>> rfft_net = ops.FFTWithSize(signal_ndim=2, inverse=False, real=True)
        >>> rfft_output = rfft_net(rfft_in)
        >>> print(rfft_output)
        [[ 1.5000000e+01+1.1920929e-07j -2.3841858e-07+5.1961522e+00j]
         [-5.0000000e+00-2.9802322e-08j  9.9999988e-01-3.4641016e+00j]]
        >>> # case IRFFT2D: signal_ndim: 2, inverse: True, real: True.
        >>> irfft_in = rfft_output
        >>> irfft_net = ops.FFTWithSize(signal_ndim=2, inverse=True, real=True, signal_sizes=rfft_in.shape)
        >>> irfft_output = irfft_net(irfft_in)
        >>> print(irfft_output)
        [[2.         1.         2.        ]
         [3.         0.99999994 5.9999995 ]]
    """

    @prim_attr_register
    def __init__(self, signal_ndim, inverse, real, norm="backward", onesided=True, signal_sizes=()):
        """Initialize FFTWithSize."""
        validator.check_value_type('signal_ndim', signal_ndim, [int], self.name)
        validator.check_value_type('inverse', inverse, [bool], self.name)
        validator.check_value_type('real', real, [bool], self.name)
        validator.check_value_type('norm', norm, [str], self.name)
        validator.check_value_type('onesided', onesided, [bool], self.name)
        validator.check_value_type('signal_sizes', signal_sizes, [tuple, list], self.name)


class Polar(Primitive):
    r"""
    Returns a complex tensor whose elements are Cartesian coordinates corresponding to the polar
    coordinates with absolute value and angle.

    .. math::

        y_{i} =  abs_{i} * cos(angle_{i}) + abs_{i} * sin(angle_{i}) * j

    Inputs:
        - **abs** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`N` means the batchsize of the input tensor,
          math:`*` means, any number of additional dimensions.
          Must be one of the following types: float32, float64.

        - **angle** (Tensor) - The shape of tensor is
          the same as the input tensor abs.
          Must be the same type as the input tensor abs.

    Outputs:
        Tensor, has the same shape and data type as `abs`.

    Raises:
        TypeError: If neither `abs` nor `angle` is a Tensor.
        TypeError: If the dtype of input is not one of: float32, float64.
        TypeError: If the dtypes of two inputs are not the same.
        ValueError: If `abs`'s shape is not the same as `angle`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> polar = ops.Polar()
        >>> x1 = Tensor(np.array([1, 2]), mindspore.float64)
        >>> x2 = Tensor(np.array([3, 4]), mindspore.float64)
        >>> output = polar(x1, x2)
        >>> print(output)
        [-0.9899925 +0.14112001j -1.30728724-1.51360499j]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Polar"""
        self.init_prim_io_names(inputs=['abs', 'angle'], outputs=['y'])


class NextAfter(Primitive):
    """
    Returns the next representable floating-point value after `x1` towards `x2` element-wise.

    Say there are two float32 numbers :math:`a`, :math:`b`, and let the
    representable delta of float32 datatype is :math:`eps`. If :math:`a < b`,
    then the next representable of :math:`a` towards :math:`b` is :math:`a+eps`,
    the next representable of :math:`b` towards :math:`a` is :math:`b-eps`.

    .. math::

        out_{i} =  nextafter({x1_{i}, x2_{i}})

    Inputs:
        - **x1** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Must be one of the following types: float32, float64.

        - **x2** (Tensor) - The shape of tensor is
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
          Must be one of the following types: float32, float64.

    Outputs:
        Tensor, has the same shape and data type as `x1`.

    Raises:
        TypeError: If neither `x1` nor `x2` is a Tensor.
        TypeError: If the dtype of `x1` and `x2` is not one of: float32, float64.
        TypeError: If the dtypes of `x1` and `x2` are not same.
        ValueError: If `x1`'s shape is not the same as `x2`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> nextafter = ops.NextAfter()
        >>> x1 = Tensor(np.asarray([0.0]), mindspore.float32)
        >>> x2 = Tensor(np.asarray([0.1]), mindspore.float32)
        >>> output = nextafter(x1, x2)
        >>> print(output)
        [1.e-45]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize NextAfter"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['output'])


class TrilIndices(Primitive):
    r"""
    Returns the indices of the lower triangular part of a `row` -by- `col` matrix in a Tensor.
    The Tensor has a shape :math:`(2, tril\_size)` where :math:`tril\_size` is the number of
    elements in the lower triangular matrix. The first row contains row coordinates of
    all indices and the second row contains column coordinates.
    Indices are ordered based on rows and then columns.

    The lower triangular part of the matrix is defined as the elements on and below the diagonal.

    Note:
        When running on CUDA, row * col must be less than 2^59 to prevent overflow during calculation.

    Args:
        row (int): number of rows in the 2-D matrix.
        col (int): number of columns in the 2-D matrix.
        offset (int, optional): diagonal offset from the main diagonal. Default: 0.
        dtype (:class:`mindspore.dtype`, optional): The specified type of output tensor.
            An optional data type of `mstype.int32` and `mstype.int64`. Default: `mstype.int32`.

    Outputs:
        - **y** (Tensor) - indices of the elements in lower triangular part of matrix. The type specified by `dtype`.
          The shape of output is :math:`(2, tril\_size)`, where :math:`tril\_size` is the number of elements in the
          lower triangular matrix.

    Raises:
        TypeError: If `row`, `col` or `offset` is not an int.
        TypeError: If `dtype` is neither int32 nor int64.
        ValueError: If `row` or `col` < 0.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> net = ops.TrilIndices(4, 3, -1, mstype.int64)
        >>> output = net()
        >>> print(output)
        [[1 2 2 3 3 3]
         [0 0 1 0 1 2]]
        >>> print(output.dtype)
        Int64
    """

    @prim_attr_register
    def __init__(self, row, col, offset=0, dtype=mstype.int32):
        """Initialize TrilIndices"""
        self.init_prim_io_names(inputs=[], outputs=['y'])
        validator.check_int(row, 0, Rel.GE, "row", self.name)
        validator.check_int(col, 0, Rel.GE, "col", self.name)
        validator.check_value_type("offset", offset, [int], self.name)
        valid_values = (mstype.int32, mstype.int64)
        validator.check_type_name("dtype", dtype, valid_values, self.name)


class MatrixTriangularSolve(Primitive):
    r"""
    Returns a new tensor with the solution of a linear equation system with an
    upper or lower triangular matrix.

    Note:
        Only GPU platforms now support the broadcast mechanism.

    Args:
        lower (bool, optional): If True, the innermost matrices in `matrix` is
            are lower triangular. Default: True.
        adjoint (bool, optional): If True, solve with the adjoint of `matrix`.
            Default: False.

    Inputs:
        - **matrix** (Tensor) - Tensor of shape :math:`(*, M, M)`,
          with float32, float64, complex64 and complex128 data type.
        - **rhs** (Tensor) - Tensor of shape :math:`(*, M, N)`,
          with float32, float64, complex64 and complex128 data type.

    Outputs:
        Tensor, has the shape of :math:`(*, M, N)` and the same data type as `matrix`.

    Raises:
        TypeError: If `matrix` or `rhs` is not a Tensor.
        TypeError: If `lower` or `adjoint` is not bool.
        ValueError: For GPU platform, if the batch sizes of `matrix` and `rhs` do not satisfy broadcasting rules.
            For other platforms, if the batch sizes of `matrix` and `rhs` are not equal.
        ValueError: If the inner-most 2 dimensions of `matrix` are not equal.
        ValueError: If the second-last dimensions of `matrix` and `rhs` are not equal.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> matrix_triangular_solve = ops.MatrixTriangularSolve(lower=True, adjoint=False)
        >>> a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
        >>> b = np.array([[1, 0],[2, 2],[1, 5],[0, 3]])
        >>> output = matrix_triangular_solve(Tensor(a, mindspore.float32), Tensor(b, mindspore.float32))
        >>> print(output)
        [[ 0.33333334  0.        ]
         [ 1.3333333   2.        ]
         [ 0.6666666   5.        ]
         [-2.3333333  -4.        ]]
    """

    @prim_attr_register
    def __init__(self, lower=True, adjoint=False):
        """Initialize MatrixTriangularSolve"""
        validator.check_value_type('adjoint', adjoint, [bool], self.name)
        validator.check_value_type('lower', lower, [bool], self.name)


class CompareAndBitpack(Primitive):
    """
    Compare values of `x` to `threshold` and pack resulting bits into a `uint8`.

    Each comparison returns a boolean true (if x_value > threshold) or and false otherwise.

    Given an `x` shaped `[s0, s1, ..., s_n]`, the output is a `uint8` tensor shaped `[s0, s1, ..., s_n / 8]`.

    Inputs:
        - **x** (Tensor) - Input tensor. Values to compare against `threshold` and bitpack. The data type must be
          bool, float16, float32, float64, int8, int16, int32, int64.
          Note: Currently, the innermost dimension of the tensor must be divisible by 8.
        - **threshold** (Tensor) - A 0D Tensor, whose data type is same as x.

    Outputs:
        Tensor, has the uint8 type.

    Raises:
        TypeError: If `x` or `threshold` is not a Tensor.
        TypeError: If the dtype of 'x' is not one of: bool, float16, float32, float64, int8, int16, int32, int64.
        TypeError: If `threshold`'s type is not as same 'x'.
        ValueError: If `threshold` is not a 0D Tensor.
        ValueError: If `x` is a 0D Tensor.
        ValueError: If the innermost dimension of `x`'s shape is not disvisible by 8.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32)
        >>> threshold = Tensor(6, mindspore.float32)
        >>> net = ops.CompareAndBitpack()
        >>> output = net(x, threshold)
        >>> print(output)
        [3]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CompareAndBitPack"""


class NanToNum(Primitive):
    """
    Replaces `NaN`, positive infinity, and negative infinity values in the `x` with the values
    specified by `nan`, `posinf`, and `neginf`, respectively. By default, NaN is replaced by 0,
    positive infinity is replaced by the largest finite value representable by the x dtype,
    and negative infinity is replaced by the smallest finite value representable by the x dtype.

    Args:
        nan (float): The value to replace `NaN`. Default value is 0.0.
        posinf (float): If a Number, the value to replace positive infinity values with. If None, positive
          infinity values are replaced with the greatest finite value representable by `x`'s dtype.
          Default value is None.
        neginf (float): if a Number, the value to replace negative infinity values with. If None, negative
          infinity values are replaced with the lowest finite value representable by `x`'s dtype.
          Default value is None.

    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. With float32 or float16 data type.

    Outputs:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> nan_to_num = ops.NanToNum()
        >>> x = Tensor(np.array([float('nan'), float('inf'), -float('inf'), 3.14]), mindspore.float32)
        >>> output = nan_to_num(x)
        >>> print(output)
        [ 0.0000000e+00  3.4028235e+38 -3.4028235e+38  3.1400001e+00]
    """

    @prim_attr_register
    def __init__(self, nan=0.0, posinf=None, neginf=None):
        """Initialize NanToNum"""
        if nan is not None:
            validator.check_value_type("nan", nan, [float], self.name)
        else:
            self.add_prim_attr("nan_none", True)
        if posinf is not None:
            validator.check_value_type("posinf", posinf, [float], self.name)
        else:
            self.add_prim_attr("posinf_none", True)
        if neginf is not None:
            validator.check_value_type("neginf", neginf, [float], self.name)
        else:
            self.add_prim_attr("neginf_none", True)


class Orgqr(Primitive):
    r"""
    Computes the first :math:`N` columns of a product of Householder matrices.

    Refer to :func:`mindspore.ops.orgqr` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[-114.6, 10.9, 1.1], [-0.304, 38.07, 69.38], [-0.45, -0.17, 62.]]), mindspore.float32)
        >>> tau = Tensor(np.array([1.55, 1.94, 0.0]), mindspore.float32)
        >>> net = ops.Orgqr()
        >>> y = net(x, tau)
        >>> print(y)
        [[-0.54999995 -0.2128925   0.8137956 ]
         [ 0.47119996 -0.8752807   0.08240613]
         [ 0.69749993  0.42560163  0.57772595]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Orgqr"""
        self.init_prim_io_names(inputs=['x', 'tau'], outputs=['y'])


class TriuIndices(Primitive):
    r"""
    Returns the indices of the upper triangular part of a `row` -by- `col` matrix in a Tensor.
    The Tensor has a shape :math:`(2, tril\_size)` where :math:`tril\_size` is the number of
    elements in the upper triangular matrix. The first row contains row coordinates of
    all indices and the second row contains column coordinates.
    Indices are ordered based on rows and then columns.

    The upper triangular part of the matrix is defined as the elements on and above the diagonal.

    Note:
        When running on CUDA, row * col must be less than 2^59 to prevent overflow during calculation.

    Args:
        row (int): number of rows in the 2-D matrix.
        col (int): number of columns in the 2-D matrix.
        offset (int, optional): diagonal offset from the main diagonal. Default: 0.
        dtype (:class:`mindspore.dtype`, optional): The specified type of output tensor.
            An optional data type of `mstype.int32` and `mstype.int64`. Default: `mstype.int32`.

    Outputs:
        - **y** (Tensor) - indices of the elements in lower triangular part of matrix. The type specified by `dtype`.
          The shape of output is :math:`(2, tril\_size)`, where :math:`tril\_size` is the number of elements in the
          lower triangular matrix.

    Raises:
        TypeError: If `row`, `col` or `offset` is not an int.
        TypeError: If `dtype` is neither int32 nor int64.
        ValueError: If `row` or `col` < 0.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> net = ops.TriuIndices(5, 4, 2, mstype.int64)
        >>> output = net()
        >>> print(output)
        [[0 0 1]
         [2 3 3]]
        >>> print(output.dtype)
        Int64
    """

    @prim_attr_register
    def __init__(self, row, col, offset=0, dtype=mstype.int32):
        """Initialize TriuIndices"""
        self.init_prim_io_names(inputs=[], outputs=['y'])
        validator.check_int(row, 0, Rel.GE, "row", self.name)
        validator.check_int(col, 0, Rel.GE, "col", self.name)
        validator.check_value_type("offset", offset, [int], self.name)
        valid_values = (mstype.int32, mstype.int64)
        validator.check_type_name("dtype", dtype, valid_values, self.name)


class Fmin(Primitive):
    """
    Computes the minimum of input tensors element-wise.

    Refer to :func:`mindspore.ops.fmin` for more detail.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> x1 = Tensor(np.array([1.0, 5.0, 3.0]), mstype.float32)
        >>> x2 = Tensor(np.array([4.0, 2.0, 6.0]), mstype.float32)
        >>> fmin = ops.Fmin()
        >>> output = fmin(x1, x2)
        >>> print(output)
        [1. 2. 3.]
    """

    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        """Initialize Fmin"""
        self.add_prim_attr('ignore_nan', True)
        self.init_prim_io_names(inputs=['x1, x2'], outputs=['y'])


class Fmax(Primitive):
    """
    Computes the maximum of input tensors element-wise.

    Refer to :func:`mindspore.ops.fmax` for more detail.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> x1 = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.float32)
        >>> x2 = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float32)
        >>> fmax = ops.Fmax()
        >>> output = fmax(x1, x2)
        >>> print(output)
        [4. 5. 6.]
    """

    __mindspore_signature__ = (sig.sig_dtype.T, sig.sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        """Initialize Fmax"""
        self.add_prim_attr('ignore_nan', True)
        self.init_prim_io_names(inputs=['x1, x2'], outputs=['y'])


class Eig(Primitive):
    """
    Computes the eigenvalues and eigenvectors of a square matrix(batch square matrices).

    Args:
        compute_v (bool, optional): If `True`, compute both eigenvalues and eigenvectors;
            If `False`, just eigenvalues will be computed. Default: False.
    Inputs:
        - **x** (Tensor) - Square matrices of shape :math:`(*, N, N)`,
          with float32, float64, complex64 or complex128 data type.

    Outputs:
        - **eigen_values** (Tensor) - Shape :math:`(*, N)`. Each inner most vector represents eigenvalues of
          the corresponding matrix. The eigenvalues may not have an order.
        - **eigen_vectors** (Tensor) - If `compute_v` is `False`, it’s an empty tensor. Otherwise, this tensor
          has shape :math:`(*, N, N)`, whose columns represent normalized (unit length) eigenvectors of corresponding
          eigenvalues.

    Raises:
        TypeError: If `compute_v` is not a bool.
        TypeError: If dtype of `x` is not one of: float64, float32, complex64 or complex128.
        TypeError: If `x` is not a Tensor.
        ValueError: If `x` is not a square(batch squares).

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[1.0, 0.0], [0.0, 2.0]]), mindspore.float32)
        >>> eig = ops.Eig(compute_v=True)
        >>> u, v = eig(input_x)
        >>> print(u)
        [1.+0.j 2.+0.j]
        >>> print(v)
        [[1.+0.j 0.+0.j]
         [0.+0.j 1.+0.j]]
    """

    @prim_attr_register
    def __init__(self, compute_v=False):
        """Initialize Eig"""
        self.init_prim_io_names(inputs=['x'], outputs=['eigen_values', 'eigen_vectors'])
        validator.check_value_type('compute_v', compute_v, [bool], self.name)


class SelfAdjointEig(Primitive):
    r"""
    Computes the eigenvalues and (optionally) eigenvectors of each inner matrix in input
    such that input[..., :, :] = v[..., :, :] * diag(e[..., :]).
    The eigenvalues are sorted in non-decreasing order.

    Args:
         compute_v(bool): If `True` then eigenvectors will be computed and returned in v;
              If `False`, only the eigenvalues will be computed. Default: True.

    Inputs:
         - **x** (Tensor) - Must be one of the following types:
           float64, float32, complex64, complex128. Tensor input of shape :math:`[...,N, N]`.

    Outputs:
         - **eigen_value** (Tensor) - Has the same type as input, the shape is :math:`[...,N]`.
         - **eigen_vector** (Tensor) - If `compute_v` is `False`, it’s an empty tensor.
           Otherwise, it has the same type and shape as input, the shape is the same as the input.

    Raises:
         TypeError: If `compute_v` is not a bool.
         TypeError: If dtype of `x` is not one of: float64, float32, complex64 or complex128.
         TypeError: If `x` is not a Tensor.
         ValueError: If `x` is not a square(batch squares).

    Supported Platforms:
         ``CPU``

    Examples:
           >>> from mindspore.ops.operations.math_ops import SelfAdjointEig
           >>> input_x = Tensor(np.array([[1.0, 0.0], [0.0, 2.0]]).astype(np.float32))
           >>> SelfAdjointEig = SelfAdjointEig()
           >>> eigen_value, eigen_vector = SelfAdjointEig(input_x)
           >>> print(eigen_value)
           [1.  2.]
           >>> print(eigen_vector)
           [[1.  0.]
            [0.  1.]]
    """

    @prim_attr_register
    def __init__(self, compute_v=True):
        """Initialize SelfAdjointEig."""
        self.init_prim_io_names(inputs=['x'], outputs=['eigen_value', 'eigen_vector'])
        validator.check_value_type("compute_v", compute_v, [bool], self.name)


class Qr(Primitive):
    """
    Returns the QR decomposition of one or more matrices. If `full_matrices` is true, compute full-sized q and r,
    If False (the default), compute the P columns of q where P is minimum of the 2 innermost dimensions of x.

    Args:
        full_matrices (bool, optional): Whether compute full-sized QR decomposition. Default: False.

    Inputs:
        - **x** (Tensor) - A matrix to be calculated. The matrix must be at least two dimensions.
          types: float16, float32, float64, complex64, complex128.
          Define the shape of x as (..., m, n), p as the minimum values of m and n.

    Outputs:
        - **q** (Tensor) - The orthonormal matrices of x.
          If `full_matrices` is true, the shape is :math:`(m, m)`, else the shape is :math:`(m, p)`.
          The dtype of `q` is same as `x`.
        - **r** (Tensor) - The upper triangular matrices of x.
          If `full_matrices` is true, the shape is :math:`(m, n)`, else the shape is :math:`(p, n)`.
          The dtype of `r` is same as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `full_matrices` is not a bool.
        ValueError: If the dimension of `x` is less than 2.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> qr_op = ops.Qr(full_matrices=False)
        >>> x = Tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]], mstype.float32)
        >>> q, r = qr_op(x)
        >>> print(q)
        [[-0.8571428   0.39428577  0.3314286 ]
        [-0.42857143 -0.90285724 -0.03428572]
        [ 0.2857143  -0.17142859  0.94285715]]
        >>> print(r)
        [[ -14.        -21.000008   13.999999]
        [   0.       -175.         70.000015]
        [   0.          0.        -34.999996]]
    """

    @prim_attr_register
    def __init__(self, full_matrices=False):
        """Initialize Qr"""
        validator.check_value_type('full_matrices', full_matrices, [bool], self.name)


class Cauchy(Primitive):
    r"""
    Create a tensor of shape `size` with random numbers drawn from Cauchy distribution.

    .. math::
        f(x)= \frac{1}{\pi} \frac{\sigma}{(x-median)^2 +\sigma^2}

    Args:
        size (list[int]): The size of tensor.
        sigma (float, optional): the location parameter, specifying the location
            of the peak of the distribution. Default: 1.0.
        median (float, optional): the scale parameter which specifies the half-width
            at half-maximum. Default: 0.0.

    Outputs:
        Tensor with cauchy distribution data. Tensor shape is size, and data type is float32.

    Raises:
        TypeError: If `sigma` is not a float.
        TypeError: If `median` is not a float.
        TypeError: If `size` is not a list.
        ValueError: If `size` list is empty.
        ValueError: If data of `size` is not a positive integer.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> size = [1]
        >>> net = ops.Cauchy(size)
        >>> y = net()
        >>> print(y)
        [0.03128606]
    """

    @prim_attr_register
    def __init__(self, size, median=0.0, sigma=1.0):
        validator.check_value_type('median', median, [float], self.name)
        validator.check_value_type('sigma', sigma, [float], self.name)
        validator.check_value_type('size', size, (list), self.name)
        for index, size_ in enumerate(size):
            validator.check_positive_int(size_, 'size[%d]' % index, self.name)


class Ormqr(Primitive):
    r"""
    Computes the matrix-matrix multiplication of a product of Householder matrices with a general matrix.
    Multiplies a(m, n) matrix C (given by other) with a matrix Q, where Q is represented using Householder
    reflectors (x, tau), which is the output of geqrf().

    Args:
        left (bool, optional): controls the order of multiplication. If true, compute op(Q)*C.
            If false, compute C*op(Q). Default: True.
        transpose(bool, optional): controls whether the matrix Q is conjugate transposed or not.Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape: (*, mn, k) where mn equals to m or n depending on the the args of `left`,
          and `*` is zero or more batch dimensions.
        - **tau** (Tensor) - Tensor of shape (*, min(mn, k)) where `*` is zero or more batch dimensions,
          and its type is the same as `x`.
        - **other** (Tensor) - Tensor of shape (*, m, n) where `*` is zero or more batch dimensions,
          and its type is the same as `x`.

    Outputs:
        - **y** (Tensor) - the output Tensor, has the same shape and data type as `other`.

    Raises:
        TypeError: If `x` or `tau` or `other` is not Tensor.
        TypeError: If dtype of `x` or `tau` or `other` is not one of: float64, float32, complex64, complex128.
        ValueError: If `x` or `other` is less than 2D.
        ValueError: If rank(x) - rank(tau) != 1.
        ValueError: If tau.shape[:-2] != x.shape[:-2]
        ValueError: If other.shape[:-2] != x.shape[:-2]
        ValueError: If left == true, other.shape[-2] < tau.shape[-1].
        ValueError: If left == true, other.shape[-2] != x.shape[-2].
        ValueError: If left == false, other.shape[-1] < tau.shape[-1].
        ValueError: If left == false, other.shape[-1] != x.shape[-2].

    Supported Platforms:
        ``GPU``

    Examples:
        >>> x = Tensor(np.array([[-114.6, 10.9, 1.1], [-0.304, 38.07, 69.38], [-0.45, -0.17, 62]]), mindspore.float32)
        >>> tau = Tensor(np.array([1.55, 1.94, 3.0]), mindspore.float32)
        >>> other = Tensor(np.array([[-114.6, 10.9, 1.1],
                                     [-0.304, 38.07, 69.38],
                                     [-0.45, -0.17, 62]]), mindspore.float32)
        >>> net = ops.Ormqr()
        >>> y = net(x, tau, other)
        >>> print(y)
        [[  63.82713   -13.823125 -116.28614 ]
         [ -53.659264  -28.157839  -70.42702 ]
         [ -79.54292    24.00183   -41.34253 ]]
    """

    @prim_attr_register
    def __init__(self, left=True, transpose=False):
        """Initialize Ormqr"""
        self.init_prim_io_names(inputs=['x', 'tau', 'other'], outputs=['y'])
        self.left = validator.check_value_type('left', left, [bool], self.name)
        self.transpose = validator.check_value_type('transpose', transpose, [bool], self.name)
        self.add_prim_attr('left', self.left)
        self.add_prim_attr('transpose', self.transpose)


class Roll(Primitive):
    """
    Rolls the elements of a tensor along an axis.

    Refer to :func:`mindspore.ops.roll` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> input_x = Tensor(np.array([0, 1, 2, 3, 4]).astype(np.float32))
        >>> op = ops.Roll(shift=2, axis=0)
        >>> output = op(input_x)
        >>> print(output)
        [3. 4. 0. 1. 2.]
        >>> input_x = Tensor(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).astype(np.float32))
        >>> op = ops.Roll(shift=-1, axis=0)
        >>> output = op(input_x)
        >>> print(output)
        [[5. 6. 7. 8. 9.]
         [0. 1. 2. 3. 4.]]
    """

    @prim_attr_register
    def __init__(self, shift, axis):
        """Initialize Roll"""
        if context.get_context("device_target") == "GPU":
            validator.check_value_type("shift", shift, [int, tuple, list], self.name)
            if not isinstance(shift, (list, tuple)):
                self.add_prim_attr('shift', [shift])
            validator.check_value_type("axis", axis, [int, tuple, list], self.name)
            if not isinstance(axis, (list, tuple)):
                self.add_prim_attr('axis', [axis])
        else:
            if isinstance(shift, (tuple, list)) and isinstance(axis, (tuple, list)):
                validator.check_equal_int(len(shift), 1, "shift size", self.name)
                validator.check_equal_int(len(axis), 1, "shift size", self.name)
                validator.check_equal_int(axis[0], 0, "axis", self.name)
            elif isinstance(shift, int) and isinstance(axis, int):
                validator.check_equal_int(axis, 0, "axis", self.name)
        self.init_prim_io_names(inputs=['input_x'], outputs=['output'])
