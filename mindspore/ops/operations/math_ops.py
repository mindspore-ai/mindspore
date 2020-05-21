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

"""Operators for math."""

import numpy as np
from ... import context
from ..._c_expression import signature_rw as sig_rw
from ..._c_expression import signature_kind as sig_kind
from ..._c_expression import signature_dtype as sig_dtype
from ..._checkparam import Validator as validator
from ..._checkparam import Rel
from ...common import dtype as mstype
from ...common.tensor import Tensor
from .._utils import get_broadcast_shape
from ..primitive import PrimitiveWithInfer, prim_attr_register, _run_op


def _infer_shape_reduce(x, axis, keep_dims, prim_name):
    """Common infer for reduce operator"""

    def reduce_one_axis(one_axis):
        validator.check_int_range('axis', one_axis, -dim, dim, Rel.INC_LEFT, prim_name)
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

    __mindspore_signature__ = (sig_dtype.T, sig_dtype.T)

    @prim_attr_register
    def __init__(self):
        """init _MathBinaryOp"""
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
        validator.check_tensor_type_same(args_type, valid_dtype, prim_name)
        return x_dtype

    def infer_dtype(self, x_dtype, y_dtype):
        return _MathBinaryOp.do_infer_dtype(x_dtype, y_dtype, mstype.number_type, self.name)


class TensorAdd(_MathBinaryOp):
    """
    Adds two input tensors element-wise.

    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be same.
    When the inputs are one tensor and one scalar, the scalar cannot be a parameter, only can be a constant,
    and the type of the scalar is the same as the data type of the tensor.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The first input is a tensor whose data type is number or a number.
        - **input_y** (Union[Tensor, Number]) - The second input is a tensor whose data type is same as 'input_x' or
          a number.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is same as 'input_x'.

    Examples:
        >>> add = P.TensorAdd()
        >>> input_x = Tensor(np.array([1,2,3]).astype(np.float32))
        >>> input_y = Tensor(np.array([4,5,6]).astype(np.float32))
        >>> add(input_x, input_y)
        [5,7,9]
    """


class AssignAdd(PrimitiveWithInfer):
    """
    Updates a `Parameter` by adding a value to it.

    Inputs:
        - **variable** (Parameter) - The `Parameter`.
        - **value** (Union[numbers.Number, Tensor]) - The value to be added to the `variable`.
          It should have the same shape as `variable` if it is a Tensor.

    Examples:
        >>> class Net(Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.AssignAdd = P.AssignAdd()
        >>>         self.variable = mindspore.Parameter(initializer(1, [1], mindspore.int64), name="global_step")
        >>>
        >>>     def construct(self, x):
        >>>         self.AssignAdd(self.variable, x)
        >>>         return self.variable
        >>>
        >>> net = Net()
        >>> value = Tensor(np.ones([1]).astype(np.int64)*100)
        >>> net(value)
    """
    __mindspore_signature__ = (
        ('variable', sig_rw.RW_WRITE, sig_kind.KIND_POSITIONAL_KEYWORD),
        ('value', sig_rw.RW_READ, sig_kind.KIND_POSITIONAL_KEYWORD)
    )

    @prim_attr_register
    def __init__(self):
        """init AssignAdd"""
        self.init_prim_io_names(inputs=['ref', 'value'], outputs=['output'])

    def infer_shape(self, variable, value):
        return value

    def infer_dtype(self, variable, value):
        args = {"value": value}
        validator.check_scalar_or_tensor_type_same(args, mstype.number_type, self.name)
        return value


class AssignSub(PrimitiveWithInfer):
    """
    Updates a `Parameter` by subtracting a value from it.

    Inputs:
        - **variable** (Parameter) - The `Parameter`.
        - **value** (Union[numbers.Number, Tensor]) - The value to be subtracted from the `variable`.
          It should have the same shape as `variable` if it is a Tensor.

    Examples:
        >>> class Net(Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.AssignSub = P.AssignSub()
        >>>         self.variable = mindspore.Parameter(initializer(1, [1], mindspore.int64), name="global_step")
        >>>
        >>>     def construct(self, x):
        >>>         self.AssignSub(self.variable, x)
        >>>         return self.variable
        >>>
        >>> net = Net()
        >>> value = Tensor(np.ones([1]).astype(np.int64)*100)
        >>> net(value)
    """

    __mindspore_signature__ = (
        ('variable', sig_rw.RW_WRITE, sig_kind.KIND_POSITIONAL_KEYWORD),
        ('value', sig_rw.RW_READ, sig_kind.KIND_POSITIONAL_KEYWORD)
    )

    @prim_attr_register
    def __init__(self):
        """init AssignSub"""

    def infer_shape(self, variable, value):
        return value

    def infer_dtype(self, variable, value):
        args = {"value": value}
        validator.check_scalar_or_tensor_type_same(args, mstype.number_type, self.name)
        return value


class _Reduce(PrimitiveWithInfer):
    """
    Definition of base class of reduction class operators.

    Args:
         keep_dims (bool): If True, keep these reduced dimensions and the length is 1.
                           If False, don't keep these dimensions.
    """

    __mindspore_signature__ = (
        ('input_x', sig_rw.RW_READ, sig_kind.KIND_POSITIONAL_KEYWORD),
        ('axis', sig_rw.RW_READ, sig_kind.KIND_POSITIONAL_KEYWORD, ()),
    )

    @prim_attr_register
    def __init__(self, keep_dims=False):
        """init Reduce"""
        validator.check_value_type('keep_dims', keep_dims, [bool], self.name)
        self.init_prim_io_names(inputs=['input_x', 'axis'], outputs=['y'])

    def __call__(self, x, axis=()):
        args = [x, axis]
        output = _run_op(self, self.name, args)
        return output

    def do_infer(self, input_x, axis, valid_dtype=mstype.number_type):
        axis_v = axis['value']
        input_shp = input_x['shape']
        args = {'input_x': input_x['dtype']}
        validator.check_tensor_type_same(args, valid_dtype, self.name)

        input_shp = _infer_shape_reduce(input_shp, axis_v, self.keep_dims, self.name)
        return {'shape': input_shp,
                'dtype': input_x['dtype'],
                'value': None}

    def __infer__(self, input_x, axis):
        return self.do_infer(input_x, axis)


class ReduceMean(_Reduce):
    """
     Reduce a dimension of a tensor by averaging all elements in the dimension.

     The dtype of the tensor to be reduced is number.

    Args:
        keep_dims (bool): If True, keep these reduced dimensions and the length is 1.
                          If False, don't keep these dimensions. Default : False.

    Inputs:
        - **input_x** (Tensor[Number]) - The input tensor.
        - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed.

    Outputs:
        Tensor, has the same dtype as the 'input_x'.

        - If axis is (), and keep_dims is false,
          the output is a 0-D tensor representing the sum of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is false,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is false,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Examples:
        >>> input_x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = P.ReduceMean(keep_dims=True)
        >>> output = op(input_x, 1)
    """


class ReduceSum(_Reduce):
    """
    Reduce a dimension of a tensor by summing all elements in the dimension.

    The dtype of the tensor to be reduced is number.

    Args:
        keep_dims (bool): If True, keep these reduced dimensions and the length is 1.
                          If False, don't keep these dimensions. Default : False.

    Inputs:
         - **input_x** (Tensor[Number]) - The input tensor.
         - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
           Only constant value is allowed.

    Outputs:
        Tensor, has the same dtype as the 'input_x'.

        - If axis is (), and keep_dims is false,
          the output is a 0-D tensor representing the sum of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is false,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is false,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Examples:
        >>> input_x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = P.ReduceSum(keep_dims=True)
        >>> output = op(input_x, 1)
    """


class ReduceAll(_Reduce):
    """
    Reduce a dimension of a tensor by the "logical and" of all elements in the dimension.

    The dtype of the tensor to be reduced is bool.

    Args:
       keep_dims (bool): If True, keep these reduced dimensions and the length is 1.
                         If False, don't keep these dimensions.
                         Default : False, don't keep these reduced dimensions.

    Inputs:
        - **input_x** (Tensor[bool]) - The input tensor.
        - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed.

    Outputs:
        Tensor, the dtype is bool.

        - If axis is (), and keep_dims is false,
          the output is a 0-D tensor representing the "logical and" of of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is false,
          and keep_dims is false, the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is false,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Examples:
        >>> input_x = Tensor(np.array([[True, False], [True, True]]))
        >>> op = P.ReduceAll(keep_dims=True)
        >>> output = op(input_x, 1)
    """

    def __infer__(self, input_x, axis):
        return self.do_infer(input_x, axis, (mstype.bool_,))


class ReduceMax(_Reduce):
    """
    Reduce a dimension of a tensor by the maximum value in this dimension.

    The dtype of the tensor to be reduced is number.

    Args:
        keep_dims (bool): If True, keep these reduced dimensions and the length is 1.
                          If False, don't keep these dimensions.
                          Default : False, don't keep these reduced dimensions.

    Inputs:
         - **input_x** (Tensor[Number]) - The input tensor.
         - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
           Only constant value is allowed.

    Outputs:
        Tensor, has the same dtype as the 'input_x'.

        - If axis is (), and keep_dims is false,
          the output is a 0-D tensor representing the maximum of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is false,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is false,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Examples:
        >>> input_x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = P.ReduceMax(keep_dims=True)
        >>> output = op(input_x, 1)
    """


class ReduceMin(_Reduce):
    """
    Reduce a dimension of a tensor by the minimum value in the dimension.

    The dtype of the tensor to be reduced is number.

    Args:
        keep_dims (bool): If True, keep these reduced dimensions and the length is 1.
                          If False, don't keep these dimensions.
                          Default : False, don't keep these reduced dimensions.

    Inputs:
        - **input_x** (Tensor[Number]) - The input tensor.
        - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed.

    Outputs:
        Tensor, has the same dtype as the 'input_x'.

        - If axis is (), and keep_dims is false,
          the output is a 0-D tensor representing the minimum of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is false,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is false,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Examples:
        >>> input_x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = P.ReduceMin(keep_dims=True)
        >>> output = op(input_x, 1)
    """


class ReduceProd(_Reduce):
    """
    Reduce a dimension of a tensor by multiplying all elements in the dimension.

    The dtype of the tensor to be reduced is number.

    Args:
        keep_dims (bool): If True, keep these reduced dimensions and the length is 1.
                          If False, don't keep these dimensions.
                          Default : False, don't keep these reduced dimensions.

    Inputs:
         - **input_x** (Tensor[Number]) - The input tensor.
         - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.

    Outputs:
        Tensor, has the same dtype as the 'input_x'.

        - If axis is (), and keep_dims is false,
          the output is a 0-D tensor representing the product of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is false,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is false,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Examples:
        >>> input_x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = P.ReduceProd(keep_dims=True)
        >>> output = op(input_x, 1)
    """


class CumProd(PrimitiveWithInfer):
    """
    Compute the cumulative product of the tensor x along axis.

    Args:
        exclusive (bool): If True, perform exclusive cumulative product. Default: False.
        reverse (bool): If True, reverse the result along axis. Default: False

    Inputs:
         - **input_x** (Tensor[Number]) - The input tensor.
         - **axis** (int) - The dimensions to compute the cumulative product.

    Outputs:
        Tensor, has the same shape and dtype as the 'input_x'.

    Examples:
        >>> input_x = Tensor(np.array([a, b, c]).astype(np.float32))
        >>> op0 = P.CumProd()
        >>> output = op0(input_x, 0) # output=[a, a * b, a * b * c]
        >>> op1 = P.CumProd(exclusive=True)
        >>> output = op1(input_x, 0) # output=[1, a, a * b]
        >>> op2 = P.CumProd(reverse=True)
        >>> output = op2(input_x, 0) # output=[a * b * c, b * c, c]
        >>> op3 = P.CumProd(exclusive=True, reverse=True)
        >>> output = op3(input_x, 0) # output=[b * c, c, 1]
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
        validator.check_tensor_type_same({'x': x_type}, mstype.number_type, cls_name)
        validator.check_subclass("axis", axis_type, mstype.int_, cls_name)
        return x_type


class MatMul(PrimitiveWithInfer):
    """
    Multiplies matrix `a` by matrix `b`.

    The rank of input tensors must be `2`.

    Args:
        transpose_a (bool): If True, `a` is transposed before multiplication. Default: False.
        transpose_b (bool): If True, `b` is transposed before multiplication. Default: False.

    Inputs:
        - **input_x** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(N, C)`. If
          `transpose_a` is True, its shape should be :math:`(N, C)` after transposing.
        - **input_y** (Tensor) - The second tensor to be multiplied. The shape of the tensor is :math:`(C, M)`. If
          `transpose_b` is True, its shape should be :math:`(C, M)` after transpose.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, M)`.

    Examples:
        >>> input_x = Tensor(np.ones(shape=[1, 3]), mindspore.float32)
        >>> input_y = Tensor(np.ones(shape=[3, 4]), mindspore.float32)
        >>> matmul = P.MatMul()
        >>> output = matmul(input_x, input_y)
    """

    @prim_attr_register
    def __init__(self, transpose_a=False, transpose_b=False):
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['output'])
        self.__setattr_flag__ = True
        cls_name = self.name
        validator.check_value_type("transpose_a", transpose_a, [bool], cls_name)
        validator.check_value_type("transpose_b", transpose_b, [bool], cls_name)

    def check_shape_size(self, x, y):
        if len(x) != 2 or len(y) != 2:
            raise ValueError('MatMul input x, y should be the same dimension size and should be '
                             + f'equal to 2, while x size = {len(x)}, y size= {len(y)}')

    def infer_shape(self, x, y):
        self.check_shape_size(x, y)
        cls_name = self.name
        # expected dimension of x, y, x:[...,a,b] y:[..., c,d], the dim size should be the same except the last two
        for i in range(len(x) - 2):
            if x[i] != y[i]:
                raise ValueError(f'For \'{cls_name}\' shape in dim[{i}] not the same, while x is {x[i]}, y is {y[i]}')

        # validate whether last two dims satifing matrix multiply
        x_last = x[-2:]
        y_last = y[-2:]

        x_col = x_last[not self.transpose_a]  # x_col = x_last[1] if (not transpose_a) else x_last[0]
        y_row = y_last[self.transpose_b]  # y_row = y_last[0] if (not transpose_b) else y_last[1]
        if x_col != y_row:
            raise ValueError(f'For \'{cls_name}\' evaluator shapes of inputs can not do this operator,'
                             + f' got {x_col} and {y_row}, with x shape {x}(transpose_a={self.transpose_a})'
                             + f', y shape {y}(transpose_b={self.transpose_b}).')
        # set attribute
        self.add_prim_attr('transpose_x1', self.transpose_a)
        self.add_prim_attr('transpose_x2', self.transpose_b)

        ret_dims = x[: -2] + [x_last[self.transpose_a], y_last[not self.transpose_b]]
        return ret_dims

    def infer_dtype(self, x, y):
        args = {"x": x, "y": y}
        validator.check_tensor_type_same(args, mstype.float_type + mstype.int_type, self.name)
        return x


class BatchMatMul(MatMul):
    """
    Computes matrix multiplication between two tensors by batch

    `result[..., :, :] = tensor(a[..., :, :]) * tensor(b[..., :, :])`.

    The two input tensors must have same rank and the rank must be `3` at least.

    Args:
        transpose_a (bool): If True, `a` is transposed on the last two dimensions before multiplication.
            Default: False.
        transpose_b (bool): If True, `b` is transposed on the last two dimensions before multiplication.
            Default: False.

    Inputs:
        - **input_x** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(*B, N, C)`,
          where :math:`*B` represents the batch size which can be multidimensional, :math:`N` and :math:`C` are the
          size of the last two dimensions. If `transpose_a` is True, its shape should be :math:`(*B, C, N)`.
        - **input_y** (Tensor) - The second tensor to be multiplied. The shape of the tensor is :math:`(*B, C, M)`. If
          `transpose_b` is True, its shape should be :math:`(*B, M, C)`.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(*B, N, M)`.

    Examples:
        >>> input_x = Tensor(np.ones(shape=[2, 4, 1, 3]), mindspore.float32)
        >>> input_y = Tensor(np.ones(shape=[2, 4, 3, 4]), mindspore.float32)
        >>> batmatmul = P.BatchMatMul()
        >>> output = batmatmul(input_x, input_y)
        >>>
        >>> input_x = Tensor(np.ones(shape=[2, 4, 3, 1]), mindspore.float32)
        >>> input_y = Tensor(np.ones(shape=[2, 4, 3, 4]), mindspore.float32)
        >>> batmatmul = P.BatchMatMul(transpose_a=True)
        >>> output = batmatmul(input_x, input_y)
    """

    @prim_attr_register
    def __init__(self, transpose_a=False, transpose_b=False):
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['output'])
        self.__setattr_flag__ = True
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

    Args:
        exclusive (bool): If True, perform exclusive mode. Default: False.
        reverse (bool): If True, perform inverse cumulative sum. Default: False.

    Inputs:
        - **input** (Tensor) - The input tensor to accumulate.
        - **axis**  (int) - The axis to accumulate the tensor's value.

    Outputs:
        Tensor, the shape of the output tensor is consistent with the input tensor's.

    Examples:
        >>> input = Tensor(np.array([[3, 4, 6, 10],[1, 6, 7, 9],[4, 3, 8, 7],[1, 3, 7, 9]]).astype(np.float32))
        >>> cumsum = P.CumSum()
        >>> output = cumsum(input, 1)
        [[ 3.  7. 13. 23.]
         [ 1.  7. 14. 23.]
         [ 4.  7. 15. 22.]
         [ 1.  4. 11. 20.]]
    """

    @prim_attr_register
    def __init__(self, exclusive=False, reverse=False):
        """init cumsum"""
        cls_name = self.name
        validator.check_value_type('exclusive', exclusive, [bool], cls_name)
        validator.check_value_type('reverse', reverse, [bool], cls_name)
        self.init_prim_io_names(inputs=['x', 'axis'], outputs=['y'])

    def __infer__(self, x, axis):
        cls_name = self.name
        x_shp = x['shape']
        validator.check_value_type('axis', axis['value'], [int], cls_name)
        valid_types = [mstype.uint8, mstype.int8, mstype.int32, mstype.float16, mstype.float32]
        validator.check_tensor_type_same({'x': x['dtype']}, valid_types, cls_name)
        return {'shape': x_shp,
                'dtype': x['dtype'],
                'value': None}


class AddN(PrimitiveWithInfer):
    """
    Computes addition of all input tensors element-wise.

    All input tensors should have the same shape.

    Inputs:
        - **input_x** (Union(tuple[Tensor], list[Tensor])) - The input tuple or list
          is made up of multiple tensors whose dtype is number or bool to be added together.

    Outputs:
        Tensor, has the same shape and dtype as each entry of the `input_x`.

    Examples:
        >>> class NetAddN(nn.Cell):
        >>>     def __init__(self):
        >>>         super(NetAddN, self).__init__()
        >>>         self.addN = P.AddN()
        >>>
        >>>     def construct(self, *z):
        >>>         return self.addN(z)
        >>>
        >>> net = NetAddN()
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> input_y = Tensor(np.array([4, 5, 6]), mindspore.float32)
        >>> net(input_x, input_y, input_x, input_y)
        Tensor([10, 14, 18], shape=(3,), dtype=mindspore.int32)
    """

    @prim_attr_register
    def __init__(self):
        self.__setattr_flag__ = True
        self.init_prim_io_names(inputs=["inputs"], outputs=["sum"])

    def infer_shape(self, inputs):
        cls_name = self.name
        validator.check_integer("inputs", len(inputs), 1, Rel.GE, cls_name)
        self.add_prim_attr('n', len(inputs))
        shp0 = inputs[0]
        for i, shp in enumerate(inputs):
            validator.check(f"shape of inputs[{i}]", shp, 'shape of inputs[0]', shp0, Rel.EQ, cls_name)
        return shp0

    def infer_dtype(self, inputs):
        cls_name = self.name
        validator.check_value_type("inputs", inputs, [tuple, list], cls_name)
        validator.check_integer("inputs", len(inputs), 1, Rel.GE, cls_name)
        args = {}
        for i, dtype in enumerate(inputs):
            args[f"inputs[{i}]"] = dtype
        validator.check_tensor_type_same(args, mstype.number_type + (mstype.bool_,), cls_name)
        return inputs[0]


class Neg(PrimitiveWithInfer):
    """
    Returns a tensor with negative values of the input tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor whose dtype is number.

    Outputs:
        Tensor, has the same shape and dtype as input.

    Examples:
        >>> neg = P.Neg()
        >>> input_x = Tensor(np.array([1, 2, -1, 2, 0, -3.5]), mindspore.float32)
        >>> result = neg(input_x)
        [-1.  -2.   1.  -2.   0.   3.5]
    """

    @prim_attr_register
    def __init__(self):
        """init Neg"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, input_x):
        return input_x

    def infer_dtype(self, input_x):
        validator.check_tensor_type_same({"input_x": input_x}, mstype.number_type, self.name)
        return input_x


class Sub(_MathBinaryOp):
    """
    Subtracts the second input tensor from the first input tensor element-wise.

    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be same.
    When the inputs are one tensor and one scalar, the scalar cannot be a parameter, only can be a constant,
    and the type of the scalar is the same as the data type of the tensor.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The first input is a tensor whose data type is number or a number.
        - **input_y** (Union[Tensor, Number]) - The second input is a tensor whose data type is same as 'input_x' or
          a number.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is same as 'input_x'.

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> input_y = Tensor(np.array([4, 5, 6]), mindspore.int32)
        >>> sub = P.Sub()
        >>> sub(input_x, input_y)
        [-3, -3, -3]
    """


class Mul(_MathBinaryOp):
    """
    Multiplies two tensors element-wise.

    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be same.
    When the inputs are one tensor and one scalar, the scalar cannot be a parameter, only can be a constant,
    and the type of the scalar is the same as the data type of the tensor.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The first input is a tensor whose data type is number or a number.
        - **input_y** (Union[Tensor, Number]) - The second input is a tensor whose data type is same as 'input_x' or
          a number.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is same as 'input_x'.

    Examples:
        >>> input_x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> input_y = Tensor(np.array([4.0, 5.0, 6.0]), mindspore.float32)
        >>> mul = P.Mul()
        >>> mul(input_x, input_y)
        [4, 10, 18]
    """
    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = x * y
            out = np.array(out, x.dtype)
            return Tensor(out)
        return None


class Square(PrimitiveWithInfer):
    """
    Returns square of a tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor whose dtype is number.

    Outputs:
        Tensor, has the same shape and dtype as the `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> square = P.Square()
        >>> square(input_x)
        [1.0, 4.0, 9.0]
    """

    @prim_attr_register
    def __init__(self):
        """init Square"""

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        validator.check_tensor_type_same({"x": x_type}, mstype.number_type, self.name)
        return x_type


class Rsqrt(PrimitiveWithInfer):
    """
    Computes reciprocal of square root of input tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input of Rsqrt. Each element should be a non-negative number.

    Outputs:
        Tensor, has the same type and shape as `input_x`.

    Examples:
        >>> input_tensor = Tensor([[4, 4], [9, 9]], mindspore.float32)
        >>> rsqrt = P.Rsqrt()
        >>> rsqrt(input_tensor)
        [[0.5, 0.5], [0.333333, 0.333333]]
    """

    @prim_attr_register
    def __init__(self):
        """init Rsqrt"""

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        validator.check_tensor_type_same({"x": x_type}, mstype.number_type, self.name)
        return x_type


class Sqrt(PrimitiveWithInfer):
    """
    Returns square root of a tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor whose dtype is number.

    Outputs:
        Tensor, has the same shape as the `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([1.0, 4.0, 9.0]), mindspore.float32)
        >>> sqrt = P.Sqrt()
        >>> sqrt(input_x)
        [1.0, 2.0, 3.0]
    """

    @prim_attr_register
    def __init__(self):
        """init Sqrt"""

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        validator.check_tensor_type_same({"x": x_type}, mstype.number_type, self.name)
        return x_type


class Reciprocal(PrimitiveWithInfer):
    """
    Returns reciprocal of a tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape as the `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> reciprocal = P.Reciprocal()
        >>> reciprocal(input_x)
        [1.0, 0.5, 0.25]
    """

    @prim_attr_register
    def __init__(self):
        """init Reciprocal"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x):
        return x

    def infer_dtype(self, x):
        validator.check_subclass("x", x, mstype.tensor, self.name)
        return x


class Pow(_MathBinaryOp):
    """
    Computes a tensor to the power of the second input.

    The first input must be a tensor, and the second input should be a tensor or a number.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be the same.
    When the inputs are one tensor and one scalar, the scalar could not be a parameter,
    only could be a constant, and the type of the scalar is the same as the data type of the tensor.

    Inputs:
        - **input_x** (Union[Tensor]) - The first input is a tensor whose data type is number.
        - **input_y** (Union[Tensor, Number]) - The second input is a tensor whose data type is same as 'input_x' or
          a number.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is same as 'input_x'.


    Inputs:
        - **input_x** (Tensor) - The input tensor.
        - **input_y** (Union[Tensor, Number]) - The exponent part. If exponent is a tensor, its shape must be able to
          broadcast to the shape of the `input_x`.

    Outputs:
        Tensor, has the same shape as the `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> input_y = 3.0
        >>> pow = P.Pow()
        >>> pow(input_x, input_y)
        [1.0, 8.0, 64.0]
        >>>
        >>> input_x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> input_y = Tensor(np.array([2.0, 4.0, 3.0]), mindspore.float32)
        >>> pow = P.Pow()
        >>> pow(input_x, input_y)
        [1.0, 16.0, 64.0]
    """


class Exp(PrimitiveWithInfer):
    """
    Returns exponential of a tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape as the `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> exp = P.Exp()
        >>> exp(input_x)
        [ 2.71828183,  7.3890561 , 54.59815003]
    """

    @prim_attr_register
    def __init__(self):
        """init Exp"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        validator.check_subclass("x", x_type, mstype.tensor, self.name)
        return x_type


class Log(PrimitiveWithInfer):
    """
    Returns the natural logarithm of a tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape as the `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> log = P.Log()
        >>> log(input_x)
        [0.0, 0.69314718, 1.38629436]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x):
        return x

    def infer_dtype(self, x):
        validator.check_subclass("x", x, mstype.tensor, self.name)
        return x


class Log1p(PrimitiveWithInfer):
    """
    Returns the natural logarithm of one plus the input tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape as the `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> log1p = P.Log1p()
        >>> log1p(input_x)
        [0.6931472, 1.0986123, 1.609438]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x):
        return x

    def infer_dtype(self, x):
        validator.check_subclass("x", x, mstype.tensor, self.name)
        return x


class Erf(PrimitiveWithInfer):
    r"""
    Computes the Gauss error function of `input_x` element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape and dtype as the `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([-1, 0, 1, 2, 3]), mindspore.float32)
        >>> erf = P.Erf()
        >>> erf(input_x)
        [-0.8427168, 0., 0.8427168, 0.99530876, 0.99997765]
    """

    @prim_attr_register
    def __init__(self):
        """init Erf"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        validator.check_tensor_type_same({"x": x_type}, [mstype.float16, mstype.float32], self.name)
        return x_type


class Erfc(PrimitiveWithInfer):
    r"""
    Computes the complementary error function of `input_x` element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape and dtype as the `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([-1, 0, 1, 2, 3]), mindspore.float32)
        >>> erfc = P.Erfc()
        >>> erfc(input_x)
        [1.8427168, 0., 0.1572832, 0.00469124, 0.00002235]
    """

    @prim_attr_register
    def __init__(self):
        """init Erfc"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        validator.check_tensor_type_same({"x": x_type}, [mstype.float16, mstype.float32], self.name)
        return x_type


class Minimum(_MathBinaryOp):
    """
    Computes the element-wise minimum of input tensors.

    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be same.
    When the inputs are one tensor and one scalar, the scalar cannot be a parameter, only can be a constant,
    and the type of the scalar is the same as the data type of the tensor.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The first input is a tensor whose data type is number or a number.
        - **input_y** (Union[Tensor, Number]) - The second input is a tensor whose data type is same as 'input_x' or
          a number.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is same as 'input_x'.

    Examples:
        >>> input_x = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.float32)
        >>> input_y = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float32)
        >>> minimum = P.Minimum()
        >>> minimum(input_x, input_y)
        [1.0, 2.0, 3.0]
    """


class Maximum(_MathBinaryOp):
    """
    Computes the element-wise maximum of input tensors.

    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be same.
    When the inputs are one tensor and one scalar, the scalar cannot be a parameter, only can be a constant,
    and the type of the scalar is the same as the data type of the tensor.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The first input is a tensor whose data type is number or a number.
        - **input_y** (Union[Tensor, Number]) - The second input is a tensor whose data type is same as 'input_x' or
          a number.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is same as 'input_x'.

    Examples:
        >>> input_x = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.float32)
        >>> input_y = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float32)
        >>> maximum = P.Maximum()
        >>> maximum(input_x, input_y)
        [4.0, 5.0, 6.0]
    """


class RealDiv(_MathBinaryOp):
    """
    Divide the first input tensor by the second input tensor in floating-point type element-wise.

    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be same.
    When the inputs are one tensor and one scalar, the scalar cannot be a parameter, only can be a constant,
    and the type of the scalar is the same as the data type of the tensor.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The first input is a tensor whose data type is number or a number.
        - **input_y** (Union[Tensor, Number]) - The second input is a tensor whose data type is same as 'input_x' or
          a number.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is same as 'input_x'.

    Examples:
        >>> input_x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> input_y = Tensor(np.array([4.0, 5.0, 6.0]), mindspore.float32)
        >>> realdiv = P.RealDiv()
        >>> realdiv(input_x, input_y)
        [0.25, 0.4, 0.5]
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
    """
    Computes the quotient of dividing the first input tensor by the second input tensor element-wise.

    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be same.
    When the inputs are one tensor and one scalar, the scalar cannot be a parameter, only can be a constant,
    and the type of the scalar is the same as the data type of the tensor.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The first input is a tensor whose data type is number or a number.
        - **input_y** (Union[Tensor, Number]) - The second input is a tensor whose data type is same as 'input_x' or
          a number.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is same as 'input_x'.

    Raises:
        ValueError: When `input_x` and `input_y` are not the same dtype.

    Examples:
        >>> input_x = Tensor(np.array([-4.0, 5.0, 6.0]), mindspore.float32)
        >>> input_y = Tensor(np.array([3.0, 2.0, 3.0]), mindspore.float32)
        >>> div = P.Div()
        >>> div(input_x, input_y)
    """

    def infer_value(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            return Tensor(x / y)
        return None


class FloorDiv(_MathBinaryOp):
    """
    Divide the first input tensor by the second input tensor element-wise and rounds down to the closest integer.

    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be same.
    When the inputs are one tensor and one scalar, the scalar cannot be a parameter, only can be a constant,
    and the type of the scalar is the same as the data type of the tensor.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The first input is a tensor whose data type is number or a number.
        - **input_y** (Union[Tensor, Number]) - The second input is a tensor whose data type is same as 'input_x' or
          a number.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is same as 'input_x'.

    Examples:
        >>> input_x = Tensor(np.array([2, 4, -1]), mindspore.int32)
        >>> input_y = Tensor(np.array([3, 3, 3]), mindspore.int32)
        >>> floor_div = P.FloorDiv()
        >>> floor_div(input_x, input_y)
        [0, 1, -1]
    """


class Floor(PrimitiveWithInfer):
    """
    Round a tensor down to the closest integer element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor. Its element data type must be float.

    Outputs:
        Tensor, has the same shape as `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([1.1, 2.5, -1.5]), mindspore.float32)
        >>> floor = P.Floor()
        >>> floor(input_x)
        [1.0, 2.0, -2.0]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_type_same({"x": x_dtype}, mstype.float_type, self.name)
        return x_dtype


class FloorMod(_MathBinaryOp):
    """
    Compute element-wise remainder of division.

    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be same.
    When the inputs are one tensor and one scalar, the scalar cannot be a parameter, only can be a constant,
    and the type of the scalar is the same as the data type of the tensor.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The first input is a tensor whose data type is number or a number.
        - **input_y** (Union[Tensor, Number]) - The second input is a tensor whose data type is same as 'input_x' or
          a number.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is same as 'input_x'.

    Examples:
        >>> input_x = Tensor(np.array([2, 4, -1]), mindspore.int32)
        >>> input_y = Tensor(np.array([3, 3, 3]), mindspore.int32)
        >>> floor_mod = P.FloorMod()
        >>> floor_mod(input_x, input_y)
        [2, 1, 2]
    """


class Acosh(PrimitiveWithInfer):
    """
    Compute inverse hyperbolic cosine of x element-wise.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`,
          and the data type of 'input_x' is number, the element in 'input_x' should be greater than or equal to 1.

    Outputs:
        Tensor, has the same shape as `input_x`.

    Examples:
        >>> acosh = P.Acosh()
        >>> input_x = Tensor(np.array([1.0, 1.5, 3.0, 100.0]), mindspore.float32)
        >>> output = acosh(input_x)
    """

    @prim_attr_register
    def __init__(self):
        """init Acosh"""

    def infer_shape(self, x):
        return x

    def infer_dtype(self, x):
        validator.check_tensor_type_same({'x': x}, mstype.number_type, self.name)
        return x


class _LogicBinaryOp(_BinaryOp):
    """
    Define logic binary operators.
    """

    @staticmethod
    def do_infer_dtype(x_dtype, y_dtype, valid_type=mstype.number_type, prim_name=None):
        args_dtype = {"x": x_dtype, "y": y_dtype}
        validator.check_tensor_type_same(args_dtype, valid_type, prim_name)
        return mstype.tensor_type(mstype.bool_)

    def infer_dtype(self, x_dtype, y_dtype):
        return _LogicBinaryOp.do_infer_dtype(x_dtype, y_dtype, prim_name=self.name)


class Equal(_LogicBinaryOp):
    """
    Computes the equivalence between two tensors element-wise.

    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be same.
    When the inputs are one tensor and one scalar, the scalar cannot be a parameter, only can be a constant,
    and the type of the scalar is the same as the data type of the tensor.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a tensor whose data type is number or bool, or
          a number or a bool object.
        - **input_y** (Union[Tensor, Number, bool]) - The second input tensor whose data type is same as 'input_x' or
          a number or a bool object.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is bool.

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> equal = P.Equal()
        >>> equal(input_x, 2.0)
        [False, True, False]
        >>>
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> input_y = Tensor(np.array([1, 2, 4]), mindspore.int32)
        >>> equal = P.Equal()
        >>> equal(input_x, input_y)
        [True, True, False]
    """

    def infer_dtype(self, x_dtype, y_dtype):
        return _LogicBinaryOp.do_infer_dtype(x_dtype, y_dtype, mstype.number_type + (mstype.bool_,), self.name)


class EqualCount(PrimitiveWithInfer):
    """
    Computes the number of the same elements of two tensors.

    The two input tensors should have same shape and same data type.

    Inputs:
        - **input_x** (Tensor) - The first input tensor.
        - **input_y** (Tensor) - The second input tensor.

    Outputs:
        Tensor, with the type same as input tensor and size as (1,).

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> input_y = Tensor(np.array([1, 2, 4]), mindspore.int32)
        >>> equal_count = P.EqualCount()
        >>> equal_count(input_x, input_y)
        [2]
    """

    @prim_attr_register
    def __init__(self):
        """init EqualCount"""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])

    def infer_shape(self, x_shape, y_shape):
        output_shape = (1,)
        return output_shape

    def infer_dtype(self, x_dtype, y_dtype):
        args = {'x': x_dtype, 'y': y_dtype}
        validator.check_tensor_type_same(args, mstype.number_type + (mstype.bool_,), self.name)
        return x_dtype


class NotEqual(_LogicBinaryOp):
    """
    Computes the non-equivalence of two tensors element-wise.

    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be same.
    When the inputs are one tensor and one scalar, the scalar cannot be a parameter, only can be a constant,
    and the type of the scalar is the same as the data type of the tensor.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a tensor whose data type is number or bool, or
          a number or a bool object.
        - **input_y** (Union[Tensor, Number, bool]) - The second input tensor whose data type is same as `input_x` or
          a number or a bool object.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is bool.

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> not_equal = P.NotEqual()
        >>> not_equal(input_x, 2.0)
        [True, False, True]
        >>>
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> input_y = Tensor(np.array([1, 2, 4]), mindspore.int32)
        >>> not_equal = P.NotEqual()
        >>> not_equal(input_x, input_y)
        [False, False, True]
    """

    def infer_dtype(self, x_dtype, y_dtype):
        return _LogicBinaryOp.do_infer_dtype(x_dtype, y_dtype, mstype.number_type + (mstype.bool_,), self.name)


class Greater(_LogicBinaryOp):
    """
    Computes the boolean value of :math:`x > y` element-wise.

    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be same.
    When the inputs are one tensor and one scalar, the scalar cannot be a parameter, only can be a constant,
    and the type of the scalar is the same as the data type of the tensor.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The first input is a tensor whose data type is number or a number.
        - **input_y** (Union[Tensor, Number]) - The second input is a tensor whose data type is same as `input_x` or
          a number.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is bool.

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> input_y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> greater = P.Greater()
        >>> greater(input_x, input_y)
        [False, True, False]
    """


class GreaterEqual(_LogicBinaryOp):
    """
    Computes the boolean value of :math:`x >= y` element-wise.

    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be same.
    When the inputs are one tensor and one scalar, the scalar cannot be a parameter, only can be a constant,
    and the type of the scalar is the same as the data type of the tensor.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The first input is a tensor whose data type is number or a number.
        - **input_y** (Union[Tensor, Number]) - The second input is a tensor whose data type is same as `input_x` or
          a number.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is bool.

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> input_y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> greater_equal = P.GreaterEqual()
        >>> greater_equal(input_x, input_y)
        [True, True, False]
    """


class Less(_LogicBinaryOp):
    """
    Computes the boolean value of :math:`x < y` element-wise.

    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be same.
    When the inputs are one tensor and one scalar, the scalar cannot be a parameter, only can be a constant,
    and the type of the scalar is the same as the data type of the tensor.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The first input is a tensor whose data type is number or a number.
        - **input_y** (Union[Tensor, Number]) - The second input is a tensor whose data type is same as `input_x` or
          a number.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is bool.

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> input_y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> less = P.Less()
        >>> less(input_x, input_y)
        [False, False, True]
    """


class LessEqual(_LogicBinaryOp):
    """
    Computes the boolean value of :math:`x <= y` element-wise.

    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be same.
    When the inputs are one tensor and one scalar, the scalar cannot be a parameter, only can be a constant,
    and the type of the scalar is the same as the data type of the tensor.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The first input is a tensor whose data type is number or a number.
        - **input_y** (Union[Tensor, Number]) - The second input is a tensor whose data type is same as `input_x` or
          a number.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is bool.

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> input_y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> less_equal = P.LessEqual()
        >>> less_equal(input_x, input_y)
        [True, False, True]
    """


class LogicalNot(PrimitiveWithInfer):
    """
    Computes the "logical NOT" of a tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor whose dtype is bool.

    Outputs:
        Tensor, the shape is same as the `input_x`, and the dtype is bool.

    Examples:
        >>> input_x = Tensor(np.array([True, False, True]), mindspore.bool_)
        >>> logical_not = P.LogicalNot()
        >>> logical_not(input_x)
        [False, True, False]
    """

    @prim_attr_register
    def __init__(self):
        """init LogicalNot"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_type_same({"x": x_dtype}, [mstype.bool_], self.name)
        return mstype.tensor_type(mstype.bool_)


class LogicalAnd(_LogicBinaryOp):
    """
    Computes the "logical AND" of two tensors element-wise.

    The inputs must be two tensors or one tensor and one bool object.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be bool.
    When the inputs are one tensor and one bool object, the bool object cannot be a parameter, only can be a constant,
    and the data type of the tensor should be bool.

    Inputs:
        - **input_x** (Union[Tensor, bool]) - The first input is a tensor whose data type is bool or a bool object.
        - **input_y** (Union[Tensor, bool]) - The second input is a tensor whose data type is bool or a bool object.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is bool.

    Examples:
        >>> input_x = Tensor(np.array([True, False, True]), mindspore.bool_)
        >>> input_y = Tensor(np.array([True, True, False]), mindspore.bool_)
        >>> logical_and = P.LogicalAnd()
        >>> logical_and(input_x, input_y)
        [True, False, False]
    """

    def infer_dtype(self, x_dtype, y_dtype):
        return _LogicBinaryOp.do_infer_dtype(x_dtype, y_dtype, (mstype.bool_,), self.name)


class LogicalOr(_LogicBinaryOp):
    """
    Computes the "logical OR" of two tensors element-wise.

    The inputs must be two tensors or one tensor and one bool object.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them should be bool.
    When the inputs are one tensor and one bool object, the bool object cannot be a parameter, only can be a constant,
    and the data type of the tensor should be bool.

    Inputs:
        - **input_x** (Union[Tensor, bool]) - The first input is a tensor whose data type is bool or a bool object.
        - **input_y** (Union[Tensor, bool]) - The second input is a tensor whose data type is bool or a bool object.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is bool.

    Examples:
        >>> input_x = Tensor(np.array([True, False, True]), mindspore.bool_)
        >>> input_y = Tensor(np.array([True, True, False]), mindspore.bool_)
        >>> logical_or = P.LogicalOr()
        >>> logical_or(input_x, input_y)
        [True, True, True]
    """

    def infer_dtype(self, x_dtype, y_dtype):
        return _LogicBinaryOp.do_infer_dtype(x_dtype, y_dtype, (mstype.bool_,), self.name)


class IsNan(PrimitiveWithInfer):
    """
    Judging which elements are nan for each position

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape of input, and the dtype is bool.

    Examples:
        >>> is_nan = P.IsNan()
        >>> input_x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> result = is_nan(input_x)
    """

    @prim_attr_register
    def __init__(self):
        """init IsNan"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        return mstype.bool_


class IsInf(PrimitiveWithInfer):
    """
    Judging which elements are inf or -inf for each position

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape of input, and the dtype is bool.

    Examples:
        >>> is_inf = P.IsInf()
        >>> input_x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> result = is_inf(input_x)
    """

    @prim_attr_register
    def __init__(self):
        """init IsInf"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        return mstype.bool_


class IsFinite(PrimitiveWithInfer):
    """
    Judging which elements are finite for each position

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape of input, and the dtype is bool.

    Examples:
        >>> is_finite = P.IsFinite()
        >>> input_x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> result = is_finite(input_x)
        [False   True   False]
    """

    @prim_attr_register
    def __init__(self):
        """init IsFinite"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("x", x_dtype, mstype.tensor, self.name)
        validator.check_tensor_type_same({'x': x_dtype}, mstype.number_type + (mstype.bool_,), self.name)
        return mstype.bool_


class FloatStatus(PrimitiveWithInfer):
    """
    Determine if the elements contains nan, inf or -inf. `0` for normal, `1` for overflow.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the shape of `(1,)`, and has the same dtype of input `mindspore.dtype.float32` or
        `mindspore.dtype.float16`.

    Examples:
        >>> float_status = P.FloatStatus()
        >>> input_x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> result = float_status(input_x)
    """

    @prim_attr_register
    def __init__(self):
        """init FloatStatus"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return [1]

    def infer_dtype(self, x_dtype):
        validator.check_tensor_type_same({'x': x_dtype}, [mstype.float32, mstype.float16], self.name)
        return x_dtype


class NPUAllocFloatStatus(PrimitiveWithInfer):
    """
    Allocates a flag to store the overflow status.

    The flag is a tensor whose shape is `(8,)` and data type is `mindspore.dtype.float32`.

    Note:
        Examples: see `NPUGetFloatStatus`.

    Outputs:
        Tensor, has the shape of `(8,)`.

    Examples:
        >>> alloc_status = P.NPUAllocFloatStatus()
        >>> init = alloc_status()
        Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], shape=(8,), dtype=mindspore.float32)
    """

    @prim_attr_register
    def __init__(self):
        """init NPUAllocFloatStatus"""
        self.add_prim_attr("_side_effect_flag", True)

    def infer_shape(self):
        return [8]

    def infer_dtype(self):
        return mstype.float32


class NPUGetFloatStatus(PrimitiveWithInfer):
    """
    Updates the flag which is the output tensor of `NPUAllocFloatStatus` with latest overflow status.

    The flag is a tensor whose shape is `(8,)` and data type is `mindspore.dtype.float32`.
    If the sum of the flag equals 0, there is no overflow happened. If the sum of the flag is bigger than 0, there
    is overflow happened.

    Inputs:
        - **input_x** (Tensor) - The output tensor of `NPUAllocFloatStatus`.

    Outputs:
        Tensor, has the same shape as `input_x`. All the elements in the tensor will be zero.

    Examples:
        >>> alloc_status = P.NPUAllocFloatStatus()
        >>> get_status = P.NPUGetFloatStatus()
        >>> init = alloc_status()
        >>> flag = get_status(init)
        Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], shape=(8,), dtype=mindspore.float32)
    """

    @prim_attr_register
    def __init__(self):
        """init NPUGetFloatStatus"""
        self.add_prim_attr("_side_effect_flag", True)

    def infer_shape(self, x_shape):
        cls_name = self.name
        validator.check_integer("len(x_shape)", len(x_shape), 1, Rel.EQ, cls_name)
        validator.check_integer("x_shape[0]", x_shape[0], 8, Rel.EQ, cls_name)
        return [8]

    def infer_dtype(self, x_dtype):
        validator.check_tensor_type_same({'x': x_dtype}, [mstype.float32], self.name)
        return mstype.float32


class NPUClearFloatStatus(PrimitiveWithInfer):
    """
    Clear the flag which stores the overflow status.

    Note:
        The flag is in the register on the `Ascend` device. It will be reset and can not be reused again after the
        `NPUClearFloatStatus` is called.

        Examples: see `NPUGetFloatStatus`.

    Inputs:
        - **input_x** (Tensor) - The output tensor of `NPUAllocFloatStatus`.

    Outputs:
        Tensor, has the same shape as `input_x`. All the elements in the tensor will be zero.

    Examples:
        >>> alloc_status = P.NPUAllocFloatStatus()
        >>> get_status = P.NPUGetFloatStatus()
        >>> clear_status = P.NPUClearFloatStatus()
        >>> init = alloc_status()
        >>> flag = get_status(init)
        >>> clear = clear_status(init)
        Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], shape=(8,), dtype=mindspore.float32)
    """

    @prim_attr_register
    def __init__(self):
        """init NPUClearFloatStatus"""
        self.add_prim_attr("_side_effect_flag", True)

    def infer_shape(self, x_shape):
        cls_name = self.name
        validator.check_integer("len(x_shape)", len(x_shape), 1, Rel.EQ, cls_name)
        validator.check_integer("x_shape[0]", x_shape[0], 8, Rel.EQ, cls_name)
        return [8]

    def infer_dtype(self, x_dtype):
        validator.check_tensor_type_same({'x': x_dtype}, [mstype.float32], self.name)
        return mstype.float32


class Cos(PrimitiveWithInfer):
    """
    Computes cosine of input element-wise.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape as `input_x`.

    Examples:
        >>> cos = P.Cos()
        >>> input_x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = cos(input_x)
    """

    @prim_attr_register
    def __init__(self):
        """init Cos"""

    def infer_shape(self, x):
        return x

    def infer_dtype(self, x):
        validator.check_tensor_type_same({'x': x}, mstype.number_type, self.name)
        return x


class ACos(PrimitiveWithInfer):
    """
    Computes arccosine of input element-wise.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape as `input_x`.

    Examples:
        >>> acos = P.ACos()
        >>> input_x = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float32)
        >>> output = acos(input_x)
    """

    @prim_attr_register
    def __init__(self):
        """init ACos"""

    def infer_shape(self, x):
        return x

    def infer_dtype(self, x):
        validator.check_tensor_type_same({'x': x}, mstype.number_type, self.name)
        return x


class Sin(PrimitiveWithInfer):
    """
    Computes sine of input element-wise.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape as `input_x`.

    Examples:
        >>> sin = P.Sin()
        >>> input_x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float32)
        >>> output = sin(input_x)
    """

    @prim_attr_register
    def __init__(self):
        """Init Sin."""

    def infer_shape(self, x):
        return x

    def infer_dtype(self, x):
        validator.check_tensor_type_same({'x': x}, mstype.number_type, self.name)
        return x


class NMSWithMask(PrimitiveWithInfer):
    """
    Select some bounding boxes in descending order of score.

    Args:
        iou_threshold (float): Specifies the threshold of overlap boxes with respect to
            IOU. Default: 0.5.

    Raises:
        ValueError: If the iou_threshold is not a float number, or if the first dimension
            of input Tensor is less than or equal to 0, or if the data type of the input
            Tensor is not float16 or float32.

    Inputs:
        - **bboxes** (Tensor) - The shape of tensor is :math:`(N, 5)`. Input bounding boxes.
          `N` is the number of input bounding boxes. Every bounding box
          contains 5 values, the first 4 values are the coordinates of bounding
          box, and the last value is the score of this bounding box.

    Outputs:
        tuple[Tensor], tuple of three tensors, they are selected_boxes, selected_idx and selected_mask.

        - **selected_boxes** (Tensor) - The shape of tensor is :math:`(N, 5)`. Bounding boxes
          list after non-max suppression calculation.
        - **selected_idx** (Tensor) - The shape of tensor is :math:`(N,)`. The indexes list of
          valid input bounding boxes.
        - **selected_mask** (Tensor) - The shape of tensor is :math:`(N,)`. A mask list of
          valid output bounding boxes.

    Examples:
        >>> bbox = np.random.rand(128, 5)
        >>> bbox[:, 2] += bbox[:, 0]
        >>> bbox[:, 3] += bbox[:, 1]
        >>> inputs = Tensor(bbox, mindspore.float32)
        >>> nms = P.NMSWithMask(0.5)
        >>> output_boxes, indices, mask = nms(inputs)
    """

    @prim_attr_register
    def __init__(self, iou_threshold=0.5):
        """Init NMSWithMask"""
        validator.check_value_type("iou_threshold", iou_threshold, [float], self.name)
        self.init_prim_io_names(inputs=['bboxes'], outputs=['selected_boxes', 'selected_idx', 'selected_mask'])
        self.is_ge = context.get_context("enable_ge")

    def infer_shape(self, bboxes_shape):
        cls_name = self.name
        validator.check_integer("bboxes rank", len(bboxes_shape), 2, Rel.EQ, cls_name)
        validator.check_integer("bboxes.shape()[0]", bboxes_shape[0], 0, Rel.GT, cls_name)
        if not self.is_ge:
            validator.check_integer("bboxes.shape()[1]", bboxes_shape[1], 8, Rel.EQ, cls_name)
            num = bboxes_shape[0]
            return ((num, 5), (num,), (num,))

        validator.check_integer("bboxes.shape()[1]", bboxes_shape[1], 5, Rel.EQ, cls_name)
        num = bboxes_shape[0]
        return (bboxes_shape, (num,), (num,))

    def infer_dtype(self, bboxes_dtype):
        validator.check_tensor_type_same({"bboxes": bboxes_dtype}, [mstype.float16, mstype.float32], self.name)
        return (bboxes_dtype, mstype.int32, mstype.bool_)


class Abs(PrimitiveWithInfer):
    """
    Returns absolute value of a tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor. The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, has the same shape as the `input_x`.

    Examples:
         >>> input_x = Tensor(np.array([-1.0, 1.0, 0.0]), mindspore.float32)
         >>> abs = P.Abs()
         >>> abs(input_x)
         [1.0, 1.0, 0.0]
    """

    @prim_attr_register
    def __init__(self):
        """init Abs"""

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        validator.check_tensor_type_same({'x': x_type}, mstype.number_type, self.name)
        return x_type

    def infer_value(self, x):
        if x is not None:
            x = x.asnumpy()
            out = np.abs(x, dtype=x.dtype)
            return Tensor(out)
        return None


class Sign(PrimitiveWithInfer):
    r"""
    Perform :math:`sign` on tensor element-wise.

    Note:
        .. math::
            sign(x) = \begin{cases} -1, &if\ x < 0 \cr
            0, &if\ x == 0 \cr
            1, &if\ x > 0\end{cases}

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape and type as the `input_x`.

    Examples:
         >>> input_x = Tensor(np.array([[2.0, 0.0, -1.0]]), mindspore.float32)
         >>> sign = P.Sign()
         >>> output = sign(input_x)
         [[1.0, 0.0, -1.0]]
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_type_same({'x': x_dtype}, mstype.number_type, self.name)
        return x_dtype


class Round(PrimitiveWithInfer):
    """
    Returns half to even of a tensor element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape and type as the `input_x`.

    Examples:
         >>> input_x = Tensor(np.array([0.8, 1.5, 2.3, 2.5, -4.5]), mindspore.float32)
         >>> round = P.Round()
         >>> round(input_x)
         [1.0, 2.0, 2.0, 2.0, -4.0]
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        validator.check_tensor_type_same({'x': x_type}, mstype.number_type, self.name)
        return x_type


class Atan2(_MathBinaryOp):
    r"""
    Returns arctangent of input_x/input_y element-wise.

    It returns :math:`\theta\ \in\ [-\pi, \pi]`
    such that :math:`x = r*\sin(\theta), y = r*\cos(\theta)`, where :math:`r = \sqrt{x^2 + y^2}`.

    Inputs:
        - **input_x** (Tensor) - The input tensor.
        - **input_y** (Tensor) - The input tensor.

    Outputs:
        Tensor, the shape is same as the shape after broadcasting, and the data type is same as `input_x`.

    Examples:
         >>> input_x = Tensor(np.array([[0, 1]]), mindspore.float32)
         >>> input_y = Tensor(np.array([[1, 1]]), mindspore.float32)
         >>> atan2 = P.Atan2()
         >>> atan2(input_x, input_y)
         [[0. 0.7853982]]
    """


class SquareSumAll(PrimitiveWithInfer):
    """
    Returns square sum all of a tensor element-wise

    Inputs:
        - **input_x1** (Tensor) - The input tensor.
        - **input_x2** (Tensor) - The input tensor same type and shape as the `input_x1`.

    Note:
        SquareSumAll only supports float16 and float32 data type.

    Outputs:
        - **output_y1** (Tensor) - The same type as the `input_x1`.
        - **output_y2** (Tensor) - The same type as the `input_x1`.

    Examples:
         >>> input_x1 = Tensor(np.random.randint([3, 2, 5, 7]), mindspore.float32)
         >>> input_x2 = Tensor(np.random.randint([3, 2, 5, 7]), mindspore.float32)
         >>> square_sum_all = P.SquareSumAll()
         >>> square_sum_all(input_x1, input_x2)
    """

    @prim_attr_register
    def __init__(self):
        """init SquareSumAll"""
    def infer_shape(self, x_shape, y_shape):
        validator.check("x1_shape", x_shape, "x2_shape", y_shape, Rel.EQ, self.name)
        return [], []

    def infer_dtype(self, x_type, y_type):
        validator.check_tensor_type_same({'x1_type': x_type}, [mstype.float16, mstype.float32], self.name)
        validator.check_tensor_type_same({'x2_type': y_type}, [mstype.float16, mstype.float32], self.name)
        return x_type, y_type
