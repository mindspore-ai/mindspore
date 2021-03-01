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

"""Inner operators."""

import numpy as np
from mindspore.common import Tensor
from ..._checkparam import Rel
from ..._checkparam import Validator as validator
from ... import context
from ...common import dtype as mstype
from ..primitive import PrimitiveWithCheck, PrimitiveWithInfer, prim_attr_register
from ..operations.math_ops import _infer_shape_reduce
from ...communication.management import GlobalComm
from .. import signature as sig


class ExtractImagePatches(PrimitiveWithInfer):
    """
    Extracts patches from images.
    The input tensor must be a 4-D tensor and the data format is NHWC.

    Args:
        ksizes (Union[tuple[int], list[int]]): The size of sliding window, must be a tuple or a list of integers,
            and the format is [1, 1, ksize_row, ksize_col].
        strides (Union[tuple[int], list[int]]): Distance between the centers of the two consecutive patches,
            must be a tuple or list of int, and the format is [1, 1, stride_row, stride_col].
        rates (Union[tuple[int], list[int]]): In each extracted patch, the gap between the corresponding dimension
            pixel positions, must be a tuple or a list of integers, and the format is [1, 1, rate_row, rate_col].
        padding (str): The type of padding algorithm, is a string whose value is "same" or "valid",
            not case sensitive. Default: "valid".

            - same: Means that the patch can take the part beyond the original image, and this part is filled with 0.

            - valid: Means that the taken patch area must be completely covered in the original image.

    Inputs:
        - **input_x** (Tensor) - A 4-D tensor whose shape is [in_batch, in_row, in_col, in_depth] and
          data type is number.

    Outputs:
        Tensor, a 4-D tensor whose data type is same as 'input_x',
        and the shape is [out_batch, out_row, out_col, out_depth], the out_batch is the same as the in_batch.
    """

    @prim_attr_register
    def __init__(self, ksizes, strides, rates, padding="valid"):
        """init"""

        def _check_tuple_or_list(arg_name, arg_val, prim_name):
            validator.check_value_type(f"{arg_name}s", ksizes, [tuple, list], self.name)
            if len(arg_val) != 4 or arg_val[0] != 1 or arg_val[1] != 1:
                raise ValueError(f"For \'{prim_name}\' the format of {arg_name}s should be [1, {arg_name}_row, "
                                 f"{arg_name}_col, 1], but got {arg_val}.")
            if not isinstance(arg_val[2], int) or not isinstance(arg_val[3], int) or arg_val[2] < 1 or arg_val[3] < 1:
                raise ValueError(f"For '{prim_name}' the {arg_name}_row and {arg_name}_col in {arg_name}s should be an "
                                 f"positive integer number, but got {arg_name}_row is {arg_val[2]}, {arg_name}_col "
                                 f"is {arg_val[3]}")

        _check_tuple_or_list("ksize", ksizes, self.name)
        _check_tuple_or_list("stride", strides, self.name)
        _check_tuple_or_list("rate", rates, self.name)
        self.padding = validator.check_string(padding.upper(), ['VALID', 'SAME'], 'padding', self.name)
        self.add_prim_attr("padding", self.padding)
        self.is_ge = context.get_context("enable_ge")

    def infer_shape(self, input_x):
        """infer shape"""
        in_batch, in_depth, in_row, in_col = input_x
        _, _, ksize_row, ksize_col = self.ksizes
        _, _, stride_row, stride_col = self.strides
        _, _, rate_row, rate_col = self.rates
        if len(input_x) != 4:
            raise ValueError("The `input_x` should be a 4-D tensor, "
                             f"but got a {len(input_x)}-D tensor whose shape is {input_x}")

        out_batch = in_batch
        out_depth = ksize_row * ksize_col * in_depth

        if self.padding == "VALID":
            out_row = \
                (in_row - (ksize_row + (ksize_row - 1) * (rate_row - 1))) // stride_row + 1
            out_col = \
                (in_col - (ksize_col + (ksize_col - 1) * (rate_col - 1))) // stride_col + 1
        else:
            out_row = (in_row - 1) // stride_row + 1
            out_col = (in_col - 1) // stride_col + 1

        out_shape = [out_batch, out_depth, out_row, out_col]
        return out_shape

    def infer_dtype(self, input_x):
        """infer dtype"""
        validator.check_tensor_dtype_valid("input_x", input_x, mstype.number_type, self.name)
        return input_x


class Range(PrimitiveWithInfer):
    r"""
    Creates a sequence of numbers.
    Set `input_x` as :math:`x_i` for each element, `output` as follows:

    .. math::
        \text{output}(x_i) = x_i * \text{delta} + \text{start}

    Args:
        start (float): If `limit` is `None`, the value acts as limit in the range and first entry
            defaults to `0`. Otherwise, it acts as first entry in the range.
        limit (float): Acts as upper limit of sequence. If `None`, defaults to the value of `start`
            while set the first entry of the range to `0`. It can not be equal to `start`.
        delta (float): Increment of the range. It can not be equal to zero. Default: 1.0.

    Inputs:
        - **input_x** (Tensor) - The assistant data. A `1-D` tensor of type float32 or int32.

    Outputs:
        Tensor, has the same shape and dtype as `input_x`.

    Examples:
        >>> range = ops.Range(1.0, 8.0, 2.0)
        >>> x = Tensor(np.array([1, 2, 3, 2]), mindspore.int32)
        >>> output = range(x)
        >>> print(output)
        [3, 5, 7, 5]
    """

    @prim_attr_register
    def __init__(self, start, limit=None, delta=1.0):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        self.delta = validator.check_value_type("delta", delta, [float], self.name)
        validator.check_value_type("start", start, [float], self.name)
        if limit is None:
            self.start = 0.0
            self.limit = start
            self.add_prim_attr("start", self.start)
            self.add_prim_attr("limit", self.limit)
        else:
            validator.check_value_type("limit", limit, [float], self.name)
        validator.check('start', self.start, 'limit', self.limit, Rel.NE, self.name)
        if self.delta == 0.0:
            raise ValueError("The input of `delta` can not be equal to zero.")
        if self.delta > 0.0 and self.start > self.limit:
            raise ValueError(f"Limit should be greater than start when delta:{self.delta} is more than zero, "
                             f"but got start:{self.start}, limit:{self.limit}")
        if self.delta < 0.0 and self.start < self.limit:
            raise ValueError(f"Start should be greater than limit when delta:{self.delta} is less than zero, "
                             f"but got start:{self.start}, limit:{self.limit}")

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, [mstype.float32, mstype.int32], self.name)
        return x_dtype

    def infer_value(self, x_value):
        return Tensor(np.arange(self.start, self.limit, self.delta), dtype=x_value.dtype)


class Quant(PrimitiveWithInfer):
    r"""
    Returns the quantized value of input_x.

    If `sqrt_mode` is False:

    .. math::
        y = round(scale * x + offset)

    If `sqrt_mode` is True:

    .. math::
        y = round(scale * x * scale + offset)

    Note:
        This operation only support Ascend 310 inference environment.

    Args:
        scale (float) : Specifies the scaling ratio.
        offset (float): Specifies the offset.
        sqrt_mode (bool) : Specifies whether to perform square root on `scale`. Default: False.
        round_mode (str): Specifies the way to round. Must be one of ["Round", "Floor", "Ceil", "Trunc"].
          Default: "Round".

    Inputs:
        - **input_x** (Tensor) : Input tensor. Its data type must be mindspore.float16 or mindspore.float32.

    Outputs:
        - Tensor: The quantized output tensor of type mindspore.int8.

    Examples:
        >>> input_x = Tensor([100.0, 150.0], mstype.float32)
        >>> quant = ops.Quant(80.0, 0.0, False, "Round")
        >>> y = quant(input_x)
    """

    @prim_attr_register
    def __init__(self, scale, offset, sqrt_mode=False, round_mode="Round"):
        self.scale = validator.check_value_type("scale", scale, [float], self.name)
        self.offset = validator.check_value_type("offset", offset, [float], self.name)
        self.sqrt_mode = validator.check_value_type("sqrt_mode", sqrt_mode, [bool], self.name)
        self.round_mode = validator.check_string(round_mode, ["Round", "Floor", "Ceil", "Trunc"],
                                                 "round_mode", self.name)

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_type):
        validator.check_subclass("input_x", x_type, mstype.tensor, self.name)
        validator.check_type_name("input_x", x_type, [mstype.float16, mstype.float32], self.name)
        return mstype.int8


class Dequant(PrimitiveWithInfer):
    r"""
    Returns the dequantized value of input_x.
    This operation will do ReLU to the dequantized value if `relu_flag` is True.

    If `sqrt_mode` is False:

    .. math::
        y = x * deq\_scale

    If `sqrt_mode` is True:

    .. math::
        y = x * deq\_scale * deq\_scale

    Note:
        This operation only support Ascend 310 inference environment.

    Args:
        sqrt_mode (bool) : Specifies whether to perform square root on `scale`. Default: False.
        relu_flag (bool): Specifies whether to perform ReLU. Default: False.

    Inputs:
        - **input_x** (Tensor) : Input tensor. Must be mindspore.int32.
        - **deq_scale** (Tensor) : Specifies the scaling ratio.
          Data type must be mindspore.float16 or mindspore.uint64

    Outputs:
        - Tensor: The quantized output tensor of type mindspore.float16.

    Examples:
        >>> input_x = Tensor([100.0, 150.0], mstype.float32)
        >>> dequant = ops.Dequant(False, False)
        >>> y = dequant(input_x)
    """

    @prim_attr_register
    def __init__(self, sqrt_mode=False, relu_flag=False):
        self.sqrt_mode = validator.check_value_type("sqrt_mode", sqrt_mode, [bool], self.name)
        self.relu_flag = validator.check_value_type("relu_flag", relu_flag, [bool], self.name)
        self.add_prim_attr("dtype", mstype.float16)

    def infer_shape(self, x_shape, deq_scale_shape):
        return x_shape

    def infer_dtype(self, x_type, deq_scale_type):
        validator.check_subclass("x", x_type, mstype.tensor, self.name)
        validator.check_type_name("x", x_type, [mstype.int32], self.name)
        validator.check_type_name("deq_scale", deq_scale_type, [mstype.float16, mstype.uint64], self.name)
        return mstype.float16


class MatrixDiag(PrimitiveWithInfer):
    """
    Returns a batched diagonal tensor with a given batched diagonal values.

    Inputs:
        - **x** (Tensor) - A tensor which to be element-wise multi by `assist`. It can be one of the following data
          types: float32, float16, int32, int8, and uint8.
        - **assist** (Tensor) - A eye tensor of the same type as `x`. It's rank must greater than or equal to 2 and
          it's last dimension must equal to the second to last dimension.

    Outputs:
        Tensor, has the same type and shape as input `assist`.

    Examples:
        >>> x = Tensor(np.array([1, -1]), mstype.float32)
        >>> assist = Tensor(np.arange(-12, 0).reshape(3, 2, 2), mindspore.float32)
        >>> matrix_diag = ops.MatrixDiag()
        >>> result = matrix_diag(x, assist)
        >>> print(result)
        [[[-12.   11.]
          [-10.    9.]]
         [[ -8.    7.]
          [ -6.    5.]]
         [[ -4.    3.]
          [ -2.    1.]]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize MatrixDiag"""

    def infer_dtype(self, x_dtype, assist_dtype):
        valid_type = [mstype.float16, mstype.float32, mstype.int32, mstype.int8, mstype.uint8]
        args = {"x": x_dtype, "assist": assist_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, valid_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape, assist_shape):
        validator.check_int(len(assist_shape), 2, Rel.GE, "assist rank", self.name)
        validator.check('rank of x', len(x_shape) + 1,
                        'rank of assist', len(assist_shape), Rel.LE, self.name)
        validator.check('assist\'s penultimate dimension', assist_shape[-2], 'assist\'s last dimension',
                        assist_shape[-1], Rel.EQ, self.name)

        r_end_dim = -len(x_shape)
        r_idx = -1
        while r_idx >= r_end_dim:
            if x_shape[r_idx] != 1:
                validator.check("reverse x dim %d" % r_idx, x_shape[r_idx], "reverse assist dim %d" %
                                assist_shape[r_idx - 1], assist_shape[r_idx - 1], Rel.EQ, self.name)
            r_idx = r_idx - 1

        return assist_shape


class MatrixDiagPart(PrimitiveWithInfer):
    r"""
    Returns the batched diagonal part of a batched tensor.

    Inputs:
        - **x** (Tensor) - The batched tensor. It can be one of the following data types:
          float32, float16, int32, int8, uint8.
        - **assist** (Tensor) - A eye tensor of the same type as `x`. With shape same as `x`.

    Outputs:
        Tensor, data type same as input `x`. The shape must be x.shape[:-2] + [min(x.shape[-2:])].

    Examples:
        >>> x = Tensor([[[-1, 0], [0, 1]], [[-1, 0], [0, 1]], [[-1, 0], [0, 1]]], mindspore.float32)
        >>> assist = Tensor(np.arange(-12, 0).reshape(3, 2, 2), mindspore.float32)
        >>> matrix_diag_part = ops.MatrixDiagPart()
        >>> result = matrix_diag_part(x, assist)
        >>> print(result)
        [[12., -9.], [8., -5.], [4., -1.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize MatrixDiagPart"""

    def infer_dtype(self, x_dtype, assist_dtype):
        valid_type = [mstype.float16, mstype.float32, mstype.int32, mstype.int8, mstype.uint8]
        args = {"x": x_dtype, "assist": assist_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, valid_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape, assist_shape):
        validator.check_int(len(x_shape), 2, Rel.GE, "x rank", self.name)
        validator.check("x shape", x_shape, "assist shape", assist_shape, Rel.EQ, self.name)

        if assist_shape[-2] < assist_shape[-1]:
            out_shape = assist_shape[:-1]
        else:
            out_shape = assist_shape[:-2] + assist_shape[-1:]
        return out_shape


class Send(PrimitiveWithInfer):
    """
    Send tensors from src_rank to the specified dest_rank.

    Note:
        Send and Recveive must be used in combination and have same sr_tag.
        Send must be used between servers.

    Args:
        sr_tag (int): A required integer identifying the send/recv message tag. The message will
                      will be received by the Receive op with the same "sr_tag".
        dest_rank (int): A required integer identifying the destination rank.
        group (str): The communication group to work on. Default: "hccl_world_group/nccl_world_group".

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Examples:
        >>> import mindspore.ops.operations as ops
        >>> import mindspore.nn as nn
        >>> from mindspore.communication import init
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>>
        >>> init()
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.depend = ops.Depend()
        >>>         self.send = ops.Send(st_tag=0, dest_rank=8, group="hccl_world_group")
        >>>
        >>>     def construct(self, x):
        >>>         out = self.depend(x, self.send(x))
        >>>         return out
        >>>
        >>> input_ = Tensor(np.ones([2, 8]).astype(np.float32))
        >>> net = Net()
        >>> output = net(input_)
    """

    @prim_attr_register
    def __init__(self, sr_tag, dest_rank, group=GlobalComm.WORLD_COMM_GROUP):
        self.rank = dest_rank
        self.sr_tag = sr_tag
        self.group = group

    def infer_shape(self, x_shape):
        self.add_prim_attr("shape", x_shape)
        return x_shape

    def infer_dtype(self, x_dtype):
        self.add_prim_attr("dtype", x_dtype)
        return x_dtype


class Receive(PrimitiveWithInfer):
    """
    receive tensors from src_rank.

    Note:
        Send and Recveive must be used in combination and have same sr_tag.
        Receive must be used between servers.

    Args:
        sr_tag (int): A required integer identifying the send/recv message tag. The message will
                      will be send by the Send op with the same "sr_tag".
        src_rank (int): A required integer identifying the source rank.
        shape (list[int]): A required list identifying the shape of the tensor to be received.
        dtype (Type): A required Type identifying the type of the tensor to be received. The supported types:
                       int8, int16, int32, float16, float32.
        group (str): The communication group to work on. Default: "hccl_world_group/nccl_world_group".

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Examples:
        >>> import mindspore.ops.operations as ops
        >>> import mindspore.nn as nn
        >>> from mindspore.communication import init
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>>
        >>> init()
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.recv = ops.Receive(st_tag=0, src_rank=0, shape=[2, 8], dtype=np.float32,
        >>>                               group="hccl_world_group")
        >>>
        >>>     def construct(self):
        >>>         out = self.recv()
        >>>         return out
        >>>
        >>> net = Net()
        >>> output = net()
    """

    @prim_attr_register
    def __init__(self, sr_tag, src_rank, shape, dtype, group=GlobalComm.WORLD_COMM_GROUP):
        self.rank = src_rank
        self.tag = sr_tag
        self.shape = shape
        self.dtype = dtype
        self.group = group
        valid_type = [mstype.float16, mstype.float32, mstype.int32, mstype.int8, mstype.uint8]
        args = {"dtype": dtype}
        validator.check_scalar_or_tensor_types_same(args, valid_type, self.name)

    def infer_shape(self, x_shape=None):
        return self.shape

    def infer_dtype(self, x_dtype=None):
        return self.dtype


class MatrixSetDiag(PrimitiveWithInfer):
    r"""
    Modifies the batched diagonal part of a batched tensor.

    Inputs:
        - **x** (Tensor) - The batched tensor. Rank k+1, where k >= 1. It can be one of the following data types:
          float32, float16, int32, int8, uint8.
        - **diagonal** (Tensor) - The diagonal values. Must have the same type as input `x`. Rank k, where k >= 1.
        - **assist** (Tensor) - A eye tensor of the same type as `x`. With shape same as `x`.

    Outputs:
        Tensor, data type same as input `x`. The shape same as `x`.

    Examples:
        >>> x = Tensor([[[-1, 0], [0, 1]], [[-1, 0], [0, 1]], [[-1, 0], [0, 1]]], mindspore.float32)
        >>> diagonal = Tensor([[-1., 2.], [-1., 1.], [-1., 1.]], mindspore.float32)
        >>> matrix_set_diag = ops.MatrixSetDiag()
        >>> result = matrix_set_diag(x, diagonal)
        >>> print(result)
        [[[-1, 0], [0, 2]], [[-1, 0], [0, 1]], [[-1, 0], [0, 1]]]

    """

    @prim_attr_register
    def __init__(self):
        """Initialize MatrixSetDiag"""

    def infer_dtype(self, x_dtype, diagonal_dtype, assist_dtype):
        valid_type = [mstype.float16, mstype.float32, mstype.int32, mstype.int8, mstype.uint8]
        args = {"x": x_dtype, "diagonal": diagonal_dtype, "assist": assist_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, valid_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape, diagonal_shape, assist_shape):
        validator.check_int(len(x_shape), 2, Rel.GE, "x rank", self.name)
        validator.check("x shape", x_shape, "assist shape", assist_shape, Rel.EQ, self.name)

        if x_shape[-2] < x_shape[-1]:
            validator.check("diagnoal shape", diagonal_shape, "x shape excluding the last dimension",
                            x_shape[:-1], Rel.EQ, self.name)
        else:
            validator.check("diagonal shape", diagonal_shape, "x shape excluding the second last dimension",
                            x_shape[:-2] + x_shape[-1:], Rel.EQ, self.name)

        return assist_shape


class ConfusionMulGrad(PrimitiveWithInfer):
    """
    `output0` is the dot product result of input0 and input1.

    `output1` is the dot product result of input0 and input1, then apply the reducesum operation on it.

    Args:
        axis (Union[int, tuple[int], list[int]]): The dimensions to reduce.
            Default:(), reduce all dimensions. Only constant value is allowed.
        keep_dims (bool):
            - If true, keep these reduced dimensions and the length as 1.
            - If false, don't keep these dimensions. Default:False.

    Inputs:
        - **input_0** (Tensor) - The input Tensor.
        - **input_1** (Tensor) - The input Tensor.
        - **input_2** (Tensor) - The input Tensor.

    Outputs:
        - **output_0** (Tensor) - The same shape as `input0`.
        - **output_1** (Tensor)

            - If axis is (), and keep_dims is false, the output is a 0-D array representing
              the sum of all elements in the input array.
            - If axis is int, set as 2, and keep_dims is false,
              the shape of output is :math:`(x_1,x_3,...,x_R)`.
            - If axis is tuple(int), set as (2,3), and keep_dims is false,
              the shape of output is :math:`(x_1,x_4,...x_R)`.

    Examples:
        >>> confusion_mul_grad = ops.ConfusionMulGrad()
        >>> input_0 = Tensor(np.random.randint(-2, 2, (2, 3)), mindspore.float32)
        >>> input_1 = Tensor(np.random.randint(0, 4, (2, 3)), mindspore.float32)
        >>> input_2 = Tensor(np.random.randint(-4, 0, (2, 3)), mindspore.float32)
        >>> output_0, output_1 = confusion_mul_grad(input_0, input_1, input_2)
        output_0:
            [[ 3.   1.   0.]
             [-6.   2.  -2.]]
        output_1:
            -3.0
    """

    @prim_attr_register
    def __init__(self, axis=(), keep_dims=False):
        self.init_prim_io_names(inputs=["input0", "input1", "input2"], outputs=["output0", "output1"])
        self.axis_ = validator.check_value_type("axis", axis, [int, tuple, list], self.name)
        self.keep_dims_ = validator.check_value_type("keep_dims", keep_dims, [bool], self.name)

    def infer_shape(self, input0_shape, input1_shape, input2_shape):
        outshape0 = input0_shape
        outshape1 = _infer_shape_reduce(input1_shape, self.axis_, self.keep_dims_, self.name)
        return outshape0, outshape1

    def infer_dtype(self, input0_dtype, input1_dtype, input2_dtype):
        validator.check_subclass("input0_dtype", input0_dtype, mstype.tensor, self.name)
        validator.check_subclass("input1_dtype", input1_dtype, mstype.tensor, self.name)
        validator.check_subclass("input2_dtype", input2_dtype, mstype.tensor, self.name)
        return input0_dtype, input1_dtype


class GpuConvertToDynamicShape(PrimitiveWithCheck):
    """
    This op is used for dynamic shape testing. Its inferred shape will be unknown
    during compile time, so that its output will appear to be dynamically shaped.
    The input will not be altered in any way. Put this operator before the operator
    being tested for dynamic shape support.

    Inputs:
        - **input** (Tensor) - The tensor used for testing.

    Outputs:
        - **output** (Tensor) - Same shape, type and value as `input`.

    Examples:
          >>> # make a model, since dynamic shape operators must be in GRAPH_MODE
          >>> class TestDynamicShapeReshapeNet(nn.Cell):
          >>>     def __init__(self):
          >>>         super(TestDynamicShapeReshapeNet, self).__init__()
          >>>         self.convert_to_dynamic_shape = inner.GpuConvertToDynamicShape()
          >>>         # suppose we are testing Reshape op
          >>>         self.reshape = P.Reshape()
          >>>
          >>>     def construct(self, input, new_shape):
          >>>         dynamic_shape_input = self.convert_to_dynamic_shape(input)
          >>>         reshaped_input = self.reshape(input, new_shape)
          >>>
          >>> context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
          >>> input = Tensor(np.array([0, 1, 2, 3])
          >>> new_shape = (2, 2)
          >>> net = TestDynamicShapeReshapeNet()
          >>> output = net(input, new_shape)
          >>> print(output)
          [[0, 1], [2, 3]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=["input"], outputs=["output"])

    def check_shape(self, input_shape):
        validator.check("input_shape rank", len(input_shape), "", 0, Rel.GT, self.name)

    def check_dtype(self, input_dtype):
        validator.check_subclass("input_dtype", input_dtype, mstype.tensor, self.name)


class ErrorOnDynamicShapeInput(PrimitiveWithInfer):
    """
    This op is used for dynamic shape testing. The only purpose of this operator is
    that it will throw a value error if the input is dynamically shaped.

    Inputs:
        - **input** (Tensor) - The tensor used for testing.

    Outputs:
        - **output** (Tensor) - Same shape, type and value as `input`.

    Examples:
          >>> # make a model, since dynamic shape operators must be in GRAPH_MODE
          >>> class AssertDynamicShapeNet(nn.Cell):
          >>>     def __init__(self):
          >>>         super(AssertDynamicShapeNet, self).__init__()
          >>>         self.convert_to_dynamic_shape = inner.GpuConvertToDynamicShape()
          >>>         self.error_on_dynamic_shape_input = inner.ErrorOnDynamicShapeInput()
          >>>
          >>>     def construct(self, input, new_shape):
          >>>         dynamic_shape_input = self.convert_to_dynamic_shape(input)
          >>>         self.error_on_dynamic_shape_input(dynamic_shape_input)
          >>>
          >>> context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
          >>> input = Tensor(np.array([0])
          >>> net = TestDynamicShapeReshapeNet()
          >>> output = net(input, new_shape)
          ValueError: Input is dynamically shaped.
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=["input"], outputs=["output"])

    def infer_shape(self, input_shape):
        shape = list(input_shape)

        for dim in shape:
            if dim == -1:
                raise ValueError("Input is dynamically shaped.")

        return input_shape

    def infer_type(self, input_dtype):
        validator.check_subclass("input_dtype", input_dtype, mstype.tensor, self.name)
        return input_dtype

    def infer_value(self, input_tensor):
        return input_tensor


class SequenceMask(PrimitiveWithCheck):
    """
    Returns a mask tensor representing the first N positions of each cell.

    If lengths has shape [d_1, d_2, ..., d_n], then the resulting tensor mask has type dtype and shape
    [d_1, d_2, ..., d_n, maxlen], with mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])

    Inputs:
        - **lengths** (Tensor) - Tensor to calculate the mask for. All values in this tensor should be
          less than or equal to `maxlen`. Values greater than `maxlen` will be treated as `maxlen`.
          Must be type int32 or int64.

        - **maxlen** (int) - size of the last dimension of returned tensor. Must be positive and same
          type as elements in `lengths`.

    Outputs:
        One mask tensor of shape lengths.shape + (maxlen,).

    Supported Platforms:
        ``GPU``

    Examples:
        >>> x = Tensor(np.array([[1, 3], [2, 0]]))
        >>> sequence_mask = ops.SequenceMask()
        >>> output = sequence_mask(x, 3)
        >>> print(output)
        [[[True, False, False],
          [True, True, True]],
         [[True, True, False],
          [False, False, False]]]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=["lengths", "maxlen"], outputs=["mask"])

    def check_shape(self, lengths_shape, maxlen_shape):
        validator.check("lengths_shape", len(lengths_shape), "", 0, Rel.GT, self.name)
        validator.check("maxlen_shape", len(maxlen_shape), "", 0, Rel.EQ, self.name)

    def check_dtype(self, lengths_dtype, maxlen_dtype):
        validator.check_subclass("lengths_dtype", lengths_dtype, mstype.tensor, self.name)
        validator.check_subclass("maxlen", maxlen_dtype, mstype.number, self.name)


class SyncBatchNorm(PrimitiveWithInfer):
    r"""
    Sync Batch Normalization for input data and updated parameters.

    Sync Batch Normalization is cross device synchronized batch normalization. Batch Normalization is
    widely used in convolutional neural networks. This operation applies Batch Normalization over input
    to avoid internal covariate shift as described in the paper `Batch Normalization: Accelerating
    Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_.
    It rescales and recenters the features using a mini-batch of data and the learned parameters which
    can be described in the following formula,

    .. math::
        y = \frac{x - mean}{\sqrt{variance + \epsilon}} * \gamma + \beta

    where :math:`\gamma` is scale, :math:`\beta` is bias, :math:`\epsilon` is epsilon.

    Args:
        epsilon (float): A small value added for numerical stability. Default: 1e-5.
        momentum (float): The hyper parameter to compute moving average for running_mean and running_var
            (e.g. :math:`new\_running\_mean = (1 - momentum) * running\_mean + momentum * current\_mean`).
            Momentum value must be [0, 1]. Default: 0.1.
        group (str): The communication group to work on. Default: "sync_bn_group0".
        device_num (int): The number of devices in each group. Default: 2.

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, C)`, with float16 or float32 data type.
        - **scale** (Tensor) - Tensor of shape :math:`(C,)`, with float16 or float32 data type.
        - **bias** (Tensor) - Tensor of shape :math:`(C,)`, has the same data type with `scale`.
        - **mean** (Tensor) - Tensor of shape :math:`(C,)`, with float16 or float32 data type.
        - **variance** (Tensor) - Tensor of shape :math:`(C,)`, has the same data type with `mean`.

    Outputs:
        Tuple of 5 Tensor, the normalized inputs and the updated parameters.

        - **output_x** (Tensor) - The same type and shape as the input_x. The shape is :math:`(N, C)`.
        - **updated_scale** (Tensor) - Tensor of shape :math:`(C,)`.
        - **updated_bias** (Tensor) - Tensor of shape :math:`(C,)`.
        - **updated_moving_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **updated_moving_variance** (Tensor) - Tensor of shape :math:`(C,)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> # This example should be run with multiple processes.
        >>> # Please refer to nn.SyncBatchNorm for direct use.
        >>> input_x = Tensor(np.ones([2, 2]), mindspore.float32)
        >>> scale = Tensor(np.ones([2]), mindspore.float32)
        >>> bias = Tensor(np.ones([2]), mindspore.float32)
        >>> mean = Tensor(np.ones([2]), mindspore.float32)
        >>> variance = Tensor(np.ones([2]), mindspore.float32)
        >>> sync_batch_norm = ops._inner_ops.SyncBatchNorm()
        >>> output = sync_batch_norm(input_x, scale, bias, mean, variance)
        >>> print(output)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 1.00000000e+00, 1.00000000e+00],
         [ 1.00000000e+00, 1.00000000e+00]]), Tensor(shape=[2], dtype=Float32, value=
         [ 1.00000000e+00, 1.00000000e+00]), Tensor(shape=[2], dtype=Float32, value=
         [ 1.00000000e+00, 1.00000000e+00]), Tensor(shape=[2], dtype=Float32, value=
         [ 1.00000000e+00, 1.00000000e+00]), Tensor(shape=[2], dtype=Float32, value=
         [ 1.00000000e+00, 1.00000000e+00]))
    """

    @prim_attr_register
    def __init__(self, epsilon=1e-5, momentum=0.1, group="sync_bn_group0", device_num=2):
        validator.check_float_range(epsilon, 0, 1, Rel.INC_RIGHT, 'epsilon', self.name)
        validator.check_float_range(momentum, 0, 1, Rel.INC_BOTH, 'momentum', self.name)
        validator.check_isinstance("group", group, str)
        validator.check_int(device_num, 2, Rel.GE, "device_num", self.name)
        self.init_prim_io_names(inputs=['x', 'scale', 'offset', 'mean', 'variance'],
                                outputs=['y', 'batch_mean', 'batch_variance', 'reserve_space_1', 'reserve_space_2'])

    def infer_shape(self, input_x, scale, bias, mean, variance):
        validator.check_equal_int(len(scale), 1, "scale rank", self.name)
        validator.check("scale shape", scale, "bias shape", bias, Rel.EQ, self.name)
        validator.check("scale shape[0]", scale[0], "input_x channel", input_x[1], Rel.EQ, self.name)
        validator.check_equal_int(len(mean), 1, "mean rank", self.name)
        validator.check("mean shape", mean, "variance shape", variance, Rel.EQ, self.name)
        validator.check("mean shape", mean, "scale shape", scale, Rel.EQ, self.name)
        return (input_x, scale, scale, scale, scale)

    def infer_dtype(self, input_x, scale, bias, mean, variance):
        validator.check_tensor_dtype_valid("input_x", input_x, [mstype.float16, mstype.float32], self.name)
        args = {"scale": scale, "bias": bias}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32], self.name)
        args_moving = {"mean": mean, "variance": variance}
        validator.check_tensors_dtypes_same_and_valid(args_moving, [mstype.float16, mstype.float32], self.name)
        return (input_x, scale, bias, input_x, input_x)


class Centralization(PrimitiveWithInfer):
    """
    Computes centralization. y = x - mean(x, axis).

    Note:
        The dimension index starts at 0 and must be in the range `[-input.ndim, input.ndim)`.

    Inputs:
        - **input_x** (Tensor) - The input tensor. The data type mast be float16 or float32.
        - **axis** (Union[Int, Tuple(Int), List(Int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed. Must be in the range [-rank(input_x), rank(input_x)).

    Outputs:
        Tensor, has the same shape and dtype as the `input_x`.

    Raises:
        TypeError: If `axis` is not one of the following types: int, list, tuple, NoneType.
        TypeError: If `axis` has non-Int elements.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> mindspore.set_seed(1)
        >>> input_x = Tensor(np.random.randn(2, 2).astype(np.float32))
        >>> centralization = ops.Centralization()
        >>> output = centralization(input_x, -1)
        >>> print(output)
        [[ 1.1180509 -1.1180508]
         [ 0.2723984 -0.2723984]]
    """

    __mindspore_signature__ = (
        sig.make_sig('input_x'),
        sig.make_sig('axis', default=())
    )

    @prim_attr_register
    def __init__(self):
        """Initialize Centralization"""
        self.init_prim_io_names(inputs=['input_x', 'axis'], outputs=['output'])

    def __infer__(self, input_x, axis):
        x_shape = list(input_x['shape'])
        x_dtype = input_x['dtype']
        axis_v = axis['value']
        rank = len(x_shape)

        args = {'input_x': input_x['dtype']}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32], self.name)

        if axis_v is None:
            raise ValueError(f"For {self.name}, axis must be const.")
        validator.check_value_type('axis', axis_v, [int, list, tuple], self.name)

        if isinstance(axis_v, int):
            validator.check_int_range(axis_v, -rank, rank, Rel.INC_LEFT, 'axis', self.name)
        elif axis:
            for index, one_axis in enumerate(axis_v):
                validator.check_value_type('axis[%d]' % index, one_axis, [int], self.name)

        out = {'shape': x_shape,
               'dtype': x_dtype,
               'value': None}
        return out


class StackInit(PrimitiveWithInfer):
    """
    Create a stack that produces tensors in first-in last-out order.

    After `StackInit`, a tensor can be pushed onto the stack using `StackPush`, and popped
    at the top of the stack using `StackPop`. Finally, the stack should be destroyed with `StackDestroy`.

    Args:
        index (int): The index of the stack.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.array([[1, 3], [2, 0]]))
        >>> index = 0
        >>> stack = ops.StackInit(index)
        >>> push = ops.StackPush(index)
        >>> pop = ops.StackPop(index, x.shape, x.dtype)
        >>> destroy = ops.StackDestroy(index)
        >>> stack()
        >>> push(x)
        >>> y = pop()
        >>> destroy()
        >>> print(y)
        [[1 3]
         [2 0]]
    """
    @prim_attr_register
    def __init__(self, index=1):
        """StackInit"""
        validator.check_value_type("index", index, [int], self.name)


class StackPush(PrimitiveWithInfer):
    """
    Push a tensor onto the stack.

    Before `StackPush`, the stack should be created using `StackInit`.
    Please refer to the usage in source code of `StackInit`.

    Args:
        index (int): The index of the stack.

    Inputs:
        - **input** (Tensor) - A tensor to be pushed onto the stack.

    Supported Platforms:
        ``Ascend``

    Examples:
        Please refer to the usage of `StackInit`.
    """
    @prim_attr_register
    def __init__(self, index=1):
        """StackPush"""
        validator.check_value_type("index", index, [int], self.name)
        self.init_prim_io_names(inputs=['input'], outputs=[])


class StackPop(PrimitiveWithInfer):
    """
    Pop the tensor at the top of the stack.

     Before `StackPop`, the stack should be created using `StackInit`.
     Please refer to the usage in source code of `StackInit`.

    Args:
        index (int): The index of the stack.
        shape (tuple): The shape of the tensor at the top of the stack.
        dtype (mindspore.dtype): The type of the tensor at the top of the stack.

    Outputs:
        - **output** (Tensor) - The tensor at the top of the stack.

    Supported Platforms:
        ``Ascend``

    Examples:
        Please refer to the usage of `StackInit`.
    """
    @prim_attr_register
    def __init__(self, index=1, shape=(1,), dtype=mstype.float32):
        """StackPop"""
        validator.check_value_type("index", index, [int], self.name)

        validator.check_value_type('shape type', shape, [list, tuple], self.name)
        validator.check_int(len(np.array(shape).shape), 1, Rel.EQ, "dim of shape", self.name)
        for elem in shape:
            validator.check_int(elem, 1, Rel.GE, 'shape element', self.name)
            validator.check_value_type('type of shape element', elem, [int], self.name)

        validator.check_type_name("dtype", dtype, (mstype.bool_,) + mstype.number_type, self.name)
        self.shape = shape
        self.dtype = dtype

        self.init_prim_io_names(inputs=[], outputs=['output'])

    def __infer__(self):
        return {'shape': (list(self.shape)),
                'dtype': (self.dtype),
                'value': None}


class StackDestroy(PrimitiveWithInfer):
    """
    Destroy the stack.

     Before `StackDestroy`, the stack should be created using `StackInit`.
     Please refer to the usage in source code of `StackInit`.

    Args:
        index (int): The index of the stack.

    Supported Platforms:
        ``Ascend``

    Examples:
        Please refer to the usage of `StackInit`.
    """
    @prim_attr_register
    def __init__(self, index=1):
        """StackDestroy"""
        validator.check_value_type("index", index, [int], self.name)
