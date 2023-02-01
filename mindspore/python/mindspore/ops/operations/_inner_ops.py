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

"""Inner operators."""
from types import FunctionType, MethodType
from collections.abc import Iterable
import numpy as np

from mindspore.common import Tensor
from mindspore.ops import composite as C
from mindspore.ops.operations.array_ops import Cast
from mindspore.ops.operations._scalar_ops import ScalarBitwiseOr, ScalarBitwiseAnd
from mindspore.ops import signature as sig
from mindspore.ops.operations.math_ops import _infer_shape_reduce
from mindspore.ops.primitive import PrimitiveWithCheck, PrimitiveWithInfer, prim_attr_register, Primitive, _run_op
from mindspore import context
from mindspore._checkparam import Rel
from mindspore._checkparam import Validator as validator
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.communication.management import GlobalComm
from mindspore.common.api import _pynative_executor
from mindspore.common._register_for_adapter import ms_adapter_registry


# Bit operation
bit_and = ScalarBitwiseAnd()
bit_or = ScalarBitwiseOr()
bit_xor = Primitive("bit_xor")
bit_left_shift = Primitive("bit_left_shift")
bit_right_shift = Primitive("bit_right_shift")
# String operation
string_lt = Primitive("string_lt")
string_gt = Primitive("string_gt")
string_le = Primitive("string_le")
string_ge = Primitive("string_ge")
string_not = Primitive("string_not")
string_in = Primitive("string_in")
string_mul = Primitive("string_mul")
string_getitem = Primitive("string_getitem")


class ExtractImagePatches(Primitive):
    r"""
    Extracts patches from images.
    The input tensor must be a 4-D tensor and the data format is NCHW.

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
        - **input_x** (Tensor) - A 4-D tensor whose shape is [in_batch, in_depth, in_row, in_col] and
          data type is number.

    Outputs:
        Tensor, a 4-D tensor whose data type is same as 'input_x',
        and the shape is [out_batch, out_depth, out_row, out_col], Where the out_batch is the same as the in_batch
        and

        .. math::
            out_depth=ksize\_row * ksize\_col * in\_depth

        and
        if 'padding' is "valid":

        .. math::
            out\_row=floor((in\_row - (ksize\_row + (ksize\_row - 1) * (rate\_row - 1))) / stride\_row) + 1
            out\_col=floor((in\_col - (ksize\_col + (ksize\_col - 1) * (rate\_col - 1))) / stride\_col) + 1

        if 'padding' is "same":

        .. math::
            out\_row=floor((in\_row - 1) / stride\_row) + 1
            out\_col=floor((in\_col - 1) / stride\_col) + 1

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    @prim_attr_register
    def __init__(self, ksizes, strides, rates, padding="valid"):
        """init"""

        def _check_tuple_or_list(arg_name, arg_val, prim_name):
            validator.check_value_type(f"{arg_name}s", arg_val, [tuple, list], self.name)
            if len(arg_val) != 4 or arg_val[0] != 1 or arg_val[1] != 1:
                raise ValueError(f"For \'{prim_name}\' the format of {arg_name}s must be [1, {arg_name}_row, "
                                 f"{arg_name}_col, 1], but got {arg_val}.")
            if not isinstance(arg_val[2], int) or not isinstance(arg_val[3], int) or arg_val[2] < 1 or arg_val[3] < 1:
                raise ValueError(f"For '{prim_name}' the {arg_name}_row and {arg_name}_col in {arg_name}s must be "
                                 f"an positive integer number, but got {arg_name}_row is {arg_val[2]}, "
                                 f"{arg_name}_col is {arg_val[3]}")

        _check_tuple_or_list("ksize", ksizes, self.name)
        _check_tuple_or_list("stride", strides, self.name)
        _check_tuple_or_list("rate", rates, self.name)
        validator.check_value_type('padding', padding, [str], self.name)
        self.padding = validator.check_string(padding.upper(), ['VALID', 'SAME'], 'padding', self.name)
        self.add_prim_attr("padding", self.padding)
        self.is_ge = context.get_context("enable_ge")


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


class Lamb(PrimitiveWithInfer):
    r"""
    LAMB optimizer algorithm.

    The Lamb optimizer is proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes
    <https://arxiv.org/abs/1904.00962>`_.

    Inputs:
        - **var** (Tensor) - Weights to be updated. The shape is :math:`(N, *)` where :math:`*` means,
          any number of additional dimensions. The data type can be float16 or float32.
        - **m** (Tensor) - The 1st moment vector in the updating formula,
          the shape and data type value should be the same as `var`.
        - **v** (Tensor) - the 2nd moment vector in the updating formula,
          the shape and data type value should be the same as `var`. Mean square gradients with the same type as `var`.
        - **lr** (float) - :math:`l` in the updating formula. The paper suggested value is :math:`10^{-8}`,
          the data type value should be the same as `var`.
        - **beta1** (float) - The exponential decay rate for the 1st moment estimations,
          the data type value should be the same as `var`. The paper suggested value is :math:`0.9`
        - **beta2** (float) - The exponential decay rate for the 2nd moment estimations,
          the data type value should be the same as `var`. The paper suggested value is :math:`0.999`
        - **epsilon** (float) - Term added to the denominator to improve numerical stability.
        - **decay** (float) - The weight decay value, must be a scalar tensor with float data type.
          Default: 0.0.
        - **global_step** (Tensor) - Tensor to record current global step.
        - **gradient** (Tensor) - Gradient, has the same shape and data type as `var`.

    Outputs:
        Tensor, the updated parameters.

        - **var** (Tensor) - The same shape and data type as `var`.

    Supported Platforms:
        ``Ascend````GPU``
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Lamb."""
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, var_shape, m_shape, v_shape, lr_shape, beta1_shape, beta2_shape,
                    epsilon_shape, decay_shape, global_step_shape, gradient_shape):
        validator.check("var_shape", var_shape, "m_shape", m_shape, Rel.EQ, self.name)
        validator.check("var_shape", var_shape, "v_shape", v_shape, Rel.EQ, self.name)
        validator.check("var_shape", var_shape, "gradient_shape", gradient_shape, Rel.EQ, self.name)
        return var_shape

    def infer_dtype(self, var_dtype, m_dtype, v_dtype, lr_dtype, beta1_dtype, beta2_dtype,
                    epsilon_dtype, decay_dtype, global_step_dtype, gradient_dtype):
        args = {"var": var_dtype, "m": m_dtype, "v": v_dtype, "grad": gradient_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32], self.name)

        args = {"lr": lr_dtype, "decay": decay_dtype, "beta1": beta1_dtype, "beta2": beta2_dtype,
                "epsilon": epsilon_dtype}
        validator.check_scalar_or_tensor_types_same(args, [mstype.float32], self.name, True)
        return var_dtype


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
        - **assist** (Tensor) - A eye tensor of the same type as `x`. It's rank must be greater than or equal to 2 and
          it's last dimension must be equal to the second to last dimension.

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
        Send and Receive must be used in combination and have same sr_tag.
        Send must be used between servers.

    Args:
        sr_tag (int): A required integer identifying the send/recv message tag. The message will
                      will be received by the Receive op with the same "sr_tag".
        dest_rank (int): A required integer identifying the destination rank.
        group (str): The communication group to work on. Default: "hccl_world_group/nccl_world_group".

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Examples:
        >>> import mindspore.ops as ops
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
    def __init__(self, sr_tag, dest_rank, group=GlobalComm.WORLD_COMM_GROUP, group_back=GlobalComm.WORLD_COMM_GROUP):
        self.rank = dest_rank
        self.sr_tag = sr_tag
        self.group = group
        self.add_prim_attr("no_eliminate", True)

    def infer_shape(self, x_shape):
        self.add_prim_attr("shape", x_shape)
        return x_shape

    def infer_dtype(self, x_dtype):
        return x_dtype


class Receive(PrimitiveWithInfer):
    """
    Receive tensors from src_rank.

    Note:
        Send and Receive must be used in combination and have same sr_tag.
        Receive must be used between servers.

    Args:
        sr_tag (int): A required integer identifying the send/recv message tag. The message will
                      will be send by the Send op with the same "sr_tag".
        src_rank (int): A required integer identifying the source rank.
        shape (list[int]): A required list identifying the shape of the tensor to be received.
        dtype (Type): A required Type identifying the type of the tensor to be received. The supported types:
                       int8, int16, int32, float16, float32.
        group (str, optional): The communication group to work on.
            Default: "hccl_world_group" on Ascend, "nccl_world_group" on GPU.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Examples:
        >>> import mindspore.ops as ops
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
    def __init__(self, sr_tag, src_rank, shape, dtype, group=GlobalComm.WORLD_COMM_GROUP,
                 group_back=GlobalComm.WORLD_COMM_GROUP):
        self.rank = src_rank
        self.tag = sr_tag
        self.shape = shape
        self.dtype = dtype
        self.group = group
        self.add_prim_attr("no_eliminate", True)
        valid_type = [mstype.float16, mstype.float32, mstype.int32, mstype.int8, mstype.uint8]
        args = {"dtype": dtype}
        validator.check_scalar_or_tensor_types_same(args, valid_type, self.name)

    def infer_shape(self, x_shape=None):
        return self.get_attr_dict()['shape']

    def infer_dtype(self, x_dtype=None):
        return self.get_attr_dict()['dtype']


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
            validator.check("diagonal shape", diagonal_shape, "x shape excluding the last dimension",
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


class ConvertToDynamic(PrimitiveWithCheck):
    """
    This op is used for dynamic rank testing. Its inferred shape will be unknown
    during compile time, so that its output will appear to be dynamically ranked.
    The input will not be altered in any way. Put this operator before the operator
    being tested for dynamic rank support.

    Args:
        is_dynamic_rank (bool): If true, convert to dynamic rank.
                                If false, convert to dynamic shape. Default: False.

    Inputs:
        - **input** (Tensor) - The tensor used for testing.

    Outputs:
        - **output** (Tensor) - Same shape, type and value as `input`.

    Supported Platforms:
        ``CPU``

    Examples:
          >>> import mindspore as ms
          >>> import mindspore.nn as nn
          >>> from mindspore.ops.operations import _inner_ops as inner
          >>> from mindspore.ops import operations as P
          >>> class TestDynamicNet(nn.Cell):
          >>>     def __init__(self):
          >>>         super(TestDynamicNet, self).__init__()
          >>>         self.convert_to_dynamic = inner.ConvertToDynamic()
          >>>         # suppose we are testing Reshape op
          >>>         self.reshape = P.Reshape()
          >>>
          >>>     def construct(self, input, new_shape):
          >>>         dynamic_input = self.convert_to_dynamic(input)
          >>>         reshaped_input = self.reshape(dynamic_input, new_shape)
          >>>
          >>> ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
          >>> input = Tensor(np.array([0, 1, 2, 3])
          >>> new_shape = (2, 2)
          >>> net = TestDynamicNet()
          >>> output = net(input, new_shape)
          >>> print(output)
          [[0, 1], [2, 3]
    """

    @prim_attr_register
    def __init__(self, is_dynamic_rank=False):
        validator.check_value_type('is_dynamic_rank', is_dynamic_rank, [bool], self.name)
        self.init_prim_io_names(inputs=["input"], outputs=["output"])

    def check_shape(self, input_shape):
        validator.check("input_shape rank", len(input_shape), "", 0, Rel.GT, self.name)

    def check_dtype(self, input_dtype):
        validator.check_subclass("input_dtype", input_dtype, mstype.tensor, self.name)


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
          >>> import mindspore as ms
          >>> import mindspore.nn as nn
          >>> from mindspore.ops.operations import _inner_ops as inner
          >>> from mindspore.ops import operations as P
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
          >>> ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
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
          >>> import mindspore as ms
          >>> import mindspore.nn as nn
          >>> from mindspore.ops.operations import _inner_ops as inner
          >>> from mindspore.ops import operations as P
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
          >>> ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
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
        """Infer the dtype of input for ErrorOnDynamicShapeInput."""
        validator.check_subclass("input_dtype", input_dtype, mstype.tensor, self.name)
        return input_dtype

    def infer_value(self, input_tensor):
        return input_tensor


class SequenceMask(PrimitiveWithCheck):
    """
    Returns a mask tensor representing the first N positions of each cell.

    If lengths has shape [d_1, d_2, ..., d_n], then the resulting tensor mask has type and shape
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
        >>> from mindspore import ops
        >>> import numpy as np
        >>> x = Tensor(np.array([[1, 3], [2, 0]]))
        >>> sequence_mask = ops.SequenceMask()
        >>> output = sequence_mask(x, 3)
        >>> print(output)
        [[[True False False]
          [True True True]]
         [[True True False]
          [False False False]]]
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


class SyncBatchNorm(Primitive):
    r"""
    Sync Batch Normalization for input data and updated parameters.

    Sync Batch Normalization is cross device synchronized Batch Normalization. Batch Normalization is
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
        self.add_prim_attr('side_effect_mem', True)
        self.add_prim_attr('format', 'NCHW')


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
        index (int): The index of the stack. Default: 1.

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
        index (int): The index of the stack. Default: 1.

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
        index (int): The index of the stack. Default: 1.
        shape (tuple): The shape of the tensor at the top of the stack. Default: (1,).
        dtype (mindspore.dtype): The type of the tensor at the top of the stack. Default: mindspore.float32.

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
        index (int): The index of the stack. Default: 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        Please refer to the usage of `StackInit`.
    """

    @prim_attr_register
    def __init__(self, index=1):
        """StackDestroy"""
        validator.check_value_type("index", index, [int], self.name)


class DynamicStitch(PrimitiveWithCheck):
    r"""
    Interleave the values from the data tensors into a single tensor.

    Inputs:
        - **indices** (Union[tuple, list]) - A Tuple or list of Tensor objects with the same shape and type.
        - **data** (Union[tuple, list]) - A Tuple or list of Tensor objects with the same shape and type.

    Outputs:
        Tensor. A stacked Tensor with the same type as `data`.

    Raises:
        TypeError: If the data types of elements in `data` or `indices` are not the same.
        ValueError: If the length of `data` or `indices` is not greater than 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x1 = Tensor([6], mstype.int32)
        >>> x2 = Tensor(np.array([4, 1]), mstype.int32)
        >>> x3 = Tensor(np.array([[5, 2], [0, 3]]), mstype.int32)
        >>> y1 = Tensor(np.array([[6, 1]]), mstype.int32)
        >>> y2 = Tensor(np.array([[41, 42], [11, 12]]), mstype.int32)
        >>> y3 = Tensor(np.array([[[51, 52], [21, 22]], [[1, 2], [31, 32]]]), mstype.int32)
        >>> stitch = ops.DynamicStitch()
        >>> output = stitch([x1, x2, x3], [y1, y2, y3])
        >>> print(output)
        [[ 1  2]
         [11 12]
         [21 22]
         [31 32]
         [41 42]
         [51 52]
         [61 62]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize DynamicStitch"""

    def check_shape(self, indices_shape, data_shape):
        validator.check_value_type("shape of indices", indices_shape, [tuple, list], self.name)
        validator.check_int(len(indices_shape), 1, Rel.GE, "len of indices_shape", self.name)
        indices_dim0 = len(indices_shape[0])
        indices_num = len(indices_shape)

        validator.check_value_type("shape of data", data_shape, [tuple, list], self.name)
        validator.check_int(len(data_shape), 1, Rel.GE, "len of data_shape", self.name)
        data_dim0 = len(data_shape[0])
        data_num = len(indices_shape)

        validator.check("size of indices", indices_num, 'size of data', data_num, Rel.EQ, self.name)

        # shape of `data` must start with shape of `indices`
        for i in range(0, indices_num):
            indices_dim = len(indices_shape[i])
            data_dim = len(data_shape[i])
            validator.check(f"dim of indices[{i}]", indices_dim, f"dim of data[{i}]", data_dim, Rel.LE, self.name)
            if data_shape[i][:indices_dim] != data_shape[i][:indices_dim]:
                raise ValueError(f"data[{i}].shape: {data_shape} does not start with indices[{i}].shape: {data_shape}")

        # the last-(data_dim0-indices_dim0)-dim of data shape must end with same shape.
        base_extra = data_dim0 - indices_dim0
        for i in range(0, data_num):
            indices_dim = len(indices_shape[i])
            data_dim = len(data_shape[i])
            extra = data_dim - indices_dim
            validator.check(f"extra dim of data[{i}]", extra,
                            f"extra dim of data[0]", base_extra, Rel.EQ, self.name)
            validator.check(f"data[0].shape[{indices_dim0}:]", data_shape[0][indices_dim0:],
                            f"data[{i}].shape[{len(indices_shape[i])}:]",
                            data_shape[i][indices_dim:], Rel.EQ, self.name)

        out_shape = [-1] + data_shape[0][indices_dim0:]
        return out_shape

    def check_dtype(self, indices_type, data_type):
        validator.check_subclass("indices[0]", indices_type[0], mstype.tensor, self.name)
        validator.check_subclass("data[0]", data_type[0], mstype.tensor, self.name)
        indices_num = len(indices_type)
        for i in range(0, indices_num):
            validator.check_tensor_dtype_valid(f'indices[{i}]', indices_type[i], mstype.int32, self.name)
            validator.check_tensor_dtype_valid(f'data[{i}]', data_type[i],
                                               mstype.number_type + (mstype.bool_,), self.name)
            validator.check(f"type of data[{i}]", data_type[i], f"type of data[0]", data_type[0], Rel.EQ, self.name)
        return data_type[0]


class DynamicBroadcastGradientArgs(Primitive):
    """
    Broadcast the two input shapes, return the dimensions that each need to be broadcast.

    Input shape `s0` and shape `s1` can be broadcast to a common shape if for each dimension pair they are either equal
    or input is one or the target dimension is -1. In case of -1 in target shape, it will be replaced by the input
    shape's value in that dimension.

    Inputs:
        - **s0** (Tensor) - A `1-D` tensor. The data type should be one of the following types: int32, int64,
          uint32, uint64.
        - **s1** (Tensor) - A `1-D` tensor with the same type as `s0`.

    Outputs:
        Tuple(Tensor), tuple of 2 tensors, r0 and r1. The first one is the index tensor and the other one is the mask
        tensor.

        - **r0** (Tensor) - The output shape is 1-D with the same type as s0.
        - **r1** (Tensor) - The output shape is 1-D with the same type as s0.

    Raises:
        ValueError: if the `s0` and `s1` are incompatible, or if a - 1 in the target shape is in an invalid
                    location.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> shape0 = (4, 2, 1)
        >>> shape1 = (2, 7)
        >>> from mindspore.ops.operations import _inner_ops
        >>> args = _inner_ops.DynamicBroadcastGradientArgs()
        >>> r0, r1 = args(Tensor(shape0), Tensor(shape1))
        >>> print(r0, r1)
        [2], [0]
    """

    @prim_attr_register
    def __init__(self):
        """Init BroadcastGradientArgs"""


class TensorCopySlices(Primitive):
    """
    Copy continues memory.

    Inputs:
        - **x** (Tensor) - The target Tensor.
        - **value** (Tensor) - The tensor to update x.
        - **begin** (tuple[int]) - A tuple which represents the location where to start. Only
          constant value is allowed.
        - **end** (tuple[int]) - A tuple or which represents the maximum location where to end.
          Only constant value is allowed.
        - **strides** (tuple[int]) - A tuple which represents the stride is continuously added
          before reaching the maximum location. Only constant value is allowed.

    Outputs:
        - **y** (Tensor), has the same shape and data type of x.

    Examples:
        >>> import numpy as np
        >>> from mindspore.ops.operations import _inner_ops
        >>> copy_slices = _inner_ops.TensorCopySlices()
        >>> out = copy_slices(Tensor(np.zeros((5, 5))), Tensor(np.ones((2, 5))), (3, 0), (5, 5), (1, 1))
        >>> print(out)
            [[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]]

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self):
        """Initialize TensorScatterUpdate"""
        self.init_prim_io_names(inputs=['x', 'value', 'begin', 'end', 'strides'], outputs=['y'])


class DSDMatmul(PrimitiveWithInfer):
    """
    The definition of the CusSquare primitive.
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['input_w1', 'input_w2', 'input_v'], outputs=['output_y'])

    def infer_shape(self, input_w1_shape, input_w2_shape, input_v_shape):
        batch_size = input_w1_shape[0]
        head = input_w1_shape[1]
        v_embedding = input_v_shape[1] * 16 // head
        seq_len = input_v_shape[0] * 16 // batch_size
        return (batch_size, head, v_embedding // 16, seq_len // 16, 16, 16)

    def infer_dtype(self, data_dtype1, data_dtype2, data_dtype3):
        return data_dtype1


class MatmulDDS(PrimitiveWithInfer):
    """MatmulDDS definition"""

    @prim_attr_register
    def __init__(self, bs, heads):
        """init MatmulDDS"""
        self.init_prim_io_names(inputs=['q', 'k', 'local_mask', 'global_mask'],
                                outputs=['local_prob', 'global_prob'])

        self.heads = heads

    def infer_shape(self, q, k, local_mask, global_mask):
        seq_len = local_mask[0] * local_mask[-1]
        bs = q[1] * q[2] // seq_len
        global_size = seq_len // 4
        size_per_head = q[0] * q[-1] // self.heads
        heads = q[0] * q[-1] // size_per_head
        block_size = local_mask[1] * local_mask[2] // bs
        block_num = seq_len // block_size
        l_size = (bs, heads, block_num, block_size // 16, block_size // 16, 16, 16)
        g_size = (bs, heads, block_num, global_size // 16, block_size // 16, 16, 16)

        return l_size, g_size

    def infer_dtype(self, q, k, local_mask, global_mask):
        return q, q


class DSDGrad(PrimitiveWithInfer):
    """
    The definition of the CusSquare primitive.
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['w1_gm', 'w2_gm', 'v_gm', 'a_gm', 'd_a_gm'],
                                outputs=['d_w1_gm', 'd_w2_gm', 'd_v_gm'])

    def infer_shape(self, input_w1_shape, input_w2_shape, input_v_shape, input_a_shape, input_da_shape):
        return input_w1_shape, input_w2_shape, input_v_shape

    def infer_dtype(self, data_dtype1, data_dtype2, data_dtype3, data_dtype4, data_dtype5):
        return data_dtype1, data_dtype1, data_dtype1


class MatmulDDSGrad(PrimitiveWithInfer):
    """MatmulDDS definition"""

    @prim_attr_register
    def __init__(self):
        """init MatmulDDS"""
        self.init_prim_io_names(inputs=['q', 'k', 'local_prob', 'global_prob', 'local_prob_grad', 'global_prob_grad'],
                                outputs=['dq', 'dk'])

    def infer_shape(self, q, k, local_prob, global_prob, local_prob_grad, global_prob_grad):
        k_size = (q[1], q[0], q[3], q[2])

        return q, k_size

    def infer_dtype(self, q, k, local_prob, global_prob, local_prob_grad, global_prob_grad):
        return q, k


class NonZeroWithValue(Primitive):
    """
    Returns the value of elements that are non-zero (in row-major order - by dimension).

    Inputs:
        - **x** (Tensor), input array of rank >= 2.

    Outputs:
         elements that are non-zero.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> op = NonZeroWithValue()
        >>> data = Tensor(np.array([[1, 0, 0], [0, 0, 1]]), mindspore.float32)
        >>> value, index, count = op(data)
        >>> print(value)
        [1.0, 1.0]
    """

    @prim_attr_register
    def __init__(self, transpose=False):
        """Initialize NonZeroWithValue"""
        validator.check_value_type("transpose", transpose, [bool], self.name)
        self.init_prim_io_names(inputs=['x'], outputs=['value', 'index', 'count'])


class NonZeroWithValueShape(Primitive):
    """
    Returns the value and index of elements that are non-zero (in row-major order - by dimension).

    Inputs:
        - **x** (Tensor), input array of rank >= 2.

    Outputs:
         elements that are non-zero.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> non_zero = NonZeroWithValue()
        >>> op = NonZeroWithValueShape()
        >>> data = Tensor(np.array([[1, 0, 0], [0, 0, 1]]), mindspore.float32)
        >>> value, index, count = non_zero(data)
        >>> out_value, out_index = op(value, index, count)
        >>> print(out_index)
        [[0, 1], [0, 2]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize NonZeroWithValueShape"""
        self.init_prim_io_names(inputs=['value', 'index', 'count'], outputs=['out_value', 'out_index'])


class DecodeImage(PrimitiveWithInfer):
    """
    Returns image data that parse from string Tensor.

    Inputs:
        - **x** (Tensor), a Tensor of type string. 0-D. The jPEG, GIF, PNG, BMP-encoded image.

    Outputs:
         A Tensor of type uint8, uint16, float.

    Supported Platforms:
        ``Ascend``

    Examples:
    """
    @prim_attr_register
    def __init__(self, channels=0, dtype=mstype.uint8, expand_animations=False, _op_max_shape="8192,8192,3",
                 _op_max_size=[8000000]):
        self.init_prim_io_names(inputs=["contents"], outputs=["image"])
        self.res_type = dtype

    def infer_shape(self, x):
        return (-1, -1, 3)

    def infer_dtype(self, x):
        return self.res_type


class SliceGetItem(Primitive):
    """
        using SliceGetItem to get slice's attribute of 'start' 'stop' 'step'
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ScatterElements"""
        self.init_prim_io_names(inputs=['slice', 'attr'], outputs=['slice_item'])

    def __call__(self, slice_value, value):
        if not isinstance(slice_value, slice):
            raise TypeError(
                "Primitive[SliceGetItem] only support to get a slice type element but got {}".format(slice_value))
        if value == "start":
            if hasattr(slice_value.start, "ndim") and slice_value.start.ndim == 1:
                return slice_value.start.item()
            return slice_value.start
        if value == "stop":
            if hasattr(slice_value.stop, "ndim") and slice_value.stop.ndim == 1:
                return slice_value.stop.item()
            return slice_value.stop
        if value == "step":
            if hasattr(slice_value.step, "ndim") and slice_value.step.ndim == 1:
                return slice_value.step.item()
            return slice_value.step
        raise AttributeError("\'slice\' object has no attribute {}".format(value))


class DynamicBroadcastTo(Primitive):
    """
    Broadcasts input tensor to a given shape.

    Inputs:
        - **input_x** (Tensor) - The input tensor. The data type should be one of the following types:
          float16, float32, int32, int8, uint8.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        - **shape** (Tensor): The target shape to broadcast.

    Outputs:
        Tensor, with the given `shape` and the same data type as `input_x`.

    Raises:
        ValueError: if the target and input shapes are incompatible.

    Supported Platforms:
        ``Ascend``
    """

    @prim_attr_register
    def __init__(self):
        """Initialize DynamicBroadcastTo"""
        self.init_prim_io_names(inputs=['x', 'shape'], outputs=['y'])


class Cummin(Primitive):
    r"""
    Returns the cumulative minimum of elements and the index.

    Refer to :func:`mindspore.ops.cummin` for more detail.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore
        >>> a = Tensor([-0.2284, -0.6628,  0.0975,  0.2680, -1.3298, -0.4220], mindspore.float32)
        >>> func = ops.Cummin(axis=0)
        >>> output = func(a)
        >>> print(output[0])
        [-0.2284 -0.6628 -0.6628 -0.6628 -1.3298 -1.3298]
        >>> print(output[1])
        [0 1 1 1 4 4]
    """
    @prim_attr_register
    def __init__(self, axis):
        """Initialize Cummin"""
        validator.check_value_type('axis', axis, [int], self.name)


class DynamicResizeNearestNeighbor(Primitive):
    r"""
    Resizes the input tensor by using the nearest neighbor algorithm.

    Resizes the input tensor to a given size by using the nearest neighbor algorithm. The nearest
    neighbor algorithm selects the value of the nearest point and does not consider the
    values of neighboring points at all, yielding a piecewise-constant interpolant.

    Note:
        The operator supports dynamic shape.

    Args:
        align_corners (bool): Whether the centers of the 4 corner pixels of the input
                              and output tensors are aligned. Default: False.

    Inputs:
        - **input_x** (Tensor) - The input tensor. The shape of the tensor is :math:`(N, C, H, W)`.
        - **size** (Union[tuple, list]): The target size. The dimension of size must be 2.

    Outputs:
        Tensor, the shape of the output tensor is  :math:`(N, C, NEW\_H, NEW\_W)`.
        The data type is the same as the `input_x`.
    """

    @prim_attr_register
    def __init__(self, align_corners=False):
        """Initialize ResizeNearestNeighbor"""
        validator.check_value_type("align_corners", align_corners, [bool], self.name)
        self.init_prim_io_names(inputs=['image_in'], outputs=['image_out'])


class PsROIPooling(PrimitiveWithInfer):
    r"""
    Position Sensitive ROI-Pooling
    Inputs:
        - feature(Tensor)
        - rois(Tensor)

        - **features** (Tensor) - The input features, whose shape must be :math:`(N, C, H, W)`.
        - **rois** (Tensor) - The shape is :math:`(rois\_n, 5)`. With data type of float16 or float32.
          `rois_n` represents the number of RoI. The size of the second dimension must be `5` and the `5` colunms
          are :math:`(image\_index, top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y)`.
          `image_index` represents the index of image. `top_left_x` and `top_left_y` represent the `x, y`
          coordinates of the top left corner of corresponding RoI, respectively. `bottom_right_x` and `bottom_right_y`
          represent the `x, y` coordinates of the bottom right corner of corresponding RoI, respectively.

    Outputs:
        - out shape(rois_num, out_channel, pool_height, pool_width), the result after pooling.
        - channel_map shape(rois_num, out_channel, pool_height, pool_width), use for back forward to compute grad
    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.ops.operations import _inner_ops as inner
        >>> features = np.random.randn(4, 21 * 7 * 7, 80, 48)
        >>> features = Tensor.from_numpy(features).astype(mindspore.float32)
        >>> rois = Tensor.from_numpy(
        >>>     np.array([
        >>>        [0.0000, 150.3563, 200.1320, 579.3563, 602.3452],
        >>>        [1.0000, 657.1263, 302.8564, 762.4214, 567.9854],
        >>>        [2.0000, 321.3122, 232.2410, 679.0281, 587.6346],
        >>>        [3.0000, 664.1630, 387.4919, 778.7322, 562.7321],
        >>>     ])).astype(mindspore.float32)
        >>> psRoIPooling = inner.PsROIPooling(pooled_height=7, pooled_width=7, num_rois=4,
        >>>                                  spatial_scale=1.0/16, out_dim=21,
        >>>                                  group_size=7)
        >>> out, channel_map = psRoIPooling(features, rois)
        >>> print(out.shape)
            [4, 21, 7, 7]
        >>> print(channel_map.shape)
            [4, 21, 7, 7]
    """

    @prim_attr_register
    def __init__(self, pooled_height, pooled_width, num_rois, spatial_scale, out_dim, group_size):
        """Initialize PsROIPooling"""
        validator.check_value_type("pooled_height", pooled_height, [int], self.name)
        validator.check_value_type("pooled_width", pooled_width, [int], self.name)
        validator.check_value_type("num_rois", pooled_width, [int], self.name)
        validator.check_value_type("spatial_scale", spatial_scale, [float], self.name)
        validator.check_value_type("out_dim", out_dim, [int], self.name)
        validator.check_value_type("group_size", group_size, [int], self.name)
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.num_rois = num_rois
        self.spatial_scale = spatial_scale
        self.out_dim = out_dim
        self.group_size = group_size

    def infer_shape(self, inputs_shape, rois_shape):
        output_shape = [self.num_rois, self.out_dim, self.pooled_height, self.pooled_width]
        output_map_shape = [self.num_rois, self.out_dim, self.pooled_height, self.pooled_width]
        return output_shape, output_map_shape

    def infer_dtype(self, inputs_type, rois_type):
        map_type = mstype.tensor_type(mstype.int32)
        return inputs_type, map_type


class ParallelResizeBilinear(PrimitiveWithInfer):
    """ParallelResizeBilinear ops"""

    @prim_attr_register
    def __init__(self, ori_image_size, split_size, src_start_w, dst_start_w, align_corners):
        """Initialize ParallelResizeBilinear."""
        validator.check_value_type("ori_image_size", ori_image_size, [list, tuple], self.name)
        validator.check_value_type("split_size", split_size, [list, tuple], self.name)
        validator.check_int(len(split_size), 2, Rel.EQ, "len of split_size", self.name)
        validator.check_value_type("src_start_w", src_start_w, [int], self.name)
        validator.check_value_type("dst_start_w", dst_start_w, [int], self.name)
        validator.check_value_type("align_corners", align_corners, [bool], self.name)
        self.ori_image_size = list(ori_image_size)
        self.split_size = list(split_size)
        self.src_start_w = src_start_w
        self.dst_start_w = dst_start_w
        self.align_corners = align_corners
        self.half_pixel_centers = False
        self.add_prim_attr('ori_image_size', self.ori_image_size)
        self.add_prim_attr('split_size', self.split_size)
        self.add_prim_attr('src_start_w', self.src_start_w)
        self.add_prim_attr('dst_start_w', self.dst_start_w)
        self.add_prim_attr('align_corners', self.align_corners)
        self.add_prim_attr('half_pixel_centers', self.half_pixel_centers)

    def __infer__(self, x, size):
        size_val = size['value']
        x_shape = x['shape']
        x_dtype = x['dtype']
        validator.check_tensor_dtype_valid("x_dtype", x_dtype, [mstype.float16, mstype.float32], self.name)
        if size_val is None:
            raise ValueError("size must be const input")
        output_shape = [x_shape[0], x_shape[1], self.split_size[0], self.split_size[1]]

        return {'shape': output_shape,
                'dtype': x_dtype,
                'value': None}


class PartitionedCall(PrimitiveWithInfer):
    """
    Pass the input tensors to the subgraph and return the output tensors.

    Inputs:
        - **inputs** (Tuple), the input tensors, which will be passed to subgraph.

    Outputs:
        - outputs(Tuple), the output tensor returned by subgraph.

    Supported Platforms:
        ``Ascend``

    Examples:
    """
    @prim_attr_register
    def __init__(self, graph, executor_type=""):
        self.add_prim_attr("executor_type", executor_type)
        self.graph = graph

    def infer_shape(self, *inputs):
        return NotImplementedError

    def infer_dtype(self, *inputs):
        return NotImplementedError


class CellBackwardHook(PrimitiveWithInfer):
    r"""
    This operator is used to hook input gradient and output gradient of Cell object.

    Note:
        This operator is only used in backward hook function of Cell object in pynative mode.

    Args:
        cell_id (str): Used to identify which cell obj the hook function registered on. For example, 'nn.Add()' is a
        cell object.

    Inputs:
        - **input** - The variable to hook.

    Outputs:
        - **output** - Returns `input` directly. `CellBackwardHook` does not affect the forward result.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.ops import GradOperation
        >>> from mindspore.ops.operations import _inner_ops as inner
        >>> ms.set_context(mode=ms.PYNATIVE_MODE)
        >>> def hook_fn(grad):
        ...     print(grad)
        ...
        >>> hook = inner.CellBackwardHook()
        >>> hook_fn_key = hook.register_backward_hook(hook_fn)
        >>> def hook_test(x, y):
        ...     z = x * y
        ...     z = hook(z)
        ...     z = z * y
        ...     return z
        ...
        >>> grad_all = GradOperation(get_all=True)
        >>> def backward(x, y):
        ...     return grad_all(hook_test)(x, y)
        ...
        >>> output = backward(Tensor(1, mindspore.float32), Tensor(2, mindspore.float32))
        (Tensor(shape=[], dtype=Float32, value= 2),)
        >>> print(output)
        (Tensor(shape=[], dtype=Float32, value= 4), Tensor(shape=[], dtype=Float32, value= 4))
        >>> hook.remove_backward_hook(hook_fn_key)
        >>> output = backward(Tensor(1, mindspore.float32), Tensor(2, mindspore.float32))
        >>> print(output)
        (Tensor(shape=[], dtype=Float32, value= 4), Tensor(shape=[], dtype=Float32, value= 4))
    """

    def __init__(self, cell_id=""):
        """Initialize CellBackwardHook"""
        super(CellBackwardHook, self).__init__(self.__class__.__name__)
        self.cell_id = cell_id
        self.add_prim_attr("cell_id", cell_id)
        self.init_attrs["cell_id"] = cell_id

    def __call__(self, args):
        if not isinstance(args, tuple):
            args = (args,)
        for arg in args:
            if isinstance(arg, Parameter) and arg.has_init:
                arg.init_data()
        return _run_op(self, self.name, args)

    def infer_shape(self, *inputs_shape):
        if len(inputs_shape) == 1:
            return inputs_shape[0]
        return inputs_shape

    def infer_dtype(self, *inputs_type):
        if len(inputs_type) == 1:
            return inputs_type[0]
        return inputs_type

    def register_backward_hook(self, hook_fn):
        r"""
        This function is used to register backward hook function. Note that this function is only supported in pynative
        mode.

        Note:
            The 'hook_fn' must be defined as the following code.
            `cell_id` is the information of registered cell. `grad_input` is the gradient passed to the cell.
            `grad_output` is the gradient computed and passed to the next cell or primitive, which may be modified by
            returning a new output gradient.
            The 'hook_fn' should have the following signature:
            hook_fn(cell_id, grad_input, grad_output) -> New output gradient or none.
            The 'hook_fn' is executed in the python environment.

        Args:
            hook_fn (Function): Python function. Backward hook function.

        Returns:
            - **key** (int) - The key of 'hook_fn'.

        Raises:
            TypeError: If the `hook_fn` is not a function of python.
        """
        if not isinstance(hook_fn, (FunctionType, MethodType)):
            raise TypeError(f"When using 'register_backward_hook(hook_fn)', the type of 'hook_fn' must be python "
                            f"function, but got {type(hook_fn)}.")
        key = self.add_backward_hook_fn(hook_fn)
        return key

    def remove_backward_hook(self, key):
        r"""
        This function is used to remove backward hook function. Note that this operation is only supported in pynative
        mode.

        Note:
            The 'key' is the object returned by 'register_backward_hook' function of the same CellBackwardHook
            operator.

        Args:
            key (int): The key corresponding to the 'hook_fn'.

        Returns:
            None.
        """
        self.remove_backward_hook_fn(key)


class Format(PrimitiveWithInfer):
    r"""
    This operator is used to format a string.

    Note:
     Current not supported to using by customer.
     Only support convert str.format() in user code and it will be converted to be Format
     operation by ME-Compiler automatically.


    Inputs:
     - **input** -
     string : the string to be formatted.
     args : the format args.

    Outputs:
     - **output** - Returns formatted string.

    Supported Platforms:
     ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['string', 'args'], outputs=['string'])

    def __infer__(self, str_, *var):
        str_value = str_["value"]
        var_value = list()
        if str_value is None and str_["dtype"] is not None:
            raise ValueError("str.format not support to input a variable.")
        for item in var:
            if item["value"] is None and item["dtype"] is not None:
                raise ValueError("str.format not support to input a variable.")
            var_value.append(item["value"])
        value = str_value.format(*var_value)
        return {'dtype': mstype.string, 'shape': [], 'value': value}


class FlattenConcat(Primitive):
    """
    Flatten input tensors and concatenate them into several chunk tensors grouped by data types.

    Args:
        fusion_size (int): Maximum memory chunk size in bytes, 0 for unlimited. Default: 0.

    Inputs:
        - **tensors** (tuple[Tensor], list[Tensor]) - The input Tensors to be flattened and concatenated.

    Outputs:
        tuple[Tensor], result chunk tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.ops.operations import _inner_ops as inner
        >>> t1 = Tensor(np.array([1]).astype(np.float32))
        >>> t2 = Tensor(np.array([2]).astype(np.float32))
        >>> t3 = Tensor(np.array([3]).astype(np.float64))
        >>> t4 = Tensor(np.array([4]).astype(np.float32))
        >>> t5 = Tensor(np.array([5]).astype(np.float64))
        >>> chunks = inner.FlattenConcat()([t1, t2, t2, t3, t4, t5])
        >>> print(chunks[0].asnumpy())
        >>> print(chunks[1].asnumpy())
        [1. 2. 4.]
        [3. 5.]
    """

    @prim_attr_register
    def __init__(self, fusion_size=0):
        """Initialize FlattenConcat"""
        validator.check_non_negative_int(fusion_size, 'fusion_size', self.name)
        self.fusion_size = fusion_size
        self.add_prim_attr('fusion_size', fusion_size)


class KMeansCentroids(PrimitiveWithInfer):
    """
    Calculate the segment_sum, segment_count, kmean_total_sum that are clustering results

    Args:
        use_actual_distance (bool): A bool value to decide whether do complete calculation of distance.

    Inputs:
        - **x** (Tensor(float32)) - Input data used for clustering
        - **y** (Tensor(float32)) - Initial centroids of clutering
        - **sum_square_y** (Tensor(float32)) - The result of preprocessing such as square, reduce and transpose of y
        - **sum_square_x** (Tensor(float32)) - The result of preprocessing such as square and reduce of x

    Outputs:
        - **segment_sum** (Tensor(float32)) - Clustering result w.r.t. each centroid
        - **segment_count** (Tensor(float32)) - Clustering count w.r.t. each centroid
        - **kmean_total_sum** (Tensor(float32)) - The sum of the distances from all vectors to ther nearest centroid

    Supported Platforms:
        ''Ascend''

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.common.dtype as mstype
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> from mindspore.ops import operations as P
        >>> ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

        >>> class Net(nn.Cell):
        >>>    def __init__(self):
        >>>        super(Net, self).__init__()
        >>>        self.reduce_sum = P.ReduceSUm(keep_dims=True)
        >>>        self.square = P.Square()
        >>>        self.transpose = P.Transpose()
        >>>        self.k_means_centroids = P.KMeansCentroids(True)

        >>>    def construct(self, x, y):
        >>>        p1 = self.reduce_sum(self.square(x), -1)
        >>>        p2 = self.transpose(self.reduce_sum(self.square(y), -1), (1, 0))
        >>>        return self.k_means_centroids(x, y, p2, p1)

        >>> def test_net():
        >>>    data_type = np.float32
        >>>    x = Tensor(np.random.uniform(-10, 10, (65536, 128)).astype(data_type))
        >>>    y = P.Ones()((1048576, 128), mstype.float32)
        >>>    net = Net()
        >>>    local_sum, local_count, local_avg_distance = net(x, y)
    """

    @prim_attr_register
    def __init__(self, use_actual_distance):
        validator.check_value_type('use_actual_distance', use_actual_distance, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'y', 'sum_square_y', 'sum_square_x'],
                                outputs=['segment_sum', 'segment_count', 'kmean_total_sum'])

    def infer_shape(self, x_shape, y_shape, sum_square_y_shape, sum_square_x_shape):
        """infer shape of primitive"""
        expected_shape_size = 2
        validator.check_int(len(x_shape), expected_shape_size, Rel.EQ, "dims of x", self.name)
        validator.check_int(len(y_shape), expected_shape_size, Rel.EQ, "dims of y", self.name)
        validator.check_int(len(sum_square_y_shape), expected_shape_size, Rel.EQ, "dims of sum_square_y", self.name)
        validator.check_int(len(sum_square_x_shape), expected_shape_size, Rel.EQ, "dims of sum_square_x", self.name)

        validator.check_int(x_shape[1], y_shape[1], Rel.EQ,
                            "the second dim of x and the second dim of y", self.name)
        validator.check_int(y_shape[0], sum_square_y_shape[1], Rel.EQ,
                            "the first dim of y and the second dim of sum_square_y", self.name)
        validator.check_int(x_shape[0], sum_square_x_shape[0], Rel.EQ,
                            "the first dim of x and the first dim of sum_square_x", self.name)
        validator.check_int(sum_square_y_shape[0], sum_square_x_shape[1], Rel.EQ,
                            "the first dim of sum_square_y and the first dim of sum_square_x",
                            self.name)
        validator.check_int(sum_square_y_shape[0], 1, Rel.EQ,
                            "the first dim of sum_square_y", self.name)

        k = y_shape[0]
        em_size = x_shape[1]
        return (k, em_size), (k, 1), (1)


class ClipByNorm(PrimitiveWithInfer):
    r"""
    Clips tensor values to a maximum :math:`L_2`-norm.

    Note:
        The output tensor of this operator remains the same with input tensor if the :math:`L_2`-norm of the input
        tensor is not greater than the argument `clip_norm`. Otherwise the output tensor will be normalized as:

        .. math::
            \text{output}(X) = \frac{\text{clip_norm} * X}{L_2(X)},

        where :math:`L_2(X)` is the :math:`L_2`-norm of :math:`X`.

    Args:
        axis (Union[None, int, tuple(int), list(int)]): Compute the `L_2`-norm along the specific dimension.
                                                       Default: None, all dimensions to calculate.

    Inputs:
        - **x** (Tensor) - Tensor of shape N-D. The type must be float16 or float32.
        - **clip_norm** (Tensor) - A scalar Tensor of shape :math:`()` or :math:`(1)`.
          Or a Tensor which shape can be broadcast to the shape of `x`. The type must be float16 or float32.

    Outputs:
        Tensor, clipped Tensor with the same shape as the `x`, whose type is float32.

    Raises:
        TypeError: If `axis` is not one of None, int, tuple(int) and list(int).
        TypeError: If dtype of `x` is neither float16 nor float32.
        TypeError: If dtype of `clip_norm` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore.ops.operations import _inner_ops as inner
        >>> clip_by_norm = inner.ClipByNorm()
        >>> x = Tensor(np.random.randint(0, 10, [4, 16]), mindspore.float32)
        >>> clip_norm = Tensor(np.array([100]).astype(np.float32))
        >>> output = clip_by_norm(x, clip_norm)
        >>> print(output.shape)
        (4, 16)
    """

    @prim_attr_register
    def __init__(self, axis=None):
        """Initialize ClipByNorm"""
        self.axis = () if axis is None else axis
        validator.check_value_type('axis', self.axis, [int, tuple, list], self.name)
        axis_check = self.axis if isinstance(self.axis, Iterable) else (self.axis,)
        for i, value in enumerate(axis_check):
            validator.check_value_type('axis[%d]' % i, value, [int], self.name)
        self.init_attrs['axis'] = self.axis
        self.add_prim_attr('axis', self.axis)
        self.init_prim_io_names(inputs=['x', 'clip_norm'], outputs=['output'])

    def infer_shape(self, x_shape, clip_norm_shape):
        """Infer shape for ClipByNorm"""
        x_dim = len(x_shape)
        axis = self.axis if isinstance(self.axis, Iterable) else (self.axis,)
        for _, value in enumerate(axis):
            validator.check_int_range(value, -x_dim, x_dim, Rel.INC_LEFT, 'axis', self.name)
        return x_shape

    def infer_dtype(self, x_type, clip_norm_type):
        """Infer data type for ClipByNorm"""
        validator.check_tensor_dtype_valid("x_type", x_type, [mstype.float16, mstype.float32], self.name)
        validator.check_tensor_dtype_valid("clip_norm_type", clip_norm_type,
                                           [mstype.float16, mstype.float32], self.name)
        return mstype.float32


class TopTypeof(Primitive):
    """
        Internal primitive method, to speed up mindspore.ops.typeof.

        Returns the top type of the input data.

        In Pynative mode, returns the top type in cache.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self):
        self.prim = Primitive('TopTypeof')
        self.typeof_cache = {
            'slice': mstype.Slice(),
            'list': mstype.List(),
            'tuple': mstype.Tuple(),
            'Tensor': mstype.tensor,
            'NoneType': mstype.none_type(),
            'int': mstype.Int(),
            'bool': mstype.Bool(),
            'ellipsis': mstype.Ellipsis_(),
            'dict': mstype.Dict()
        }

    def __call__(self, x):
        index_type = type(x).__name__
        if 'Tensor' in index_type:
            index_type = 'Tensor'
        if index_type in self.typeof_cache:
            return self.typeof_cache.get(index_type)
        return _pynative_executor.constant_folding(self.prim, x)


class MixedPrecisionCast(Primitive):
    r"""
    Internal primitive method, to achieve mindspore.functional.mixed_precision_cast.

    Note:
        This internal primitive method used to do mixed precision conversion.
        Only the input object with float dtype will be cast.

    Inputs:
        - **dtype** (Union[Float16, Float32]) - The data type of the output object.
        - **input** (Union[Tensor, Tuple, Dictionary, KeywordArg]) - The object to be cast.

    Outputs:
        Object, its dtype is the same as `dtype` and shape is the same as 'input'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.common import dtype as mstype
        >>> from mindspore.ops.operations import _inner_ops as inner
        >>> x = Tensor(np.ones([2, 3], dtype=np.float32))
        >>> out = inner.MixedPrecisionCast(mstype.float16, x)
        >>> print(out.dtype)
        Float16
    """

    @prim_attr_register
    def __init__(self):
        """Initialize MixedPrecisionCast"""
        self.init_prim_io_names(inputs=['dst_dtype', 'input_x'], outputs=['output'])
        self.cast = Cast()
        self.hyper_map = C.HyperMap()

    def __call__(self, dst_dtype, x):
        def cast_inner(data):
            if isinstance(data, Tensor) and data.dtype in (mstype.float16, mstype.float32, mstype.float64):
                return self.cast(data, dst_dtype)
            return data

        return self.hyper_map(cast_inner, x)


class CheckBprop(PrimitiveWithInfer):
    """
    Checks whether the data type and the shape of corresponding elements from tuples x and y are the same.

    Args:
        prim_to_check (str): The name of the primitive being checked. Default: ''.

    Inputs:
        - **input_x** (tuple[Tensor]) - The `input_x` contains the outputs of bprop to be checked.
        - **input_y** (tuple[Tensor]) - The `input_y` contains the inputs of bprop to check against.

    Outputs:
        Tuple[Tensor], the `input_x`,
        if data type and shape of corresponding elements from `input_x` and `input_y` are the same.

    Raises:
        TypeError: If `input_x` or `input_y` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.op = ops.CheckBprop()
        ...     def construct(self, x, y):
        ...         return self.op(x, y)
        ...
        >>> net = Net()
        >>> input_x = (Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32),)
        >>> input_y = (Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32),)
        >>> output = net(input_x, input_y)
        >>> print(output)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 2.00000000e+00,  2.00000000e+00],
         [ 2.00000000e+00,  2.00000000e+00]]),)
    """

    @prim_attr_register
    def __init__(self, prim_to_check=""):
        """Initialize CheckBprop"""
        self.prim_to_check = prim_to_check

    def infer_shape(self, xshapes, yshapes):
        """infer shape"""
        tips = f"user defined method 'bprop'"
        validator.check_value_type('grads', xshapes, (tuple,), tips)
        validator.check_value_type('params', yshapes, (tuple,), tips)
        if not len(xshapes) == len(yshapes):
            raise ValueError(f"For {tips} the number of return values(gradients) must be equal to "
                             f"the number of input arguments except 'out' and 'dout', "
                             f"which is:{len(yshapes)} but got {len(xshapes)}.")
        checking_range = len(yshapes)
        for i in range(checking_range):
            xshape = xshapes[i]
            yshape = yshapes[i]
            if not xshape or not yshape:
                continue
            if xshape != yshape:
                raise ValueError(f"For {tips}, the {i}th return value(gradient of the {i}th argument) "
                                 f"should have the same shape as the {i}th argument, "
                                 f"which is:{yshape}, but got: {xshape}.")
        return xshapes

    def infer_dtype(self, xdtypes, ydtypes):
        """infer dtype"""
        tips = f"user defined method 'bprop'"
        validator.check_value_type('grads', xdtypes, (tuple,), tips)
        validator.check_value_type('params', ydtypes, (tuple,), tips)
        if not len(xdtypes) == len(ydtypes):
            raise ValueError(f"For {tips}, the number of return values(gradients) must be equal to "
                             f"the number of input arguments except 'out' and 'dout', "
                             f"which is:{len(ydtypes)} but got {len(xdtypes)}.")
        checking_range = len(ydtypes)
        for i in range(checking_range):
            xdtype = xdtypes[i]
            ydtype = ydtypes[i]
            if isinstance(xdtype, mstype.anything_type) or isinstance(ydtype, mstype.anything_type):
                continue
            if isinstance(ydtype, mstype.function_type):
                if not isinstance(xdtype, mstype.env_type_type):
                    raise TypeError(f"For {tips}, the {i}th return value(gradient of the {i}th argument) type "
                                    f"should be {mstype.env_type_type}, but got {xdtype}.")
            if xdtype != ydtype:
                raise TypeError(f"For {tips}, the {i}th return value(gradient of the {i}th argument) "
                                f"should have the same dtype as the {i}th argument, "
                                f"which is:{ydtype}, but got: {xdtype}.")
        return xdtypes

check_bprop = CheckBprop()


class SameTypeShape(PrimitiveWithInfer):
    """
    Checks whether the data type and shape of two tensors are the same.

    Refer to :func:`mindspore.ops.same_type_shape` for more detail.

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

same_type_shape_ = SameTypeShape()


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

        value = mstype._issubclass_(sub_type_t, type_v)  # pylint: disable=W0212

        out = {'shape': (),
               'dtype': mstype.type_type,
               'value': value}
        return out


issubclass_ = IsSubClass()


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
            value = mstype._issubclass_(sub_type_t, type_v)  # pylint: disable=W0212

        out = {'shape': (),
               'dtype': mstype.type_type,
               'value': value}
        return out


class ConvertToAdapterTensor(Primitive):
    """
    Convert a tensor from MindSpore's Tensor type to MSAdapter's Tensor type,
    where MSAdapter's Tensor is a subclass of MindSpore's Tensor.

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        A tensor, whose type is MSAdapter's Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([1, 2 ,3])
        >>> x = ops.ConvertToAdapterTensor()(x)
        >>> print(x)
        [1 2 3]
    """
    @prim_attr_register
    def __init__(self):
        """Initialize"""

    def __call__(self, x):
        """run in PyNative mode"""
        return ms_adapter_registry.tensor(x, inner=True)

convert_to_adapter_tensor = ConvertToAdapterTensor()


class ConvertToMsTensor(Primitive):
    """
    Convert a tensor from MSAdapter's Tensor type to MindSpore's Tensor type,
    where MSAdapter's Tensor is a subclass of MindSpore's Tensor.

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        A tensor, whose type is MindSpore's Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([1, 2 ,3])
        >>> x = ops.ConvertToMsTensor()(x)
        >>> print(x)
        [1 2 3]
    """
    @prim_attr_register
    def __init__(self):
        """Initialize"""

    def __call__(self, x):
        """run in PyNative mode"""
        return Tensor(x)

convert_to_ms_tensor = ConvertToMsTensor()


class GetGrad(Primitive):
    """
        Use the position id or Parameter object to get the gradient from the output
        which returned by the :func:`mindspore.ops.grad`.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ScatterElements"""
        self.init_prim_io_names(
            inputs=['gradients', 'x'], outputs=['gradient'])

    def __call__(self, gradients, x):
        if not isinstance(x, int) and not isinstance(x, Parameter):
            raise TypeError(
                f"For `get_grad`, the `x` should be an integer or a Parameter, but got {x}")
        hash_id = x
        if isinstance(x, Parameter):
            hash_id = x.name
        output = None

        def _get_grad(grads, identifier):
            if isinstance(grads, tuple):
                if len(grads) != 2 or identifier != grads[0]:
                    for gradient in grads:
                        _get_grad(gradient, identifier)
                else:
                    nonlocal output
                    output = grads[1]
                    return

        _get_grad(gradients, hash_id)
        if output is None:
            raise ValueError(
                f"Can not find the gradient for position or Parameter {x}")
        return output


class IsParameter(PrimitiveWithInfer):
    """
        Check if input is `Parameter`
    """
    @prim_attr_register
    def __init__(self):
        """Initialize IsParameter"""

    def __call__(self, x):
        return isinstance(x, Parameter)

    def __infer__(self, x):
        return {'shape': [],
                'dtype': mstype.bool_,
                'value': isinstance(x['dtype'], mstype.ref_type)}
