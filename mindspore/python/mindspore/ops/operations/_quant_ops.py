# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http:  // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Operators for quantization."""
from __future__ import absolute_import
from functools import partial

import mindspore.context as context
from mindspore import _checkparam as validator
from mindspore.ops.primitive import Primitive, PrimitiveWithInfer, prim_attr_register
from mindspore.common import dtype as mstype
from mindspore.common.dtype import QuantDtype


def _support_te():
    try:
        import te  # pylint: disable=unused-import
        return True
    # pylint: disable=broad-except
    except Exception:
        return False

if context.get_context('device_target') == "Ascend" and _support_te():
    import mindspore.ops._op_impl._custom_op

__all__ = ["MinMaxUpdatePerLayer",
           "MinMaxUpdatePerChannel",
           "FakeLearnedScaleQuantPerLayer",
           "FakeLearnedScaleQuantPerLayerGrad",
           "FakeLearnedScaleQuantPerLayerGradD",
           "FakeLearnedScaleQuantPerLayerGradDReduce",
           "FakeLearnedScaleQuantPerChannel",
           "FakeLearnedScaleQuantPerChannelGrad",
           "FakeLearnedScaleQuantPerChannelGradD",
           "FakeLearnedScaleQuantPerChannelGradDReduce",
           "FakeQuantWithMinMaxVars",
           "FakeQuantWithMinMaxVarsGradient",
           "FakeQuantWithMinMaxVarsPerChannel",
           "FakeQuantWithMinMaxVarsPerChannelGradient",
           "FakeQuantPerLayer",
           "FakeQuantPerLayerGrad",
           "FakeQuantPerChannel",
           "FakeQuantPerChannelGrad",
           "BatchNormFold",
           "BatchNormFoldGrad",
           "CorrectionMul",
           "CorrectionMulGrad",
           "CorrectionMulGradReduce",
           "BatchNormFold2",
           "BatchNormFold2Grad",
           "BatchNormFoldD",
           "BatchNormFoldGradD",
           "BatchNormFold2D",
           "BatchNormFold2GradD",
           "BatchNormFold2GradReduce",
           "IFMR",
           "ActsULQ",
           "ActsULQInputGrad",
           "ActULQClampMinGrad",
           "ActULQClampMaxGrad",
           "WtsARQ",
           "FakeQuantParam",
           ]


class FakeQuantParam(Primitive):
    r"""
    Define the operation for storing quant parameter. This operation passes through input tensor to output tensor
        without any calculation.

    Args:
        quant_dtype (QuantDtype) - The valid data type of the input tensor.
        quant_algo_name (str) - Define the name of quant algorithm. Use
            `FakeQuantParam.attr_value_linear_quant_algo_name` for linear quantization specially.
        is_per_channel (bool) - Define whether quant parameter is per-channel or per-layer.
        kwargs (dict): Other quant parameter in key-value form. Please use classmethod `linear_quant_param` to create a
            linear quantization specially because key of scale and zero-point is pre-defined by MindSpore.

    Inputs:
        - *input_x* (Tensor) : Input tensor.

    Outputs:
        - Tensor: Output tensor same with `input_x`.

    Examples:
        >>> input_tensor = mindspore.Tensor(numpy.random.rand(1, 16, 5, 5), mindspore.dtype.float32)
        >>> fake_quant_param_op = FakeQuantParam.linear_quant_param(mindspore.common.dtype.QuantDtype.INT8,
        >>>                                                         0.5, 1)
        >>> output_tensor = fake_quant_param_op(input_tensor)
    """

    attr_key_linear_quant_scale = "linear_quant_scale"
    attr_key_linear_quant_zero_point = "linear_quant_zero_point"

    attr_value_linear_quant_algo_name = "linear_quant_algo"

    @prim_attr_register
    def __init__(self, quant_dtype: QuantDtype, quant_algo_name: str, is_per_channel: bool, **kwargs):
        self.add_prim_attr("quant_algo_name", quant_algo_name)
        self.add_prim_attr("is_per_channel", is_per_channel)
        self.add_prim_attr("quant_dtype", quant_dtype.value())
        for key, value in kwargs.items():
            self.add_prim_attr(key, value)

    @classmethod
    def linear_quant_param(cls, quant_dtype, scale, zp, is_per_channel=False, **kwargs):
        """
        Create a linear quantization operator based on scale and zero-point parameter.
        """
        validator.check_value_type("scale", scale, [float, tuple, list], "FakeQuantParam")
        if isinstance(scale, float):
            scale_list = [scale]
        else:
            scale_list = scale
        validator.check_value_type("zero_point", zp, [int, tuple, list], "FakeQuantParam")
        if isinstance(zp, int):
            zp_list = [zp]
        else:
            zp_list = zp
        validator.check_value_type("is_per_channel", is_per_channel, [bool], "FakeQuantParam")
        kwargs[FakeQuantParam.attr_key_linear_quant_scale] = scale_list
        kwargs[FakeQuantParam.attr_key_linear_quant_zero_point] = zp_list
        return cls(quant_dtype, FakeQuantParam.attr_value_linear_quant_algo_name, is_per_channel, **kwargs)


class MinMaxUpdatePerLayer(PrimitiveWithInfer):
    r"""
    Updates min and max per layer.

    Args:
        ema (bool): Uses EMA algorithm update value min and max. Default: ``False``.
        ema_decay (int) : EMA algorithm decay parameter. Default: 0.999.

    Inputs:
        - **x** (Tensor) : float32 Tensor representing the shape of the output tensor.
        - **min** (Tensor) : Value of the min range of the input data x.
        - **max** (Tensor) : Value of the max range of the input data x.

    Outputs:
        - Tensor: Simulates quantize tensor of x.

    Examples:
        >>> input_tensor = Tensor(np.random.rand(3, 16, 5, 5), mstype.float32)
        >>> min_tensor = Tensor(np.array([-6]), mstype.float32)
        >>> max_tensor = Tensor(np.array([6]), mstype.float32)
        >>> output_tensor = MinMaxUpdatePerLayer(num_bits=8)(input_tensor, min_tensor, max_tensor)
    """
    support_quant_bit = [4, 7, 8]

    @prim_attr_register
    def __init__(self, ema=False, ema_decay=0.999):
        """Initialize FakeQuantMinMaxPerLayerUpdate OP"""
        if context.get_context('device_target') == "Ascend":
            from mindspore.ops._op_impl._custom_op import minmax_update_perlayer
        if ema and not ema_decay:
            raise ValueError(
                f"For '{self.name}' attr \'ema\' and \'ema_decay\' should set together.")

        self.ema = validator.check_value_type('ema', ema, (bool,), self.name)
        self.ema_decay = validator.check_float_range(ema_decay, 0, 1, validator.INC_BOTH, 'ema_decay', self.name)
        self.init_prim_io_names(inputs=['x', 'min', 'max'],
                                outputs=['min_up', 'max_up'])

    def infer_shape(self, x_shape, min_shape, max_shape):
        validator.check_int(len(x_shape), 1, validator.GE, "x rank", self.name)
        validator.check("min shape", min_shape, "max shape",
                        max_shape, validator.EQ, self.name)
        validator.check_equal_int(len(min_shape), 1, "min shape", self.name)
        return min_shape, max_shape

    def infer_dtype(self, x_type, min_type, max_type):
        tuple(map(partial(validator.check_tensor_dtype_valid,
                          valid_dtypes=(mstype.float16, mstype.float32), prim_name=self.name),
                  ("x", "min", "max"),
                  (x_type, min_type, max_type)))
        return min_type, max_type


class MinMaxUpdatePerChannel(PrimitiveWithInfer):
    r"""
     Updates min and max per channel.

    Args:
        ema (bool): Uses EMA algorithm update value min and max. Default: ``False``.
        ema_decay (int) : EMA algorithm decay parameter. Default: 0.999.
        channel_axis (int): Quantization by channel axis. Ascend backend only supports 0 or 1. Default: 1.

    Inputs:
        - **x** (Tensor) : float32 Tensor representing the shape of the output tensor.
        - **min** (Tensor) : Value of the min range of the input data x.
        - **max** (Tensor) : Value of the max range of the input data x.

    Outputs:
        - Tensor: Simulates quantize tensor of x.

    Examples:
        >>> x = Tensor(np.random.rand(3, 16, 5, 5), mstype.float32)
        >>> min_value = Tensor(np.random.uniform(-1, 1, size=16), mstype.float32)
        >>> max_value = Tensor(np.random.uniform(-1, 1, size=16), mstype.float32)
        >>> output_tensor = MinMaxUpdatePerChannel(num_bits=8)(x, min_value, max_value)
    """
    support_quant_bit = [4, 7, 8]
    ascend_support_x_rank = [2, 4]

    @prim_attr_register
    def __init__(self, ema=False, ema_decay=0.999, channel_axis=1):
        """Initialize FakeQuantPerChannelUpdate OP for Ascend"""
        self.is_ascend = context.get_context('device_target') == "Ascend"
        if self.is_ascend:
            from mindspore.ops._op_impl._custom_op import minmax_update_perchannel
        if ema and not ema_decay:
            raise ValueError(
                f"For '{self.name}' attr \'ema\' and \'ema_decay\' should set together.")

        self.ema = validator.check_value_type('ema', ema, (bool,), self.name)
        self.ema_decay = validator.check_float_range(ema_decay, 0, 1, validator.INC_BOTH, 'ema_decay', self.name)
        if self.is_ascend:
            self.channel_axis = validator.check_int_range(channel_axis, 0, 1, validator.INC_BOTH,
                                                          'channel_axis', self.name)
        else:
            self.channel_axis = validator.check_non_negative_int(channel_axis, 'channel_axis', self.name)
        self.init_prim_io_names(
            inputs=['x', 'min', 'max'], outputs=['min_up', 'max_up'])

    def infer_shape(self, x_shape, min_shape, max_shape):
        if self.is_ascend and len(x_shape) not in self.ascend_support_x_rank:
            raise ValueError(f"For '{self.name}' x rank must be in '{self.ascend_support_x_rank}'")
        if not self.is_ascend:
            validator.check_int(len(x_shape), 1, validator.GE, "x rank", self.name)
        validator.check("min shape", min_shape, "max shape",
                        max_shape, validator.EQ, self.name)
        validator.check_equal_int(len(min_shape), 1, "min shape", self.name)
        return min_shape, max_shape

    def infer_dtype(self, x_type, min_type, max_type):
        tuple(map(partial(validator.check_tensor_dtype_valid,
                          valid_dtypes=(mstype.float16, mstype.float32), prim_name=self.name),
                  ("x", "min", "max"),
                  (x_type, min_type, max_type)))
        return min_type, max_type


class FakeLearnedScaleQuantPerLayer(PrimitiveWithInfer):
    r"""
    Simulates the quantize and dequantize operations of the fake learned scale quant per-layer case in training time.

    Args:
        quant_delay (int): Quantilization delay parameter. Before delay step in training time not update
            simulate quantization aware function. After delay step in training time begin simulate the aware
            quantize function. Default: 0.
        neg_trunc (bool): Whether the quantization algorithm uses negative truncation or not. Default: ``False``.
        training (bool): Training the network or not. Default: ``True``.

    Inputs:
        - **input_x** (Tensor) : Input tensor that needs to be quantified.
        - **alpha** (Tensor) : Value of the max clipping range of the input data `input_x`.
        - **quant_max** (Tensor) : Value of the quantization range.

    Outputs:
        - Tensor: Simulates quantize tensor of `input_x`, with the same type and shape as the `input_x`.

    Examples:
        >>> input_tensor = Tensor(np.random.rand(3, 16, 5, 5), mstype.float32)
        >>> alpha_tensor = Tensor(np.array([6]), mstype.float32)
        >>> quant_max_tensor = Tensor(np.array([127]), mstype.float32)
        >>> output_tensor = FakeLearnedScaleQuantPerLayer()(input_tensor, alpha_tensor, quant_max_tensor)
    """
    @prim_attr_register
    def __init__(self,
                 quant_delay=0,
                 neg_trunc=False,
                 training=True):
        """init FakeLearnedScaleQuantPerLayer OP"""
        if context.get_context('device_target') == "Ascend":
            from mindspore.ops._op_impl._custom_op import fake_learned_scale_quant_perlayer

        self.quant_delay = validator.check_non_negative_int(
            quant_delay, 'quant_delay', self.name)
        self.neg_trunc = validator.check_value_type(
            'neg_trunc', neg_trunc, (bool,), self.name)
        self.training = validator.check_value_type(
            'training', training, (bool,), self.name)
        self.init_prim_io_names(inputs=['input_x', 'alpha', 'quant_max'],
                                outputs=['out'])

    def infer_shape(self, input_x_shape, alpha_shape, quant_max_shape):
        validator.check_int(len(input_x_shape), 1, validator.GE, "input_x rank", self.name)
        validator.check_int(len(alpha_shape), 1, validator.GE, "alpha rank", self.name)
        validator.check_int(len(quant_max_shape), 1, validator.GE, "quant max rank", self.name)
        return input_x_shape

    def infer_dtype(self, input_x_type, alpha_type, quant_max_type):
        if context.get_context('device_target') == "GPU":
            valid_dtypes = (mstype.float32,)
        else:
            valid_dtypes = (mstype.float16, mstype.float32)
        tuple(map(partial(validator.check_tensor_dtype_valid, valid_dtypes=valid_dtypes, prim_name=self.name),
                  ("input_x", "alpha", "quant_max"),
                  (input_x_type, alpha_type, quant_max_type)))
        return input_x_type


class FakeLearnedScaleQuantPerLayerGrad(PrimitiveWithInfer):
    r"""
    Performs grad of FakeLearnedScaleQuantPerLayer operation.

    Examples:
        >>> fake_learned_scale_grad = FakeLearnedScaleQuantPerLayerGrad()
        >>> dout = Tensor(np.array([[-2.3, 1.2], [5.7, 0.2]]), mindspore.float32)
        >>> input_x = Tensor(np.array([[18, -23], [0.2, 6]]), mindspore.float32)
        >>> _alpha = Tensor(np.array([6]), mindspore.float32)
        >>> _quant_max = Tensor(np.array([127]), mindspore.float32)
        >>> result = fake_learned_scale_grad(dout, input_x, _min, _max)
    """

    @prim_attr_register
    def __init__(self,
                 quant_delay=0,
                 neg_trunc=False):
        self.quant_delay = validator.check_non_negative_int(
            quant_delay, 'quant_delay', self.name)
        self.neg_trunc = validator.check_value_type(
            'neg_trunc', neg_trunc, (bool,), self.name)
        self.init_prim_io_names(
            inputs=['dout', 'x', 'alpha', 'quant_max'], outputs=['dx', 'dalpha'])

    def infer_shape(self, dout_shape, x_shape, alpha_shape, quant_max_shape):
        validator.check("dout shape", dout_shape, "x_shape", x_shape, validator.EQ, self.name)
        validator.check_int(len(alpha_shape), 1, validator.GE, "alpha rank", self.name)
        validator.check_int(len(quant_max_shape), 1, validator.GE, "quant max rank", self.name)
        return dout_shape, alpha_shape

    def infer_dtype(self, dout_type, x_type, alpha_type, quant_max_type):
        if context.get_context('device_target') == "GPU":
            valid_dtypes = (mstype.float32,)
        else:
            valid_dtypes = (mstype.float16, mstype.float32)
        tuple(map(partial(validator.check_tensor_dtype_valid, valid_dtypes=valid_dtypes, prim_name=self.name),
                  ("dout", "x", "alpha", "quant_max"),
                  (dout_type, x_type, alpha_type, quant_max_type)))
        return dout_type, alpha_type


class FakeLearnedScaleQuantPerLayerGradD(PrimitiveWithInfer):
    r"""
    Performs input grad of FakeLearnedScaleQuantPerLayer operation.
    """

    @prim_attr_register
    def __init__(self,
                 neg_trunc=False):
        from mindspore.ops._op_impl._custom_op import fake_learned_scale_quant_perlayer_grad
        self.neg_trunc = validator.check_value_type(
            'neg_trunc', neg_trunc, (bool,), self.name)
        self.init_prim_io_names(
            inputs=['dout', 'x', 'alpha', 'quant_max'], outputs=['dx', 'dalpha'])

    def infer_shape(self, dout_shape, x_shape, alpha_shape, quant_max_shape):
        validator.check("dout shape", dout_shape, "x_shape", x_shape, validator.EQ, self.name)
        validator.check_int(len(alpha_shape), 1, validator.GE, "alpha rank", self.name)
        validator.check_int(len(quant_max_shape), 1, validator.GE, "quant max rank", self.name)
        return dout_shape, dout_shape

    def infer_dtype(self, dout_type, x_type, alpha_type, quant_max_type):
        valid_dtypes = (mstype.float16, mstype.float32)
        tuple(map(partial(validator.check_tensor_dtype_valid, valid_dtypes=valid_dtypes, prim_name=self.name),
                  ("dout", "x", "alpha", "quant_max"),
                  (dout_type, x_type, alpha_type, quant_max_type)))
        return dout_type, dout_type


class FakeLearnedScaleQuantPerLayerGradDReduce(PrimitiveWithInfer):
    r"""
    Performs alpha grad reduce of FakeLearnedScaleQuantPerLayer operation.
    """

    @prim_attr_register
    def __init__(self):
        from mindspore.ops._op_impl._custom_op import fake_learned_scale_quant_perlayer_grad_reduce
        self.init_prim_io_names(
            inputs=['dout_alpha'], outputs=['dalpha'])

    def infer_shape(self, dout_alpha_shape):
        return (1,)

    def infer_dtype(self, dout_alpha_type):
        valid_dtypes = (mstype.float16, mstype.float32)
        validator.check_tensor_dtype_valid("dout_alpha", dout_alpha_type, valid_dtypes, self.name)
        return dout_alpha_type


class FakeLearnedScaleQuantPerChannel(PrimitiveWithInfer):
    r"""
    Simulates the quantize and dequantize operations of the fake learned scale quant per-channel case in training time.

    Args:
        quant_delay (int): Quantilization delay parameter. Before delay step in training time not update
            simulate quantization aware function. After delay step in training time begin simulate the aware
            quantize function. Default: 0.
        neg_trunc (bool): Whether the quantization algorithm uses negative truncation or not. Default: ``False``.
        training (bool): Training the network or not. Default: ``True``.
        channel_axis (int): Quantization by channel axis. Ascend backend only supports 0 or 1. Default: 1.

    Inputs:
        - **input_x** (Tensor) : Input tensor that needs to be quantified.
        - **alpha** (Tensor) : Value of the max clipping range of the input data `input_x`.
        - **quant_max** (Tensor) : Value of the quantization range.

    Outputs:
        - Tensor: Simulates quantize tensor of `input_x`, with the same type and shape as the `input_x`.

    Examples:
        >>> input_tensor = Tensor(np.random.rand(3, 16, 5, 5), mstype.float32)
        >>> alpha_tensor = Tensor(np.array([6]*3), mstype.float32)
        >>> quant_max_tensor = Tensor(np.array([127]), mstype.float32)
        >>> output_tensor = FakeLearnedScaleQuantPerChannel()(input_tensor, alpha_tensor, quant_max_tensor)
    """
    ascend_support_x_rank = [2, 4]

    @prim_attr_register
    def __init__(self,
                 quant_delay=0,
                 neg_trunc=False,
                 training=True,
                 channel_axis=1):
        """init FakeLearnedScaleQuantPerChannel OP"""
        if context.get_context('device_target') == "Ascend":
            from mindspore.ops._op_impl._custom_op import fake_learned_scale_quant_perchannel
        self.is_ascend = context.get_context('device_target') == "Ascend"
        self.quant_delay = validator.check_non_negative_int(
            quant_delay, 'quant_delay', self.name)
        self.neg_trunc = validator.check_value_type(
            'neg_trunc', neg_trunc, (bool,), self.name)
        self.training = validator.check_value_type(
            'training', training, (bool,), self.name)
        if self.is_ascend:
            self.channel_axis = validator.check_int_range(channel_axis, 0, 1, validator.INC_BOTH,
                                                          'channel_axis', self.name)
        else:
            self.channel_axis = validator.check_non_negative_int(channel_axis, 'channel_axis', self.name)
        self.init_prim_io_names(inputs=['input_x', 'alpha', 'quant_max'],
                                outputs=['out'])

    def infer_shape(self, input_x_shape, alpha_shape, quant_max_shape):
        if self.is_ascend and len(input_x_shape) not in self.ascend_support_x_rank:
            raise ValueError(f"For '{self.name}' x rank must be in '{self.ascend_support_x_rank}'")
        if not self.is_ascend:
            validator.check_int(len(input_x_shape), 1, validator.GE, "input_x rank", self.name)
        if len(input_x_shape) == 1:
            self.channel_axis = 0

        validator.check_equal_int(alpha_shape[0], input_x_shape[self.channel_axis], "alpha rank", self.name)
        validator.check_int(len(quant_max_shape), 1, validator.GE, "quant max rank", self.name)
        return input_x_shape

    def infer_dtype(self, input_x_type, alpha_type, quant_max_type):
        if context.get_context('device_target') == "GPU":
            valid_dtypes = (mstype.float32,)
        else:
            valid_dtypes = (mstype.float16, mstype.float32)
        tuple(map(partial(validator.check_tensor_dtype_valid, valid_dtypes=valid_dtypes, prim_name=self.name),
                  ("input_x", "alpha", "quant_max"),
                  (input_x_type, alpha_type, quant_max_type)))
        return input_x_type


class FakeLearnedScaleQuantPerChannelGrad(PrimitiveWithInfer):
    r"""
    Performs grad of FakeLearnedScaleQuantPerChannel operation.

    Examples:
        >>> fake_learned_scale_grad = FakeLearnedScaleQuantPerChannelGrad()
        >>> dout = Tensor(np.array([[-2.3, 1.2], [5.7, 0.2]]), mindspore.float32)
        >>> input_x = Tensor(np.array([[18, -23], [0.2, 6]]), mindspore.float32)
        >>> _alpha = Tensor(np.array([6]*2), mindspore.float32)
        >>> _quant_max = Tensor(np.array([127]), mindspore.float32)
        >>> result = fake_learned_scale_grad(dout, input_x, _min, _max)
    """

    @prim_attr_register
    def __init__(self,
                 quant_delay=0,
                 neg_trunc=False,
                 channel_axis=1):
        self.quant_delay = validator.check_non_negative_int(
            quant_delay, 'quant_delay', self.name)
        self.neg_trunc = validator.check_value_type(
            'neg_trunc', neg_trunc, (bool,), self.name)
        self.channel_axis = validator.check_non_negative_int(channel_axis, 'channel axis', self.name)
        self.init_prim_io_names(
            inputs=['dout', 'x', 'alpha', 'quant_max'], outputs=['dx', 'dalpha'])

    def infer_shape(self, dout_shape, x_shape, alpha_shape, quant_max_shape):
        validator.check("dout shape", dout_shape, "x_shape", x_shape, validator.EQ, self.name)
        return dout_shape, alpha_shape

    def infer_dtype(self, dout_type, x_type, alpha_type, quant_max_type):
        if context.get_context('device_target') == "GPU":
            valid_dtypes = (mstype.float32,)
        else:
            valid_dtypes = (mstype.float16, mstype.float32)
        tuple(map(partial(validator.check_tensor_dtype_valid, valid_dtypes=valid_dtypes, prim_name=self.name),
                  ("dout", "x", "alpha", "quant_max"),
                  (dout_type, x_type, alpha_type, quant_max_type)))
        return dout_type, alpha_type


class FakeLearnedScaleQuantPerChannelGradD(PrimitiveWithInfer):
    r"""
    Performs input grad of FakeLearnedScaleQuantPerChannel operation.
    """

    @prim_attr_register
    def __init__(self,
                 neg_trunc=False,
                 channel_axis=1):
        from mindspore.ops._op_impl._custom_op import fake_learned_scale_quant_perchannel_grad
        self.neg_trunc = validator.check_value_type(
            'neg_trunc', neg_trunc, (bool,), self.name)
        self.channel_axis = validator.check_non_negative_int(channel_axis, 'channel axis', self.name)
        self.init_prim_io_names(
            inputs=['dout', 'x', 'alpha', 'quant_max'], outputs=['dx', 'dalpha'])

    def infer_shape(self, dout_shape, x_shape, alpha_shape, quant_max_shape):
        validator.check("dout shape", dout_shape, "x_shape", x_shape, validator.EQ, self.name)
        validator.check_int(len(alpha_shape), 1, validator.GE, "alpha rank", self.name)
        validator.check_int(len(quant_max_shape), 1, validator.GE, "quant max rank", self.name)
        return dout_shape, dout_shape

    def infer_dtype(self, dout_type, x_type, alpha_type, quant_max_type):
        valid_dtypes = (mstype.float16, mstype.float32)
        tuple(map(partial(validator.check_tensor_dtype_valid, valid_dtypes=valid_dtypes, prim_name=self.name),
                  ("dout", "x", "alpha", "quant_max"),
                  (dout_type, x_type, alpha_type, quant_max_type)))
        return dout_type, dout_type


class FakeLearnedScaleQuantPerChannelGradDReduce(PrimitiveWithInfer):
    r"""
    Performs alpha grad reduce of FakeLearnedScaleQuantPerChannel operation.
    """

    @prim_attr_register
    def __init__(self, channel_axis=1):
        from mindspore.ops._op_impl._custom_op import fake_learned_scale_quant_perchannel_grad_reduce
        self.channel_axis = validator.check_non_negative_int(channel_axis, 'channel axis', self.name)
        self.init_prim_io_names(
            inputs=['dout_alpha'], outputs=['dalpha'])

    def infer_shape(self, dout_alpha_shape):
        return (dout_alpha_shape[self.channel_axis],)

    def infer_dtype(self, dout_alpha_type):
        valid_dtypes = (mstype.float16, mstype.float32)
        validator.check_tensor_dtype_valid("dout_alpha", dout_alpha_type, valid_dtypes, self.name)
        return dout_alpha_type


class FakeQuantWithMinMaxVars(PrimitiveWithInfer):
    r"""
    Fake-quantize the input by min and max.

    Args:
        num_bits (int): Quantization bitwidth; between 2 and 16. Default: 8.
        narrow_range (bool): Whether the quantization algorithm uses narrow range or not.
            if True, the quantization range is [0, 2^num_bits-1]. Otherwise, the quantization
            range is [1, 2^num_bits-1]. Default: ``False``.

    Inputs:
        - **x** (Tensor) - Float32 tensor representing the shape of the output tensor.
        - **min** (Tensor) - Value of the min range of the input data x.
        - **max** (Tensor) - Value of the max range of the input data x.

    Outputs:
        - Tensor, the data type and shape of output tensor is the same as input x.

    Examples:
        >>> input_tensor = Tensor(np.random.rand(3, 16, 5, 5), mstype.float32)
        >>> min_tensor = Tensor(np.array([-6]), mstype.float32)
        >>> max_tensor = Tensor(np.array([6]), mstype.float32)
        >>> output_tensor = FakeQuantWithMinMaxVars(num_bits=8, narrow_range=False)(
        ...                 input_tensor, min_tensor, max_tensor)
        >>> output_tensor # shape: (3, 16, 5, 5)  data type: mstype.float32
    """

    @prim_attr_register
    def __init__(self,
                 num_bits=8,
                 narrow_range=False):
        self.num_bits = validator.check_positive_int(num_bits, 'num_bits', self.name)
        self.num_bits = validator.check_int_range(self.num_bits, 2, 16, validator.INC_BOTH, 'num_bits', self.name)
        self.narrow_range = validator.check_value_type(
            'narrow_range', narrow_range, (bool,), self.name)

    def check_broadcast(self, min_shape, input_shape):
        shape_val = 1
        for shape in input_shape:
            shape_val = shape_val * shape
        if min_shape[0] > 1 and min_shape[0] != shape_val:
            raise ValueError(f"For '{self.name}', the shape of \'min\' cannot broadcast to the shape of \'x\'.")

    def infer_shape(self, x_shape, min_shape, max_shape):
        validator.check_int(len(x_shape), 1, validator.GE, "x rank", self.name)
        validator.check("min shape", min_shape, "max shape", max_shape, validator.EQ, self.name)
        validator.check_int(len(min_shape), 1, validator.EQ, "min shape", self.name)
        self.check_broadcast(min_shape, x_shape)
        return x_shape

    def infer_dtype(self, x_type, min_type, max_type):
        tuple(map(partial(validator.check_tensor_dtype_valid,
                          valid_dtypes=(mstype.float16, mstype.float32), prim_name=self.name),
                  ("x", "min", "max"),
                  (x_type, min_type, max_type)))
        return x_type


class FakeQuantWithMinMaxVarsGradient(PrimitiveWithInfer):
    r"""
    Performs grad of FakeQuantWithMinMaxVars operation.

    Args:
        num_bits (int): Quantization bitwidth; between 2 and 16, inclusive. Default: 8.
        narrow_range (bool): Whether the quantization algorithm uses narrow range or not.
            if True, the quantization range is [0, 2^num_bits-1]. Otherwise, the quantization
            range is [1, 2^num_bits-1]. Default: ``False``.

    Inputs:
        - **gradients** (Tensor) - The gradient above the FakeQuantWithMinMaxVars.
        - **x** (Tensor) - Float32 tensor representing the shape of the output tensor.
        - **min** (Tensor) - Value of the min range of the input data x.
        - **max** (Tensor) - Value of the max range of the input data x.

    Outputs:
        - **backprops_wrt_x** (Tensor) - The gradient of input x, with the same shape and date type as input x.
        - **backprops_wrt_min** (Tensor) - The gradient of input min, with the same shape and date type as input min.
        - **backprops_wrt_max** (Tensor) - The gradient of input max, with the same shape and date type as input max.

    Examples:
        >>> gradients = Tensor(np.random.rand(3, 16, 5, 5), mstype.float32)
        >>> input_tensor = Tensor(np.random.rand(3, 16, 5, 5), mstype.float32)
        >>> min_tensor = Tensor(np.array([-6]), mstype.float32)
        >>> max_tensor = Tensor(np.array([6]), mstype.float32)
        >>> x_gradient, min_gradient, max_gradient = FakeQuantWithMinMaxVarsGradient(num_bits=8,narrow_range=False)
        ...                                          (gradients, input_tensor, min_tensor, max_tensor)
        >>> x_gradient   # shape: (3, 16, 5, 5)  data type: mstype.float32
        >>> min_gradient # shape: (1,)           data type: mstype.float32
        >>> max_gradient # shape: (1,)           data type: mstype.float32
    """

    @prim_attr_register
    def __init__(self,
                 num_bits=8,
                 narrow_range=False):
        self.num_bits = validator.check_positive_int(num_bits, 'num_bits', self.name)
        self.num_bits = validator.check_int_range(self.num_bits, 2, 16, validator.INC_BOTH, 'num_bits', self.name)
        self.narrow_range = validator.check_value_type(
            'narrow_range', narrow_range, (bool,), self.name)

    def check_broadcast(self, min_shape, input_shape):
        shape_val = 1
        for shape in input_shape:
            shape_val = shape_val * shape
        if min_shape[0] > 1 and min_shape[0] != shape_val:
            raise ValueError(f"For '{self.name}', the shape of \'min\' cannot broadcast to the shape of \'x\'.")

    def infer_shape(self, dout_shape, x_shape, min_shape, max_shape):
        validator.check_int(len(x_shape), 1, validator.GE, "x rank", self.name)
        validator.check("dout shape", dout_shape, "x shape", x_shape, validator.EQ, self.name)
        validator.check("min shape", min_shape, "max shape", max_shape, validator.EQ, self.name)
        validator.check_int(len(min_shape), 1, validator.EQ, "min shape", self.name)
        self.check_broadcast(min_shape, x_shape)
        return x_shape, min_shape, max_shape

    def infer_dtype(self, dout_type, x_type, min_type, max_type):
        tuple(map(partial(validator.check_tensor_dtype_valid,
                          valid_dtypes=(mstype.float16, mstype.float32), prim_name=self.name),
                  ('dout', "x", "min", "max"),
                  (dout_type, x_type, min_type, max_type)))
        return x_type, min_type, max_type


class FakeQuantWithMinMaxVarsPerChannel(PrimitiveWithInfer):
    r"""
    Fake-quantize the input and one of shape: [d], [b, d], [b, h, w, d] by per-channel min and max

    Args:
        num_bits (int): Quantization bitwidth; between 2 and 16, inclusive. Default: 8.
        narrow_range (bool): Whether the quantization algorithm uses narrow range or not.
            if True, the quantization range is [0, 2^num_bits-1]. Otherwise, the quantization
            range is [1, 2^num_bits-1]. Default: ``False``.

    Inputs:
        - **x** (Tensor) - Float32 tensor representing the shape of the output tensor.
        - **min** (Tensor) - Value of the min range of the input data x.
        - **max** (Tensor) - Value of the max range of the input data x.

    Outputs:
        - Tensor, the data type and shape of output tensor is the same as input x.

    Examples:
        >>> input_tensor = Tensor(np.random.rand(3, 16, 3, 4), mstype.float32)
        >>> min_tensor = Tensor(np.array([-6, -1, -2, -3]), mstype.float32)
        >>> max_tensor = Tensor(np.array([6, 1, 2, 3]), mstype.float32)
        >>> output_tensor = FakeQuantWithMinMaxVars(num_bits=8, narrow_range=False)(
        ...                 input_tensor, min_tensor, max_tensor)
        >>> output_tensor # shape: (3, 16, 3, 4)  data type: mstype.float32
    """

    @prim_attr_register
    def __init__(self,
                 num_bits=8,
                 narrow_range=False):
        self.num_bits = validator.check_positive_int(num_bits, 'num_bits', self.name)
        self.num_bits = validator.check_int_range(self.num_bits, 2, 16, validator.INC_BOTH, 'num_bits', self.name)
        self.narrow_range = validator.check_value_type(
            'narrow_range', narrow_range, (bool,), self.name)

    def infer_shape(self, x_shape, min_shape, max_shape):
        validator.check_int(len(x_shape), 1, validator.GE, "x rank", self.name)
        validator.check("min shape", min_shape, "max shape", max_shape, validator.EQ, self.name)
        validator.check_int(len(min_shape), 1, validator.EQ, "min shape", self.name)
        validator.check("min shape", min_shape[0], "x shape", x_shape[-1], validator.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_type, min_type, max_type):
        tuple(map(partial(validator.check_tensor_dtype_valid,
                          valid_dtypes=(mstype.float16, mstype.float32), prim_name=self.name),
                  ("x", "min", "max"),
                  (x_type, min_type, max_type)))
        return x_type


class FakeQuantWithMinMaxVarsPerChannelGradient(PrimitiveWithInfer):
    r"""
    Performs grad of FakeQuantWithMinMaxVars operation.

    Args:
        num_bits (int): Quantization bitwidth; between 2 and 16, inclusive. Default: 8.
        narrow_range (bool): Whether the quantization algorithm uses narrow range or not.
            if True, the quantization range is [0, 2^num_bits-1]. Otherwise, the quantization
            range is [1, 2^num_bits-1]. Default: ``False``.

    Inputs:
        - **gradients** (Tensor) - The gradient above the FakeQuantWithMinMaxVars.
        - **x** (Tensor) - Float32 tensor representing the shape of the output tensor.
        - **min** (Tensor) - Value of the min range of the input data x.
        - **max** (Tensor) - Value of the max range of the input data x.

    Outputs:
        - **backprops_wrt_x** (Tensor) - The gradient of input x, with the same shape and date type as input x.
        - **backprops_wrt_min** (Tensor) - The gradient of input min, with the same shape and date type as input min.
        - **backprops_wrt_max** (Tensor) - The gradient of input max, with the same shape and date type as input max.

    Examples:
        >>> gradients = Tensor(np.random.rand(3, 16, 3, 4), mstype.float32)
        >>> input_tensor = Tensor(np.random.rand(3, 16, 3, 4), mstype.float32)
        >>> min_tensor = Tensor(np.array([-6, -1, -2, -3]), mstype.float32)
        >>> max_tensor = Tensor(np.array([6, 1, 2, 3]), mstype.float32)
        >>> x_gradient, min_gradient, max_gradient = FakeQuantWithMinMaxVarsPerChannelGradient(
        ...                                          num_bits=8, narrow_range=False)(
        ...                                          gradients, input_tensor, min_tensor, max_tensor)
        >>> x_gradient   # shape: (3, 16, 3, 4)  data type: mstype.float32
        >>> min_gradient # shape: (4,)           data type: mstype.float32
        >>> max_gradient # shape: (4,)           data type: mstype.float32
    """

    @prim_attr_register
    def __init__(self,
                 num_bits=8,
                 narrow_range=False):
        self.num_bits = validator.check_positive_int(num_bits, 'num_bits', self.name)
        self.num_bits = validator.check_int_range(self.num_bits, 2, 16, validator.INC_BOTH, 'num_bits', self.name)
        self.narrow_range = validator.check_value_type(
            'narrow_range', narrow_range, (bool,), self.name)

    def infer_shape(self, dout_shape, x_shape, min_shape, max_shape):
        validator.check_int(len(x_shape), 1, validator.GE, "x rank", self.name)
        validator.check("dout shape", dout_shape, "x shape", x_shape, validator.EQ, self.name)
        validator.check("min shape", min_shape, "max shape", max_shape, validator.EQ, self.name)
        validator.check_int(len(min_shape), 1, validator.EQ, "min shape", self.name)
        validator.check("min shape", min_shape[0], "x shape", x_shape[-1], validator.EQ, self.name)
        return x_shape, min_shape, max_shape

    def infer_dtype(self, dout_type, x_type, min_type, max_type):
        tuple(map(partial(validator.check_tensor_dtype_valid,
                          valid_dtypes=(mstype.float16, mstype.float32), prim_name=self.name),
                  ("dout", "x", "min", "max"),
                  (dout_type, x_type, min_type, max_type)))
        return x_type, min_type, max_type


def _fake_quant_per_infer_dtype(prim_name, x_type, min_type, max_type):
    if context.get_context('device_target') == "GPU":
        valid_dtypes = (mstype.float32,)
    else:
        valid_dtypes = (mstype.float16, mstype.float32)
    tuple(map(partial(validator.check_tensor_dtype_valid, valid_dtypes=valid_dtypes, prim_name=prim_name),
              ("x", "min", "max"),
              (x_type, min_type, max_type)))
    return x_type


def _fake_quant_per_grad_infer_dtype(prim_name, dout_type, x_type, min_type, max_type):
    if context.get_context('device_target') == "GPU":
        valid_dtypes = (mstype.float32,)
    else:
        valid_dtypes = (mstype.float16, mstype.float32)
    tuple(map(partial(validator.check_tensor_dtype_valid, valid_dtypes=valid_dtypes, prim_name=prim_name),
              ("dout", "x", "min", "max"),
              (dout_type, x_type, min_type, max_type)))
    return dout_type


class FakeQuantPerLayer(PrimitiveWithInfer):
    r"""
    Simulates the quantize and dequantize operations in training time.

    Args:
        num_bits (int) : Number bits for quantization aware. Default: 8.
        ema (bool): Uses EMA algorithm update value min and max. Default: ``False``.
        ema_decay (int) : EMA algorithm decay parameter. Default: 0.999.
        quant_delay (int): Quantilization delay parameter. Before delay step in training time not update
            simulate quantization aware function. After delay step in training time begin simulate the aware
            quantize function. Default: 0.
        symmetric (bool): Whether the quantization algorithm is symmetric or not. Default: ``False``.
        narrow_range (bool): Whether the quantization algorithm uses narrow range or not. Default: ``False``.
        training (bool): Training the network or not. Default: ``True``.

    Inputs:
        - **x** (Tensor) : float32 Tensor representing the shape of the output tensor.
        - **min** (Tensor) : Value of the min range of the input data x.
        - **max** (Tensor) : Value of the max range of the input data x.

    Outputs:
        - Tensor: Simulates quantize tensor of x.

    Examples:
        >>> input_tensor = Tensor(np.random.rand(3, 16, 5, 5), mstype.float32)
        >>> min_tensor = Tensor(np.array([-6]), mstype.float32)
        >>> max_tensor = Tensor(np.array([6]), mstype.float32)
        >>> output_tensor = FakeQuantPerLayer(num_bits=8)(input_tensor, min_tensor, max_tensor)
    """
    support_quant_bit = [4, 7, 8]

    @prim_attr_register
    def __init__(self,
                 num_bits=8,
                 ema=False,
                 ema_decay=0.999,
                 quant_delay=0,
                 symmetric=False,
                 narrow_range=False,
                 training=True):
        """Initialize FakeQuantPerLayer OP"""
        if context.get_context('device_target') == "Ascend":
            from mindspore.ops._op_impl._custom_op import fake_quant_perlayer
        if num_bits not in self.support_quant_bit:
            raise ValueError(
                f"For '{self.name}' attr \'num_bits\' is not support.")
        if ema and not ema_decay:
            raise ValueError(
                f"For '{self.name}' attr \'ema\' and \'ema_decay\' should set together.")

        self.ema = validator.check_value_type('ema', ema, (bool,), self.name)
        self.symmetric = validator.check_value_type(
            'symmetric', symmetric, (bool,), self.name)
        self.narrow_range = validator.check_value_type(
            'narrow_range', narrow_range, (bool,), self.name)
        self.training = validator.check_value_type('training', training, (bool,), self.name)
        self.ema_decay = validator.check_float_range(ema_decay, 0, 1, validator.INC_BOTH, 'ema_decay', self.name)
        self.num_bits = validator.check_positive_int(num_bits, 'num_bits', self.name)
        self.quant_delay = validator.check_non_negative_int(quant_delay, 'quant_delay', self.name)
        self.init_prim_io_names(inputs=['x', 'min', 'max'],
                                outputs=['out'])

    def infer_shape(self, x_shape, min_shape, max_shape):
        validator.check_int(len(x_shape), 1, validator.GE, "x rank", self.name)
        validator.check("min shape", min_shape, "max shape", max_shape, validator.EQ, self.name)
        validator.check_equal_int(len(min_shape), 1, "min shape", self.name)
        return x_shape

    def infer_dtype(self, x_type, min_type, max_type):
        return _fake_quant_per_infer_dtype(self.name, x_type, min_type, max_type)


class FakeQuantPerLayerGrad(PrimitiveWithInfer):
    r"""
    Performs grad of FakeQuantPerLayer operation.

    Examples:
        >>> fake_min_max_grad = FakeQuantPerLayerGrad()
        >>> dout = Tensor(np.array([[-2.3, 1.2], [5.7, 0.2]]), mindspore.float32)
        >>> input_x = Tensor(np.array([[18, -23], [0.2, 6]]), mindspore.float32)
        >>> _min = Tensor(np.array([-4]), mindspore.float32)
        >>> _max = Tensor(np.array([2]), mindspore.float32)
        >>> result = fake_min_max_grad(dout, input_x, _min, _max)
    """
    support_quant_bit = [4, 7, 8]

    @prim_attr_register
    def __init__(self,
                 num_bits=8,
                 quant_delay=0,
                 symmetric=False,
                 narrow_range=False):
        if context.get_context('device_target') == "Ascend":
            from mindspore.ops._op_impl._custom_op import fake_quant_perlayer_grad
        if num_bits not in self.support_quant_bit:
            raise ValueError(
                f"For '{self.name}' attr \'num_bits\' is not support.")

        self.num_bits = validator.check_positive_int(num_bits, 'num_bits', self.name)
        self.quant_delay = validator.check_value_type(
            'quant_delay', quant_delay, (int,), self.name)
        self.symmetric = validator.check_value_type(
            'symmetric', symmetric, (bool,), self.name)
        self.narrow_range = validator.check_value_type(
            'narrow_range', narrow_range, (bool,), self.name)
        self.init_prim_io_names(
            inputs=['dout', 'x', 'min', 'max'], outputs=['dx'])

    def infer_shape(self, dout_shape, x_shape, min_shape, max_shape):
        validator.check("dout shape", dout_shape, "x shape",
                        x_shape, validator.EQ, self.name)
        validator.check("min shape", min_shape, "max shape",
                        max_shape, validator.EQ, self.name)
        validator.check_equal_int(len(min_shape), 1, "min shape", self.name)
        return dout_shape

    def infer_dtype(self, dout_type, x_type, min_type, max_type):
        return _fake_quant_per_grad_infer_dtype(self.name, dout_type, x_type, min_type, max_type)


class FakeQuantPerChannel(PrimitiveWithInfer):
    r"""
    Simulates the quantize and dequantize operations in training time base on per channel.

    Args:
        num_bits (int) : Number bits to quantilization. Default: 8.
        ema (bool): Uses EMA algorithm update tensor min and tensor max. Default: ``False``.
        ema_decay (int) : EMA algorithm decay parameter. Default: 0.999.
        quant_delay (int): Quantilization delay  parameter. Before delay step in training time not
            update the weight data to simulate quantize operation. After delay step in training time
            begin simulate the quantize operation. Default: 0.
        symmetric (bool): Whether the quantization algorithm is symmetric or not. Default: ``False``.
        narrow_range (bool): Whether the quantization algorithm uses narrow range or not. Default: ``False``.
        training (bool): Training the network or not. Default: ``True``.
        channel_axis (int): Quantization by channel axis. Ascend backend only supports 0 or 1. Default: 1.

    Inputs:
        - **x** (Tensor) : 4-D float32 Tensor representing the shape of the output tensor.
        - **min** (int, float) : Value of the min range of the input data.
        - **max** (int, float) : Value of the max range of the input data.

    Outputs:
        - Tensor, has the same type as input.

    Examples:
        >>> fake_quant = FakeQuantPerChannel()
        >>> input_x = Tensor(np.array([3, 4, 5, -2, -3, -1]).reshape(3, 2), mindspore.float32)
        >>> _min = Tensor(np.linspace(-2, 2, 12).reshape(3, 2, 2), mindspore.float32)
        >>> _max = Tensor(np.linspace(8, 12, 12).reshape(3, 2, 2), mindspore.float32)
        >>> result = fake_quant(input_x, _min, _max)
    """
    support_quant_bit = [4, 7, 8]
    ascend_support_x_rank = [2, 3, 4]

    @prim_attr_register
    def __init__(self,
                 num_bits=8,
                 ema=False,
                 ema_decay=0.999,
                 quant_delay=0,
                 symmetric=False,
                 narrow_range=False,
                 training=True,
                 channel_axis=1):
        """Initialize FakeQuantPerChannel OP"""
        self.is_ascend = context.get_context('device_target') == "Ascend"
        if self.is_ascend:
            from mindspore.ops._op_impl._custom_op import fake_quant_perchannel
        if num_bits not in self.support_quant_bit:
            raise ValueError(
                f"For '{self.name}' Attr \'num_bits\' is not support.")
        if ema and not ema_decay:
            raise ValueError(
                f"For '{self.name}' attr \'ema\' and \'ema_decay\' should set together.")

        self.ema = validator.check_value_type('ema', ema, (bool,), self.name)
        self.symmetric = validator.check_value_type(
            'symmetric', symmetric, (bool,), self.name)
        self.narrow_range = validator.check_value_type(
            'narrow_range', narrow_range, (bool,), self.name)
        self.training = validator.check_value_type(
            'training', training, (bool,), self.name)
        self.ema_decay = validator.check_float_range(ema_decay, 0, 1, validator.INC_BOTH, 'ema_decay', self.name)
        self.num_bits = validator.check_positive_int(num_bits, 'num_bits', self.name)
        self.quant_delay = validator.check_non_negative_int(quant_delay, 'quant_delay', self.name)
        self.channel_axis = validator.check_non_negative_int(channel_axis, 'channel_axis', self.name)
        self.init_prim_io_names(inputs=['x', 'min', 'max'], outputs=['out'])

    def infer_shape(self, x_shape, min_shape, max_shape):
        if self.is_ascend and len(x_shape) not in self.ascend_support_x_rank:
            raise ValueError(f"For '{self.name}' x rank must be in '{self.ascend_support_x_rank}'")
        if not self.is_ascend:
            validator.check_int(len(x_shape), 1, validator.GE, "x rank", self.name)
        if len(x_shape) == 1:
            self.channel_axis = 0
        validator.check("min shape", min_shape, "max shape", max_shape, validator.EQ, self.name)
        validator.check_equal_int(min_shape[0], x_shape[self.channel_axis], "min shape", self.name)
        validator.check_equal_int(max_shape[0], x_shape[self.channel_axis], "max shape", self.name)
        return x_shape

    def infer_dtype(self, x_type, min_type, max_type):
        return _fake_quant_per_infer_dtype(self.name, x_type, min_type, max_type)


class FakeQuantPerChannelGrad(PrimitiveWithInfer):
    r"""
    Performs grad of FakeQuantPerChannel operation.

    Examples:
        >>> fqmmpc_grad = FakeQuantPerChannelGrad()
        >>> input_x = Tensor(np.random.randint(-4, 4, (2, 3, 4)), mindspore.float32)
        >>> dout = Tensor(np.random.randint(-2, 2, (2, 3, 4)), mindspore.float32)
        >>> _min = Tensor(np.random.randint(-8, 2, (2, 3, 4)), mindspore.float32)
        >>> _max = Tensor(np.random.randint(-2, 8, (2, 3, 4)), mindspore.float32)
        >>> result = fqmmpc_grad(dout, input_x, _min, _max)
    """
    support_quant_bit = [4, 7, 8]

    @prim_attr_register
    def __init__(self,
                 num_bits=8,
                 quant_delay=0,
                 symmetric=False,
                 narrow_range=False,
                 channel_axis=1):
        """Initialize FakeQuantPerChannelGrad Fill"""
        if context.get_context('device_target') == "Ascend":
            from mindspore.ops._op_impl._custom_op import fake_quant_perchannel_grad
        if num_bits not in self.support_quant_bit:
            raise ValueError(
                f"For '{self.name}' attr \'num_bits\' is not support.")

        self.num_bits = validator.check_positive_int(num_bits, 'num_bits', self.name)
        self.quant_delay = validator.check_value_type(
            'quant_delay', quant_delay, (int,), self.name)
        self.symmetric = validator.check_value_type(
            'symmetric', symmetric, (bool,), self.name)
        self.narrow_range = validator.check_value_type(
            'narrow_range', narrow_range, (bool,), self.name)
        self.channel_axis = validator.check_non_negative_int(channel_axis, 'channel axis', self.name)
        self.init_prim_io_names(
            inputs=['dout', 'x', 'min', 'max'], outputs=['dx'])

    def infer_shape(self, dout_shape, x_shape, min_shape, max_shape):
        validator.check("dout shape", dout_shape, "x shape", x_shape)
        validator.check("min shape", min_shape, "max shape", max_shape)
        return dout_shape

    def infer_dtype(self, dout_type, x_type, min_type, max_type):
        return _fake_quant_per_grad_infer_dtype(self.name, dout_type, x_type, min_type, max_type)


class BatchNormFold(PrimitiveWithInfer):
    """
    Batch Normalization folded.

    Args:
        momentum (float): Momentum value must be [0, 1]. Default: 0.9.
        epsilon (float): A small float number to avoid dividing by 0. 1e-5 if dtype in
            float32 else 1e-3. Default: 1e-5.
        is_training (bool): In training mode set True, else set False. Default: ``True``.
        freeze_bn (int): Delay in steps at which computation switches from regular batch
            norm to frozen mean and std. Default: 0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C)`.
        - **mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **variance** (Tensor) - Tensor of shape :math:`(C,)`.
        - **global_step** (Tensor) - Tensor to record current global step.

    Outputs:
        Tuple of 4 Tensor, the normalized input and the updated parameters.

        - **batch_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **batch_std** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_std** (Tensor) - Tensor of shape :math:`(C,)`.

    Examples:
        >>> batch_norm_fold = P.BatchNormFold()
        >>> input_x = Tensor(np.array([1, 2, -1, -2, -2, 1]).reshape(2, 3), mindspore.float32)
        >>> mean = Tensor(np.array([0.5, -1, 1,]), mindspore.float32)
        >>> variance = Tensor(np.array([0.36, 0.4, 0.49]), mindspore.float32)
        >>> global_step = Tensor(np.arange(6), mindspore.int32)
        >>> batch_mean, batch_std, running_mean, running_std = batch_norm_fold(input_x, mean, variance, global_step)
    """
    channel_axis = 1

    @prim_attr_register
    def __init__(self, momentum=0.9, epsilon=1e-5, is_training=True, freeze_bn=0):
        """Initialize batch norm fold layer"""
        self.momentum = validator.check_float_range(momentum, 0, 1, validator.INC_BOTH, 'momentum', self.name)
        self.epsilon = validator.check_positive_float(epsilon, 'epsilon', self.name)
        self.is_training = validator.check_value_type('is_training', is_training, (bool,), self.name)
        self.freeze_bn = validator.check_value_type('freeze_bn', freeze_bn, (int,), self.name)

        self.init_prim_io_names(inputs=['x', 'mean', 'variance', 'global_step'],
                                outputs=['batch_mean', 'batch_std', 'running_mean', 'running_std'])

    def infer_shape(self, x_shape, mean_shape, variance_shape, global_step_shape):
        validator.check("mean shape", mean_shape, "gamma_shape", variance_shape, validator.EQ, self.name)
        validator.check("mean_shape[0]", mean_shape[0], "input channel",
                        x_shape[self.channel_axis], validator.EQ, self.name)
        validator.check_equal_int(len(global_step_shape), 1, "global step shape len", self.name)
        return mean_shape, mean_shape, mean_shape, mean_shape

    def infer_dtype(self, x_type, mean_type, variance_type, global_step_type):
        validator.check("input type", x_type, "mean type", mean_type)
        validator.check("input type", x_type, "variance type", variance_type)
        args = {"x": x_type, "mean": mean_type, "variance": variance_type}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float16, mstype.float32), self.name)
        validator.check_tensor_dtype_valid("global_step", global_step_type, (mstype.int32,), self.name)
        return x_type, x_type, x_type, x_type


class BatchNormFoldGrad(PrimitiveWithInfer):
    r"""
    Performs grad of BatchNormFold operation.

    Examples:
        >>> batch_norm_fold_grad = ops.BatchNormFoldGrad()
        >>> d_batch_mean = Tensor(np.random.randint(-2., 2., (1, 2, 2, 3)), mindspore.float32)
        >>> d_batch_std = Tensor(np.random.randn(1, 2, 2, 3), mindspore.float32)
        >>> input_x = Tensor(np.random.randint(0, 256, (4, 1, 4, 6)), mindspore.float32)
        >>> batch_mean = Tensor(np.random.randint(-8., 8., (1, 2, 2, 3)), mindspore.float32)
        >>> batch_std = Tensor(np.random.randint(0, 12, (1, 2, 2, 3)), mindspore.float32)
        >>> global_step = Tensor([2], mindspore.int32)
        >>> result = batch_norm_fold_grad(d_batch_mean, d_batch_std, input_x, batch_mean, batch_std, global_step)
    """
    channel_axis = 1

    @prim_attr_register
    def __init__(self, epsilon=1e-5, is_training=True, freeze_bn=0):
        """Initialize BatchNormGrad layer"""
        self.is_training = validator.check_value_type('is_training', is_training, (bool,), self.name)
        self.freeze_bn = validator.check_value_type('freeze_bn', freeze_bn, (int,), self.name)
        self.epsilon = validator.check_positive_float(epsilon, 'epsilon', self.name)
        self.init_prim_io_names(inputs=['d_batch_mean', 'd_batch_std', 'x', 'batch_mean', 'batch_std', 'global_step'],
                                outputs=['dx'])

    def infer_shape(self, d_batch_mean_shape, d_batch_std_shape, x_shape, batch_mean_shape, batch_std_shape,
                    global_step_shape):
        validator.check("d_batch_mean shape", d_batch_mean_shape,
                        "d_batch_std shape", d_batch_std_shape, validator.EQ, self.name)
        validator.check("d_batch_mean shape", d_batch_mean_shape,
                        "batch_mean shape", batch_mean_shape, validator.EQ, self.name)
        validator.check("d_batch_mean shape", d_batch_mean_shape,
                        "batch_std shape", batch_std_shape, validator.EQ, self.name)
        validator.check("d_batch_mean_shape[0]", d_batch_mean_shape[0],
                        "input channel", x_shape[self.channel_axis], validator.EQ, self.name)
        validator.check_equal_int(len(global_step_shape), 1, "global step shape len", self.name)
        return x_shape

    def infer_dtype(self, d_batch_mean_type, d_batch_std_type, x_type, batch_mean_type, batch_std_type,
                    global_step_type):
        args = {"input": x_type, "d_batch_mean": d_batch_mean_type, "d_batch_std": d_batch_std_type,
                "batch_mean": batch_mean_type, "batch_std": batch_std_type}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float16, mstype.float32), self.name)
        validator.check_tensor_dtype_valid("global_step", global_step_type, (mstype.int32,), self.name)
        return x_type


class CorrectionMul(PrimitiveWithInfer):
    """
    Scales the weights with a correction factor to the long term statistics
    prior to quantization. This ensures that there is no jitter in the quantized weights
    due to batch to batch variation.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C)`.
        - **batch_std** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_std** (Tensor) - Tensor of shape :math:`(C,)`.

    Outputs:
        - **out** (Tensor) - Tensor has the same shape as x.

    Examples:
        >>> correction_mul = ops.CorrectionMul()
        >>> input_x = Tensor(np.random.randint(-8, 12, (3, 4)), mindspore.float32)
        >>> batch_std = Tensor(np.array([1.5, 3, 2]), mindspore.float32)
        >>> running_std = Tensor(np.array([2, 1.2, 0.5]), mindspore.float32)
        >>> out = correction_mul(input_x, batch_std, running_std)
    """

    @prim_attr_register
    def __init__(self, channel_axis=0):
        """Initialize correction mul layer"""
        if context.get_context('device_target') == "Ascend":
            from mindspore.ops._op_impl._custom_op import correction_mul
        self.channel_axis = channel_axis
        self.init_prim_io_names(inputs=['x', 'batch_std', 'running_std'],
                                outputs=['out'])

    def infer_shape(self, x_shape, batch_std_shape, running_std_shape):
        validator.check("batch_std shape", batch_std_shape, "running_std shape",
                        running_std_shape, validator.EQ, self.name)
        validator.check("batch_std_shape[0]", batch_std_shape[0], "x_shape channel size", x_shape[self.channel_axis],
                        validator.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_type, batch_std_type, running_std_type):
        args = {"x": x_type, "batch_std": batch_std_type, "running_std": running_std_type}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float16, mstype.float32), self.name)
        return x_type


class CorrectionMulGrad(PrimitiveWithInfer):
    r"""
    Performs grad of CorrectionMul operation.

    Examples:
        >>> correction_mul_grad = ops.CorrectionMulGrad()
        >>> dout = Tensor(np.array([1.5, -2.2, 0.7, -3, 1.6, 2.8]).reshape(2, 1, 1, 3), mindspore.float32)
        >>> input_x = Tensor(np.random.randint(0, 256, (2, 1, 1, 3)), mindspore.float32)
        >>> gamma = Tensor(np.array([0.2, -0.2, 2.5, -1.]).reshape(2, 1, 2), mindspore.float32)
        >>> running_std = Tensor(np.array([1.2, 0.1, 0.7, 2.3]).reshape(2, 1, 2), mindspore.float32)
        >>> result = correction_mul_grad(dout, input_x, gamma, running_std)
    """

    @prim_attr_register
    def __init__(self, channel_axis=0):
        """Initialize correction mul layer"""
        if context.get_context('device_target') == "Ascend":
            from mindspore.ops._op_impl._custom_op import correction_mul_grad
        self.channel_axis = channel_axis
        self.init_prim_io_names(inputs=['dout', 'x', 'gamma', 'running_std'],
                                outputs=['dx', 'mul_dx'])

    def infer_shape(self, dout_shape, x_shape, gamma_shape, running_std_shape):
        validator.check("dout shape", dout_shape, "x_shape x", x_shape, validator.EQ, self.name)
        validator.check("gamma_shape[0]", gamma_shape[0], "dout channel size", dout_shape[self.channel_axis],
                        validator.EQ, self.name)
        validator.check("running_std_shape[0]", running_std_shape[0],
                        "dout channel size", dout_shape[self.channel_axis], validator.EQ, self.name)
        if context.get_context('device_target') == "Ascend":
            return x_shape, x_shape
        return x_shape, gamma_shape

    def infer_dtype(self, dout_type, x_type, gamma_type, running_std_type):
        args = {"dout": dout_type, "x": x_type, "gamma": gamma_type, "running_std": running_std_type}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float16, mstype.float32), self.name)
        if context.get_context('device_target') == "Ascend":
            return x_type, x_type
        return x_type, gamma_type


class CorrectionMulGradReduce(PrimitiveWithInfer):
    r"""
    Performs grad reduce of CorrectionMul operation.

    Examples:
        >>> correction_mul_grad_rd = ops.CorrectionMulGradReduce()
        >>> dout = Tensor(np.array([1.5, -2.2, 0.7, -3, 1.6, 2.8]).reshape(2, 1, 1, 3), mindspore.float32)
        >>> input_x = Tensor(np.random.randint(0, 256, (2, 1, 1, 3)), mindspore.float32)
        >>> gamma = Tensor(np.array([0.2, -0.2, 2.5, -1.]).reshape(2, 1, 2), mindspore.float32)
        >>> running_std = Tensor(np.array([1.2, 0.1, 0.7, 2.3]).reshape(2, 1, 2), mindspore.float32)
        >>> result = correction_mul_grad_rd(dout, input_x, gamma, running_std)
    """

    @prim_attr_register
    def __init__(self, channel_axis=0):
        """Initialize correction mul reduce layer"""
        if context.get_context('device_target') == "Ascend":
            from mindspore.ops._op_impl._custom_op import correction_mul_grad
        self.channel_axis = channel_axis
        self.init_prim_io_names(inputs=['mul_dx'],
                                outputs=['d_gamma'])

    def infer_shape(self, mul_dx_shape):
        return [mul_dx_shape[self.channel_axis]]

    def infer_dtype(self, mul_dx_type):
        return mul_dx_type


class BatchNormFold2(PrimitiveWithInfer):
    """
    Scales the bias with a correction factor to the long term statistics
    prior to quantization. This ensures that there is no jitter in the quantized bias
    due to batch to batch variation.

    Inputs:
        - **x** (Tensor)  - Tensor of shape :math:`(N, C)`.
        - **beta** (Tensor) - Tensor of shape :math:`(C,)`.
        - **gamma** (Tensor) - Tensor of shape :math:`(C,)`.
        - **batch_std** (Tensor) - Tensor of shape :math:`(C,)`.
        - **batch_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_std** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **global_step** (Tensor) - Tensor to record current global step.

    Outputs:
        - **y** (Tensor) - Tensor has the same shape as x.

    Examples:
        >>> batch_norm_fold2 = ops.BatchNormFold2()
        >>> input_x = Tensor(np.random.randint(-6, 6, (4, 3)), mindspore.float32)
        >>> beta = Tensor(np.array([0.2, -0.1, 0.25]), mindspore.float32)
        >>> gamma = Tensor(np.array([-0.1, -0.25, 0.1]), mindspore.float32)
        >>> batch_std = Tensor(np.array([0.1, 0.2, 0.1]), mindspore.float32)
        >>> batch_mean = Tensor(np.array([0, 0.05, 0.2]), mindspore.float32)
        >>> running_std = Tensor(np.array([0.1, 0.1, 0.3]), mindspore.float32)
        >>> running_mean = Tensor(np.array([-0.1, 0, -0.1]), mindspore.float32)
        >>> global_step = Tensor(np.random.randint(1, 8, (8, )), mindspore.int32)
        >>> result = batch_norm_fold2(input_x, beta, gamma, batch_std, batch_mean,
        >>>                           running_std, running_mean, global_step)
    """
    channel_axis = 1

    @prim_attr_register
    def __init__(self, freeze_bn=0):
        """Initialize conv2d fold layer"""
        self.freeze_bn = validator.check_value_type('freeze_bn', freeze_bn, (int,), self.name)
        self.init_prim_io_names(inputs=['x', 'beta', 'gamma', 'batch_std', 'batch_mean',
                                        'running_std', 'running_mean', 'global_step'],
                                outputs=['y'])

    def infer_shape(self, x_shape, beta_shape, gamma_shape, batch_std_shape, running_std_shape, batch_mean_shape,
                    running_mean_shape, global_step_shape):
        validator.check("batch_std shape", batch_std_shape, "running_std shape",
                        running_std_shape, validator.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "batch_mean shape",
                        batch_mean_shape, validator.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "beta shape", beta_shape, validator.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "running_mean shape", running_mean_shape,
                        validator.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "batch_mean shape", gamma_shape, validator.EQ, self.name)
        validator.check("batch_std_shape[0]", batch_std_shape[0], "x_shape channel size", x_shape[self.channel_axis],
                        validator.EQ, self.name)
        validator.check_equal_int(len(global_step_shape), 1, "global step shape len", self.name)
        return x_shape

    def infer_dtype(self, x_type, beta_type, gamma_type, batch_std_type, running_std_type, batch_mean_type,
                    running_mean_type, global_step_type):
        args = {"batch_std": batch_std_type, "running_std": running_std_type, "batch_mean": batch_mean_type,
                "beta": beta_type, "running_mean": running_mean_type, "gamma": gamma_type, "x": x_type}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float16, mstype.float32), self.name)
        validator.check_tensor_dtype_valid("global_step", global_step_type, (mstype.int32,), self.name)
        return x_type


class BatchNormFold2Grad(PrimitiveWithInfer):
    r"""
    Performs grad of BatchNormFold2 operation.

    Examples:
        >>> bnf2_grad = ops.BatchNormFold2Grad()
        >>> input_x = Tensor(np.arange(3*3*12*12).reshape(6, 3, 6, 12), mindspore.float32)
        >>> dout = Tensor(np.random.randint(-32, 32, (6, 3, 6, 12)), mindspore.float32)
        >>> gamma = Tensor(np.random.randint(-4, 4, (3, 1, 1, 2)), mindspore.float32)
        >>> batch_std = Tensor(np.random.randint(0, 8, (3, 1, 1, 2)), mindspore.float32)
        >>> batch_mean = Tensor(np.random.randint(-6, 6, (3, 1, 1, 2)), mindspore.float32)
        >>> running_std = Tensor(np.linspace(0, 2, 6).reshape(3, 1, 1, 2), mindspore.float32)
        >>> running_mean = Tensor(np.random.randint(-3, 3, (3, 1, 1, 2)), mindspore.float32)
        >>> global_step = Tensor(np.array([-2]), mindspore.int32)
        >>> result = bnf2_grad(dout, input_x, gamma, batch_std, batch_mean, running_std, running_mean, global_step)
    """
    channel_axis = 1

    @prim_attr_register
    def __init__(self, freeze_bn=0):
        """Initialize MulFold layer"""
        self.freeze_bn = freeze_bn
        self.init_prim_io_names(inputs=['dout', 'x', 'gamma',
                                        'batch_std', 'batch_mean',
                                        'running_std', 'running_mean', 'global_step'],
                                outputs=['d_batch_std', 'd_batch_mean', 'd_beta', 'd_gamma', 'dx'])

    def infer_shape(self, dout_shape, x_shape, gamma_shape,
                    batch_std_shape, batch_mean_shape,
                    running_std_shape, running_mean_shape, global_step_shape):
        validator.check("batch_std shape", batch_std_shape, "batch_mean shape",
                        batch_mean_shape, validator.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "running_std shape",
                        running_std_shape, validator.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "running_mean shape", running_mean_shape,
                        validator.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "gamma shape", gamma_shape, validator.EQ, self.name)
        validator.check("batch_std size", batch_std_shape[0], "dout channel size", dout_shape[self.channel_axis],
                        validator.EQ, self.name)
        validator.check_equal_int(len(global_step_shape), 1, "global step shape len", self.name)
        return gamma_shape, gamma_shape, gamma_shape, gamma_shape, x_shape

    def infer_dtype(self, dout_type, x_type, gamma_type,
                    batch_std_type, batch_mean_type,
                    running_std_type, running_mean_type, global_step_type):
        validator.check("batch_std type", batch_std_type,
                        "batch_mean type", batch_mean_type)
        validator.check("batch_std type", batch_std_type,
                        "gamma type", gamma_type)
        validator.check("batch_std type", batch_std_type,
                        "running_std type", running_std_type)
        validator.check("batch_std type", batch_std_type,
                        "running_mean type", running_mean_type)
        validator.check("batch_std_type", batch_std_type,
                        "dout type", dout_type)
        args = {"batch_std": batch_std_type, "batch_mean": batch_mean_type, "gamma": gamma_type,
                "running_std": running_std_type, "running_mean": running_mean_type, "dout": dout_type}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float16, mstype.float32), self.name)
        validator.check_tensor_dtype_valid("global_step", global_step_type, (mstype.int32,), self.name)
        return gamma_type, gamma_type, gamma_type, gamma_type, gamma_type


class BatchNormFoldD(PrimitiveWithInfer):
    """Performs grad of _BatchNormFold operation."""

    @prim_attr_register
    def __init__(self, momentum=0.9, epsilon=1e-5, is_training=True, freeze_bn=0):
        """Initialize _BatchNormFold layer"""
        from mindspore.ops._op_impl._custom_op import batchnorm_fold
        self.momentum = validator.check_float_range(momentum, 0, 1, validator.INC_BOTH, 'momentum', self.name)
        self.epsilon = validator.check_positive_float(epsilon, 'epsilon', self.name)
        self.is_training = validator.check_value_type('is_training', is_training, (bool,), self.name)
        self.freeze_bn = validator.check_value_type('freeze_bn', freeze_bn, (int,), self.name)
        self.data_format = "NCHW"
        self.init_prim_io_names(inputs=['x', 'x_sum', 'x_square_sum', 'mean', 'variance'],
                                outputs=['batch_mean', 'batch_std', 'running_mean', 'running_std',
                                         'mean_updated', 'variance_updated'])

    def infer_shape(self, x_shape, x_sum_shape, x_square_sum_shape, mean_shape, variance_shape):
        validator.check("mean shape", mean_shape, "gamma_shape", variance_shape, validator.EQ, self.name)
        validator.check("mean_shape[0]", mean_shape[0], "input channel", x_shape[1], validator.EQ, self.name)
        return x_shape, mean_shape, mean_shape, mean_shape, mean_shape, mean_shape, mean_shape

    def infer_dtype(self, x_type, x_sum_type, x_square_sum_type, mean_type, variance_type):
        validator.check("input type", x_type, "mean type", mean_type)
        validator.check("input type", x_type, "variance type", variance_type)
        args = {"x": x_type, "mean": mean_type, "variance": variance_type}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float16, mstype.float32), self.name)
        return x_type, x_type, x_type, x_type, x_type, x_type, x_type


class BatchNormFoldGradD(PrimitiveWithInfer):
    """Performs grad of BatchNormFold operation."""

    @prim_attr_register
    def __init__(self, epsilon=1e-5, is_training=True, freeze_bn=0):
        """Initialize _BatchNormFoldGrad layer"""
        from mindspore.ops._op_impl._custom_op import batchnorm_fold_grad
        self.epsilon = validator.check_positive_float(epsilon, 'epsilon', self.name)
        self.is_training = validator.check_value_type('is_training', is_training, (bool,), self.name)
        self.freeze_bn = validator.check_value_type('freeze_bn', freeze_bn, (int,), self.name)
        self.init_prim_io_names(inputs=['d_batch_mean', 'd_batch_std', 'x', 'batch_mean', 'batch_std'],
                                outputs=['dx'])

    def infer_shape(self, d_batch_mean_shape, d_batch_std_shape, x_shape, batch_mean_shape, batch_std_shape):
        validator.check("d_batch_mean shape", d_batch_mean_shape, "d_batch_std shape", d_batch_std_shape)
        validator.check("d_batch_mean shape", d_batch_mean_shape, "batch_mean shape", batch_mean_shape)
        validator.check("d_batch_mean shape", d_batch_mean_shape, "batch_std shape", batch_std_shape)
        validator.check("x_shape shape", d_batch_mean_shape[0], "input channel", x_shape[1])
        return x_shape

    def infer_dtype(self, d_batch_mean_type, d_batch_std_type, x_type, batch_mean_type, batch_std_type):
        validator.check("input type", x_type, "d_batch_mean type", d_batch_mean_type)
        validator.check("input type", x_type, "d_batch_std type", d_batch_std_type)
        validator.check("input type", x_type, "batch_mean type", batch_mean_type)
        validator.check("input type", x_type, "batch_std type", batch_std_type)
        validator.check_tensor_dtype_valid("input type", x_type, (mstype.float16, mstype.float32), self.name)
        return x_type


class BatchNormFold2D(PrimitiveWithInfer):
    """
    Scales the bias with a correction factor to the long term statistics
    prior to quantization. This ensures that there is no jitter in the quantized bias
    due to batch to batch variation.

    Inputs:
        - **x** (Tensor)  - Tensor of shape :math:`(N, C)`.
        - **beta** (Tensor) - Tensor of shape :math:`(C,)`.
        - **gamma** (Tensor) - Tensor of shape :math:`(C,)`.
        - **batch_std** (Tensor) - Tensor of shape :math:`(C,)`.
        - **batch_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_std** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **global_step** (Tensor) - Tensor to record current global step.

    Outputs:
        - **y** (Tensor) - Tensor has the same shape as x.

    """
    channel_axis = 1

    @prim_attr_register
    def __init__(self, freeze_bn=0):
        """Initialize conv2d fold layer"""
        from mindspore.ops._op_impl._custom_op import batchnorm_fold2
        self.init_prim_io_names(inputs=['x', 'beta', 'gamma', 'batch_std', 'batch_mean', 'running_std'],
                                outputs=['y'])

    def infer_shape(self, x_shape, beta_shape, gamma_shape, batch_std_shape, running_std_shape, batch_mean_shape):
        validator.check("batch_std shape", batch_std_shape, "running_std shape",
                        running_std_shape, validator.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "batch_mean shape",
                        batch_mean_shape, validator.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "beta shape", beta_shape, validator.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "batch_mean shape", gamma_shape, validator.EQ, self.name)
        validator.check("batch_std_shape[0]", batch_std_shape[0], "x_shape channel size", x_shape[self.channel_axis],
                        validator.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_type, beta_type, gamma_type, batch_std_type, running_std_type, batch_mean_type):
        args = {"batch_std": batch_std_type, "running_std": running_std_type, "batch_mean": batch_mean_type,
                "beta": beta_type, "gamma": gamma_type, "x": x_type}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float16, mstype.float32), self.name)
        return x_type


class BatchNormFold2GradD(PrimitiveWithInfer):
    """Performs grad of BatchNormFold2 operation."""
    channel_axis = 1

    @prim_attr_register
    def __init__(self, freeze_bn=False):
        """Initialize MulFold layer"""
        from mindspore.ops._op_impl._custom_op import batchnorm_fold2_grad
        self.freeze_bn = freeze_bn
        self.init_prim_io_names(
            inputs=['dout', 'dout_reduce', 'dout_x_reduce', 'gamma', 'batch_std', 'batch_mean', 'running_std'],
            outputs=['d_batch_std', 'd_batch_mean', 'd_gamma', 'dx'])

    def infer_shape(self, dout_shape, dout_reduce_shape, dout_x_reduce_shape, gamma_shape, batch_std_shape,
                    batch_mean_shape, running_std_shape):
        validator.check("batch_std shape", batch_std_shape, "batch_mean shape",
                        batch_mean_shape, validator.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "running_std shape",
                        running_std_shape, validator.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "gamma shape", gamma_shape, validator.EQ, self.name)
        validator.check("batch_std size", batch_std_shape[0], "dout channel size", dout_shape[self.channel_axis],
                        validator.EQ, self.name)
        return gamma_shape, gamma_shape, gamma_shape, dout_shape

    def infer_dtype(self, dout_type, dout_reduce_type, dout_x_reduce_type, gamma_type, batch_std_type,
                    batch_mean_type, running_std_type):
        validator.check("batch_std type", batch_std_type,
                        "batch_mean type", batch_mean_type)
        validator.check("batch_std type", batch_std_type,
                        "gamma type", gamma_type)
        validator.check("batch_std type", batch_std_type,
                        "running_std type", running_std_type)
        validator.check("batch_std_type", batch_std_type,
                        "dout type", dout_type)
        args = {"batch_std": batch_std_type, "batch_mean": batch_mean_type, "gamma": gamma_type,
                "running_std": running_std_type, "dout": dout_type}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float16, mstype.float32), self.name)
        return gamma_type, gamma_type, gamma_type, gamma_type


class BatchNormFold2GradReduce(PrimitiveWithInfer):
    """Performs grad of CorrectionAddGrad operation."""
    channel_axis = 1

    @prim_attr_register
    def __init__(self, freeze_bn=False):
        """Initialize MulFold layer"""
        from mindspore.ops._op_impl._custom_op import batchnorm_fold2_grad_reduce
        self.freeze_bn = freeze_bn
        self.init_prim_io_names(inputs=['dout', 'x'],
                                outputs=['dout_reduce', 'dout_x_reduce'])

    def infer_shape(self, dout_shape, x_shape):
        validator.check("dout shape", dout_shape, "x shape", x_shape, validator.EQ, self.name)
        return (dout_shape[self.channel_axis],), (dout_shape[self.channel_axis],)

    def infer_dtype(self, dout_type, x_type):
        validator.check("dout type", dout_type, "x type", x_type)
        return dout_type, dout_type


class ActsULQ(PrimitiveWithInfer):
    """
    The ActsULQ(Activation universal learnable quantization).

    Args:
        fixed_min (bool): whether fix clamp min to zero.
        num_bits (int): The bits num used for quantize.

    Inputs:
        - **x** (Tensor) - A Tensor of feature map. With float16 or float32 data type.
        - **clamp_min** (Tensor) - A Tensor of clamp min with the same type as x.
        - **clamp_max** (Tensor) - A Tensor of clamp max with the same type as x.

    Outputs:
        - **y** (Tensor) - A tensor of fake quant of feature map with the same type as `w`.
        - **clamp_min** (Tensor) - A tensor of boolean masks if data in feature map >= clamp_min.
        - **clamp_max** (Tensor) - A tensor of boolean masks if data in feature map <= clamp_max.
        - **x_clamped_loss** (Tensor) - A tensor of clamped loss.

    Examples:
        >>> data_type = np.float32
        >>> x= np.random.uniform(-10, 10, (32, 120)).astype(data_type)
        >>> clamp_max = 0.7 * np.max(x)
        >>> clamp_min = 0.7 * np.min(x)
        >>> clamp_max = np.array([clamp_max], dtype=data_type)
        >>> clamp_min = np.array([clamp_min], dtype=data_type)
        >>> acts_ulq = Q.ActsULQ(fixed_mini=True, num_bits=8)
        >>> quant_x, clamp_min_mask, clamp_max_mask, x_clamped_loss = acts_ulq(Tensor(x), Tensor( clamp_min),
                                                                               Tensor(clamp_max))
    """
    @prim_attr_register
    def __init__(self, fixed_min=False, num_bits=8):
        validator.check_value_type("fixed_min", fixed_min, [bool], self.name)
        validator.check_value_type("num_bits", num_bits, [int], self.name)
        validator.check_int(num_bits, 8, validator.EQ, "value of num_bits", self.name)

    def infer_shape(self, x_shape, clamp_min_shape, clamp_max_shape):
        """infer shape of primitive"""
        validator.check_int(len(clamp_min_shape), len(x_shape), validator.EQ, "dims of clamp_min", self.name)
        validator.check_int(len(clamp_max_shape), len(x_shape), validator.EQ, "dims of clamp_max", self.name)

        x_shape_len = len(x_shape)
        for i in range(x_shape_len):
            validator.check_int(clamp_min_shape[i], 1, validator.EQ, "dims of clamp_min", self.name)
            validator.check_int(clamp_max_shape[i], 1, validator.EQ, "dims of clamp_max", self.name)

        return x_shape, x_shape, x_shape, x_shape

    def infer_dtype(self, x_dtype, clamp_min_dtype, clamp_max_dtype):
        """infer dtype of primitive"""
        valid_types = [mstype.float32, mstype.float16]
        validator.check_tensor_dtype_valid("x", x_dtype, valid_types, self.name)
        validator.check_tensor_dtype_valid("clamp_min", clamp_min_dtype, valid_types, self.name)
        validator.check_tensor_dtype_valid("clamp_max", clamp_max_dtype, valid_types, self.name)

        return x_dtype, mstype.bool_, mstype.bool_, x_dtype


class ActsULQInputGrad(PrimitiveWithInfer):
    """
    The ActsULQInputGrad(grad of ActsULQ).

    Inputs:
        - **y_grad** (Tensor) - A Tensor of grad. With float16 or float32 data type.

    Outputs:
        - **x_grad** (Tensor) - A tensor of data grad with the same type as `y_grad`.
    """
    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, y_grad_shape, clamp_min_mask_shape, clamp_max_mask_shape):
        return y_grad_shape

    def infer_dtype(self, y_grad_type, clamp_min_mask_type, clamp_max_mask_type):
        valid_types = [mstype.float32, mstype.float16]
        validator.check_tensor_dtype_valid("y_grad", y_grad_type, valid_types, self.name)
        return y_grad_type


class ActULQClampMinGrad(PrimitiveWithInfer):
    """
    The ActULQClampMinGrad(Activation Universal Linear Quantization on Clamp Minimum Gradient)

    Inputs:
        - **y_grad** (Tensor) - A tensor of gradient, with float16 or float32 type.
        - **clamp_min_mask** - A tensor of mask, only support int8 type.
        - **x_clamped_loss** - A tensor of loss, with the same type as "y_grad".

    Outputs:
        - **clamp_min_grad** - A tensor of clamp minimum gradient, with the same type as "y_grad".
          The length of tensor is 1.

    Examples:
        >>> data_type = np.float32
        >>> y_grad = np.random.uniform(-10, 10, (32, 120)).astype(data_type)
        >>> clamp_min_mask = np.where(np.random.rand(32, 120) >= 0.5, 1, 0)
        >>> x_clamped_loss = np.random.uniform(-10, 10, (32, 120)).astype(data_type)
        >>> act_ulq_clamp_min_grad = Q.ActULQClampMinGrad()
        >>> clamp_min_grad = act_ulq_clamp_min_grad(Tensor(y_grad), Tensor(clamp_min_mask, mindspore.bool_),
                                                           Tensor(x_clamped_loss))
    """
    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, input_x, input_y, input_z):
        input_x_len = len(input_x)
        output_shape = []
        for _ in range(input_x_len):
            output_shape.append(1)
        return tuple(output_shape)

    def infer_dtype(self, input_x, input_y, input_z):
        return mstype.float32


class ActULQClampMaxGrad(PrimitiveWithInfer):
    """
    The ActULQClampMaxGrad(Activation Universal Linear Quantization on Clamp Maximum Gradient)

    Inputs:
        - **y_grad** (Tensor) - A tensor of gradient, with float16 or float32 type.
        - **clamp_max_mask** - A tensor of mask, only support int8 type.
        - **x_clamped_loss** - A tensor of loss, with the same type as "y_grad".

    Outputs:
        - **clamp_max_grad** - A tensor of clamp maximum gradient, with the same type as "y_grad".
          The length of tensor is 1.

    Examples:
        >>> data_type = np.float32
        >>> y_grad = np.random.uniform(-10, 10, (32, 120)).astype(data_type)
        >>> clamp_max_mask = np.where(np.random.rand(32, 120) >= 0.5, 1, 0)
        >>> x_clamped_loss = np.random.uniform(-10, 10, (32, 120)).astype(data_type)
        >>> act_ulq_clamp_max_grad = Q.ActULQClampMaxGrad()
        >>> clamp_max_grad = act_ulq_clamp_max_grad(Tensor(y_grad), Tensor(clamp_max_mask, mindspore.bool_),
                                                    Tensor(x_clamped_loss))
    """
    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, input_x, input_y, input_z):
        input_x_len = len(input_x)
        output_shape = []
        for _ in range(input_x_len):
            output_shape.append(1)
        return tuple(output_shape)

    def infer_dtype(self, input_x, input_y, input_z):
        return mstype.float32


class WtsARQ(PrimitiveWithInfer):
    """
    The WtsARQ(Weights Adaptive Range Quantization).

    Args:
        num_bits (int): The bits num used for quantize.
        offset_flag (bool): Whether use offset for quantize.

    Inputs:
        - **w** (Tensor) - A Tensor of weights. With float16 or float32 data type.

    Outputs:
        - **scale** (Tensor) - A tensor of optimal scale, has the same type as `w`.
        - **offset** (Tensor) - A tensor of optimal offset, has the same type as `w`.
        - If axis is [],
          the shape of scale and offset is :math:`(1, )`.
        - If axis is [0],
          the shape of scale and offset is :math:`(w_1, )`.
        - If axis is [1],
          the shape of scale and offset is :math:`(w_2, )`.
        - **y** (Tensor) - A tensor of fakequant weights, has the same type and shape as `w`.

    Examples:
        >>> data = Tensor(np.random.rand(1, 3, 6, 4).astype(np.float32))
        >>> wts_arq = Q.WtsARQ(axes=[0], num_bits=8, offset_flag=False)
        >>> scale, offset, y = wts_arq(data)
    """
    @prim_attr_register
    def __init__(self, num_bits, offset_flag):
        validator.check_value_type("num_bits", num_bits, [int], self.name)
        validator.check_int(num_bits, 8, validator.EQ, "value of num_bits", self.name)
        validator.check_value_type("offset_flag", offset_flag, [bool], self.name)

    def infer_shape(self, w_shape, w_min_shape, w_max_shape):
        validator.check_int(len(w_min_shape), len(w_shape), validator.EQ, "dims of w_min", self.name)
        validator.check_int(len(w_max_shape), len(w_shape), validator.EQ, "dims of w_max", self.name)
        return w_shape

    def infer_dtype(self, w_dtype, w_min_dtype, w_max_dtype):
        valid_types = [mstype.float32, mstype.float16]
        validator.check_tensor_dtype_valid("w", w_dtype, valid_types, self.name)
        validator.check_tensor_dtype_valid("w_min", w_min_dtype, valid_types, self.name)
        validator.check_tensor_dtype_valid("w_max", w_max_dtype, valid_types, self.name)
        return w_dtype


class IFMR(Primitive):
    """
    The TFMR(Input Feature Map Reconstruction).

    Args:
        min_percentile (float): Min init percentile. Default: 0.999999.
        max_percentile (float): Max init percentile. Default: 0.999999.
        search_range Union[list(float), tuple(float)]: Range of searching. Default: [0.7, 1.3].
        search_step (float): Step size of searching. Default: 0.01.
        with_offset (bool): Whether using offset. Default: ``True``.

    Inputs:
        - **data** (Tensor) - A Tensor of feature map. With float16 or float32 data type.
        - **data_min** (Tensor) - A Tensor of min value of feature map, the shape is :math:`(1)`.
          With float16 or float32 data type.
        - **data_max** (Tensor) - A Tensor of max value of feature map, the shape is :math:`(1)`.
          With float16 or float32 data type.
        - **cumsum** (Tensor) - A `1-D` Tensor of cumsum bin of data. With int32 data type.

    Outputs:
        - **scale** (Tensor) - A tensor of optimal scale, the shape is :math:`(1)`. Data dtype is float32.
        - **offset** (Tensor) - A tensor of optimal offset, the shape is :math:`(1)`. Data dtype is float32.

    Examples:
        >>> data = Tensor(np.random.rand(1, 3, 6, 4).astype(np.float32))
        >>> data_min = Tensor([0.1], mindspore.float32)
        >>> data_max = Tensor([0.5], mindspore.float32)
        >>> cumsum = Tensor(np.random.rand(4).astype(np.int32))
        >>> ifmr = Q.IFMR(min_percentile=0.2, max_percentile=0.9, search_range=(1.0, 2.0),
        ...               search_step=1.0, with_offset=False)
        >>> output = ifmr(data, data_min, data_max, cumsum)
        >>> print(output)
        (Tensor(shape=[1], dtype=Float32, value= [7.87401572e-03]),
         Tensor(shape=[1], dtype=Float32, value= [0.00000000e+00]))
    """

    @prim_attr_register
    def __init__(self, min_percentile=0.999999, max_percentile=0.999999, search_range=(0.7, 1.3), search_step=0.01,
                 with_offset=True):
        self.init_prim_io_names(
            inputs=['data', 'data_min', 'data_max', 'cumsum'], outputs=['scale', 'offset'])
        validator.check_value_type("min_percentile", min_percentile, [float], self.name)
        validator.check_value_type("max_percentile", max_percentile, [float], self.name)
        validator.check_value_type("search_range", search_range, [list, tuple], self.name)
        for item in search_range:
            validator.check_positive_float(item, "item of search_range", self.name)
        validator.check('search_range[1]', search_range[1], 'search_range[0]', search_range[0], validator.GE, self.name)
        validator.check_value_type("search_step", search_step, [float], self.name)
        validator.check_value_type("offset_flag", with_offset, [bool], self.name)
