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

"""Operators for gradients."""

from ..._c_expression import signature_rw as sig_rw
from ..._c_expression import signature_kind as sig_kind
from ..primitive import Primitive, PrimitiveWithInfer, prim_attr_register
from ..._checkparam import ParamValidator as validator
from ..._checkparam import Rel, check_int_positive, check_bool
from ...common import dtype as mstype


class AbsGrad(PrimitiveWithInfer):
    """Computes gradients for abs operation."""

    @prim_attr_register
    def __init__(self):
        """init AbsGrad"""

    def infer_shape(self, y, dy):
        return y

    def infer_dtype(self, y, dy):
        return y


class ACosGrad(PrimitiveWithInfer):
    """
    Computes ACosGrad of input element-wise.

    Returns:
        Tensor, has the same type as input.
    """

    @prim_attr_register
    def __init__(self):
        """init ACosGrad"""

    def infer_shape(self, x, dout):
        validator.check_param_equal("x", x, "dout", dout)
        return x

    def infer_dtype(self, x, dout):
        args = {"x": x, "dout": dout}
        validator.check_type_same(args, mstype.number_type)
        return x


class BatchNormGrad(PrimitiveWithInfer):
    """Performs grad of BatchNorm operation."""

    @prim_attr_register
    def __init__(self, is_training=False, epsilon=1e-5):
        self.is_training = validator.check_type('is_training', is_training, (bool,))
        self.epsilon = validator.check_number_range('epsilon', epsilon, 0, 1, Rel.INC_RIGHT)
        self.add_prim_attr('data_format', "NCHW")

    def infer_shape(self, y_backprop_shape, x_shape, scale_shape, reserve_1_shape, reserve_2_shape, reserve_3_shape):
        validator.check("BatchNorm y_backprop_shape", y_backprop_shape, "BatchNorm x_shape", x_shape)
        return (x_shape, scale_shape, scale_shape, reserve_1_shape, reserve_2_shape)

    def infer_dtype(self, y_backprop_type, x_type, scale_type, reserve_1_type, reserve_2_type, reserve_3_type):
        return (x_type, scale_type, scale_type, reserve_1_type, reserve_2_type)


class BiasAddGrad(Primitive):
    """Computes gradients of BiasAdd."""

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['dout'], outputs=['output'])
        self.add_prim_attr('data_format', 'NCHW')

    def __call__(self, d_output):
        raise NotImplementedError


class BinaryCrossEntropyGrad(PrimitiveWithInfer):
    """Computes gradients for `BinaryCrossEntropy` operation."""
    @prim_attr_register
    def __init__(self, reduction='mean'):
        self.reduction = validator.check_string('reduction', reduction, ['none', 'mean', 'sum'])

    def infer_shape(self, x_shape, y_shape, doutput_shape, weight_shape):
        validator.check_param_equal('x_shape', x_shape, 'y_shape', y_shape)
        if weight_shape:
            validator.check_param_equal('y_shape', y_shape, 'weight_shape', weight_shape)
        return x_shape

    def infer_dtype(self, x_type, y_type, doutput_type, weight_type):
        args = {'x_type': x_type, 'y_type': y_type, 'doutput_type': doutput_type}
        validator.check_type_same(args, (mstype.float16, mstype.float32))
        if weight_type:
            validator.check_two_types_same('x_type', x_type, 'weight_type', weight_type)
        return x_type


class Conv2DBackpropFilter(PrimitiveWithInfer):
    """
    Computes the gradients of convolution with respect to the filter.

    Args:
        out_channel (int): The dimensionality of the output space.
        kernel_size (Union[int, tuple[int]]): The size of the convolution window.
        pad_mode (str): "valid", "same", "pad" the mode to fill padding. Default: "valid".
        pad (int): The pad value to fill. Default: 0.
        mode (int): 0 Math convolutiuon, 1 cross-correlation convolution ,
                    2 deconvolution, 3 depthwise convolution. Default: 1.
        stride (int): The stride to apply conv filter. Default: 1.
        dilation (int): Specifies the dilation rate to use for dilated convolution. Default: 1.
        group (int): Splits input into groups. Default: 1.

    Returns:
        Tensor, the gradients of convolution.
    """

    @prim_attr_register
    def __init__(self,
                 out_channel,
                 kernel_size,
                 pad_mode="valid",
                 pad=0,
                 pad_list=(0, 0, 0, 0),
                 mode=1,
                 stride=1,
                 dilation=1,
                 group=1):
        """init Convolution"""
        self.init_prim_io_names(inputs=['out_backprop', 'input', 'filter_sizes'], outputs=['output'])
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.mode = mode
        pad_mode = pad_mode.upper()
        self.add_prim_attr('pad_mode', pad_mode)
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.group = group
        self.add_prim_attr('data_format', "NCHW")

    def __infer__(self, doutput, x, w_size):
        w_size_v = w_size['value']
        validator.check_type('w_size', w_size_v, [tuple])
        for i, dim_len in enumerate(w_size_v):
            validator.check_type("w_size[%d]" % i, dim_len, [int])
        validator.check_typename('x_dtype', x['dtype'], [mstype.int8, mstype.int32, mstype.float16, mstype.float32])
        validator.check_two_types_same('doutput_dtype', doutput['dtype'], 'x_dtype', x['dtype'])
        out = {
            'value': None,
            'shape': w_size_v,
            'dtype': doutput['dtype'],
            }
        return out


class DepthwiseConv2dNativeBackpropFilter(PrimitiveWithInfer):
    """
    Returns the gradient of filter for DepthwiseConv2dNative.

    Applies depthwise conv2d for the input, which will generate more channels with channel_multiplier.

    Refer to class DepthwiseConv2dNative for more details.

    Args:
        channel_multiplier (int): The multipiler for the original output conv.
        kernel_size (int or tuple): The size of the conv kernel.
        mode (int): 0 Math convolutiuon, 1 cross-correlation convolution,
                       2 deconvolution,3 depthwise convolution. Defaul: 3.
        pad_mode (str): The mode to fill padding which can be: "valid", "same" or "pad". Default: "valid".
        pad (int): The pad value to fill. Default: 0.
        pads (tuple): The pad list like (top, bottom, left, right). Default: (0, 0, 0, 0).
        stride (int): The stride to apply conv filter. Default: 1.
        dilation (int): Specifies the space to use between kernel elements. Default: 1.
        group (int): Splits input into groups. Default: 1.

    Returns:
        Tensor, the value is the gradient of filter for DepthwiseConv2dNative.
    """

    @prim_attr_register
    def __init__(self,
                 channel_multiplier,
                 kernel_size,
                 pad_mode="valid",
                 pad=0,
                 pads=(0, 0, 0, 0),
                 mode=3,
                 stride=1,
                 dilation=1,
                 group=1):
        """init Convolution"""
        self.init_prim_io_names(inputs=['input', 'filter_size', 'dout'], outputs=['output'])
        self.channel_multiplier = channel_multiplier
        self.kernel_size = kernel_size
        self.mode = mode
        self.pad_mode = pad_mode
        self.pad = pad
        self.pads = pads
        self.stride = stride
        self.dilation = dilation
        self.group = group
        self.add_prim_attr('data_format', "NCHW")

    def __call__(self, x, w_size, dout):
        raise NotImplementedError

    def __infer__(self, x, w_size, dout):
        w_size_v = w_size['value']
        args = {'x_dtype': x['dtype'], 'dout_type': dout['dtype']}
        validator.check_type_same(args, mstype.number_type)
        out = {
            'value': None,
            'shape': w_size_v,
            'dtype': dout['dtype'],
        }
        return out


class DepthwiseConv2dNativeBackpropInput(PrimitiveWithInfer):
    """
    Returns the gradient of input for DepthwiseConv2dNative.

    Applies depthwise conv2d for the input, which will generate more channels with channel_multiplier.

    Args:
        channel_multiplier (int): The multipiler for the original output conv.
        kernel_size (int or tuple): The size of the conv kernel.
        mode (int): 0 Math convolutiuon, 1 cross-correlation convolution ,
                    2 deconvolution,3 depthwise convolution. Default: 3.
        pad_mode (str):  "valid", "same", "pad" the mode to fill padding. Default: "valid".
        pad (int): the pad value to fill. Default: 0.
        pads (tuple): The pad list like (top, bottom, left, right). Default: (0, 0, 0, 0).
        stride (int): the stride to apply conv filter. Default: 1.
        dilation (int): Specifies the space to use between kernel elements. Default: 1.
        group (int): Splits input into groups. Default: 1.

    Returns:
        Tensor, the value is the gradient of input for DepthwiseConv2dNative.
    """

    @prim_attr_register
    def __init__(self,
                 channel_multiplier,
                 kernel_size,
                 pad_mode="valid",
                 pad=0,
                 pads=(0, 0, 0, 0),
                 mode=3,
                 stride=1,
                 dilation=1,
                 group=1):
        """init Convolution"""
        self.init_prim_io_names(inputs=['input_size', 'filter', 'dout'], outputs=['output'])
        self.channel_multiplier = channel_multiplier
        self.kernel_size = kernel_size
        self.mode = mode
        self.pad_mode = pad_mode
        self.pad = pad
        self.pads = pads
        self.stride = stride
        self.dilation = dilation
        self.group = group
        self.add_prim_attr('data_format', "NCHW")

    def __call__(self, x_size, w, dout):
        raise NotImplementedError

    def __infer__(self, x_size, w, dout):
        args = {'w_dtype': w['dtype'], 'dout_type': dout['dtype']}
        validator.check_type_same(args, mstype.number_type)
        x_size_v = x_size['value']
        out = {
            'value': None,
            'shape': x_size_v,
            'dtype': dout['dtype'],
        }
        return out


class FlattenGrad(PrimitiveWithInfer):
    """Performs gradients of Flatten."""

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x', 'shape'], outputs=['output'])

    def __infer__(self, *args):
        out = {
            'value': None,
            'shape': args[1]['value'],
            'dtype': args[0]['dtype'],
            }
        return out


class FusedBatchNormGrad(Primitive):
    """Gradients of FusedBatchNorm operation."""

    @prim_attr_register
    def __init__(self, epsilon=0.0, momentum=0.1):
        self.init_prim_io_names(inputs=['dy', 'x', 'scale', 'save_mean', 'save_inv_variance'],
                                outputs=['dx', 'bn_scale', 'bn_bias'])

    def __call__(self, dy, x, scale, save_mean, save_inv_variance):
        raise NotImplementedError


class GeluGrad(PrimitiveWithInfer):
    """Gradients of Gelu operation."""

    @prim_attr_register
    def __init__(self):
        """init GeluGrad"""

    def infer_shape(self, y_backprop_shape, x_shape, y_shape):
        return x_shape

    def infer_dtype(self, y_backprop_dtype, x_dtype, y_dtype):
        validator.check_typename("y_backprop_dtype", y_backprop_dtype, (mstype.float16, mstype.float32))
        validator.check_typename("x_dtype", x_dtype, (mstype.float16, mstype.float32))
        validator.check_typename("y_dtype", y_dtype, (mstype.float16, mstype.float32))
        return x_dtype


class _PoolGrad(PrimitiveWithInfer):
    """Gradients of the max/avg pool operation."""

    @prim_attr_register
    def __init__(self, ksize, strides, padding="VALID"):
        self.init_prim_io_names(inputs=['x_origin', 'out_origin', 'grad'], outputs=['output'])

        validator.check_type('ksize', ksize, [int, tuple])
        validator.check_type('strides', strides, [int, tuple])
        self.padding = validator.check_string('padding', padding.upper(), ['VALID', 'SAME'])
        self.add_prim_attr("padding", self.padding)
        self.is_maxpoolgradwithargmax = (self.name == "MaxPoolGradWithArgmax")
        if not self.is_maxpoolgradwithargmax:
            self.add_prim_attr('data_format', "NCHW")

        if isinstance(ksize, int):
            validator.check_integer("ksize", ksize, 1, Rel.GE)
            if self.is_maxpoolgradwithargmax:
                self.ksize = (1, ksize, ksize, 1)
            else:
                self.ksize = (1, 1, ksize, ksize)
        else:
            ksize_error = ValueError(f"The 'ksize' passed to operator {self.name} should be an positive int number"
                                     f"or a tuple of two or four positive int numbers, but got {ksize}")
            if len(ksize) != 2 and len(ksize) != 4:
                raise ksize_error
            for ksize_val in ksize:
                if not isinstance(ksize_val, int) or (ksize_val <= 0):
                    raise ksize_error
            if len(ksize) == 2 and self.is_maxpoolgradwithargmax:
                self.ksize = (1, ksize[0], ksize[1], 1)
            elif len(ksize) == 2 and not self.is_maxpoolgradwithargmax:
                self.ksize = (1, 1, ksize[0], ksize[1])
            else:
                self.ksize = ksize
        self.add_prim_attr("ksize", self.ksize)

        if isinstance(strides, int):
            validator.check_integer("strides", strides, 1, Rel.GE)
            if self.is_maxpoolgradwithargmax:
                self.strides = (1, strides, strides, 1)
            else:
                self.strides = (1, 1, strides, strides)
        else:
            strides_error = ValueError(f"The 'strides' passed to operator {self.name} should be an positive int number"
                                       f"or a tuple of two or four positive int numbers, but got {strides}")
            if len(strides) != 2 and len(strides) != 4:
                raise strides_error
            for strides_val in strides:
                if not isinstance(strides_val, int) or (strides_val <= 0):
                    raise strides_error
            if len(strides) == 2 and self.is_maxpoolgradwithargmax:
                self.strides = (1, strides[0], strides[1], 1)
            elif len(strides) == 2 and not self.is_maxpoolgradwithargmax:
                self.strides = (1, 1, strides[0], strides[1])
            else:
                self.strides = strides
        self.add_prim_attr("strides", self.strides)


class AvgPoolGrad(_PoolGrad):
    """Gradients of the avg pool operation."""

    @prim_attr_register
    def __init__(self, ksize=1, strides=1, padding="VALID"):
        super(AvgPoolGrad, self).__init__(ksize, strides, padding)

    def __infer__(self, origin_input, dout):
        out = {
            'value': None,
            'shape': tuple(origin_input['value']),
            'dtype': dout['dtype'],
        }

        return out


class AvgPoolGradGpu(_PoolGrad):
    """Gradients of the avg pool operation for gpu."""

    @prim_attr_register
    def __init__(self, ksize=1, strides=1, padding="VALID"):
        super(AvgPoolGradGpu, self).__init__(ksize, strides, padding)

    def infer_shape(self, x1_shape, x2_shape, grad_shape):
        return x1_shape

    def infer_dtype(self, x1_dtype, x2_dtype, grad_dtype):
        return x1_dtype


class MaxPoolGrad(_PoolGrad):
    """Performs gradients of the max pool operation."""

    @prim_attr_register
    def __init__(self, ksize=1, strides=1, padding="VALID"):
        super(MaxPoolGrad, self).__init__(ksize, strides, padding)

    def infer_shape(self, x1_shape, x2_shape, grad_shape):
        return x1_shape

    def infer_dtype(self, x1_dtype, x2_dtype, grad_dtype):
        return x1_dtype


class MaximumGrad(Primitive):
    """Grad for maximum."""

    @prim_attr_register
    def __init__(self, grad_x=True, grad_y=True):
        """Init MaximumGrad"""

    def __call__(self, x, y, dout):
        raise NotImplementedError


class MaxPoolGradWithArgmax(_PoolGrad):
    """Computes the gradients of MaxPoolWithArgmax."""

    @prim_attr_register
    def __init__(self, ksize=1, strides=1, padding="VALID",):
        self.init_prim_io_names(inputs=['x', 'grad', 'argmax'], outputs=['output'])
        super(MaxPoolGradWithArgmax, self).__init__(ksize, strides, padding)

    def infer_shape(self, x_shape, grad_shape, argmax_shape):
        if not grad_shape:
            raise TypeError("The dout of MaxPoolGradWithArgmax should be a Tensor.")
        return x_shape

    def infer_dtype(self, x_dtype, grad_dtype, argmax_dtype):
        return grad_dtype


class MinimumGrad(Primitive):
    """Grad for minimum."""

    @prim_attr_register
    def __init__(self, grad_x=True, grad_y=True):
        """Init MinimumGrad"""

    def __call__(self, x, y, dout):
        raise NotImplementedError


class L2NormalizeGrad(PrimitiveWithInfer):
    r"""
    Gradients of L2 normalize.

    Args:
        axis (int): The begin axis for the input to apply L2 normalize. Default: 0.
        epsilon (float): A small value added for numerical stability. Default: 1e-4.

    Inputs:
        - **input_x** (Tensor) - Should be the input `weight` of forward operator L2Normalize.
        - **out** (Tensor) - Should be the output of forward operator L2Normalize.
        - **dout** (Tensor) - The backprop of the next layer.

    Outputs:
        Tensor, gradients of L2Normalize `input_x`.
    """

    @prim_attr_register
    def __init__(self, axis=0, epsilon=1e-4):
        validator.check_type('axis', axis, [int])
        validator.check_type('epsilon', epsilon, [int, float])

    def infer_shape(self, input_x, out, dout):
        validator.check_param_equal('input_x', input_x, 'out', out)
        validator.check_param_equal('input_x', input_x, 'dout', dout)
        return input_x

    def infer_dtype(self, input_x, out, dout):
        args = {'input_x': input_x, 'out': out, 'dout': dout}
        validator.check_type_same(args, mstype.number_type)
        return input_x


class LayerNormGrad(Primitive):
    """
    Applies the layer normalization to the input array.

    This operator will calculate the input gradients of layernorm.

    Args:
        begin_norm_axis (int): The begin axis for the input to apply layernorm. Default: 1.
        begin_params_axis (int): The begin axis for the parameter input to apply layernorm. Default: 1.

    Returns:
        tuple[int], tuple of 3 values (the gradients of layernorm input,  gamma, beta).
    """

    @prim_attr_register
    def __init__(self, begin_norm_axis=1, begin_params_axis=1):
        """init"""
        self.begin_norm_axis = validator.check_type('begin_norm_axis', begin_norm_axis, [int])
        self.begin_params_axis = validator.check_type('begin_params_axis', begin_params_axis, [int])

    def __call__(self, x, dy, variance, mean, gamma):
        raise NotImplementedError


class LogSoftmaxGrad(PrimitiveWithInfer):
    """Computes gradient for the Log Softmax activation."""

    @prim_attr_register
    def __init__(self, axis=-1):
        """init LogSoftmaxGrad"""
        validator.check_type("axis", axis, [int])

    def infer_shape(self, dout, logits):
        rank = len(logits)
        validator.check_int_range('axis', self.axis, -rank - 1, rank, Rel.INC_BOTH)
        return logits

    def infer_dtype(self, dout, logits):
        validator.check_subclass("logits", logits, mstype.tensor)
        return logits


class LSTMGradData(PrimitiveWithInfer):
    """Computes the data gradients of LSTM."""

    @prim_attr_register
    def __init__(self, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout):
        self.input_size = check_int_positive(input_size)
        self.hidden_size = check_int_positive(hidden_size)
        self.num_layers = check_int_positive(num_layers)
        self.has_bias = check_bool(has_bias)
        self.bidirectional = check_bool(bidirectional)
        self.dropout = validator.check_type("dropout", dropout, [float])
        self.dropout = validator.check_number_range('dropout', dropout, 0, 1, Rel.INC_BOTH)

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

    def infer_shape(self, y_shape, dy_shape, dhy_shape, dcy_shape, w_shape,
                    hx_shape, cx_shape, reserve_shape, state_shape):
        # dhy and dcy should be same shape
        validator.check_integer("h_shape", len(dhy_shape), 3, Rel.EQ)
        validator.check_integer("h_shape", len(dhy_shape), len(dcy_shape), Rel.EQ)
        validator.check_integer("h_shape[0]", dhy_shape[0], dcy_shape[0], Rel.EQ)
        validator.check_integer("h_shape[1]", dhy_shape[1], dcy_shape[1], Rel.EQ)
        validator.check_integer("h_shape[2]", dhy_shape[2], dcy_shape[2], Rel.EQ)

        validator.check_integer("h_shape[0]", dhy_shape[0], self.num_layers * self.num_directions, Rel.EQ)
        validator.check_integer("h_shape[2]", dhy_shape[2], self.hidden_size, Rel.EQ)

        # dy: (seq_len, batch_size, hidden_size * num_directions)
        validator.check_integer("dy_shape", len(dy_shape), 3, Rel.EQ)
        validator.check_integer("dy[1]", dy_shape[1], dhy_shape[1], Rel.EQ)
        validator.check_integer("dy[2]", dy_shape[2], self.hidden_size * self.num_directions, Rel.EQ)

        # (seq_len, batch_size, input_size)
        dx_shape = (y_shape[0], y_shape[1], self.input_size)
        dhx_shape = dhy_shape
        dcx_shape = dcy_shape

        return (dx_shape, dhx_shape, dcx_shape)

    def infer_dtype(self, y_dtype, dy_dtype, dhy_dtype, dcy_dtype, w_dtype,
                    hx_dtype, cx_dtype, reserve_dtype, state_dtype):
        validator.check_typename("dy_dtype", dy_dtype, (mstype.float32, mstype.float16))
        validator.check_typename("dhy_dtype", dhy_dtype, (mstype.float32, mstype.float16))
        validator.check_typename("dcy_dtype", dcy_dtype, (mstype.float32, mstype.float16))
        validator.check_typename("datatype", dy_dtype, (dhy_dtype.element_type(),))
        validator.check_typename("datatype", dy_dtype, (dcy_dtype.element_type(),))
        return (dy_dtype, dy_dtype, dy_dtype)


class LSTMGradWeight(PrimitiveWithInfer):
    """Computes the weight gradients of LSTM."""

    @prim_attr_register
    def __init__(self, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout):
        self.input_size = check_int_positive(input_size)
        self.hidden_size = check_int_positive(hidden_size)
        self.num_layers = check_int_positive(num_layers)
        self.has_bias = check_bool(has_bias)
        self.bidirectional = check_bool(bidirectional)
        self.dropout = validator.check_type("dropout", dropout, [float])
        self.dropout = validator.check_number_range('dropout', dropout, 0, 1, Rel.INC_BOTH)

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

    def infer_shape(self, x_shape, hx_shape, y_shape, reserve_shape, state_shape):
        weight_size = 0
        gate_size = 4 * self.hidden_size
        for layer in range(self.num_layers):
            for _ in range(self.num_directions):
                input_layer_size = self.input_size if layer == 0 else self.hidden_size * self.num_directions
                weight_size += gate_size * input_layer_size
                weight_size += gate_size * self.hidden_size
                if self.has_bias:
                    weight_size += 2 * gate_size

        return (weight_size, 1, 1)

    def infer_dtype(self, x_dtype, hx_dtype, y_dtype, reserve_dtype, state_dtype):
        return hx_dtype


class PReLUGrad(PrimitiveWithInfer):
    r"""
    Gradients of PReLU operation.

    Inputs:
        - **y_backprop** (Tensor) - Representing the backprop of the next layer.
        - **input_x** (Tensor) - Should be the input `input_x` of forward operator PRelu.
        - **weight** (Tensor) - Float Tensor, w > 0, should be the input `weight` of forward operator PRelu.

    Outputs:
        Tensor, with the same type as `input_x`.
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, y_backprop_shape, A_shape, w_shape):
        return y_backprop_shape, w_shape

    def infer_dtype(self, y_backprop_dtype, A_dtype, w_dtype):
        validator.check_typename("y_backprop_dtype", y_backprop_dtype, (mstype.float16, mstype.float32))
        validator.check_typename("A_dtype", A_dtype, (mstype.float16, mstype.float32))
        validator.check_typename("w_dtype", w_dtype, (mstype.float16, mstype.float32))
        return y_backprop_dtype, w_dtype


class ReluGrad(Primitive):
    """Performs grad of Relu operation."""

    @prim_attr_register
    def __init__(self):
        """init ReluGrad"""
        self.init_prim_io_names(inputs=['y_backprop', 'x'], outputs=['output'])

    def __call__(self, y_backprop, x):
        raise NotImplementedError


class ReLU6Grad(PrimitiveWithInfer):
    """Performs grad of ReLU6 operation."""

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['y_grad', 'x'], outputs=['output'])

    def __call__(self, y_grad, x):
        raise NotImplementedError

    def infer_shape(self, y_grad_shape, x_shape):
        return x_shape

    def infer_dtype(self, y_grad_dtype, x_dtype):
        validator.check_typename("y_grad_dtype", y_grad_dtype, (mstype.float16, mstype.float32))
        validator.check_typename("x_dtype", x_dtype, (mstype.float16, mstype.float32))
        return x_dtype


class EluGrad(PrimitiveWithInfer):
    """Performs grad of Elu operation."""

    @prim_attr_register
    def __init__(self):
        """Init EluGrad"""

    def infer_shape(self, y_grad_shape, x_shape):
        return x_shape

    def infer_dtype(self, y_grad_dtype, x_dtype):
        args_type = {'y_grad': y_grad_dtype, 'x': x_dtype}
        validator.check_args_tensor(args_type)
        args_dtype = {'y_grad_dtype': y_grad_dtype, 'x_dtype': x_dtype}
        validator.check_type_same(args_dtype, mstype.float_type)
        return x_dtype


class ResizeBilinearGrad(PrimitiveWithInfer):
    """Performs grad of ResizeBilinear operation."""

    @prim_attr_register
    def __init__(self, align_corners=False):
        """init"""

    def infer_shape(self, dout_shape, orig_shape):
        return orig_shape

    def infer_dtype(self, dout_dtype, orig_type):
        return dout_dtype


class ResizeNearestNeighborGrad(PrimitiveWithInfer):
    """
    Compute gradient of `ResizeNearestNeighbor` operator.

    Note:
        The shape of input parameter `size` must be (height, width).

    Args:
        align_corners (bool): Whether the centers of the 4 corner pixels of the input
            and output tensors are aligned. Default: False.
    """

    @prim_attr_register
    def __init__(self, align_corners=False):
        """Init ResizeNearestNeighborGrad"""
        self.init_prim_io_names(inputs=['grads', 'size'], outputs=['y'])

    def __infer__(self, grads, size):
        shp = (grads['shape'][0],) + (grads['shape'][1],) + size['value']
        return {'shape': shp,
                'dtype': grads['dtype'],
                'value': None}


class ROIAlignGrad(PrimitiveWithInfer):
    """
    ROIAlignGrad operator.

    Args:
       pooled_height (int): The output feature height.
       pooled_width (int): The output feature width.
       spatial_scale (float): The feature stride.
       sample_num (int): Number of sampling points. Default: 2.
    """

    @prim_attr_register
    def __init__(self, xdiff_shape, pooled_height, pooled_width, spatial_scale, sample_num=2):
        """init ROIAlignGrad"""
        validator.check_type("pooled_height", pooled_height, [int])
        validator.check_type("pooled_width", pooled_width, [int])
        validator.check_type("spatial_scale", spatial_scale, [float])
        validator.check_type("sample_num", sample_num, [int])
        validator.check_type("xdiff_shape", xdiff_shape, [tuple])
        self.xdiff_shape = xdiff_shape
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.spatial_scale = spatial_scale
        self.sample_num = sample_num

    def infer_shape(self, ydiff_shape, rois_shape):
        return self.xdiff_shape

    def infer_dtype(self, ydiff_type, rois_type):
        return ydiff_type


class SigmoidGrad(PrimitiveWithInfer):
    """Gets the gradient of Sigmoid operation."""

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, out, dout):
        return out

    def infer_dtype(self, out, dout):
        validator.check_typename("dout dtype", dout, (mstype.float16, mstype.float32))
        validator.check_typename("out dtype", out, (mstype.float16, mstype.float32))
        args = {"out type": out, "dout type": dout}
        validator.check_type_same(args, mstype.number_type)
        return out


class SigmoidCrossEntropyWithLogitsGrad(PrimitiveWithInfer):
    """Computes the gradients of `SigmoidCrossEntropyWithLogits`."""

    @prim_attr_register
    def __init__(self):
        """Init SigmoidCrossEntropyWithLogitsGrad"""
        self.init_prim_io_names(inputs=['x', 'y', 'dout'], outputs=['x_grad'])

    def infer_shape(self, x_shape, y_shape, dout_shape):
        validator.check_param_equal("x_shape", x_shape, "y_shape", y_shape)
        validator.check_param_equal("x_shape", x_shape, "dout_shape", dout_shape)
        return x_shape

    def infer_dtype(self, x_dtype, y_dtype, dout_dtype):
        args = {"x_dtype": x_dtype, "y_dtype": y_dtype, 'dout_dtype': dout_dtype}
        validator.check_type_same(args, mstype.number_type)
        return dout_dtype


class SliceGrad(PrimitiveWithInfer):
    """Reverse of slice."""

    @prim_attr_register
    def __init__(self):
        """init SliceGrad"""
        self.init_prim_io_names(inputs=['dy', 'x', 'begin', 'size'], outputs=['dx'])

    def __infer__(self, dy, x, begin, size):
        dy_shape, x_shape, size_value = dy['shape'], x['shape'], size['value']
        dy_shape_len = len(dy_shape)
        for i in range(dy_shape_len):
            validator.check(f'dy_shape[{i}]', dy_shape[i], f'x_shape[{i}]', x_shape[i], Rel.LE)
            validator.check(f'dy_shape[{i}]', dy_shape[i], f'size_shape[{i}]', size_value[i], Rel.EQ)
        return {'shape': x_shape,
                'dtype': x['dtype'],
                'value': None}


class SmoothL1LossGrad(PrimitiveWithInfer):
    """Computes gradient for prediction on SmoothL1Loss."""

    @prim_attr_register
    def __init__(self, sigma=1.0):
        pass

    def infer_shape(self, prediction, target, dloss):
        validator.check_param_equal('prediction', prediction, 'target', target)
        validator.check_param_equal('prediction', prediction, 'dloss', dloss)
        return prediction

    def infer_dtype(self, prediction, target, dloss):
        args = {"prediction": prediction, "target": target, 'dloss': dloss}
        validator.check_type_same(args, mstype.number_type)
        return dloss


class StridedSliceGrad(PrimitiveWithInfer):
    """
    Performs grad of StridedSlice operation.

    Args:
        begin_mask (int): Start indexing the slice. Default: 0.
        end_mask (int): End indexing the slice. Default: 0.
        ellipsis_mask (int): An int32 mask. Default: 0.
        new_axis_mask (int): An int32 mask. Default: 0.
        shrink_axis_mask (int): An int32 mask. Default: 0.

    Returns:
        Tensor, has the same shape of input.
    """

    @prim_attr_register
    def __init__(self,
                 begin_mask=0,
                 end_mask=0,
                 ellipsis_mask=0,
                 new_axis_mask=0,
                 shrink_axis_mask=0):
        """init StrideSliceGrad"""
        validator.check_type('begin_mask', begin_mask, [int])
        validator.check_type('end_mask', end_mask, [int])
        validator.check_type('ellipsis_mask', ellipsis_mask, [int])
        validator.check_type('new_axis_mask', new_axis_mask, [int])
        validator.check_type('shrink_axis_mask', shrink_axis_mask, [int])
        self.init_prim_io_names(inputs=['dy', 'shapex', 'begin', 'end', 'strides'], outputs=['output'])

    def __infer__(self, dy, shapex, begin, end, strides):
        return {'shape': shapex['value'],
                'dtype': dy['dtype'],
                'value': None}


class TanhGrad(PrimitiveWithInfer):
    """Computes gradient of hyperbolic tangent of input element-wise."""

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, out, dout):
        return out

    def infer_dtype(self, out, dout):
        validator.check_subclass("out", out, mstype.tensor)
        validator.check_subclass("dout", dout, mstype.tensor)
        args = {"out type": out, "dout type": dout}
        validator.check_type_same(args, mstype.number_type)
        return out


class RefToEmbed(Primitive):
    r"""
    Make a key from Ref.

    The Key is a symbolic_key, is a embedding on Parameter, which is used as a key of the variable in env_type,
    and get items by operation `env_get_item` with the symbolic_key instance. The `Parameter` is a ref.

    Inputs:
        - **input** (Ref) - Target ref, ref is short for reference. The value of a Parameter is a ref.

    Outputs:
        symbolic_key, made from the Ref.

    Examples:
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.weight = mindspore.Parameter(1.0, name='weight')
        >>>
        >>>     def construct(self):
        >>>         key = RefToEmbed()(self.weight)
        >>>         return key, self.weight
    """
    __mindspore_signature__ = (
        ('variable', sig_rw.RW_REF, sig_kind.KIND_POSITIONAL_KEYWORD),
    )
    @prim_attr_register
    def __init__(self):
        pass
