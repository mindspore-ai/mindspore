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

"""Operators for gradients."""
import math
from functools import partial
from mindspore._checkparam import _check_3d_int_or_tuple
from .. import signature as sig
from ..primitive import Primitive, PrimitiveWithInfer, prim_attr_register
from ..._checkparam import Validator as validator, Rel
from .._utils import get_concat_offset
from ...common import dtype as mstype
from .. import functional as F
from ... import context


class AbsGrad(PrimitiveWithInfer):
    """Computes gradients for abs operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize AbsGrad"""

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
        """Initialize ACosGrad"""

    def infer_shape(self, x, dout):
        validator.check("x shape", x, "dout shape", dout, Rel.EQ, self.name)
        return x

    def infer_dtype(self, x, dout):
        args = {"x": x, "dout": dout}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        return x


class AcoshGrad(PrimitiveWithInfer):
    """Performs grad of Acosh operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize AcoshGrad"""

    def infer_shape(self, x, dout):
        validator.check("x shape", x, "dout shape", dout, Rel.EQ, self.name)
        return x

    def infer_dtype(self, x, dout):
        args = {"x": x, "dout": dout}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        return x


class AsinGrad(PrimitiveWithInfer):
    """
    Computes AsinGrad of input element-wise.

    Returns:
        Tensor, has the same type as input.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize AsinGrad"""

    def infer_shape(self, x, dout):
        validator.check("x shape", x, "dout shape", dout, Rel.EQ, self.name)
        return x

    def infer_dtype(self, x, dout):
        args = {"x": x, "dout": dout}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        return x


class AsinhGrad(PrimitiveWithInfer):
    """Performs grad of Asinh operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize AsinhGrad"""

    def infer_shape(self, x, dout):
        validator.check("x shape", x, "dout shape", dout, Rel.EQ, self.name)
        return x

    def infer_dtype(self, x, dout):
        args = {"x": x, "dout": dout}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        return x


class ReciprocalGrad(PrimitiveWithInfer):
    """Performs grad of Reciprocal operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize ReciprocalGrad"""

    def infer_shape(self, x_shape, dout_shape):
        validator.check("x shape", x_shape, "dout shape", dout_shape, Rel.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_dtype, dout_dtype):
        args = {"x": x_dtype, "dout": dout_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32], self.name)
        return x_dtype


class RsqrtGrad(PrimitiveWithInfer):
    """Performs grad of Rsqrt operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize RsqrtGrad"""

    def infer_shape(self, x_shape, dout_shape):
        validator.check("x shape", x_shape, "dout shape", dout_shape, Rel.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_dtype, dout_dtype):
        args = {"x": x_dtype, "dout": dout_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32, mstype.int32, mstype.int8],
                                                      self.name)
        return x_dtype


class SoftmaxGrad(PrimitiveWithInfer):
    """Performs grad of Softmax operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize SoftmaxGrad"""

    def infer_shape(self, x_shape, dout_shape):
        validator.check("x shape", x_shape, "dout shape", dout_shape, Rel.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_dtype, dout_dtype):
        args = {"x": x_dtype, "dout": dout_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32], self.name)
        return x_dtype


class SqrtGrad(PrimitiveWithInfer):
    """Performs grad of Sqrt operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize SqrtGrad"""

    def infer_shape(self, x_shape, dout_shape):
        validator.check("x shape", x_shape, "dout shape", dout_shape, Rel.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_dtype, dout_dtype):
        args = {"x": x_dtype, "dout": dout_dtype}
        valid_types = [mstype.float16, mstype.float32, mstype.float64]
        validator.check_tensors_dtypes_same_and_valid(args, valid_types, self.name)
        return x_dtype


class BatchNormGrad(PrimitiveWithInfer):
    """Performs grad of BatchNorm operation."""

    @prim_attr_register
    def __init__(self, is_training=False, epsilon=1e-5, data_format='NCHW'):
        self.is_training = validator.check_value_type('is_training', is_training, (bool,), self.name)
        self.epsilon = validator.check_float_range(epsilon, 0, 1, Rel.INC_RIGHT, 'epsilon', self.name)
        self.data_format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.name)

    def infer_shape(self, y_backprop_shape, x_shape, scale_shape, save_mean_shape, save_variance_shape, reserve):
        validator.check("BatchNorm y_backprop_shape", y_backprop_shape, "BatchNorm x_shape", x_shape)
        return (x_shape, scale_shape, scale_shape)

    def infer_dtype(self, y_backprop_type, x_type, scale_type, save_mean_shape, save_variance_shape, reserve):
        return (x_type, scale_type, scale_type)


class SyncBatchNormGrad(PrimitiveWithInfer):
    """Performs grad of SyncBatchNorm operation."""

    @prim_attr_register
    def __init__(self, epsilon=1e-5, group="group0", device_num=2):
        validator.check_float_range(epsilon, 0, 1, Rel.INC_RIGHT, 'epsilon', self.name)
        if not isinstance(group, str):
            raise TypeError("The group attr of SyncBatchNormGrad should be str.")
        validator.check_int(device_num, 2, Rel.GE, "device_num", self.name)

    def infer_shape(self, y_backprop_shape, x_shape, scale_shape, save_mean_shape, save_variance_shape):
        validator.check("BatchNorm y_backprop_shape", y_backprop_shape, "BatchNorm x_shape", x_shape)
        return (x_shape, scale_shape, scale_shape)

    def infer_dtype(self, y_backprop_type, x_type, scale_type, save_mean_shape, save_variance_shape):
        return (x_type, scale_type, scale_type)


class BiasAddGrad(PrimitiveWithInfer):
    """Computes gradients of BiasAdd."""

    @prim_attr_register
    def __init__(self, data_format="NCHW"):
        self.init_prim_io_names(inputs=['dout'], outputs=['output'])
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC', 'NCDHW'], 'format', self.name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError("NHWC format only support in GPU target.")
        if self.format == "NCDHW":
            self.format = "NCHW"
        self.add_prim_attr('data_format', self.format)

    def infer_shape(self, d_output):
        channel = d_output[-1] if self.format == "NHWC" else d_output[1]
        return (channel,)

    def infer_dtype(self, dout_dtype):
        return dout_dtype


class KLDivLossGrad(PrimitiveWithInfer):
    """Computes gradients for `KLDivLoss` operation."""

    @prim_attr_register
    def __init__(self, reduction='mean'):
        self.reduction = validator.check_string(reduction, ['none', 'mean', 'sum'], 'reduction', self.name)

    def infer_shape(self, x_shape, y_shape, doutput_shape):
        validator.check('x_shape', x_shape, 'y_shape', y_shape, Rel.EQ, self.name)
        return x_shape, y_shape

    def infer_dtype(self, x_type, y_type, doutput_type):
        args = {'x_type': x_type, 'y_type': y_type, 'doutput_type': doutput_type}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float16, mstype.float32), self.name)
        return x_type, y_type


class BinaryCrossEntropyGrad(PrimitiveWithInfer):
    """Computes gradients for `BinaryCrossEntropy` operation."""

    @prim_attr_register
    def __init__(self, reduction='mean'):
        self.reduction = validator.check_string(reduction, ['none', 'mean', 'sum'], 'reduction', self.name)

    def infer_shape(self, x_shape, y_shape, doutput_shape, weight_shape):
        validator.check('x_shape', x_shape, 'y_shape', y_shape, Rel.EQ, self.name)
        if weight_shape:
            validator.check('y_shape', y_shape, 'weight_shape', weight_shape, Rel.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_type, y_type, doutput_type, weight_type):
        args = {'x_type': x_type, 'y_type': y_type, 'doutput_type': doutput_type}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float16, mstype.float32), self.name)
        if weight_type:
            validator.check('x_type', x_type, 'weight_type', weight_type, Rel.EQ, TypeError)
        return x_type


class ConcatOffset(PrimitiveWithInfer):
    """primitive for computing Concat's gradient."""

    @prim_attr_register
    def __init__(self, N=2, axis=0):
        """Initialize ConcatOffset"""

    def __infer__(self, input_x):
        axis = self.axis
        x_shp = input_x['shape']
        x_type = input_x['dtype']
        offset, _, axis = get_concat_offset(x_shp, x_type, axis, self.name)
        self.add_prim_attr('T', x_type[0].element_type())
        offset_values = []
        for i in range(len(x_shp)):
            values = []
            for j in range(len(x_shp[0])):
                value = 0
                if j == axis:
                    value = offset[i]
                values.append(value)
            offset_values.append(tuple(values))
        out = {'shape': None,
               'dtype': None,
               'value': tuple(offset_values)}
        return out


class Conv3DBackpropFilter(PrimitiveWithInfer):
    """
    Computes the gradients of convolution 3D with respect to the filter.

    Args:
        out_channel (int): The dimension of the output.
        kernel_size (Union[int, tuple[int]]): The kernel size of the 3D convolution.
        mode (int): Modes for different convolutions. Not currently used.
        pad_mode (str): Modes to fill padding. It could be "valid", "same", or "pad". Default: "valid".
        pad (Union(int, tuple[int])): The pad value to be filled. Default: 0. If `pad` is an integer, the paddings of
                    head, tail, top, bottom, left and right are the same, equal to pad. If `pad` is a tuple of four
                    integers, the padding of head, tail, top, bottom, left and right equal to pad[0], pad[1], pad[2],
                    pad[3], pad[4] and pad[5] correspondingly.
        stride (Union(int, tuple[int])): The stride to be applied to the convolution filter. Default: 1.
        dilation (Union(int, tuple[int])): Specifies the space to use between kernel elements. Default: 1.
        group (int): Splits input into groups. Default: 1.
        data_format (str): The optional value for data format. Currently only support 'NCDHW'.

    Inputs:
        - **x** (Tensor) - The input of the convolution, then the shape is :math:`(C_{out}, C_{in}, D_{in}, K_1, K_2)`.
          Currently dout data type only support float16 and float32.
        - **dout** (Tensor) - The gradients w.r.t the output of the convolution. The shape conforms to the default
          data_format :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`. Currently dout data type only support float16
          and float32.
        - **w_size** (tuple(int)) - A tuple describes the shape of the weight which conforms to the format
          :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, the gradients w.r.t the weight of convolution 3D. It has the same shape as the weight.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.ones([16, 32, 13, 37, 33]), mindspore.float16)
        >>> dout = Tensor(np.ones([16, 32, 10, 32, 32]), mindspore.float16)
        >>> w = Tensor(np.ones([32, 32, 4, 6, 2]), mindspore.float16)
        >>> conv3d_backprop_input = P.Conv3DBackpropInput(out_channel=4, kernel_size=(4, 6, 2))
        >>> output = conv3d_backprop_input(x, dout, F.shape(w))
        >>> print(output.shape)
        (32, 32, 4, 6, 2)
    """


    @prim_attr_register
    def __init__(self,
                 out_channel,
                 kernel_size,
                 mode=1,
                 pad_mode="valid",
                 pad=0,
                 stride=(1, 1, 1, 1, 1),
                 dilation=(1, 1, 1, 1, 1),
                 group=1,
                 data_format="NCDHW"):
        """Initialize Convolution"""
        self.init_prim_io_names(inputs=['x', 'out_backprop', 'filter_size'], outputs=['y'])
        self.out_channel = validator.check_positive_int(out_channel, 'out_channel', self.name)
        self.kernel_size = _check_3d_int_or_tuple('kernel_size', kernel_size, self.name)
        self.stride = _check_3d_int_or_tuple('stride', stride, self.name, allow_five=True, ret_five=True)
        self.add_prim_attr('strides', self.stride)
        self.dilation = _check_3d_int_or_tuple('dilation', dilation, self.name, allow_five=True, ret_five=True)
        self.add_prim_attr('dilations', self.dilation)
        validator.check_value_type('pad', pad, (int, tuple), self.name)
        if isinstance(pad, int):
            pad = (pad,) * 6
        validator.check_equal_int(len(pad), 6, 'pad size', self.name)
        self.add_prim_attr('pad', self.pad)
        self.pad_list = pad
        self.add_prim_attr('pad_list', self.pad_list)

        self.pad_mode = validator.check_string(pad_mode.lower(), ['valid', 'same', 'pad'], 'pad_mode', self.name)
        if self.pad_mode != 'pad' and self.pad_list != (0, 0, 0, 0, 0, 0):
            raise ValueError(f"For '{self.name}', when pad is not 0, pad_mode should be set as 'pad'.")
        if self.pad_mode == 'pad':
            for item in pad:
                validator.check_non_negative_int(item, 'pad item', self.name)
        self.add_prim_attr('pad_mode', self.pad_mode)

        self.mode = validator.check_equal_int(mode, 1, 'mode', self.name)
        self.add_prim_attr('mode', self.mode)
        self.group = validator.check_positive_int(group, 'group', self.name)
        self.add_prim_attr('groups', self.group)
        self.format = validator.check_string(data_format, ['NCDHW'], 'format', self.name)
        self.add_prim_attr('data_format', self.format)

    def __infer__(self, x, doutput, w_size):
        w_size_v = w_size['value']
        validator.check_value_type('w_size', w_size_v, [tuple], self.name)
        for i, dim_len in enumerate(w_size_v):
            validator.check_value_type("w_size[%d]" % i, dim_len, [int], self.name)
        args = {"x": x['dtype'], "doutput": doutput['dtype']}
        valid_dtypes = [mstype.float16, mstype.float32]
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)

        validator.check("filter's batch", w_size_v[0], "dout's channel", doutput['shape'][1], Rel.EQ, self.name)
        validator.check("filter's channel", w_size_v[1], "input_size's channel", x['shape'][1], Rel.EQ, self.name)
        validator.check("input_size's batch", x['shape'][0], "dout's batch", doutput['shape'][0], Rel.EQ, self.name)

        # infer shape
        x_shape = x['shape']
        dout_shape = doutput['shape']
        kernel_d = self.kernel_size[0]
        kernel_h = self.kernel_size[1]
        kernel_w = self.kernel_size[2]
        stride_d = self.stride[2]
        stride_h = self.stride[3]
        stride_w = self.stride[4]
        dilation_d = self.dilation[2]
        dilation_h = self.dilation[3]
        dilation_w = self.dilation[4]
        # The pad_mode is valid by default. If pad_mode is not valid or same, then pad.
        if self.pad_mode == "valid":
            self.pad_list = (0, 0, 0, 0, 0, 0)
        if self.pad_mode == "same":
            pad_needed_d = max(0, (dout_shape[2] - 1) * stride_d + dilation_d * (kernel_d - 1) + 1 - x_shape[2])
            pad_head = math.floor(pad_needed_d / 2)
            pad_tail = pad_needed_d - pad_head

            pad_needed_h = max(0, (dout_shape[3] - 1) * stride_h + dilation_h * (kernel_h - 1) + 1 - x_shape[3])
            pad_top = math.floor(pad_needed_h / 2)
            pad_bottom = pad_needed_h - pad_top

            pad_needed_w = max(0, (dout_shape[4] - 1) * stride_w + dilation_w * (kernel_w - 1) + 1 - x_shape[4])
            pad_left = math.floor(pad_needed_w / 2)
            pad_right = pad_needed_w - pad_left
            self.pad_list = (pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right)

        self.add_prim_attr('pad_list', self.pad_list)
        out = {
            'value': None,
            'shape': w_size_v,
            'dtype': mstype.float32,
        }
        return out


class Conv2DBackpropFilter(PrimitiveWithInfer):
    """
    Computes the gradients of convolution with respect to the filter.

    Args:
        out_channel (int): The dimensionality of the output space.
        kernel_size (Union[int, tuple[int]]): The size of the convolution window.
        pad_mode (str): Modes to fill padding. It could be "valid", "same", or "pad". Default: "valid".
        pad (Union(int, tuple[int])): The pad value to be filled. Default: 0. If `pad` is an integer, the paddings of
                    top, bottom, left and right are the same, equal to pad. If `pad` is a tuple of four integers, the
                    padding of top, bottom, left and right equal to pad[0], pad[1], pad[2], and pad[3] correspondingly.
        pad_list (tuple): The pad list like (top, bottom, left, right). Default: (0, 0, 0, 0).
        mode (int): Modes for different convolutions. 0 Math convolutiuon, 1 cross-correlation convolution ,
                    2 deconvolution, 3 depthwise convolution. Default: 1.
        stride (tuple): The stride to be applied to the convolution filter. Default: (1, 1).
        dilation (tuple): Specifies the dilation rate to be used for the dilated convolution. Default: (1, 1, 1, 1).
        group (int): Splits input into groups. Default: 1.
        data_format (str) - The format of input and output data. It should be 'NHWC' or 'NCHW'，\
            default is 'NCHW'.

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
                 stride=(1, 1),
                 dilation=(1, 1, 1, 1),
                 group=1,
                 data_format="NCHW"):
        """Initialize Convolution"""
        self.init_prim_io_names(inputs=['out_backprop', 'input', 'filter_sizes'], outputs=['output'])
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.mode = mode
        pad_mode = pad_mode.upper()
        self.add_prim_attr('pad_mode', pad_mode)
        if isinstance(pad, int):
            pad = (pad,) * 4
        else:
            validator.check_equal_int(len(pad), 4, 'pad size', self.name)
        self.add_prim_attr("pad", pad)
        if isinstance(stride, tuple) and len(stride) == 4:
            self.stride = (stride[2], stride[3])
            self.add_prim_attr('stride', self.stride)
        self.dilation = dilation
        self.group = group
        self.add_prim_attr('groups', group)
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError("NHWC format only support in GPU target.")
        self.add_prim_attr('data_format', self.format)

    def __infer__(self, doutput, x, w_size):
        w_size_v = w_size['value']
        validator.check_value_type('w_size', w_size_v, [tuple], self.name)
        for i, dim_len in enumerate(w_size_v):
            validator.check_value_type("w_size[%d]" % i, dim_len, [int], self.name)
        args = {"x": x['dtype'], "doutput": doutput['dtype']}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.int8, mstype.int32, mstype.float16, mstype.float32],
                                                      self.name)
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
        channel_multiplier (int): The multiplier for the original output conv.
        kernel_size (int or tuple): The size of the conv kernel.
        mode (int): Modes for different convolutions. 0 Math convolutiuon, 1 cross-correlation convolution,
                       2 deconvolution,3 depthwise convolution. Default: 3.
        pad_mode (str): The mode to fill padding which can be: "valid", "same" or "pad". Default: "valid".
        pad (Union(int, tuple[int])): The pad value to be filled. Default: 0. If `pad` is an integer, the paddings of
                    top, bottom, left and right are the same, equal to pad. If `pad` is a tuple of four integers, the
                    padding of top, bottom, left and right equal to pad[0], pad[1], pad[2], and pad[3] correspondingly.
        pad_list (tuple): The pad list like (top, bottom, left, right). Default: (0, 0, 0, 0).
        stride (int): The stride to be applied to the convolution filter. Default: 1.
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
                 pad_list=(0, 0, 0, 0),
                 mode=3,
                 stride=1,
                 dilation=1,
                 group=1):
        """Initialize Convolution"""
        self.init_prim_io_names(inputs=['input', 'filter_size', 'dout'], outputs=['output'])
        self.channel_multiplier = channel_multiplier
        self.kernel_size = kernel_size
        self.mode = mode
        self.pad_mode = pad_mode
        if isinstance(pad, int):
            pad = (pad,) * 4
        else:
            validator.check_equal_int(len(pad), 4, 'pad size', self.name)
        self.add_prim_attr("pad", pad)
        self.pad_list = pad_list
        self.stride = stride
        self.dilation = dilation
        self.group = group
        self.add_prim_attr('data_format', "NCHW")

    def __call__(self, x, w_size, dout):
        raise NotImplementedError

    def __infer__(self, x, w_size, dout):
        w_size_v = w_size['value']
        args = {'x': x['dtype'], 'dout': dout['dtype']}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
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
        channel_multiplier (int): The multiplier for the original output conv.
        kernel_size (int or tuple): The size of the conv kernel.
        mode (int): Modes for different convolutions. 0 Math convolutiuon, 1 cross-correlation convolution ,
                    2 deconvolution,3 depthwise convolution. Default: 3.
        pad_mode (str):  Modes to fill padding. It could be "valid", "same", or "pad". Default: "valid".
        pad (Union(int, tuple[int])): The pad value to be filled. Default: 0. If `pad` is an integer, the paddings of
                    top, bottom, left and right are the same, equal to pad. If `pad` is a tuple of four integers, the
                    padding of top, bottom, left and right equal to pad[0], pad[1], pad[2], and pad[3] correspondingly.
        pad_list (tuple): The pad list like (top, bottom, left, right). Default: (0, 0, 0, 0).
        stride (int): The stride to be applied to the convolution filter. Default: 1.
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
                 pad_list=(0, 0, 0, 0),
                 mode=3,
                 stride=1,
                 dilation=1,
                 group=1):
        """Initialize Convolution"""
        self.init_prim_io_names(inputs=['input_size', 'filter', 'dout'], outputs=['output'])
        self.channel_multiplier = channel_multiplier
        self.kernel_size = kernel_size
        self.mode = mode
        self.pad_mode = pad_mode
        if isinstance(pad, int):
            pad = (pad,) * 4
        else:
            validator.check_equal_int(len(pad), 4, 'pad size', self.name)
        self.add_prim_attr("pad", pad)
        self.pad_list = pad_list
        self.stride = stride
        self.dilation = dilation
        self.group = group
        self.add_prim_attr('data_format', "NCHW")

    def __call__(self, x_size, w, dout):
        raise NotImplementedError

    def __infer__(self, x_size, w, dout):
        args = {'w': w['dtype'], 'dout': dout['dtype']}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        x_size_v = x_size['value']
        out = {
            'value': None,
            'shape': x_size_v,
            'dtype': dout['dtype'],
        }
        return out


class DropoutGrad(PrimitiveWithInfer):
    """
    The gradient of Dropout. During training, randomly zeroes some of the elements
    of the input tensor with probability.

    Args:
        keep_prob (float): The keep rate, between 0 and 1, e.g. keep_prob = 0.9,
          means dropping out 10% of input units.

    Inputs:
        - **shape** (tuple[int]) - The shape of target mask.

    Outputs:
        Tensor, the value of generated mask for input shape.

    Examples:
        >>> dropout_grad = ops.DropoutGrad(keep_prob=0.5)
        >>> in = Tensor((20, 16, 50, 50))
        >>> out = dropout_grad(in)
    """

    @prim_attr_register
    def __init__(self, keep_prob=0.5):
        self.keep_prob = validator.check_float_range(keep_prob, 0, 1, Rel.INC_RIGHT, "keep_prob", self.name)

    def infer_shape(self, dy_shape, mask_shape):
        return dy_shape

    def infer_dtype(self, dy_dtype, mask_dtype):
        valid_dtypes = (mstype.float16, mstype.float32)
        validator.check_subclass("mask", mask_dtype, mstype.tensor, self.name)
        validator.check_tensor_dtype_valid("dy", dy_dtype, valid_dtypes, self.name)
        return dy_dtype


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


class InstanceNormGrad(PrimitiveWithInfer):
    """Gradients of InstanceNorm operation."""

    @prim_attr_register
    def __init__(self, epsilon=0.0, momentum=0.1):
        self.init_prim_io_names(inputs=['dy', 'x', 'gamma', 'save_mean', 'save_variance'],
                                outputs=['dx', 'bn_gamma', 'bn_beta'])

    def infer_shape(self, y_backprop_shape, x_shape, gamma_shape, save_mean_shape, save_variance_shape):
        return (x_shape, gamma_shape, gamma_shape)

    def infer_dtype(self, y_backprop_type, x_type, gamma_type, save_mean_type, save_variance_type):
        return (x_type, gamma_type, gamma_type)


class UniqueGrad(Primitive):
    """Gradients of Unique operation."""

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['dy', 'y'], outputs=['dx'])

    def __call__(self, dy, x, scale, save_mean, save_inv_variance):
        raise NotImplementedError


class BNTrainingReduceGrad(PrimitiveWithInfer):
    """Gradients of FusedBatchNorm operation."""

    @prim_attr_register
    def __init__(self, epsilon=0.0001):
        _inputs = ['grads', 'x', 'diff_scale', 'diff_offset', 'scale', 'batch_mean', 'batch_variance']
        self.init_prim_io_names(inputs=_inputs, outputs=['y'])

    def infer_shape(self, grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance):
        return grads

    def infer_dtype(self, grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance):
        return grads


class BNTrainingUpdateGrad(PrimitiveWithInfer):
    """Gradients of FusedBatchNorm operation."""

    @prim_attr_register
    def __init__(self, epsilon=0.0001):
        self.init_prim_io_names(inputs=['grads', 'x', 'batch_mean', 'batch_variance'],
                                outputs=['diff_scale', 'diff_offset'])

    def infer_shape(self, grads, x, batch_mean, batch_variance):
        return (batch_mean, batch_variance)

    def infer_dtype(self, grads, x, batch_mean, batch_variance):
        return (batch_mean, batch_variance)


class GeLUGrad(PrimitiveWithInfer):
    """Gradients of GeLU operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize GeLUGrad"""

    def infer_shape(self, y_backprop_shape, x_shape, y_shape):
        return x_shape

    def infer_dtype(self, y_backprop_dtype, x_dtype, y_dtype):
        tuple(map(partial(validator.check_tensor_dtype_valid,
                          valid_dtypes=(mstype.float16, mstype.float32), prim_name=self.name),
                  ("y_backprop", "x", "y"),
                  (y_backprop_dtype, x_dtype, y_dtype)))
        return x_dtype


class FastGeLUGrad(PrimitiveWithInfer):
    """Gradients of FastGeLU operation."""

    @prim_attr_register
    def __init__(self):
        """init FastGeLUGrad"""

    def infer_shape(self, y_backprop_shape, x_shape):
        return x_shape

    def infer_dtype(self, y_backprop_dtype, x_dtype):
        tuple(map(partial(validator.check_tensor_dtype_valid,
                          valid_dtypes=(mstype.float16, mstype.float32), prim_name=self.name),
                  ("y_backprop", "x"),
                  (y_backprop_dtype, x_dtype)))
        return x_dtype


class _PoolGrad(PrimitiveWithInfer):
    """Gradients of the max/avg pool operation."""

    @prim_attr_register
    def __init__(self, kernel_size, strides, pad_mode="VALID", data_format="NCHW"):
        self.init_prim_io_names(inputs=['x_origin', 'out_origin', 'grad'], outputs=['output'])

        validator.check_value_type('kernel_size', kernel_size, [int, tuple], self.name)
        validator.check_value_type('strides', strides, [int, tuple], self.name)
        self.pad_mode = validator.check_string(pad_mode.upper(), ['VALID', 'SAME'], 'pad_mode', self.name)
        self.add_prim_attr("pad_mode", self.pad_mode)
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError("NHWC format only support in GPU target.")
        self.is_maxpoolgradwithargmax = (self.name == "MaxPoolGradWithArgmax")
        if not self.is_maxpoolgradwithargmax:
            self.add_prim_attr('data_format', self.format)

        def _grad_check_int_or_tuple(arg_name, arg_val, is_argmax):
            validator.check_value_type(arg_name, arg_val, (int, tuple), self.name)
            error_msg = ValueError(f"For '{self.name}' the '{arg_name}' should be an positive int number "
                                   f"or a tuple of two or four positive int numbers, but got {arg_val}")
            if isinstance(arg_val, int):
                ret = (1, arg_val, arg_val, 1) if is_argmax else (1, 1, arg_val, arg_val)
            elif len(arg_val) == 2:
                ret = (1, arg_val[0], arg_val[1], 1) if is_argmax else (1, 1, arg_val[0], arg_val[1])
            elif len(arg_val) == 4:
                ret = arg_val
            else:
                raise error_msg
            # whether all elements of tuple are positive integers
            for item in ret:
                if not isinstance(item, int) or item <= 0:
                    raise error_msg
            return ret

        kernel_size = _grad_check_int_or_tuple("kernel_size", kernel_size, self.is_maxpoolgradwithargmax)
        self.kernel_size = kernel_size if self.format == "NCHW" else [kernel_size[0], kernel_size[2],
                                                                      kernel_size[3], kernel_size[1]]
        self.add_prim_attr("kernel_size", self.kernel_size)

        strides = _grad_check_int_or_tuple("strides", strides, self.is_maxpoolgradwithargmax)
        self.strides = strides if self.format == "NCHW" else [strides[0], strides[2], strides[3], strides[1]]
        self.add_prim_attr("strides", self.strides)


class AvgPoolGradVm(_PoolGrad):
    """Gradients of the avg pool operation for vm."""

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="VALID"):
        super(AvgPoolGradVm, self).__init__(kernel_size, strides, pad_mode)
        self.init_prim_io_names(inputs=['x_origin', 'grad', 'mean_matrix', 'kernel_matrix'], outputs=['output'])

    def __infer__(self, origin_input, dout, mean_matrix, kernel_matrix):
        out = {
            'value': None,
            'shape': tuple(origin_input['value']),
            'dtype': dout['dtype'],
        }

        return out


class AvgPoolGrad(_PoolGrad):
    """Gradients of the avg pool operation."""

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="VALID", data_format="NCHW"):
        super(AvgPoolGrad, self).__init__(kernel_size, strides, pad_mode, data_format)

    def infer_shape(self, x1_shape, x2_shape, grad_shape):
        return x1_shape

    def infer_dtype(self, x1_dtype, x2_dtype, grad_dtype):
        return x1_dtype


class MaxPoolGrad(_PoolGrad):
    """Performs gradients of the max pool operation."""

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="VALID", data_format="NCHW"):
        super(MaxPoolGrad, self).__init__(kernel_size, strides, pad_mode, data_format)

    def infer_shape(self, x1_shape, x2_shape, grad_shape):
        return x1_shape

    def infer_dtype(self, x1_dtype, x2_dtype, grad_dtype):
        return x1_dtype


class MaxPoolGradGrad(_PoolGrad):
    r"""
    Performs gradients of the MaxPoolGrad operation.

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the maximum value,
            is an int number that represents height and width are both kernel_size, or a tuple
            of two int numbers that represent height and width respectively. Default: 1.
        strides (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.

    Inputs:
        - **origin_input** (Tensor) - Tensor with data format "NCHW", data type must be float16.
        - **origin_output** (Tensor) - Data type same as `origin_input`.
        - **grad** (Tensor) - Data type same as `origin_input`.

    Outputs:
        Tensor, with data type same as `origin_input`.

    """

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="VALID"):
        super(MaxPoolGradGrad, self).__init__(kernel_size, strides, pad_mode)

    def infer_shape(self, x1_shape, x2_shape, grad_shape):
        return x2_shape

    def infer_dtype(self, x1_dtype, x2_dtype, grad_dtype):
        args = {'x1_dtype': x1_dtype, 'x2_dtype': x2_dtype, 'grad_dtype': grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16], self.name)
        return x2_dtype


def _get_max_pool3d_grad_pads_by_pad_mode(input_shape, kernel_size, strides, pad_mode):
    """
    helper for get max pool3d grad pads by pad_mode
    """
    def get_pad(origin_shape, ksize, stride):
        tail = origin_shape % stride
        pad = (ksize - tail) if tail > 0 else (ksize - stride)
        pad = max(pad, 0)
        pad1 = int(pad / 2)
        pad2 = int(pad / 2) + pad % 2
        return pad1, pad2

    _, _, d, h, w = input_shape
    _, _, kd, kh, kw = kernel_size
    _, _, strd, strh, strw = strides

    pads = (0, 0, 0, 0, 0, 0)
    if pad_mode == 'SAME':
        pads_d = get_pad(d, kd, strd)
        pads_h = get_pad(h, kh, strh)
        pads_w = get_pad(w, kw, strw)
        pads = pads_d + pads_h + pads_w
    return pads

class MaxPool3DGrad(PrimitiveWithInfer):
    """Gradients of the max pool3d operation."""

    @prim_attr_register
    def __init__(self, kernel_size=(1, 1, 1, 1, 1), strides=(1, 1, 1, 1, 1), pad_mode='VALID', data_format="NCDHW"):
        validator.check_value_type('kernel_size', kernel_size, [int, tuple], self.name)
        validator.check_value_type('strides', strides, [int, tuple], self.name)
        validator.check_value_type('pad_mode', pad_mode, [str], self.name)
        self.format = validator.check_string(data_format, ['NCDHW'], 'format', self.name)
        self.pad_mode = validator.check_string(pad_mode.upper(), ['VALID', 'SAME'], 'pad_mode', self.name)
        self.kernel_size = _check_3d_int_or_tuple("kernel_size", kernel_size, self.name, allow_five=True, ret_five=True)
        self.add_prim_attr("kernel_size", self.kernel_size)
        self.strides = _check_3d_int_or_tuple("strides", strides, self.name, allow_five=True, ret_five=True)
        self.add_prim_attr("strides", self.strides)

    def infer_shape(self, x_shape, y_shape, grad_shape):
        validator.check_equal_int(len(x_shape), 5, "x rank", self.name)
        return x_shape

    def infer_dtype(self, x_dtype, y_dtype, grad_dtype):
        args = {'x_dtype': x_dtype, 'y_dtype': y_dtype, 'grad_dtype': grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32], self.name)
        return mstype.tensor_type(mstype.float32)


class MaxPool3DGradGrad(PrimitiveWithInfer):
    """Gradients of the max pool3d grad operation."""

    @prim_attr_register
    def __init__(self, kernel_size=(1, 1, 1, 1, 1), strides=(1, 1, 1, 1, 1), pad_mode='VALID', data_format="NCDHW"):
        validator.check_value_type('kernel_size', kernel_size, [int, tuple], self.name)
        validator.check_value_type('strides', strides, [int, tuple], self.name)
        validator.check_value_type('pad_mode', pad_mode, [str], self.name)
        self.format = validator.check_string(data_format, ['NCDHW'], 'format', self.name)
        self.pad_mode = validator.check_string(pad_mode.upper(), ['VALID', 'SAME'], 'pad_mode', self.name)
        self.kernel_size = _check_3d_int_or_tuple("kernel_size", kernel_size, self.name, allow_five=True, ret_five=True)
        self.add_prim_attr("kernel_size", self.kernel_size)
        self.strides = _check_3d_int_or_tuple("strides", strides, self.name, allow_five=True, ret_five=True)
        self.add_prim_attr("strides", self.strides)

    def infer_shape(self, x_shape, y_shape, grad_shape):
        validator.check_equal_int(len(x_shape), 5, "x rank", self.name)
        pad_list = _get_max_pool3d_grad_pads_by_pad_mode(x_shape, self.kernel_size, self.strides, self.pad_mode)
        for pad in pad_list:
            validator.check_non_negative_int(pad, 'element of pad_list', self.name)
        self.add_prim_attr("pad_list", pad_list)
        return y_shape

    def infer_dtype(self, x_dtype, y_dtype, grad_dtype):
        args = {'x_dtype': x_dtype, 'y_dtype': y_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32], self.name)
        validator.check_tensor_dtype_valid('grad_dtype', grad_dtype, [mstype.float16, mstype.float32], self.name)
        return x_dtype


class MaximumGrad(Primitive):
    """Grad for maximum."""

    @prim_attr_register
    def __init__(self, grad_x=True, grad_y=True):
        """Initialize MaximumGrad"""


class MaxPoolGradWithArgmax(_PoolGrad):
    """Computes the gradients of MaxPoolWithArgmax."""

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="VALID"):
        self.init_prim_io_names(inputs=['x', 'grad', 'argmax'], outputs=['output'])
        super(MaxPoolGradWithArgmax, self).__init__(kernel_size, strides, pad_mode)

    def infer_shape(self, x_shape, grad_shape, argmax_shape):
        if not grad_shape:
            raise TypeError("The dout of MaxPoolGradWithArgmax should be a Tensor.")
        return x_shape

    def infer_dtype(self, x_dtype, grad_dtype, argmax_dtype):
        return grad_dtype


class MaxPoolGradGradWithArgmax(_PoolGrad):
    r"""
    Computes the gradients of MaxPoolGradWithArgmax.

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the maximum value,
            is an int number that represents height and width are both kernel_size, or a tuple
            of two int numbers that represent height and width respectively. Default: 1.
        strides (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.

    Inputs:
        - **x** (Tensor) - Tensor with data format "NCHW", data type must be float16.
        - **grad** (Tensor) - Data type same as `x`.
        - **argmax** (Tensor) - Data type must be uint16 or int64.

    Outputs:
        Tensor, with data type same as `x`.

    """

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="VALID"):
        self.init_prim_io_names(inputs=['x', 'grad', 'argmax'], outputs=['output'])
        super(MaxPoolGradGradWithArgmax, self).__init__(kernel_size, strides, pad_mode)

    def infer_shape(self, x_shape, grad_shape, argmax_shape):
        if not grad_shape:
            raise TypeError("The dout of MaxPoolGradGradWithArgmax should be a Tensor.")
        return x_shape

    def infer_dtype(self, x_dtype, grad_dtype, argmax_dtype):
        args = {'x_dtype': x_dtype, 'grad_dtype': grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16], self.name)
        return grad_dtype


class MinimumGrad(Primitive):
    """Grad for minimum."""

    @prim_attr_register
    def __init__(self, grad_x=True, grad_y=True):
        """Initialize MinimumGrad"""

    def __call__(self, x, y, dout):
        raise NotImplementedError


class L2NormalizeGrad(PrimitiveWithInfer):
    r"""
    Gradients of L2 normalize.

    Args:
        axis (Union[list(int), tuple(int), int]): The begin axis for the input to apply L2 normalize. Default: 0.
        epsilon (float): A small value added for numerical stability. Default: 1e-4.

    Inputs:
        - **input_x** (Tensor) - Must be the input `weight` of forward operator L2Normalize.
        - **out** (Tensor) - Must be the output of forward operator L2Normalize.
        - **dout** (Tensor) - The backprop of the next layer.

    Outputs:
        Tensor, gradients of L2Normalize `input_x`.
    """

    @prim_attr_register
    def __init__(self, axis=0, epsilon=1e-4):
        axis = [axis] if isinstance(axis, int) else axis
        validator.check_value_type('axis', axis, [list, tuple], self.name)
        validator.check_value_type('epsilon', epsilon, [int, float], self.name)
        self.add_prim_attr('axis', axis)
        self.init_attrs['axis'] = axis
        if len(axis) != 1:
            raise TypeError("The length of axis must be 1, later will support multiple axis!")

    def infer_shape(self, input_x, out, dout):
        validator.check('input_x shape', input_x, 'out shape', out, Rel.EQ, self.name)
        validator.check('input_x shape', input_x, 'dout shape', dout, Rel.EQ, self.name)
        return input_x

    def infer_dtype(self, input_x, out, dout):
        args = {'input_x': input_x, 'out': out, 'dout': dout}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
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
        self.begin_norm_axis = validator.check_value_type('begin_norm_axis', begin_norm_axis, [int], self.name)
        self.begin_params_axis = validator.check_value_type('begin_params_axis', begin_params_axis, [int], self.name)

    def __call__(self, x, dy, variance, mean, gamma):
        raise NotImplementedError


class LayerNormGradGrad(PrimitiveWithInfer):
    """
    Gets the gradient of LayerNormGrad operation.

    Args:
        begin_norm_axis (int): The begin axis for the input to apply layernorm. Default: 1.
        begin_params_axis (int): The begin axis for the parameter input to apply layernorm. Default: 1.

    Returns:
        tuple[int], tuple of 3 values (the gradients of layernormgrad input, dy, gamma).
    """

    @prim_attr_register
    def __init__(self, begin_norm_axis=1, begin_params_axis=1):
        """init"""
        self.begin_norm_axis = validator.check_value_type('begin_norm_axis', begin_norm_axis, [int], self.name)
        self.begin_params_axis = validator.check_value_type('begin_params_axis', begin_params_axis, [int], self.name)

    def __call__(self, x, dy, variance, mean, gamma, grad_dx, grad_dg, grad_db):
        raise NotImplementedError

    def infer_shape(self, x, dy, variance, mean, gamma, grad_dx, grad_dg, grad_db):
        return x, dy, gamma

    def infer_dtype(self, x, dy, variance, mean, gamma, grad_dx, grad_dg, grad_db):
        return x, dy, gamma


class LogSoftmaxGrad(PrimitiveWithInfer):
    """Computes gradient for the Log Softmax activation."""

    @prim_attr_register
    def __init__(self, axis=-1):
        """Initialize LogSoftmaxGrad"""
        validator.check_value_type("axis", axis, [int], self.name)

    def infer_shape(self, dout, logits):
        rank = len(logits)
        validator.check_int_range(self.axis, -rank - 1, rank, Rel.INC_BOTH, 'axis', self.name)
        return logits

    def infer_dtype(self, dout, logits):
        validator.check_subclass("logits", logits, mstype.tensor, self.name)
        return logits


class LSTMGradData(PrimitiveWithInfer):
    """Computes the data gradients of LSTM."""

    @prim_attr_register
    def __init__(self, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout):
        self.input_size = validator.check_positive_int(input_size, 'input_size', self.name)
        self.hidden_size = validator.check_positive_int(hidden_size, 'hidden_size', self.name)
        self.num_layers = validator.check_positive_int(num_layers, 'num_layers', self.name)
        self.has_bias = validator.check_value_type('has_bias', has_bias, (bool,), self.name)
        self.bidirectional = validator.check_value_type('bidirectional', bidirectional, (bool,), self.name)
        self.dropout = validator.check_value_type("dropout", dropout, [float], self.name)
        self.dropout = validator.check_float_range(dropout, 0, 1, Rel.INC_BOTH, 'dropout', self.name)

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

    def infer_shape(self, y_shape, dy_shape, dhy_shape, dcy_shape, w_shape,
                    hx_shape, cx_shape, reserve_shape, state_shape):
        # dhy and dcy should be same shape
        validator.check_equal_int(len(dhy_shape), 3, "h_shape", self.name)
        validator.check_equal_int(len(dhy_shape), len(dcy_shape), "h_shape", self.name)
        validator.check_equal_int(dhy_shape[0], dcy_shape[0], "h_shape[0]", self.name)
        validator.check_equal_int(dhy_shape[1], dcy_shape[1], "h_shape[1]", self.name)
        validator.check_equal_int(dhy_shape[2], dcy_shape[2], "h_shape[2]", self.name)

        validator.check_int(dhy_shape[0], self.num_layers * self.num_directions, Rel.EQ, "h_shape[0]", self.name)
        validator.check_equal_int(dhy_shape[2], self.hidden_size, "h_shape[2]", self.name)

        # dy: (seq_len, batch_size, hidden_size * num_directions)
        validator.check_equal_int(len(dy_shape), 3, "dy_shape", self.name)
        validator.check_equal_int(dy_shape[1], dhy_shape[1], "dy[1]", self.name)
        validator.check_int(dy_shape[2], self.hidden_size * self.num_directions, Rel.EQ, "dy[2]", self.name)

        # (seq_len, batch_size, input_size)
        dx_shape = (y_shape[0], y_shape[1], self.input_size)
        dhx_shape = dhy_shape
        dcx_shape = dcy_shape

        return (dx_shape, dhx_shape, dcx_shape)

    def infer_dtype(self, y_dtype, dy_dtype, dhy_dtype, dcy_dtype, w_dtype,
                    hx_dtype, cx_dtype, reserve_dtype, state_dtype):
        args = {"dy": dy_dtype, "dhy": dhy_dtype, "dcy": dcy_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float32, mstype.float16), self.name)
        return (dy_dtype, dy_dtype, dy_dtype)


class LSTMGradWeight(PrimitiveWithInfer):
    """Computes the weight gradients of LSTM."""

    @prim_attr_register
    def __init__(self, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout):
        self.input_size = validator.check_positive_int(input_size, 'input_size', self.name)
        self.hidden_size = validator.check_positive_int(hidden_size, 'hidden_size', self.name)
        self.num_layers = validator.check_positive_int(num_layers, 'num_layers', self.name)
        self.has_bias = validator.check_value_type('has_bias', has_bias, (bool,), self.name)
        self.bidirectional = validator.check_value_type('bidirectional', bidirectional, (bool,), self.name)
        self.dropout = validator.check_value_type("dropout", dropout, [float], self.name)
        self.dropout = validator.check_float_range(dropout, 0, 1, Rel.INC_BOTH, 'dropout', self.name)

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


class LSTMGrad(PrimitiveWithInfer):
    """Computes the data and weight gradients of LSTM."""

    @prim_attr_register
    def __init__(self, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout):
        self.input_size = validator.check_positive_int(input_size, 'input_size', self.name)
        self.hidden_size = validator.check_positive_int(hidden_size, 'hidden_size', self.name)
        self.num_layers = validator.check_positive_int(num_layers, 'num_layers', self.name)
        self.has_bias = validator.check_value_type('has_bias', has_bias, (bool,), self.name)
        self.bidirectional = validator.check_value_type('bidirectional', bidirectional, (bool,), self.name)
        self.dropout = validator.check_value_type("dropout", dropout, [float], self.name)
        self.dropout = validator.check_float_range(dropout, 0, 1, Rel.INC_BOTH, 'dropout', self.name)

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

    def infer_shape(self, x_shape, hx_shape, cx_shape, w_shape, y_shape, hy_shape, cy_shape, dy_shape, dhy_shape,
                    dcy_shape, reserve_shape):
        # dhy and dcy should be same shape
        validator.check_equal_int(len(dhy_shape), 3, "h_shape", self.name)
        validator.check_equal_int(len(dhy_shape), len(dcy_shape), "h_shape", self.name)
        validator.check_equal_int(dhy_shape[0], dcy_shape[0], "h_shape[0]", self.name)
        validator.check_equal_int(dhy_shape[1], dcy_shape[1], "h_shape[1]", self.name)
        validator.check_equal_int(dhy_shape[2], dcy_shape[2], "h_shape[2]", self.name)

        validator.check_int(dhy_shape[0], self.num_layers * self.num_directions, Rel.EQ, "h_shape[0]", self.name)
        validator.check_equal_int(dhy_shape[2], self.hidden_size, "h_shape[2]", self.name)

        # dy: (seq_len, batch_size, hidden_size * num_directions)
        validator.check_equal_int(len(dy_shape), 3, "dy_shape", self.name)
        validator.check_equal_int(dy_shape[1], dhy_shape[1], "dy[1]", self.name)
        validator.check_int(dy_shape[2], self.hidden_size * self.num_directions, Rel.EQ, "dy[2]", self.name)

        # (seq_len, batch_size, input_size)
        dx_shape = (y_shape[0], y_shape[1], self.input_size)
        dhx_shape = dhy_shape
        dcx_shape = dcy_shape
        weight_size = 0
        gate_size = 4 * self.hidden_size
        for layer in range(self.num_layers):
            for _ in range(self.num_directions):
                input_layer_size = self.input_size if layer == 0 else self.hidden_size * self.num_directions
                weight_size += gate_size * input_layer_size
                weight_size += gate_size * self.hidden_size
                if self.has_bias:
                    weight_size += gate_size

        return (dx_shape, dhx_shape, dcx_shape, (weight_size, 1, 1))

    def infer_dtype(self, x_dtype, hx_dtype, cx_dtype, w_dtype, y_dtype, hy_dtype, cy_dtype, dy_dtype, dhy_dtype,
                    dcy_dtype, reserve_dtype):
        return (dy_dtype, dy_dtype, dy_dtype, hx_dtype)


class DynamicRNNGrad(PrimitiveWithInfer):
    """Computes the input gradients of DynamicRNN."""

    @prim_attr_register
    def __init__(self,
                 cell_type='LSTM',
                 direction='UNIDIRECTIONAL',
                 cell_depth=1,
                 use_peephole=False,
                 keep_prob=1.0,
                 cell_clip=-1.0,
                 num_proj=0,
                 time_major=True,
                 forget_bias=0.0):
        self.forget_bias = validator.check_value_type("forget_bias", forget_bias, [float], self.name)

    def infer_shape(self, x_shape, w_shape, b_shape, y_shape, init_h_shape, init_c_shape, h_shape,
                    c_shape, dy_shape, dh_shape, dc_shape, i_shape, j_shape, f_shape, o_shape, tanhc_shape):
        validator.check_equal_int(len(x_shape), 3, "x_shape", self.name)
        num_step, batch_size, input_size = x_shape
        hidden_size = w_shape[-1] // 4
        if w_shape[-1] % 4 != 0:
            raise ValueError(f"For {self.name}, w_shape[-1] should multiple of 4.")
        validator.check("w_shape[0]", w_shape[0], "input_size + hidden_size",
                        input_size + hidden_size, Rel.EQ, self.name)
        valid_shape = [num_step, batch_size, hidden_size]
        validator.check("b_shape[0]", b_shape[0], "w_shape[1]", w_shape[1], Rel.EQ, self.name)
        validator.check("y_shape", y_shape, "excepted shape", valid_shape, Rel.EQ, self.name)
        validator.check("h_shape", h_shape, "excepted shape", valid_shape, Rel.EQ, self.name)
        validator.check("c_shape", c_shape, "excepted shape", valid_shape, Rel.EQ, self.name)
        validator.check("i_shape", i_shape, "excepted shape", valid_shape, Rel.EQ, self.name)
        validator.check("j_shape", j_shape, "excepted shape", valid_shape, Rel.EQ, self.name)
        validator.check("f_shape", f_shape, "excepted shape", valid_shape, Rel.EQ, self.name)
        validator.check("o_shape", o_shape, "excepted shape", valid_shape, Rel.EQ, self.name)
        validator.check("tanhc_shape", tanhc_shape, "excepted shape", valid_shape, Rel.EQ, self.name)
        validator.check("dy_shape", dy_shape, "excepted shape", valid_shape, Rel.EQ, self.name)
        validator.check("dh_shape", dh_shape, "excepted shape", [batch_size, hidden_size], Rel.EQ, self.name)
        validator.check("dc_shape", dc_shape, "excepted shape", [batch_size, hidden_size], Rel.EQ, self.name)

        return w_shape, (w_shape[1],), x_shape, dh_shape, dc_shape

    def infer_dtype(self, x_dtype, w_dtype, b_dtype, y_dtype, init_h_dtype, init_c_dtype, h_dtype,
                    c_dtype, dy_dtype, dh_dtype, dc_dtype, i_dtype, j_dtype, f_dtype, o_dtype, tanhc_dtype):
        return x_dtype, x_dtype, x_dtype, x_dtype, x_dtype


class DynamicGRUV2Grad(PrimitiveWithInfer):
    r"""
    Computes the input gradients of DynamicGRUV2.

    Args:
        direction (str): A string identifying the direction in the op. Default: 'UNIDIRECTIONAL'.
            Only 'UNIDIRECTIONAL' is currently supported.
        cell_depth (int): An integer identifying the cell depth in the op. Default: 1.
        keep_prob (float): A float identifying the keep prob in the op. Default: 1.0.
        cell_clip (float): A float identifying the cell clip in the op. Default: -1.0.
        num_proj (int): An integer identifying the num proj in the op. Default: 0.
        time_major (bool): A bool identifying the time major in the op. Default: True.
        gate_order (str): An string identifying the gate order in weight and bias. Default: 'rzh.
            'zrh' is another option.
        reset_after (bool): An bool identifying whether to apply reset gate after matrix multiplication. Default: True.

    Inputs:
        - **x** (Tensor) - Current words. Tensor of shape :math:`(num_step, batch_size, input_size)`.
          The data type must be float16 or float32.
        - **weight_input** (Tensor) - Weight. Tensor of shape :math:`(input_size, 3 x hidden_size)`.
          The data type must be float16 or float32.
        - **weight_hidden** (Tensor) - Bias. Tensor of shape :math:`(hidden_size, 3 x hidden_size)`.
          The data type must be float16 or float32.
        - **y** (Tensor) - A Tensor of shape :math:
          if num_proj > 0 `(num_step, batch_size, min(hidden_size, num_proj)`,
          if num_proj == 0 `(num_step, batch_size, hidden_size)`.
          The data type must be float16 or float32.
        - **init_h** (Tensor) - Hidden state of initial time.
          Tensor of shape :math:`(batch_size, hidden_size)`.
          The data type must be float16 or float32.
        - **h** (Tensor) - A Tensor of shape :math:`(num_step, batch_size, hidden_size)`.
          The data type must be float16 or float32.
        - **dy** (Tensor) - Gradient of `y`, has the same shape and data type as `y`.
        - **dh** (Tensor) - Gradient of `h`, has the same shape and data type as `init_h`.
        - **update** (Tensor) - A Tensor of shape :math:`(num_step, batch_size, hidden_size)`.
          The data type must be float16 or float32.
        - **reset** (Tensor) - A Tensor of shape :math:`(num_step, batch_size, hidden_size)`.
          The data type must be float16 or float32.
        - **new** (Tensor) - A Tensor of shape :math:`(num_step, batch_size, hidden_size)`.
          The data type must be float16 or float32.
        - **hidden_new** (Tensor) - A Tensor of shape :math:`(num_step, batch_size, hidden_size)`.
          The data type must be float16 or float32.
        - **seq_length** (Tensor) - The length of each batch. Tensor of shape :math:`(batch_size)`.
          Only `None` is currently supported.
        - **mask** (Tensor) - A 4-D Tensor. The data type must be float16 or float32.

    Outputs:
        - **dw_input** (Tensor) - A Tensor has the same shape as `weight_input`.
          Has the same type with input `x`.
        - **dw_hidden** (Tensor) - A Tensor has the same shape as `weight_hidden`.
          Has the same type with input `x`.
        - **db_input** (Tensor) - A Tensor of shape :math:`(3 x hidden_size)`.
          Has the same type with input `x`.
        - **db_hidden** (Tensor) - A Tensor of shape :math:`(3 x hidden_size)`.
          Has the same type with input `x`.
        - **dx** (Tensor) - A Tensor of shape :math:`(num_step, batch_size, hidden_size)`.
          Has the same type with input `x`.
        - **dh_prev** (Tensor) - A Tensor of shape :math:`(batch_size, hidden_size)`.
          Has the same type with input `x`.
    """

    @prim_attr_register
    def __init__(self,
                 direction='UNIDIRECTIONAL',
                 cell_depth=1,
                 keep_prob=1.0,
                 cell_clip=-1.0,
                 num_proj=0,
                 time_major=True,
                 gate_order="rzh",
                 reset_after=True):
        self.cell_depth = validator.check_value_type("cell_depth", cell_depth, [int], self.name)
        self.keep_prob = validator.check_value_type("keep_prob", keep_prob, [float], self.name)
        self.cell_clip = validator.check_value_type("cell_clip", cell_clip, [float], self.name)
        self.num_proj = validator.check_non_negative_int(num_proj, "num_proj", self.name)
        self.time_major = validator.check_value_type("time_major", time_major, [bool], self.name)
        self.direction = validator.check_string(direction, ['UNIDIRECTIONAL'], "direction", self.name)
        self.gate_order = validator.check_string(gate_order, ['zrh', 'rzh'], "gate_order", self.name)
        self.reset_after = validator.check_value_type("reset_after", reset_after, [bool], self.name)

    def infer_shape(self, x_shape, winput_shape, whidden_shape, y_shape, init_h_shape, h_shape,
                    dy_shape, dh_shape, update_shape, reset_shape, new_shape, hnew_shape, seq_shape, mask_shape):
        validator.check_int(len(x_shape), 3, Rel.EQ, "x shape", self.name)
        validator.check_int(len(winput_shape), 2, Rel.EQ, "weight input shape rank", self.name)
        validator.check_int(len(whidden_shape), 2, Rel.EQ, "weight hidden shape rank", self.name)
        validator.check_int(len(y_shape), 3, Rel.EQ, "y shape rank", self.name)
        num_step, batch_size, input_size = x_shape
        hidden_size = whidden_shape[0]
        validator.check("weight_hidden_shape[-1]", whidden_shape[-1], "3 * hidden_size",
                        3 * hidden_size, Rel.EQ, self.name)
        validator.check("weight_input_shape", winput_shape, "excepted shape",
                        [input_size, 3 * hidden_size], Rel.EQ, self.name)
        if self.num_proj > 0:
            valid_y_shape = [num_step, batch_size, min(hidden_size, self.num_proj)]
        else:
            valid_y_shape = [num_step, batch_size, hidden_size]
        validator.check("y_shape", y_shape, "excepted shape", valid_y_shape, Rel.EQ, self.name)

        validator.check("init_h_shape", init_h_shape, "excepted shape",
                        [batch_size, hidden_size], Rel.EQ, self.name)
        valid_shape = [num_step, batch_size, hidden_size]
        validator.check("h_shape", h_shape, "excepted shape", valid_shape, Rel.EQ, self.name)
        validator.check("dy_shape", dy_shape, "excepted shape", valid_shape, Rel.EQ, self.name)
        validator.check("dh_shape", dh_shape, "excepted shape",
                        [batch_size, hidden_size], Rel.EQ, self.name)
        validator.check("update_shape", update_shape, "excepted shape", valid_shape, Rel.EQ, self.name)
        validator.check("reset_shape", reset_shape, "excepted shape", valid_shape, Rel.EQ, self.name)
        validator.check("new_shape", new_shape, "excepted shape", valid_shape, Rel.EQ, self.name)
        validator.check("hnew_shape", hnew_shape, "excepted shape", valid_shape, Rel.EQ, self.name)
        if seq_shape is not None:
            validator.check("seq_shape", seq_shape, "batch_size", batch_size, Rel.EQ, self.name)

        dx_shape = (num_step, batch_size, input_size)
        dh_shape = (batch_size, hidden_size)
        dwinput_shape = (input_size, 3 * hidden_size)
        dwhidden_shape = (hidden_size, 3 * hidden_size)
        db_shape = (3 * hidden_size,)
        return dwinput_shape, dwhidden_shape, db_shape, db_shape, dx_shape, dh_shape

    def infer_dtype(self, x_dtype, winput_dtype, whidden_dtype, y_dtype, init_h_dtype, h_dtype,
                    dy_dtype, dh_dtype, update_dtype, reset_dtype, new_dtype, hnew_dtype, seq_dtype, mask_dtype):
        valid_types = (mstype.float16, mstype.float32)
        args = {"y_dtype": y_dtype, "h_dtype": h_dtype, "dy_dtype": dy_dtype,
                "dh_dtype": dh_dtype, "update_dtype": update_dtype, "reset_dtype": reset_dtype,
                "new_dtype": new_dtype, "hnew_dtype": hnew_dtype}
        validator.check_tensor_dtype_valid("x_dtype", x_dtype, valid_types, self.name)
        validator.check_tensor_dtype_valid("winput_dtype", winput_dtype, valid_types, self.name)
        validator.check_tensor_dtype_valid("whidden_dtype", whidden_dtype, valid_types, self.name)
        validator.check_tensor_dtype_valid("init_h_dtype", init_h_dtype, valid_types, self.name)
        validator.check_tensors_dtypes_same_and_valid(args, valid_types, self.name)
        if seq_dtype is not None:
            validator.check_tensor_dtype_valid("seq_dtype", seq_dtype, valid_types, self.name)
        if mask_dtype is not None:
            validator.check_tensor_dtype_valid("mask_dtype", mask_dtype, valid_types, self.name)
        return x_dtype, x_dtype, x_dtype, x_dtype, x_dtype, x_dtype


class PReLUGrad(PrimitiveWithInfer):
    r"""
    Gradients of PReLU operation.

    Note:
        1-dimensional input_x is not supported.

    Inputs:
        - **y_backprop** (Tensor) - Representing the backprop of the next layer.
        - **input_x** (Tensor) - Must be the input `input_x` of forward operator PRelu.
        - **weight** (Tensor) - Float Tensor, w > 0, must be the input `weight` of forward operator PRelu.

    Outputs:
        Tensor, with the same type as `input_x`.
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, y_backprop_shape, A_shape, w_shape):
        if len(A_shape) == 1:
            raise ValueError(f'For \'{self.name}\' input_x rank 1 is not supported.')
        return y_backprop_shape, w_shape

    def infer_dtype(self, y_backprop_dtype, A_dtype, w_dtype):
        tuple(map(partial(validator.check_tensor_dtype_valid,
                          valid_dtypes=(mstype.float16, mstype.float32), prim_name=self.name),
                  ('y_backprop', "input_x", "weight"),
                  (y_backprop_dtype, A_dtype, w_dtype)))
        return y_backprop_dtype, w_dtype


class ReluGrad(Primitive):
    """Performs grad of Relu operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize ReluGrad"""
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
        valid_dtypes = (mstype.float16, mstype.float32)
        validator.check_tensor_dtype_valid("y_grad", y_grad_dtype, valid_dtypes, self.name)
        validator.check_tensor_dtype_valid("x", x_dtype, valid_dtypes, self.name)
        return x_dtype


class ReluGradV2(PrimitiveWithInfer):
    """Performs grad of ReLUV2 operation."""

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['gradients', 'mask'], outputs=['output'])

    def __call__(self, gradients, mask):
        raise NotImplementedError

    def infer_shape(self, gradients_shape, mask_shape):
        return gradients_shape

    def infer_dtype(self, gradients_dtype, mask_dtype):
        validator.check_tensor_dtype_valid('gradients', gradients_dtype, mstype.number_type, self.name)
        validator.check_tensor_dtype_valid('mask', mask_dtype, (mstype.uint8,), self.name)
        return gradients_dtype


class EluGrad(PrimitiveWithInfer):
    """Performs grad of Elu operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize EluGrad"""

    def infer_shape(self, y_grad_shape, x_shape):
        return x_shape

    def infer_dtype(self, y_grad_dtype, x_dtype):
        args = {'y_grad': y_grad_dtype, 'x': x_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.float_type, self.name)
        return x_dtype


class GatherDGrad(PrimitiveWithInfer):
    """Performs grad of GatherD operation."""

    @prim_attr_register
    def __init__(self, dim=0, shape=None):
        """Initialize GatherDGrad"""
        validator.check_is_int(dim, int)
        self.add_prim_attr("dim", dim)
        self.out_shape = shape
        self.init_prim_io_names(inputs=['index', 'grad'], outputs=['output'])

    def infer_shape(self, index_shape, grad_shape):
        return self.out_shape

    def infer_dtype(self, index_dtype, grad_dtype):
        return grad_dtype


class ResizeBilinearGrad(PrimitiveWithInfer):
    """Performs grad of ResizeBilinear operation."""

    @prim_attr_register
    def __init__(self, align_corners=False):
        """init"""

    def infer_shape(self, dout_shape, orig_shape):
        return orig_shape

    def infer_dtype(self, dout_dtype, orig_type):
        return orig_type


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
        """Initialize ResizeNearestNeighborGrad"""
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
        """Initialize ROIAlignGrad"""
        validator.check_value_type("pooled_height", pooled_height, [int], self.name)
        validator.check_value_type("pooled_width", pooled_width, [int], self.name)
        validator.check_value_type("spatial_scale", spatial_scale, [float], self.name)
        validator.check_value_type("sample_num", sample_num, [int], self.name)
        validator.check_value_type("xdiff_shape", xdiff_shape, [tuple], self.name)
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
        args = {'out': out, 'dout': dout}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        return out


class HSigmoidGrad(PrimitiveWithInfer):
    """Gets the gradient of HSigmoid operation."""

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['y_grad', 'x'], outputs=['output'])

    def infer_shape(self, y_grad_shape, x_shape):
        return x_shape

    def infer_dtype(self, y_grad_dtype, x_dtype):
        valid_dtypes = (mstype.float16, mstype.float32)
        validator.check_tensor_dtype_valid("y_grad", y_grad_dtype, valid_dtypes, self.name)
        validator.check_tensor_dtype_valid("x", x_dtype, valid_dtypes, self.name)
        return x_dtype


class HSwishGrad(PrimitiveWithInfer):
    """Gets the gradient of HSwish operation."""

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['y_grad', 'x'], outputs=['output'])

    def infer_shape(self, y_grad_shape, x_shape):
        return x_shape

    def infer_dtype(self, y_grad_dtype, x_dtype):
        valid_dtypes = (mstype.float16, mstype.float32)
        validator.check_tensor_dtype_valid("y_grad", y_grad_dtype, valid_dtypes, self.name)
        validator.check_tensor_dtype_valid("x", x_dtype, valid_dtypes, self.name)
        return x_dtype


class SigmoidCrossEntropyWithLogitsGrad(PrimitiveWithInfer):
    """Computes the gradients of `SigmoidCrossEntropyWithLogits`."""

    @prim_attr_register
    def __init__(self):
        """Initialize SigmoidCrossEntropyWithLogitsGrad"""
        self.init_prim_io_names(inputs=['x', 'y', 'dout'], outputs=['x_grad'])

    def infer_shape(self, x_shape, y_shape, dout_shape):
        validator.check("x_shape", x_shape, "y_shape", y_shape, Rel.EQ, self.name)
        validator.check("x_shape", x_shape, "dout_shape", dout_shape, Rel.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_dtype, y_dtype, dout_dtype):
        args = {"x_dtype": x_dtype, "y_dtype": y_dtype, 'dout_dtype': dout_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        return dout_dtype


class SliceGrad(PrimitiveWithInfer):
    """Reverse of slice."""

    @prim_attr_register
    def __init__(self):
        """Initialize SliceGrad"""
        self.init_prim_io_names(inputs=['dy', 'x', 'begin', 'size'], outputs=['dx'])

    def __infer__(self, dy, x, begin, size):
        dy_shape, x_shape, size_value = dy['shape'], x['shape'], size['value']
        dy_shape_len = len(dy_shape)
        for i in range(dy_shape_len):
            validator.check(f'dy_shape[{i}]', dy_shape[i], f'x_shape[{i}]', x_shape[i], Rel.LE, self.name)
            validator.check(f'dy_shape[{i}]', dy_shape[i], f'size_shape[{i}]', size_value[i], Rel.EQ, self.name)
        return {'shape': x_shape,
                'dtype': x['dtype'],
                'value': None}


class NLLLossGrad(PrimitiveWithInfer):
    """Computes the gradients of `NLLLoss`."""

    @prim_attr_register
    def __init__(self, reduction="mean"):
        """Initialize NLLLoss"""
        self.init_prim_io_names(inputs=['x', 'target', "weight"], outputs=['loss'])
        self.reduction = validator.check_string(reduction, ['none', 'sum', 'mean'], 'reduction', self.name)
        self.add_prim_attr('reduction', self.reduction)

    def infer_shape(self, x_shape, y_grad_shape, t_shape, w_shape, tw_shape):
        validator.check_int(len(x_shape), [1, 2], Rel.IN, "x rank", self.name)
        validator.check_int(len(t_shape), 1, Rel.EQ, "target rank", self.name)
        validator.check_int(len(w_shape), 1, Rel.EQ, "weight rank", self.name)
        validator.check(f"input_shape[0]", x_shape[0], "target_shape", t_shape[0], Rel.EQ, self.name)
        if len(x_shape) == 1:
            validator.check(f"input_shape[0]", x_shape[0], "weight_shape", w_shape[0], Rel.EQ, self.name)
        else:
            validator.check(f"input_shape[1]", x_shape[1], "weight_shape", w_shape[0], Rel.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_dtype, y_grad_dtype, t_dtype, w_dtype, tw_dtype):
        valid_dtypes = (mstype.float16, mstype.float32)
        validator.check_tensor_dtype_valid("x_dtype", x_dtype, valid_dtypes, self.name)
        validator.check_tensor_dtype_valid("y_grad_dtype", y_grad_dtype, valid_dtypes, self.name)
        validator.check_tensor_dtype_valid("t_dtype", t_dtype, mstype.int32, self.name)
        validator.check_tensor_dtype_valid("w_dtype", w_dtype, valid_dtypes, self.name)
        validator.check_tensor_dtype_valid("tw_dtype", tw_dtype, valid_dtypes, self.name)
        validator.check('tw_shape_dtype', tw_dtype, 'w_shape_dtype', w_dtype, Rel.EQ, self.name)
        return x_dtype


class SmoothL1LossGrad(PrimitiveWithInfer):
    """Computes gradient for prediction on SmoothL1Loss."""

    @prim_attr_register
    def __init__(self, beta=1.0):
        pass

    def infer_shape(self, prediction, target, dloss):
        validator.check('prediction shape', prediction, 'target shape', target, Rel.EQ, self.name)
        validator.check('prediction shape', prediction, 'dloss shape', dloss, Rel.EQ, self.name)
        return prediction

    def infer_dtype(self, prediction, target, dloss):
        args = {"prediction": prediction, "target": target, 'dloss': dloss}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
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
        """Initialize StridedSliceGrad"""
        validator.check_value_type('begin_mask', begin_mask, [int], self.name)
        validator.check_value_type('end_mask', end_mask, [int], self.name)
        validator.check_value_type('ellipsis_mask', ellipsis_mask, [int], self.name)
        validator.check_value_type('new_axis_mask', new_axis_mask, [int], self.name)
        validator.check_value_type('shrink_axis_mask', shrink_axis_mask, [int], self.name)
        self.init_prim_io_names(inputs=['dy', 'shapex', 'begin', 'end', 'strides'], outputs=['output'])

    def __infer__(self, dy, shapex, begin, end, strides):
        validator.check_tensor_dtype_valid("dy", dy['dtype'], mstype.number_type + (mstype.bool_,), self.name)

        for idx, item in enumerate(shapex['value']):
            validator.check_value_type("shapex[%d]" % idx, item, [int], self.name)
        for idx, item in enumerate(begin['value']):
            validator.check_value_type("begin[%d]" % idx, item, [int], self.name)
        for idx, item in enumerate(end['value']):
            validator.check_value_type("end[%d]" % idx, item, [int], self.name)
        for idx, item in enumerate(strides['value']):
            validator.check_value_type("strides[%d]" % idx, item, [int], self.name)

        return {'shape': shapex['value'],
                'dtype': dy['dtype'],
                'value': None}


class SoftplusGrad(PrimitiveWithInfer):
    """Computes gradient for the Log Softmax activation."""

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['dout', 'x'], outputs=['output'])

    def infer_shape(self, dout_shape, x_shape):
        validator.check("x_shape", x_shape, "dout_shape", dout_shape, Rel.EQ, self.name)
        return x_shape

    def infer_dtype(self, dout_dtype, x_dtype):
        args = {"x_dtype": x_dtype, "dout_dtype": dout_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.float_type, self.name)
        return x_dtype


class TanhGrad(PrimitiveWithInfer):
    """Computes gradient of hyperbolic tangent of input element-wise."""

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, out, dout):
        return out

    def infer_dtype(self, out, dout):
        args = {"out": out, "dout": dout}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        return out


class MirrorPadGrad(PrimitiveWithInfer):
    """Gradients of MirrorPad operation."""

    @prim_attr_register
    def __init__(self, mode="REFLECT"):
        """Initialize MirrorPad"""
        validator.check_string(mode, ['REFLECT', 'SYMMETRIC'], 'mode', self.name)
        self.mode = mode

    def __infer__(self, dout, paddings):
        validator.check_subclass("dout", dout['dtype'], mstype.tensor, self.name)
        validator.check_subclass("paddings", paddings['dtype'], mstype.tensor, self.name)
        validator.check("paddings rank", len(paddings['shape']), "expected", 2, Rel.EQ, self.name)
        validator.check("paddings dim_1", paddings['shape'][1], "expected", 2, Rel.EQ, self.name)

        if paddings['value'] is None:
            raise ValueError(f"For {self.name}, paddings must be const.")
        paddings_value = paddings['value'].asnumpy()
        y_shape = ()
        dout_shape = dout['shape']
        for i, val in enumerate(dout_shape):
            y_shape += (val - paddings_value[i][0] - paddings_value[i][1],)
        return {'shape': y_shape,
                'dtype': dout['dtype'],
                'value': None}


class EmbeddingLookupCommGrad(PrimitiveWithInfer):
    """
    Performs the gradient for the communication part of EmbeddingLookup operator.

    This works ONLY when 'reduce_scatter_flag' is True in 'EmbeddingLookup'. Roughly speaking,
    this primitive is implemented by StridedSlice --> _HostAllGather --> Concat. This primitive runs on host.
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['dy', 'split_num'], outputs=['output'])
        self.add_prim_attr('primitive_target', 'CPU')

    def __infer__(self, dy, split_num):
        """
        This primitive is implemented by three steps:
            1) Splits the 'dy' along dimension 0 into 'split_num' parts.
            2) For each part, perform _HostAllGather((0, 1, 2, 3, 4, 5, 6, 7)) on the host.
            3) After _HostAllGather, there are still 'split_num' parts in each process. Then, perform Concat on them
              along dimension 0.

        The output shape of this primitive: shape(output)[0] == shape(dy)[0] * 8
        """
        dy_shape = tuple(dy['shape'])
        split_num_value = split_num['value']
        validator.check_value_type("split_num_value", split_num_value, [int], self.name)
        dy_shape_all = F.tuple_setitem(dy_shape, 0, dy_shape[0] * 8)
        return {'shape': dy_shape_all,
                'dtype': dy['dtype'],
                'value': None}


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
        sig.make_sig('variable', sig.sig_rw.RW_REF),
    )

    @prim_attr_register
    def __init__(self):
        pass


class AtanGrad(PrimitiveWithInfer):
    """
    Computes AtanGrad of input element-wise.

    Returns:
        Tensor, has the same type as input.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize AtanGrad"""

    def infer_shape(self, x, dout):
        validator.check("x shape", x, "dout shape", dout, Rel.EQ, self.name)
        return x

    def infer_dtype(self, x, dout):
        args = {"x": x, "dout": dout}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        return x


class BasicLSTMCellCStateGrad(PrimitiveWithInfer):
    """Computes the state gradients of BasicLSTMCell."""

    @prim_attr_register
    def __init__(self, forget_bias, activation):
        self.forget_bias = validator.check_value_type("forget_bias", forget_bias, [float], self.name)
        self.activation = validator.check_string(activation, ['tanh'], "activation", self.name)

    def infer_shape(self, c_shape, dht_shape, dct_shape, it_shape, jt_shape, ft_shape, ot_shape, tanhct_shape):
        # dhy and dcy should be same shape
        validator.check_equal_int(len(c_shape), 2, "c rank", self.name)
        validator.check("dht rank", len(dht_shape), "c rank", len(c_shape), Rel.EQ, self.name)
        validator.check("dct rank", len(dct_shape), "c rank", len(c_shape), Rel.EQ, self.name)
        validator.check("it rank", len(it_shape), "c rank", len(c_shape), Rel.EQ, self.name)
        validator.check("jt rank", len(jt_shape), "c rank", len(c_shape), Rel.EQ, self.name)
        validator.check("ft rank", len(ft_shape), "c rank", len(c_shape), Rel.EQ, self.name)
        validator.check("ot rank", len(ot_shape), "c rank", len(c_shape), Rel.EQ, self.name)
        validator.check("tanhct rank", len(tanhct_shape), "c rank", len(c_shape), Rel.EQ, self.name)
        validator.check("dht shape", dht_shape, "c shape", c_shape, Rel.EQ, self.name)
        validator.check("dct shape", dct_shape, "c shape", c_shape, Rel.EQ, self.name)
        validator.check("it shape", it_shape, "c shape", c_shape, Rel.EQ, self.name)
        validator.check("jt shape", jt_shape, "c shape", c_shape, Rel.EQ, self.name)
        validator.check("ft shape", ft_shape, "c shape", c_shape, Rel.EQ, self.name)
        validator.check("ot shape", ot_shape, "c shape", c_shape, Rel.EQ, self.name)
        validator.check("tanhct shape", tanhct_shape, "c shape", c_shape, Rel.EQ, self.name)

        dgate_shape = (c_shape[0], 4 * c_shape[1])
        dct_1_shape = c_shape

        return (dgate_shape, dct_1_shape)

    def infer_dtype(self, c_dtype, dht_dtype, dct_dtype, it_dtype, jt_dtype, ft_dtype, ot_dtype, tanhct_dtype):
        validator.check_subclass("c", c_dtype, [mstype.tensor], self.name)
        validator.check_subclass("dht", dht_dtype, [mstype.tensor], self.name)
        validator.check_subclass("dct", dct_dtype, [mstype.tensor], self.name)
        validator.check_subclass("it", it_dtype, [mstype.tensor], self.name)
        validator.check_subclass("jt", jt_dtype, [mstype.tensor], self.name)
        validator.check_subclass("ft", ft_dtype, [mstype.tensor], self.name)
        validator.check_subclass("ot", ot_dtype, [mstype.tensor], self.name)
        validator.check_subclass("tanhct", tanhct_dtype, [mstype.tensor], self.name)
        validator.check_type_name("c", c_dtype, [mstype.float16, mstype.float32], self.name)
        validator.check_type_name("dht", dht_dtype, [mstype.float16, mstype.float32], self.name)
        validator.check_type_name("dct", dct_dtype, [mstype.float16, mstype.float32], self.name)
        validator.check_type_name("it", it_dtype, [mstype.float16, mstype.float32], self.name)
        validator.check_type_name("jt", jt_dtype, [mstype.float16, mstype.float32], self.name)
        validator.check_type_name("ft", ft_dtype, [mstype.float16, mstype.float32], self.name)
        validator.check_type_name("ot", ot_dtype, [mstype.float16, mstype.float32], self.name)
        validator.check_type_name("tanhct", tanhct_dtype, [mstype.float16, mstype.float32], self.name)
        return (c_dtype, c_dtype)


class BasicLSTMCellWeightGrad(PrimitiveWithInfer):
    """Computes the weight gradients of BasicLSTM."""
    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, x_shape, h_shape, dgate_shape):
        validator.check_equal_int(len(x_shape), 2, "x rank", self.name)
        validator.check("h rank", len(h_shape), " x rank", len(x_shape), Rel.EQ, self.name)
        validator.check("dgate rank", len(dgate_shape), "x rank", len(x_shape), Rel.EQ, self.name)
        validator.check("h_shape[0]", h_shape[0], "x_shape[0]", x_shape[0], Rel.EQ, self.name)
        validator.check("dgate_shape[0]", dgate_shape[0], "h_shape[0]", h_shape[0], Rel.EQ, self.name)
        validator.check("dgate_shape[1]", dgate_shape[1], "4*h_shape[1]", 4 * h_shape[1], Rel.EQ, self.name)
        input_size = x_shape[1]
        hidden_size = h_shape[1]
        dw_shape = (input_size + hidden_size, 4 * hidden_size)
        db_shape = (4 * hidden_size,)
        return (dw_shape, db_shape)

    def infer_dtype(self, x_dtype, h_dtype, dgate_dtype):
        validator.check_subclass("x", x_dtype, mstype.tensor, self.name)
        validator.check_subclass("h", h_dtype, mstype.tensor, self.name)
        validator.check_subclass("dgate", dgate_dtype, mstype.tensor, self.name)
        validator.check_type_name("x", x_dtype, [mstype.float16, mstype.float32], self.name)
        validator.check_type_name("h", h_dtype, [mstype.float16, mstype.float32], self.name)
        validator.check_type_name("dgate", dgate_dtype, [mstype.float16, mstype.float32], self.name)
        return (x_dtype, x_dtype)


class BasicLSTMCellInputGrad(PrimitiveWithInfer):
    """Computes the input gradients of BasicLSTM."""

    @prim_attr_register
    def __init__(self, keep_prob):
        self.keep_prob = validator.check_value_type("keep_prob", keep_prob, [float], self.name)
        self.keep_prob = validator.check_float_range(keep_prob, 0.0, 1.0, Rel.INC_BOTH, "keep_prob", self.name)

    def infer_shape(self, dgate_shape, w_shape):
        validator.check_equal_int(len(dgate_shape), 2, "dgate rank", self.name)
        validator.check_equal_int(len(w_shape), 2, "w rank", self.name)
        validator.check("dgate_shape[1]", dgate_shape[1], "w_shape[1]", w_shape[1], Rel.EQ, self.name)
        batch_size = dgate_shape[0]
        hidden_size = dgate_shape[1] // 4
        input_size = w_shape[0] - hidden_size
        dxt_shape = (batch_size, input_size)
        dht_shape = (batch_size, hidden_size)
        return (dxt_shape, dht_shape)

    def infer_dtype(self, dgate_dtype, w_dtype):
        validator.check_subclass("dgate", dgate_dtype, mstype.tensor, self.name)
        validator.check_subclass("w", w_dtype, mstype.tensor, self.name)
        validator.check_type_name("dgate", dgate_dtype, [mstype.float16, mstype.float32], self.name)
        validator.check_type_name("w", w_dtype, [mstype.float16, mstype.float32], self.name)
        return (dgate_dtype, dgate_dtype)


class InvGrad(PrimitiveWithInfer):
    """Computes gradients for inv operation."""

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, x, grad):
        validator.check("x_shape", x, "grad_shape", grad, Rel.EQ, self.name)
        return x

    def infer_dtype(self, x, grad):
        validator.check_type_name("dgate", x, [mstype.float16, mstype.float32, mstype.int32, mstype.int8], self.name)
        validator.check_type_name("grad", grad, [mstype.float16, mstype.float32, mstype.int32, mstype.int8], self.name)
        return x


class LRNGrad(PrimitiveWithInfer):
    """Computes gradients for LRN operation."""

    @prim_attr_register
    def __init__(self, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5):
        self.init_prim_io_names(inputs=['grads', 'x', 'y'], outputs=['z'])
        validator.check_value_type("depth_radius", depth_radius, [int], self.name)
        validator.check_value_type("bias", bias, [float], self.name)
        validator.check_value_type("alpha", alpha, [float], self.name)
        validator.check_value_type("beta", beta, [float], self.name)

    def infer_dtype(self, grads, x, y):
        args = {"grads": grads, "x": x, "y": y}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float16, mstype.float32,), self.name)
        return x

    def infer_shape(self, grads, x, y):
        return x
