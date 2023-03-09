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

"""Operators for gradients."""
from __future__ import absolute_import

from __future__ import division
from mindspore._checkparam import _check_3d_int_or_tuple
from mindspore.ops.operations.nn_ops import _check_positive_int_or_tuple
from mindspore.ops import signature as sig
from mindspore.ops._utils import get_concat_offset
from mindspore.ops.primitive import Primitive, PrimitiveWithInfer, prim_attr_register
import mindspore.context as context
from mindspore._checkparam import Validator as validator, Rel
from mindspore.common import dtype as mstype
from mindspore.communication.management import GlobalComm
from mindspore.common._utils import is_shape_unknown, is_dim_unknown


class SparseFillEmptyRowsGrad(Primitive):
    """Performs grad of SparseFillEmptyRows operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize SparseFillEmptyRowsGrad."""
        self.init_prim_io_names(inputs=['reverse_index_map', 'grad_values'],
                                outputs=['y_values', 'y_default_value'])


class AbsGrad(PrimitiveWithInfer):
    """Computes gradients for abs operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize AbsGrad"""


class ACosGrad(Primitive):
    """
    Computes ACosGrad of input element-wise.

    Returns:
        Tensor, has the same type as input.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ACosGrad"""
        self.init_prim_io_names(inputs=['y', 'dy'], outputs=['z'])


class LogitGrad(Primitive):
    """
    Computes LogitGrad of input element-wise.

    Returns:
        Tensor, has the same type as input.
    """
    @prim_attr_register
    def __init__(self, eps=-1.0):
        """Initialize Exp"""
        self.init_prim_io_names(inputs=['grad', 'input'], outputs=['dx'])
        validator.check_value_type("eps", eps, [float], self.name)
        self.add_prim_attr('eps', eps)


class AcoshGrad(Primitive):
    """Performs grad of Acosh operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize AcoshGrad"""
        self.init_prim_io_names(inputs=['y', 'dy'], outputs=['z'])


class AsinGrad(Primitive):
    """
    Computes AsinGrad of input element-wise.

    Returns:
        Tensor, has the same type as input.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize AsinGrad"""
        self.init_prim_io_names(inputs=['y', 'dy'], outputs=['z'])


class AsinhGrad(Primitive):
    """Performs grad of Asinh operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize AsinhGrad"""
        self.init_prim_io_names(inputs=['y', 'dy'], outputs=['z'])


class ReciprocalGrad(Primitive):
    """Performs grad of Reciprocal operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize ReciprocalGrad"""
        self.init_prim_io_names(inputs=['y', 'dy'], outputs=['z'])


class RsqrtGrad(Primitive):
    """Performs grad of Rsqrt operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize RsqrtGrad"""


class ScaleAndTranslateGrad(Primitive):
    """Performs grad of ScaleAndTranslate operation."""

    @prim_attr_register
    def __init__(self, kernel_type="lanczos3", antialias=True):
        """Initialize ScaleAndTranslateGrad"""
        validator.check_value_type("kernel_type", kernel_type, [str], self.name)
        validator.check_string(kernel_type, ["lanczos1", "lanczos3", "lanczos5", "gaussian", "box", "triangle",
                                             "keyscubic", "mitchellcubic"], "kernel_type", self.name)
        validator.check_value_type("antialias", antialias, [bool], self.name)


class SoftmaxGrad(ReciprocalGrad):
    """Performs grad of Softmax operation."""


class SqrtGrad(Primitive):
    """Performs grad of Sqrt operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize SqrtGrad"""
        self.init_prim_io_names(inputs=['y', 'dy'], outputs=['z'])


class BatchNormGrad(Primitive):
    """Performs grad of BatchNorm operation."""

    @prim_attr_register
    def __init__(self, is_training=False, epsilon=1e-5, data_format='NCHW'):
        self.is_training = validator.check_value_type('is_training', is_training, (bool,), self.name)
        self.epsilon = validator.check_float_range(epsilon, 0, 1, Rel.INC_RIGHT, 'epsilon', self.name)
        self.data_format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.name)


class BatchNormGradGrad(Primitive):
    """Performs grad of BatchNormGrad operation."""

    @prim_attr_register
    def __init__(self, is_training=False, epsilon=1e-5, data_format='NCHW'):
        self.is_training = validator.check_value_type('is_training', is_training, (bool,), self.name)
        self.epsilon = validator.check_float_range(epsilon, 0, 1, Rel.INC_RIGHT, 'epsilon', self.name)
        self.data_format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.name)


class SyncBatchNormGrad(Primitive):
    """Performs grad of SyncBatchNorm operation."""

    @prim_attr_register
    def __init__(self, epsilon=1e-5, group="group0", device_num=2):
        validator.check_float_range(epsilon, 0, 1, Rel.INC_RIGHT, 'epsilon', self.name)
        if not isinstance(group, str):
            raise TypeError("The group attr of SyncBatchNormGrad must be str.")
        validator.check_int(device_num, 2, Rel.GE, "device_num", self.name)


class BiasAddGrad(Primitive):
    """Computes gradients of BiasAdd."""

    @prim_attr_register
    def __init__(self, data_format="NCHW"):
        self.init_prim_io_names(inputs=['dout'], outputs=['output'])
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC', 'NCDHW'], 'format', self.name)
        if self.format == "NCDHW":
            self.format = "NCHW"
        self.add_prim_attr('data_format', self.format)


class KLDivLossGrad(Primitive):
    """Computes gradients for `KLDivLoss` operation."""

    @prim_attr_register
    def __init__(self, reduction='mean'):
        device_target = context.get_context("device_target")
        if device_target == "CPU":
            support_mode = ['none', 'mean', 'batchmean', 'sum']
        elif device_target == "GPU":
            support_mode = ['none', 'mean', 'sum']
        elif device_target == "Ascend":
            support_mode = ['none', 'mean', 'batchmean', 'sum']
        else:
            raise ValueError(f"'{self.name}' unknown device target: '{device_target}'")
        self.reduction = validator.check_string(reduction, support_mode, 'reduction', self.name)


class BinaryCrossEntropyGrad(Primitive):
    """Computes gradients for `BinaryCrossEntropy` operation."""

    @prim_attr_register
    def __init__(self, reduction='mean'):
        self.reduction = validator.check_string(reduction, ['none', 'mean', 'sum'], 'reduction', self.name)


class LuUnpackGrad(Primitive):
    """Computes gradients for `LuUnpack` operation."""

    @prim_attr_register
    def __init__(self, L_grad_flag, U_grad_flag):
        validator.check_value_type("L_grad_flag", L_grad_flag, [bool], self.name)
        validator.check_value_type("U_grad_flag", U_grad_flag, [bool], self.name)
        self.add_prim_attr("cust_aicpu", self.name)


class ConcatOffset(PrimitiveWithInfer):
    """primitive for computing Concat's gradient."""

    @prim_attr_register
    def __init__(self, N=2, axis=0):
        """Initialize ConcatOffset"""

    def __infer__(self, input_x):
        axis = self.axis
        x_shp = input_x['shape']
        x_type = input_x['dtype']
        self.add_prim_attr('T', x_type[0].element_type())

        # input_x is dynamic rank
        rank = -1
        is_dyn_rank = False
        for _, sh in enumerate(x_shp):
            if is_dim_unknown(sh):
                is_dyn_rank = True
            else:
                rank = len(sh)
        if is_dyn_rank:
            return {
                'shape': [len(x_shp), rank],
                'dtype': mstype.int64,
                'value': None
            }

        # if the dimension of input_x on the axis is dynamic
        if axis < -rank or axis >= rank:
            raise ValueError("For 'ConcatOffset', 'axis' must be in range [{}, {}), but got {}"
                             .format(-rank, rank, axis))
        if axis < 0:
            axis = axis + rank
        for each in x_shp:
            if each[axis] == -1:
                return {
                    'shape': [len(x_shp), len(x_shp[0])],
                    'dtype': mstype.int64,
                    'value': None
                }

        offset, _, axis = get_concat_offset(x_shp, x_type, axis, self.name)
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


class Conv3DBackpropFilter(Primitive):
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
        self.add_prim_attr('pad', pad)
        self.pad_list = pad
        self.add_prim_attr('pad_list', self.pad_list)

        validator.check_value_type('pad_mode', pad_mode, [str], self.name)
        self.pad_mode = validator.check_string(pad_mode.lower(), ['valid', 'same', 'pad'], 'pad_mode', self.name)
        if self.pad_mode != 'pad' and self.pad_list != (0, 0, 0, 0, 0, 0):
            raise ValueError(f"For '{self.name}', when pad is not 0, pad_mode must be set as 'pad'.")
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


class Conv2DBackpropFilter(Primitive):
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
        mode (int): Modes for different convolutions. 0 Math convolution, 1 cross-correlation convolution ,
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
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError("NHWC format only support in GPU target.")
        self.add_prim_attr('data_format', self.format)
        self.stride = _check_positive_int_or_tuple('stride', stride, self.name, allow_four=True, ret_four=True)
        self.add_prim_attr('stride', self.stride)
        self.dilation = _check_positive_int_or_tuple('dilation', dilation, self.name, allow_four=True, ret_four=True)
        self.add_prim_attr('dilation', self.dilation)
        self.group = group
        self.add_prim_attr('groups', group)
        if pad_list:
            for x in pad_list:
                if x != -1:
                    validator.check_non_negative_int(x, 'element of pad_list', self.name)
        self.pad_list = pad_list


class DepthwiseConv2dNativeBackpropFilter(PrimitiveWithInfer):
    """
    Returns the gradient of filter for DepthwiseConv2dNative.

    Applies depthwise conv2d for the input, which will generate more channels with channel_multiplier.

    Refer to class DepthwiseConv2dNative for more details.

    Args:
        channel_multiplier (int): The multiplier for the original output conv.
        kernel_size (int or tuple): The size of the conv kernel.
        mode (int): Modes for different convolutions. 0 Math convolution, 1 cross-correlation convolution,
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
        mode (int): Modes for different convolutions. 0 Math convolution, 1 cross-correlation convolution ,
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


class DropoutGrad(Primitive):
    """
    The gradient of Dropout. During training, randomly zeroes some of the elements
    of the input tensor with probability.

    Args:
        keep_prob (float): The keep rate, between 0 and 1, e.g. keep_prob = 0.9,
          means dropping out 10% of input units. Default: 0.5.

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


class InstanceNormV2Grad(Primitive):
    """Gradients of InstanceNormV2 operation."""

    @prim_attr_register
    def __init__(self, is_training=True, epsilon=1e-5):
        self.init_prim_io_names(inputs=['dy', 'x', 'gamma', 'mean', 'variance', 'save_mean', 'save_variance'],
                                outputs=['pd_x', 'pd_gamma', 'pd_beta'])
        validator.check_is_float(epsilon, 'epsilon', self.name)
        validator.check_float_range(epsilon, 0, 1, Rel.INC_RIGHT, 'epsilon', self.name)
        validator.check_bool(is_training, "is_training", self.name)


class EinsumGrad(PrimitiveWithInfer):
    """Gradients of Einsum."""

    @prim_attr_register
    def __init__(self, equation):
        self.add_prim_attr('equation', equation)

    def infer_shape(self, x_shapes, dout_shape):
        out_shape = ()
        for dim in x_shapes:
            out_shape += (dim,)
        return out_shape

    def infer_dtype(self, x_types, dout_shape):
        out_type = ()
        for cur_type in x_types:
            out_type += (cur_type,)
        return out_type


class UniqueGrad(Primitive):
    """Gradients of Unique operation."""

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['dy', 'y'], outputs=['dx'])


class BNTrainingReduceGrad(Primitive):
    """Gradients of FusedBatchNorm operation."""

    @prim_attr_register
    def __init__(self, epsilon=0.0001, data_format='NCHW'):
        self.data_format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.name)
        _inputs = ['grads', 'x', 'diff_scale', 'diff_offset', 'scale', 'batch_mean', 'batch_variance']
        self.init_prim_io_names(inputs=_inputs, outputs=['y'])


class BNTrainingUpdateGrad(Primitive):
    """Gradients of FusedBatchNorm operation."""

    @prim_attr_register
    def __init__(self, epsilon=0.0001, data_format='NCHW'):
        self.data_format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.name)
        self.init_prim_io_names(inputs=['grads', 'x', 'batch_mean', 'batch_variance'],
                                outputs=['diff_scale', 'diff_offset'])


class NeighborExchangeV2Grad(PrimitiveWithInfer):
    """"Gradients of NeighborExchangeV2 operation."""

    @prim_attr_register
    def __init__(self, send_rank_ids, send_lens, recv_rank_ids, recv_lens, data_format,
                 group=GlobalComm.WORLD_COMM_GROUP):
        self.init_prim_io_names(inputs=['dy'], outputs=['dx'])
        self.send_rank_ids = send_rank_ids
        self.recv_rank_ids = recv_rank_ids
        self.send_lens = send_lens
        self.recv_lens = recv_lens
        self.format = validator.check_string(data_format, ['NCHW'], 'format', self.name)
        self.add_prim_attr('no_elimilate', True)

    def __infer__(self, dy):
        dy_shape = dy['shape']
        validator.check(f'dy_shape.size()', len(dy_shape), f'4', 4, Rel.EQ, self.name)
        if self.send_rank_ids[5] != -1 or self.send_rank_ids[6] != -1 or self.send_rank_ids[7] != -1:
            dy_shape[3] -= self.send_lens[2]

        if self.send_rank_ids[1] != -1 or self.send_rank_ids[2] != -1 or self.send_rank_ids[3] != -1:
            dy_shape[3] -= self.send_lens[3]

        if self.send_rank_ids[0] != -1 or self.send_rank_ids[1] != -1 or self.send_rank_ids[7] != -1:
            dy_shape[2] -= self.send_lens[0]

        if self.send_rank_ids[3] != -1 or self.send_rank_ids[4] != -1 or self.send_rank_ids[5] != -1:
            dy_shape[2] -= self.send_lens[1]

        return {'shape': dy_shape,
                'dtype': dy['dtype'],
                'value': None}


class GeLUGrad(Primitive):
    """Gradients of GeLU operation."""

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['dy', 'x', 'y'], outputs=['z'])


class FastGeLUGrad(Primitive):
    """Gradients of FastGeLU operation."""

    @prim_attr_register
    def __init__(self):
        """init FastGeLUGrad"""


class _PoolGrad(PrimitiveWithInfer):
    """Gradients of the max/avg pool operation."""

    @prim_attr_register
    def __init__(self, kernel_size, strides, pad_mode="VALID", data_format="NCHW"):
        self.init_prim_io_names(inputs=['x_origin', 'out_origin', 'grad'], outputs=['output'])

        validator.check_value_type('kernel_size', kernel_size, [int, tuple], self.name)
        validator.check_value_type('strides', strides, [int, tuple], self.name)
        validator.check_value_type('pad_mode', pad_mode, [str], self.name)
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
            error_msg = ValueError(f"For '{self.name}' the '{arg_name}' must be an positive int number "
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
        strides = _grad_check_int_or_tuple("strides", strides, self.is_maxpoolgradwithargmax)
        if self.format == "NCHW":
            self.kernel_size = kernel_size
            self.strides = strides
        else:
            self.kernel_size = [kernel_size[0], kernel_size[2], kernel_size[3], kernel_size[1]]
            self.strides = [strides[0], strides[2], strides[3], strides[1]]
        self.add_prim_attr("kernel_size", self.kernel_size)
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


class AvgPoolGradGe(_PoolGrad):
    """Gradients of the avg pool operation for ge."""

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="VALID", data_format="NCHW"):
        super(AvgPoolGradGe, self).__init__(kernel_size, strides, pad_mode, data_format)

    def __infer__(self, origin_input, dout):
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


class AvgPoolGradV1(Primitive):
    """Gradients of the AvgPoolV1 operation."""

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="VALID", data_format="NCHW"):
        validator.check_value_type('kernel_size', kernel_size, [int, tuple], self.name)
        validator.check_value_type('strides', strides, [int, tuple], self.name)
        self.pad_mode = validator.check_string(
            pad_mode.upper(), ['VALID', 'SAME'], 'pad_mode', self.name)
        self.add_prim_attr("pad_mode", self.pad_mode)
        self.format = validator.check_string(
            data_format, ['NCHW', 'NHWC'], 'format', self.name)
        self.add_prim_attr('data_format', self.format)

        def _avgpoolgrad_check_int_or_tuple(argname, argval):
            validator.check_value_type(argname, argval, (int, tuple), self.name)
            errormsg = ValueError(f"For '{self.name}' the '{argname}' should be an positive int number "
                                  f"or a tuple of two or four positive int numbers, but got {argval}")
            if isinstance(argval, int):
                ret = (1, 1, argval, argval)
            elif len(argval) == 2:
                ret = (1, 1, argval[0], argval[1])
            elif len(argval) == 4:
                ret = argval
            else:
                raise errormsg
            # whether all elements of tuple are positive integers?
            for it in ret:
                if not isinstance(it, int) or it <= 0:
                    raise errormsg
            return ret

        self.kernel_size = _avgpoolgrad_check_int_or_tuple(
            "kernel_size", kernel_size)
        self.strides = _avgpoolgrad_check_int_or_tuple("strides", strides)

        self.kernel_size_adapt = self.kernel_size if self.format == "NCHW" else (
            self.kernel_size[0], self.kernel_size[2], self.kernel_size[3], self.kernel_size[1])
        self.strides_adapt = self.strides if self.format == "NCHW" else (
            self.strides[0], self.strides[2], self.strides[3], self.strides[1])

        # If length of some attrs is 4 we regard it as legal, either by using the op directly,
        # or passed from an instance of forward op AvgPoolV1.
        if len(self.kernel_size) == 4:
            self.kernel_size_adapt = self.kernel_size
        if len(self.strides) == 4:
            self.strides_adapt = self.strides

        self.add_prim_attr("kernel_size", self.kernel_size_adapt)
        self.add_prim_attr("strides", self.strides_adapt)


class AdaptiveAvgPool2DGrad(Primitive):
    """Gradients of the adaptive avg pool 2D operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize AdaptiveAvgPool2DGrad"""
        self.init_prim_io_names(inputs=['input_grad', 'orig_input_shape'], outputs=['output_grad'])


class AdaptiveAvgPool3DGrad(Primitive):
    """Performs grad of AdaptiveAvgPool3D operation."""
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['y_grad', 'orig_input_shape'], outputs=['x_grad'])


class AvgPool3DGrad(Primitive):
    """Gradients of the avg pool3d operation."""

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pads=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=0, data_format="NCDHW", pad_mode="pad"):
        self.init_prim_io_names(inputs=['origin_input_shape', 'grads'], outputs=['output'])
        self.kernel_size = _check_3d_int_or_tuple('kernel_size', kernel_size, self.name, allow_five=True, ret_five=True)
        self.add_prim_attr('kernel_size', self.kernel_size)
        self.strides = _check_3d_int_or_tuple('strides', strides, self.name, allow_five=True, ret_five=True)
        self.add_prim_attr('strides', self.strides)
        validator.check_value_type('pad_mode', pad_mode, [str], self.name)
        self.pad_mode = validator.check_string(pad_mode.upper(), ['VALID', 'SAME', 'PAD'], 'pad_mode', self.name)
        validator.check_value_type('pads', pads, (int, tuple), self.name)
        if isinstance(pads, int):
            pads = (pads,) * 6
        validator.check_equal_int(len(pads), 6, 'pad size', self.name)
        for item in pads:
            validator.check_non_negative_int(item, 'pad item', self.name)
        self.add_prim_attr('pad_list', pads)
        self.ceil_mode = validator.check_value_type('ceil_mode', ceil_mode, bool, self.name)
        self.count_include_pad = validator.check_value_type('count_include_pad', count_include_pad, bool, self.name)
        self.divisor_override = validator.check_value_type('divisor_override', divisor_override, int, self.name)
        self.format = validator.check_string(data_format, ['NCDHW'], 'format', self.name)


class AdaptiveMaxPool2DGrad(Primitive):
    """Gradients of the adaptive max pool 2D operation."""
    @prim_attr_register
    def __init__(self):
        """Initialize AdaptiveMaxPool2DGrad"""
        self.init_prim_io_names(inputs=['y_grad', 'x', 'argmax'], outputs=['x_grad'])


class MaxPoolGrad(_PoolGrad):
    """Performs gradients of the max pool operation."""

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="VALID", data_format="NCHW"):
        super(MaxPoolGrad, self).__init__(kernel_size, strides, pad_mode, data_format)

    def infer_shape(self, x1_shape, x2_shape, grad_shape):
        return x1_shape

    def infer_dtype(self, x1_dtype, x2_dtype, grad_dtype):
        return x1_dtype


class MaxPoolGradV1(Primitive):
    """Performs gradients of the MaxPoolV1 operation."""

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="VALID", data_format="NCHW"):
        self.init_prim_io_names(
            inputs=['x_origin', 'out_origin', 'grad'], outputs=['output'])

        validator.check_value_type('kernel_size', kernel_size, [int, tuple], self.name)
        validator.check_value_type('strides', strides, [int, tuple], self.name)
        self.pad_mode = validator.check_string(
            pad_mode.upper(), ['VALID', 'SAME'], 'pad_mode', self.name)
        self.add_prim_attr("pad_mode", self.pad_mode)
        self.format = validator.check_string(
            data_format, ['NCHW', 'NHWC'], 'format', self.name)
        self.add_prim_attr('data_format', self.format)

        def _grad_check_int_or_tuple(arg_name, arg_val):
            validator.check_value_type(
                arg_name, arg_val, (int, tuple), self.name)
            error_msg = ValueError(f"For '{self.name}' the '{arg_name}' should be an positive int number "
                                   f"or a tuple of two or four positive int numbers, but got {arg_val}")
            if isinstance(arg_val, int):
                ret = (1, 1, arg_val, arg_val)
            elif len(arg_val) == 2:
                ret = (1, 1, arg_val[0], arg_val[1])
            elif len(arg_val) == 4:
                ret = arg_val
            else:
                raise error_msg
            # whether all elements of tuple are positive integers
            for item in ret:
                if not isinstance(item, int) or item <= 0:
                    raise error_msg
            return ret

        self.kernel_size = _grad_check_int_or_tuple("kernel_size", kernel_size)
        self.strides = _grad_check_int_or_tuple("strides", strides)

        kernel_size_adapted = self.kernel_size if self.format == 'NCHW' else (
            self.kernel_size[0], self.kernel_size[2], self.kernel_size[3], self.kernel_size[1])
        strides_adapted = self.strides if self.format == 'NCHW' else (
            self.strides[0], self.strides[2], self.strides[3], self.strides[1])

        if len(kernel_size) == 4:
            kernel_size_adapted = kernel_size
        if len(strides) == 4:
            strides_adapted = strides

        self.add_prim_attr("kernel_size", kernel_size_adapted)
        self.add_prim_attr("strides", strides_adapted)


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
        - **origin_input** (Tensor) - Tensor with data format "NCHW".
          For Ascend, data type must be float16. For CPU and GPU, data type support float16 and float32.
        - **origin_output** (Tensor) - Data type same as `origin_input`.
        - **grad** (Tensor) - Data type and shape same as `origin_input`.

    Outputs:
        Tensor, with data type same as `origin_input`. Shape same as `origin_output`.

    Raises:
        TypeError: If kernel_size is neither int nor a tuple of 2/4 int numbers.
        TypeError: If strides is neither int nor a tuple of 2/4 int numbers.
        TypeError: If pad_mode is not string.
        ValueError: If pad_mode is neither "same" nor "valid"(not case sensitive).
        TypeError: For Ascend, input data type is not float16. For CPU or GPU, input data type is neither
        float16 nor float32.
        ValueError: If the rank of `origin_input`, `origin_output` or `grad` is not equal to 4.
        ValueError: If data types of all inputs are not equal.
        ValueError: If the shapes of `origin_input` and `grad` are not equal.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
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


class MaxPool3DGrad(Primitive):
    """Gradients of the max pool3d operation."""

    @prim_attr_register
    def __init__(self, kernel_size=(1, 1, 1, 1, 1), strides=(1, 1, 1, 1, 1),
                 pad_mode='VALID', pad_list=0, data_format="NCDHW"):
        self.init_prim_io_names(inputs=['x_origin', 'out_origin', 'grad'], outputs=['output'])
        validator.check_value_type('kernel_size', kernel_size, [int, tuple], self.name)
        validator.check_value_type('strides', strides, [int, tuple], self.name)
        validator.check_value_type('pad_mode', pad_mode, [str], self.name)
        self.format = validator.check_string(data_format, ['NCDHW'], 'format', self.name)
        if pad_mode.upper() == 'PAD':
            pad_mode = 'CALCULATED'
        self.pad_mode = validator.check_string(pad_mode.upper(), ['VALID', 'SAME', 'CALCULATED'], 'pad_mode', self.name)
        self.kernel_size = _check_3d_int_or_tuple("kernel_size", kernel_size, self.name,
                                                  allow_five=True, ret_five=True)
        self.add_prim_attr("kernel_size", self.kernel_size)
        self.strides = _check_3d_int_or_tuple("strides", strides, self.name, allow_five=True, ret_five=True)
        self.add_prim_attr("strides", self.strides)
        validator.check_value_type('pad_list', pad_list, (int, tuple), self.name)
        self.pad_list = pad_list
        if isinstance(self.pad_list, int):
            self.pad_list = (self.pad_list,) * 6
        if len(self.pad_list) == 3:
            self.pad_list = (pad_list[0], pad_list[0], pad_list[1], pad_list[1], pad_list[2], pad_list[3])
        if len(self.pad_list) != 3 and len(self.pad_list) != 6:
            raise ValueError(f"For `maxpool3d` attr 'pad_list' must be an positive int number or a tuple of "
                             f"three or six positive int numbers, but got `{len(self.pad_list)}` numbers.")
        if self.pad_mode != 'CALCULATED' and self.pad_list != (0, 0, 0, 0, 0, 0):
            raise ValueError(f"For '{self.name}', when pad_list is not 0, pad_mode must be set as 'pad'.")
        if self.pad_mode == 'CALCULATED':
            for item in self.pad_list:
                validator.check_non_negative_int(item, 'pad_list item', self.name)
        self.add_prim_attr("pad_list", self.pad_list)


class MaxPool3DGradGrad(PrimitiveWithInfer):
    r"""Gradients of the max pool3d grad operation.

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the maximum value,
            is an int number that represents depth, height and width are both kernel_size, or a tuple
            of two int numbers that represent depth, height and width respectively. Default: 1.
        strides (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the depth, height and width of movement are both strides, or a tuple of two int numbers that
            represent depth, height and width of movement respectively. Default: 1.
        pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. The depth, height and width of the output will be the
              same as the input. The total number of padding will be calculated in depth, horizontal and
              vertical directions and evenly distributed to front and back, top and bottom, left and
              right if possible. Otherwise, the last extra padding will be done from the back, the bottom
              and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.

    Inputs:
        - **origin_input** (Tensor) - Tensor with data format "NCDHW".
          For Ascend, data type must be float16. For CPU and GPU, data type support float16 and float32.
        - **origin_output** (Tensor) - Data type same as `origin_input`.
        - **grad** (Tensor) - Data type and shape same as `origin_input`.

    Outputs:
        Tensor, with data type same as `origin_input`. Shape same as `origin_output`.

    Raises:
        TypeError: If kernel_size is neither int nor a tuple of 3/5 int numbers.
        TypeError: If strides is neither int nor a tuple of 3/5 int numbers.
        TypeError: If pad_mode is not string.
        ValueError: If pad_mode is neither "same" nor "valid"(not case sensitive).
        TypeError: For Ascend, input data type is not float16. For CPU or GPU, input data type is neither
        float16 nor float32.
        ValueError: If the rank of `origin_input`, `origin_output` or `grad` is not equal to 5.
        ValueError: If data types of all inputs are not equal.
        ValueError: If the shapes of `origin_input` and `grad` are not equal.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self, kernel_size=(1, 1, 1, 1, 1), strides=(1, 1, 1, 1, 1), pad_mode='VALID', data_format="NCDHW"):
        validator.check_value_type('kernel_size', kernel_size, [int, tuple], self.name)
        validator.check_value_type('strides', strides, [int, tuple], self.name)
        validator.check_value_type('pad_mode', pad_mode, [str], self.name)
        self.format = validator.check_string(data_format, ['NCDHW'], 'format', self.name)
        self.pad_mode = validator.check_string(pad_mode.upper(), ['VALID', 'SAME'], 'pad_mode', self.name)
        self.kernel_size = _check_3d_int_or_tuple("kernel_size", kernel_size, self.name,
                                                  allow_five=True, ret_five=True)
        self.add_prim_attr("kernel_size", self.kernel_size)
        self.strides = _check_3d_int_or_tuple("strides", strides, self.name, allow_five=True, ret_five=True)
        self.add_prim_attr("strides", self.strides)

    def infer_shape(self, x_shape, y_shape, grad_shape):
        validator.check_equal_int(len(x_shape), 5, "x rank", self.name)
        validator.check('x_shape', x_shape, 'grad_shape', grad_shape, prim_name=self.name)
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
        self.init_prim_io_names(inputs=['x1', 'x2', 'grads'], outputs=['y1', 'y2'])


class MaximumGradGrad(Primitive):
    """Grad for maximum grad."""

    @prim_attr_register
    def __init__(self, grad_x=True, grad_y=True):
        """Initialize MaximumGradGrad"""
        super().__init__("MaximumGradGrad")
        self.init_prim_io_names(inputs=['x1', 'x2', 'dy1', 'dy2'], outputs=['sopd_x1', 'sopd_x2', 'sopd_grad'])


class MaxPoolGradWithArgmax(Primitive):
    """Computes the gradients of MaxPoolWithArgmax."""
    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="VALID", data_format="NCHW"):
        self.init_prim_io_names(inputs=['x_origin', 'out_origin', 'grad'], outputs=['output'])
        validator.check_value_type('kernel_size', kernel_size, [int, tuple], self.name)
        validator.check_value_type('strides', strides, [int, tuple], self.name)
        validator.check_value_type('pad_mode', pad_mode, [str], self.name)
        self.pad_mode = validator.check_string(pad_mode.upper(), ['VALID', 'SAME'], 'pad_mode', self.name)
        self.add_prim_attr("pad_mode", self.pad_mode)
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError("NHWC format only support in GPU target.")
        self.is_maxpoolgradwithargmax = (self.name == "MaxPoolGradWithArgmax")
        if not self.is_maxpoolgradwithargmax:
            self.add_prim_attr('data_format', self.format)

        def _grad_check_int_or_tuple(arg_name, arg_val):
            validator.check_value_type(arg_name, arg_val, (int, tuple), self.name)
            error_msg = ValueError(f"For '{self.name}' the '{arg_name}' must be an positive int number "
                                   f"or a tuple of two or four positive int numbers, but got {arg_val}")
            if isinstance(arg_val, int):
                ret = (1, arg_val, arg_val, 1)
            elif len(arg_val) == 2:
                ret = (1, arg_val[0], arg_val[1], 1)
            elif len(arg_val) == 4:
                ret = arg_val
            else:
                raise error_msg
            # whether all elements of tuple are positive integers
            for item in ret:
                if not isinstance(item, int) or item <= 0:
                    raise error_msg
            return ret

        kernel_size = _grad_check_int_or_tuple("kernel_size", kernel_size)
        self.kernel_size = kernel_size
        self.add_prim_attr("kernel_size", self.kernel_size)

        strides = _grad_check_int_or_tuple("strides", strides)
        self.strides = strides
        self.add_prim_attr("strides", self.strides)


class MaxPool3DGradWithArgmax(Primitive):
    """Gradients of the maxpool3Dwithargmax operation."""

    @prim_attr_register
    def __init__(self, ksize, strides, pads, dilation=(1, 1, 1), ceil_mode=False, data_format="NCDHW"):
        self.init_prim_io_names(inputs=['x', 'grads', 'argmax'], outputs=['y'])
        validator.check_value_type('ceil_mode', ceil_mode, bool, self.name)
        validator.check_value_type('data_format', data_format, str, self.name)
        self.data_format = validator.check_string(data_format, ['NCDHW'], 'data_format', self.name)
        self.ksize = _check_3d_int_or_tuple("ksize", ksize, self.name, ret_five=False)
        self.add_prim_attr('ksize', self.ksize)
        self.strides = _check_3d_int_or_tuple("strides", strides, self.name, ret_five=False)
        self.add_prim_attr('strides', self.strides)
        self.pads = _check_3d_int_or_tuple("pads", pads, self.name, greater_zero=False, ret_five=False)
        self.add_prim_attr('pads', self.pads)
        self.dilation = _check_3d_int_or_tuple("dilation", dilation, self.name, allow_five=True, ret_five=False)
        self.add_prim_attr('dilation', self.dilation)


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
        - **x** (Tensor) - Tensor with data format "NCHW".
          For Ascend, data type must be float16. For CPU and GPU, data type support float16 and float32.
        - **grad** (Tensor) - Data type and shape same as `x`.
        - **argmax** (Tensor) - Data type must be int32 or int64.

    Outputs:
        Tensor, with data type same as `x`. Shape same as `argmax`.

    Raises:
        TypeError: If kernel_size is neither int nor a tuple of 2/4 int numbers.
        TypeError: If strides is neither int nor a tuple of 2/4 int numbers.
        TypeError: If pad_mode is not string.
        ValueError: If pad_mode is neither "same" nor "valid"(not case sensitive).
        TypeError: For Ascend, the data types of `x` and `grad` are not float16.
        For CPU or GPU, the data types of `x` and `grad` are neither float16 nor float32.
        TypeError: The data type of `argmax` is neither int32 nor int64.
        ValueError: If the rank of `x`, `grad` or `argmax` is not equal to 4.
        ValueError: If the shapes of `x` and `grad` are not equal.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="VALID"):
        self.init_prim_io_names(inputs=['x', 'grad', 'argmax'], outputs=['output'])
        super(MaxPoolGradGradWithArgmax, self).__init__(kernel_size, strides, pad_mode)

    def infer_shape(self, x_shape, grad_shape, argmax_shape):
        if not grad_shape:
            raise TypeError("The dout of MaxPoolGradGradWithArgmax must be a Tensor.")
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
        self.init_prim_io_names(inputs=['x1', 'x2', 'grads'], outputs=['y1', 'y2'])


class MinimumGradGrad(Primitive):
    """Grad for minimum_grad."""
    @prim_attr_register
    def __init__(self):
        """Initialize MinimumGradGrad"""
        super().__init__("MinimumGradGrad")
        self.init_prim_io_names(inputs=['x1', 'x2', 'grad_y1', 'grad_y2'],
                                outputs=['sopd_x1', 'sopd_x2', 'sopd_grads'])


class L2NormalizeGrad(Primitive):
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


class LayerNormGrad(Primitive):
    """
    Applies the layer Normalization to the input array.

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


class LayerNormGradGrad(Primitive):
    """
    Gets the gradient of LayerNormGrad operation.

    Args:
        begin_norm_axis (int): The begin axis for the input to apply layernorm. Default: 1.
        begin_params_axis (int): The begin axis for the parameter input to apply layernorm. Default: 1.

    Inputs:
        - **x** (Tensor) - The input tensor to be normalized, float32 or float16.
        - **dy** (Tensor) - The gradient of LayerNorm's output y, float32 or float16.
        - **variance** (Tensor) - The variance of x, float32 or float16.
        - **mean** (Tensor) - The mean of x, float32 or float16.
        - **gamma** (Tensor) - The original value of weight gamma initialized in LayerNorm, float32 or float16.
          Default: 'ones'.
        - **d_dx** (Tensor) - The gradient of dx, where dx is the gradient of LayerNorm's input x, float32 or float16.
        - **d_dg** (Tensor) - The gradient of dg, where dg is the gradient of LayerNorm's weight gamma,
          float32 or float16.
        - **d_db** (Tensor) - The gradient of db, where db is the gradient of LayerNorm's weight beta,
          float32 or float16.

    Returns:
        Tuple[Tensor], tuple of 3 Tensors (the gradients of layernormgrad x, dy, gamma).

    Raises:
        TypeError: If the 8 inputs don't have the same dtype.
        ValueError: If x, dy, d_dx don't have the same shape.
        ValueError: If variance, mean don't have the same shape.
        ValueError: If gamma, d_dg, d_db don't have the same shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self, begin_norm_axis=1, begin_params_axis=1):
        """init"""
        self.begin_norm_axis = validator.check_value_type('begin_norm_axis', begin_norm_axis, [int], self.name)
        self.begin_params_axis = validator.check_value_type('begin_params_axis', begin_params_axis, [int], self.name)
        self.init_prim_io_names(inputs=['x', 'dy', 'variance', 'mean', 'gamma', 'd_dx', 'd_dg', 'd_db'],
                                outputs=['sopd_x', 'sopd_dy', 'sopd_gamma'])


class LogSoftmaxGrad(Primitive):
    """Computes gradient for the Log Softmax activation."""

    @prim_attr_register
    def __init__(self, axis=-1):
        """Initialize LogSoftmaxGrad"""
        validator.check_value_type("axis", axis, [int], self.name)


class LSTMGradData(Primitive):
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


class LSTMGradWeight(Primitive):
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


class LSTMGrad(Primitive):
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


class DynamicRNNGrad(Primitive):
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


class GruGradData(PrimitiveWithInfer):
    """Computes the data gradients of GRU."""

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

    def infer_shape(self, y_shape, dy_shape, dhy_shape, w_shape,
                    hx_shape, reserve_shape, state_shape):
        # dhy and dcy should be same shape
        validator.check_equal_int(len(dhy_shape), 3, "h_shape", self.name)

        validator.check_int(dhy_shape[0], self.num_layers * self.num_directions, Rel.EQ, "h_shape[0]", self.name)
        validator.check_equal_int(dhy_shape[2], self.hidden_size, "h_shape[2]", self.name)

        validator.check_equal_int(len(dy_shape), 3, "dy_shape", self.name)
        validator.check_equal_int(dy_shape[1], dhy_shape[1], "dy[1]", self.name)
        validator.check_int(dy_shape[2], self.hidden_size * self.num_directions, Rel.EQ, "dy[2]", self.name)

        dx_shape = (y_shape[0], y_shape[1], self.input_size)
        dhx_shape = dhy_shape

        return (dx_shape, dhx_shape)

    def infer_dtype(self, y_dtype, dy_dtype, dhy_dtype, w_dtype,
                    hx_dtype, reserve_dtype, state_dtype):
        args = {"dy": dy_dtype, "dhy": dhy_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float32, mstype.float16), self.name)
        return (dy_dtype, dy_dtype)


class GruGradWeight(PrimitiveWithInfer):
    """Computes the weight gradients of GRU."""

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
        gate_size = 3 * self.hidden_size
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


class GRUV2Grad(Primitive):
    """Computes the grad gradients of GRU."""

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


class DynamicGRUV2Grad(Primitive):
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
        - **x** (Tensor) - Current words. Tensor of shape :math:`(num\_step, batch\_size, input\_size)`.
          The data type must be float16 or float32.
        - **weight_input** (Tensor) - Weight. Tensor of shape :math:`(input\_size, 3 x hidden\_size)`.
          The data type must be float16 or float32.
        - **weight_hidden** (Tensor) - Bias. Tensor of shape :math:`(hidden\_size, 3 x hidden\_size)`.
          The data type must be float16 or float32.
        - **y** (Tensor) - A Tensor of shape :math:
          if num_proj > 0 `(num_step, batch_size, min(hidden_size, num_proj)`,
          if num_proj == 0 `(num_step, batch_size, hidden_size)`.
          The data type must be float16 or float32.
        - **init_h** (Tensor) - Hidden state of initial time.
          Tensor of shape :math:`(batch\_size, hidden\_size)`.
          The data type must be float16 or float32.
        - **h** (Tensor) - A Tensor of shape :math:`(num\_step, batch\_size, hidden\_size)`.
          The data type must be float16 or float32.
        - **dy** (Tensor) - Gradient of `y`, has the same shape and data type as `y`.
        - **dh** (Tensor) - Gradient of `h`, has the same shape and data type as `init_h`.
        - **update** (Tensor) - A Tensor of shape :math:`(num\_step, batch\_size, hidden\_size)`.
          The data type must be float16 or float32.
        - **reset** (Tensor) - A Tensor of shape :math:`(num\_step, batch\_size, hidden\_size)`.
          The data type must be float16 or float32.
        - **new** (Tensor) - A Tensor of shape :math:`(num\_step, batch\_size, hidden\_size)`.
          The data type must be float16 or float32.
        - **hidden_new** (Tensor) - A Tensor of shape :math:`(num\_step, batch\_size, hidden\_size)`.
          The data type must be float16 or float32.
        - **seq_length** (Tensor) - The length of each batch. Tensor of shape :math:`(batch\_size)`.
          Only `None` is currently supported.
        - **mask** (Tensor) - A 4-D Tensor. The data type must be float16 or float32.

    Outputs:
        - **dw_input** (Tensor) - A Tensor has the same shape as `weight_input`.
          Has the same type with input `x`.
        - **dw_hidden** (Tensor) - A Tensor has the same shape as `weight_hidden`.
          Has the same type with input `x`.
        - **db_input** (Tensor) - A Tensor of shape :math:`(3 x hidden\_size)`.
          Has the same type with input `x`.
        - **db_hidden** (Tensor) - A Tensor of shape :math:`(3 x hidden\_size)`.
          Has the same type with input `x`.
        - **dx** (Tensor) - A Tensor of shape :math:`(num\_step, batch\_size, hidden\_size)`.
          Has the same type with input `x`.
        - **dh_prev** (Tensor) - A Tensor of shape :math:`(batch\_size, hidden\_size)`.
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
        self.init_prim_io_names(inputs=[
            "x", "weight_input", "weight_hidden", "y", "init_h", "h", "dy",
            "dh", "update", "reset", "new", "hidden_new", "seq_length", "mask"
        ],
                                outputs=[
                                    "dw_input", "dw_hidden", "db_input",
                                    "db_hidden", "dx", "dh_prev"
                                ])


class PReLUGrad(Primitive):
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


class RandomGammaGrad(Primitive):
    r"""
    Computes the derivative of a random sample of Gamma with respect to alpha.:

    Inputs:
        - **alpha** (Tensor) - α is the shape parameter of RandomGamma distribution.
        It must be greater than 0. Must be one of the following types: float32, float64.
        - **sample** (Tensor) - The sample of random gamma tensor. Must be one of the
        following types: float32, float64.

    Outputs:
        The dtype is the same type as alpha.
        The output shape is derived from the input through broadcasting.

    Raises:
        TypeError: If data type of `alpha` and `sample` is not float32 or float64.
        TypeError: If data type of `alpha` and `sample` is not same.
        ValueError: If the shape last dim of `sample` and `alpha` is not equal.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> alpha = Tensor(np.array([1., 0.6, 3., 26.]), mstype.float32)
        >>> sample = Tensor(np.array([6., 7, 11., 0.5]), mstype.float32)
        >>> randomgammagrad = ops.RandomGammaGrad()
        >>> output = randomgammagrad(alpha, sample)
        >>> print(output)
        [2.5142431 3.4334087 1.8847835 0.07780622]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize RandomGammaGrad"""
        self.init_prim_io_names(inputs=['alpha', 'sample'], outputs=['output'])
        self.add_prim_attr("side_effect_hidden", True)


class ReluGrad(Primitive):
    """Performs grad of Relu operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize ReluGrad"""
        self.init_prim_io_names(inputs=['y_backprop', 'x'], outputs=['output'])


class SiLUGrad(Primitive):
    """Performs grad of SiLU operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize SiLUGrad"""
        self.init_prim_io_names(inputs=['dout', 'out'], outputs=['output'])


class ReLU6Grad(Primitive):
    """Performs grad of ReLU6 operation."""

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['y_grad', 'x'], outputs=['output'])


class ReluGradV2(Primitive):
    """Performs grad of ReLUV2 operation."""

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['gradients', 'mask'], outputs=['output'])


class EluGrad(Primitive):
    """Performs grad of Elu operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize EluGrad"""
        self.init_prim_io_names(inputs=['y_backprop', 'x'], outputs=['output'])


class GatherDGrad(Primitive):
    """Performs grad of GatherD operation."""

    @prim_attr_register
    def __init__(self, dim=0, shape=None):
        """Initialize GatherDGrad"""
        validator.check_is_int(dim, int)
        self.add_prim_attr("dim", dim)
        self.dim = dim
        self.out_shape = shape
        self.init_prim_io_names(inputs=['index', 'grad'], outputs=['output'])


class GatherDGradV2(Primitive):
    """Performs grad of GatherD operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize GatherDGradV2"""
        self.init_prim_io_names(inputs=['x', 'dim', 'index', 'grad'], outputs=['output'])


class ResizeBilinearGrad(Primitive):
    """Performs grad of ResizeBilinear operation."""

    @prim_attr_register
    def __init__(self, align_corners=False, half_pixel_centers=False):
        """init"""
        validator.check_value_type("align_corners", align_corners, [bool], self.name)
        validator.check_value_type("half_pixel_centers", half_pixel_centers, [bool], self.name)
        self.align_corners = validator.check_value_type("align_corners", align_corners, [bool], self.name)
        self.half_pixel_centers = validator.check_value_type("half_pixel_centers",
                                                             half_pixel_centers, [bool], self.name)
        self.init_prim_io_names(inputs=['grads', 'original_image'], outputs=['y'])
        if half_pixel_centers and align_corners:
            raise ValueError(f"If half_pixel_centers is True, align_corners must be False, but got {align_corners}")


class ResizeNearestNeighborGrad(Primitive):
    """
    Compute gradient of `ResizeNearestNeighbor` operator.

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


class ResizeLinear1DGrad(Primitive):
    """
    Compute gradient of `ResizeLinear1D` operator.

    Note:
        This is an experimental feature and is subjected to change.

    Args:
        coordinate_transformation_mode (string): Default is 'align_corners'. Describes how to transform the coordinate
            in the resized tensor to the coordinate in the original tensor. Other optional: 'half_pixel'.
    """

    @prim_attr_register
    def __init__(self, coordinate_transformation_mode="align_corners"):
        """Initialize ResizeLinear1DGrad"""
        self.init_prim_io_names(
            inputs=['grads', 'input_x'], outputs=['y'])
        validator.check_value_type(
            "coordinate_transformation_mode", coordinate_transformation_mode, [str], self.name)
        validator.check_string(coordinate_transformation_mode, ["align_corners", "half_pixel"],
                               "coordinate_transformation_mode", self.name)


class ResizeNearestNeighborV2Grad(Primitive):
    """
    Compute gradient of `ResizeNearestNeighborV2` operator.

    Args:
        align_corners (bool): Whether the centers of the 4 corner pixels of the input
            and output tensors are aligned. Default: False.
        half_pixel_centers (bool): Default: False.
        data_format: An optional `string` that describes the format of the input `x` Defaults to `NHWC`.
    """

    @prim_attr_register
    def __init__(self, align_corners=False, half_pixel_centers=False, data_format='NHWC'):
        """Initialize ResizeNearestNeighborV2Grad"""
        self.init_prim_io_names(inputs=['grads', 'size'], outputs=['y'])

        validator.check_value_type('align_corners', align_corners, [bool], self.name)
        validator.check_value_type('half_pixel_centers', half_pixel_centers, [bool], self.name)
        validator.check_value_type('data_format', data_format, [str], self.name)
        self.format = validator.check_string(data_format, ['NHWC', 'NCHW'], 'data_format', self.name)
        self.add_prim_attr('data_format', self.format)


class UpsampleNearest3DGrad(Primitive):
    """
    Upsample the 3-D gradient data  with the nearest neighbor interpolation algorithm.

    Args:
        input_size (listInt): An required listInt. contain 5 elements: [min_batch, channels, depth, height, width].
            Must: input_size[0] == grad_output_tensor_size[0], input_size[1] == grad_output_tensor_size[1].
        output_size (listInt): An optional listInt. Defaults to none.
            contain 3 elements: depth, height, width. The number of elements of 'output_size' should
            be the same as the rank of input 'grad_output'. Only one of 'scales' and 'output_size' can be specified.
            Must: grad_output_tensor_size[2] == floor(input_size[2] * scales[0]) == output_size[0]
            grad_output_tensor_size[3] == floor(input_size[3] * scales[1]) == output_size[1]
            grad_output_tensor_size[4] == floor(input_size[4] * scales[2]) == output_size[2].
        scales (listFloat): An optional listFloat. Defaults to none.
            The scale array along each dimension, contain 3 elements: scale_depth, scale_height, scale_width.
            The number of elements of 'scales' should be the same as the rank of input 'grad_output'.
            One of 'scales' and 'output_size' MUST be specified and it is an error if both are specified.

    Inputs:
        - **grad_output** (Tensor) - Tensor of shape [N, C, D, H, W], Must be one of the following types:
          float16, float32, float64.

    Outputs:
        Tensor, A 5-D tensor. Has the same type as input grad_output, shape depends on x and output_size/scales.

    Raises:
        TypeError: If `input_size` is not listInt.
        TypeError: When `output_size` is not none and `output_size` is not listInt.
        TypeError: When `scales` is not none and `scales` is not listFloat.
        TypeError: If dtype of `x` is not int [float16, float32, float64].
        ValueError: If any value of `input_size` is negative.
        ValueError: If any value of `output_size` is negative.
        ValueError: If any value of `scales` is negative.
        ValueError: If shape of `x` is not 5D.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self, input_size, output_size=None, scales=None):
        """Initialize UpsampleNearest3DGrad."""
        self.init_prim_io_names(inputs=['grad_output'], outputs=['y'])
        validator.check_value_type("input_size", input_size, [list, tuple], self.name)
        validator.check_positive_int_sequence(input_size, "input_size", self.name)
        if output_size is not None:
            validator.check_value_type("output_size", output_size, [list, tuple], self.name)
            validator.check_positive_int_sequence(output_size, "output_size", self.name)
        else:
            self.add_prim_attr("output_size", [])
        if scales is not None:
            validator.check_value_type("scales", scales, [list, tuple], self.name)
            validator.check_positive_float_sequence(scales, "scales", self.name)
        else:
            self.add_prim_attr("scales", [])


class ROIAlignGrad(Primitive):
    """
    ROIAlignGrad operator.

    Args:
       pooled_height (int): The output feature height.
       pooled_width (int): The output feature width.
       spatial_scale (float): The feature stride.
       sample_num (int): Number of sampling points. Default: 2.
    """

    @prim_attr_register
    def __init__(self, pooled_height, pooled_width, spatial_scale, sample_num=2):
        """Initialize ROIAlignGrad"""
        self.init_prim_io_names(inputs=["dy", "rois", "xdiff_shape"], outputs=["dx"])
        validator.check_value_type("pooled_height", pooled_height, [int], self.name)
        validator.check_value_type("pooled_width", pooled_width, [int], self.name)
        validator.check_value_type("spatial_scale", spatial_scale, [float], self.name)
        validator.check_value_type("sample_num", sample_num, [int], self.name)
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.spatial_scale = spatial_scale
        self.sample_num = sample_num


class PsROIPoolingGrad(PrimitiveWithInfer):
    """
    PsROIPoolingGrad operator.
    """

    @prim_attr_register
    def __init__(self, batch_size, channels, height, width, num_rois,
                 pooled_height, pooled_width, spatial_scale, out_dim):
        """Initialize PsROIPoolingGrad"""
        validator.check_value_type("batch_size", batch_size, [int], self.name)
        validator.check_value_type("channels", channels, [int], self.name)
        validator.check_value_type("height", height, [int], self.name)
        validator.check_value_type("width", width, [int], self.name)
        validator.check_value_type("num_rois", num_rois, [int], self.name)
        validator.check_value_type("pooled_height", pooled_height, [int], self.name)
        validator.check_value_type("pooled_width", pooled_width, [int], self.name)
        validator.check_value_type("spatial_scale", spatial_scale, [float], self.name)
        validator.check_value_type("out_dim", out_dim, [int], self.name)
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.num_rois = num_rois
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.spatial_scale = spatial_scale
        self.out_dim = out_dim

    def infer_shape(self, ydiff_shape, rois_shape, mapping_channel_shape):
        return [self.batch_size, self.channels, self.height, self.width]

    def infer_dtype(self, ydiff_type, rois_type, mapping_channel_type):
        return ydiff_type


class SigmoidGrad(Primitive):
    """Gets the gradient of Sigmoid operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize SigmoidGrad"""
        self.init_prim_io_names(inputs=['y', 'dy'], outputs=['output'])


class _ActivationGrad(PrimitiveWithInfer):
    """_ActivationGrad base class."""

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


class HSwishGrad(Primitive):
    """Gets the gradient of HSwish operation."""
    @prim_attr_register
    def __init__(self):
        """Initialize HSwishGrad"""
        self.init_prim_io_names(inputs=['y_grad', 'x'], outputs=['output'])


class SigmoidCrossEntropyWithLogitsGrad(Primitive):
    """Computes the gradients of `SigmoidCrossEntropyWithLogits`."""

    @prim_attr_register
    def __init__(self):
        """Initialize SigmoidCrossEntropyWithLogitsGrad"""
        self.init_prim_io_names(inputs=['x', 'y', 'dout'], outputs=['x_grad'])


class SliceGrad(PrimitiveWithInfer):
    """Reverse of slice."""

    @prim_attr_register
    def __init__(self):
        """Initialize SliceGrad"""
        self.init_prim_io_names(inputs=['dy', 'x', 'begin', 'size'], outputs=['dx'])

    def __infer__(self, dy, x, begin, size):
        dy_shape, x_shape, size_value, begin_v = dy['shape'], x['shape'], size['value'], begin['value']
        dy_shape_len = len(dy_shape)
        if size_value is not None and not is_shape_unknown(x_shape) and not is_shape_unknown(dy_shape):
            size_value = list(size_value)
            for i in range(dy_shape_len):
                if size_value[i] == -1:
                    size_value[i] = x_shape[i] - begin_v[i]
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
        self.init_prim_io_names(inputs=['x', 'loss_grad', 'target', 'weight', 'total_weight'], outputs=['x_grad'])
        self.reduction = validator.check_string(reduction, ['none', 'sum', 'mean'], 'reduction', self.name)
        self.add_prim_attr('reduction', self.reduction)


class SmoothL1LossGrad(Primitive):
    """Computes gradient for prediction on SmoothL1Loss."""

    @prim_attr_register
    def __init__(self, beta=1.0, reduction='none'):
        self.add_prim_attr('sigma', self.beta)
        self.reduction = validator.check_string(
            reduction, ['none', 'sum', 'mean'], 'reduction', self.name)


class SoftMarginLossGrad(Primitive):
    """Computes gradient for prediction on SoftMarginLoss."""

    @prim_attr_register
    def __init__(self, reduction="mean"):
        self.init_prim_io_names(inputs=['predict', 'label', "dout"], outputs=['gradient'])
        self.reduction = validator.check_string(reduction, ['none', 'sum', 'mean'], 'reduction', self.name)


class StridedSliceV2Grad(Primitive):
    """
    Performs grad of StridedSliceV2 operation.

    Inputs:
        - **shapex** (Tensor) - StridedSliceV2 shape of input
        - **begin** (tuple[int]) - A tuple which represents the location where to start. Only
          constant value is allowed.
        - **end** (tuple[int]) - A tuple or which represents the maximum location where to end.
          Only constant value is allowed.
        - **strides** (tuple[int]) - A tuple which represents the stride is continuously added
          before reaching the maximum location. Only constant value is allowed.
        - **dy** (Tensor) - The output of StridedSliceV2

    Outputs:
        Tensor, the shape same as the input of StridedSliceV2
    """

    @prim_attr_register
    def __init__(self,
                 begin_mask=0,
                 end_mask=0,
                 ellipsis_mask=0,
                 new_axis_mask=0,
                 shrink_axis_mask=0):
        """Initialize StridedSliceV2Grad"""
        self.set_const_input_indexes([0])
        self.init_prim_io_names(inputs=['shapex', 'begin', 'end', 'strides', 'dy'], outputs=['output'])


class StridedSliceGrad(Primitive):
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


class SoftplusGrad(Primitive):
    """Computes gradient for the Softplus activation."""

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['gradients', 'features'], outputs=['backprops'])


class TanhGrad(Primitive):
    """Computes gradient of hyperbolic tangent of input element-wise."""

    @prim_attr_register
    def __init__(self):
        """Initialize TanhGrad"""
        self.init_prim_io_names(inputs=['y', 'dy'], outputs=['z'])


class MirrorPadGrad(Primitive):
    """Gradients of MirrorPad operation."""

    @prim_attr_register
    def __init__(self, mode="REFLECT"):
        """Initialize MirrorPad"""
        self.set_const_input_indexes([1])
        self.init_prim_io_names(inputs=['dy', 'paddings'], outputs=['output'])
        validator.check_string(mode, ['REFLECT', 'SYMMETRIC'], 'mode', self.name)
        self.mode = mode


class PadV3Grad(Primitive):
    """Gradients of PadV3 operation."""

    @prim_attr_register
    def __init__(self, mode='reflect', paddings_contiguous=True):
        """Initialize Padv3Grad"""
        self.add_prim_attr("cust_aicpu", self.name)
        self.init_prim_io_names(inputs=['x', 'paddings'], outputs=['y'])
        validator.check_string(mode, ['reflect', 'edge', 'circular'], 'mode', self.name)
        validator.check_bool(paddings_contiguous, "paddings_contiguous", self.name)
        self.set_const_input_indexes([1])
        self.mode = mode
        self.paddings_contiguous = paddings_contiguous


class EmbeddingLookupCommGrad(PrimitiveWithInfer):
    """
    Performs the gradient for the communication part of EmbeddingLookup operator.

    This works ONLY when 'reduce_scatter_flag' is True in 'EmbeddingLookup'. Roughly speaking,
    this primitive is implemented by StridedSlice --> _HostAllGather --> Concat. This primitive runs on host.
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['dy', 'split_num'], outputs=['output'])
        self.set_device('CPU')
        self.tuple_setitem = Primitive('tuple_setitem')

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
        dy_shape_all = self.tuple_setitem(dy_shape, 0, dy_shape[0] * 8)
        return {'shape': dy_shape_all,
                'dtype': dy['dtype'],
                'value': None}


class RefToEmbed(Primitive):
    r"""
    Make a key from Ref.

    The Key is a symbolic_key, is a embedding on Parameter, which is used as a key of the variable in env_type,
    and get items by operation `EnvironGet` with the symbolic_key instance. The `Parameter` is a ref.

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


class AtanGrad(Primitive):
    """
    Computes AtanGrad of input element-wise.

    Returns:
        Tensor, has the same type as input.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize AtanGrad"""


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


class InvGrad(Primitive):
    """Computes gradients for inv operation."""

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x', 'grad'], outputs=['y'])


class LRNGrad(Primitive):
    """Computes gradients for LRN operation."""

    @prim_attr_register
    def __init__(self, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5):
        self.init_prim_io_names(inputs=['grads', 'x', 'y'], outputs=['z'])
        validator.check_value_type("depth_radius", depth_radius, [int], self.name)
        validator.check_value_type("bias", bias, [float], self.name)
        validator.check_value_type("alpha", alpha, [float], self.name)
        validator.check_value_type("beta", beta, [float], self.name)


class MvlgammaGrad(Primitive):
    r"""
    Computes gradients for Mvlgamma.

    The following tex shows the mathematical calculation process of Mvlgamma:

    .. math::

        \log (\Gamma_{p}(a))=C+\sum_{i=1}^{p} \log (\Gamma(a-\frac{i-1}{2}))

    where :math:`C = \log(\pi) \times \frac{p(p-1)}{4}` and :math:`\Gamma(\cdot)` is the Gamma function.

    Args:
        p(int): The number of dimensions. And the value of `p` must be greater than or equal to 1.

    Inputs:
        - **y_grad** (Tensor) - The input gradient.
        - **x** (Tensor) - The input of Mvlgamma with data type of float32 or float64.

    Outputs:
        Tensor, has the same shape and type as `x`.

    Raises:
        TypeError: If dtype of `y_grad or `x` is neither float32 nor float64.
        TypeError: If `p` is not an int.
        ValueError: If p is not greater than or equal to 1.
        ValueError: If all elements of `x` are not greater than (p-1)/2.

    Supported Platforms:
        ``Ascend`` ``CPU``
    """

    @prim_attr_register
    def __init__(self, p):
        self.init_prim_io_names(inputs=['y_grad', 'x'], outputs=['x_grad'])
        self.p = validator.check_value_type('p', p, [int], self.name)


class MaskedSelectGrad(PrimitiveWithInfer):
    """Computes gradient for MaskedSelect."""

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, x, mask, grad):
        return x

    def infer_dtype(self, x, mask, grad):
        return x


class SoftShrinkGrad(Primitive):
    r"""
          Gradients for SoftShrink operation.

          Args:
              lambd – The \lambdaλ (must be no less than zero) value for the Softshrink formulation. Default: 0.5.

          Inputs:
              - **input_grad** (Tensor) - The input gradient.
              - **input_x** (Tensor) - The input of SoftShrink with data type of float16 or float32.
                Any number of additional dimensions.

          Outputs:
              output - Tensor, has the same shape and data type as input_x.

          Raises:
              TypeError: If lambd is not a float.
              TypeError: If dtype of input_x is neither float16 nor float32.
              ValueError: If lambd is less than to 0.

          Supported Platforms:
              ``Ascend``
      """

    @prim_attr_register
    def __init__(self, lambd=0.5):
        self.init_prim_io_names(inputs=['input_grad', 'input_x'], outputs=['output'])
        validator.check_value_type("lambd", lambd, [float], self.name)
        validator.check_number("lambd", lambd, 0, Rel.GE, self.name)


class CdistGrad(Primitive):
    """Computes gradient for Cdist."""

    @prim_attr_register
    def __init__(self, p=2.0):
        validator.check_value_type("p", p, [float], self.name)
        self.init_prim_io_names(inputs=['grad', 'input_x', 'input_y', 'cdist'], outputs=['output'])


class PdistGrad(Primitive):
    """Computes gradient for Pdist operation.

    Args:
        p (float): the p value for the Pdist formulation. Default: 2.0.

    Inputs:
        - **y_grad** (Tensor) - The gradients of loss to output of Pdist function.
        - **x** (Tensor) - Input tensor of shape :math:`(N, M)`.
        Must be the input `x` of the forward operator Pdist.
        - **y** (Tensor) - Input tensor of shape :math:`(N*(N-1)/2)`.
        Must be the output `y` of the forward operator Pdist.

    Outputs:
        Tensor, with the same shape and dtype as `x`.

    Raises:
        TypeError: If one of `y_grad`, `x` and `y` is not a Tensor.
        TypeError: If dtype of `y_grad`, `x` and `y` are not all float16, float32 or float64.
        TypeError: If `p` is not a float.
        ValueError: If `p` is a negative float.
        ValueError: If shape of `y_grad` is not same as `y`.
        ValueError: If dimension of `x` is not 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self, p=2.0):
        validator.check_value_type("p", p, [float], self.name)
        if p < 0:
            raise ValueError('Pdist p must be a non-negative value, but got `{p}`.')
        self.init_prim_io_names(inputs=['y_grad', 'x', 'y'], outputs=['x_grad'])


class MultilabelMarginLossGrad(Primitive):
    """
    Compute the gradients of MultilabelMarginLoss operation.

    Args:
        reduction (str): Apply specific reduction method to the output: 'none', 'mean', 'sum'. Default: "mean".

    Inputs:
        - **y_grad** (Tensor) - The gradients of loss to output of MultilabelMarginLoss function, with
          the same shape and data type as forward output `y`.
        - **x** (Tensor) - Predict data. Tensor of shape :math:`(C)` or :math:`(N, C)`, where :math:`N`
          is the batch size and :math:`C` is the number of classes. Data type must be float16 or float32.
        - **target** (Tensor) - Ground truth data, with the same shape as `x`, data type must be int32 and
          label targets padded by -1.
        - **is_target** (Tensor) - Forward output tensor for backward input, with the same shape and
          data type as `target`.

    Outputs:
        The shape of output :math:`(C)` or :math:`(N, C)`, with the same shape and data type as `x`.

    Raises:
        TypeError: If `x` or `target` or `y_grad` is not a Tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.
        TypeError: If dtype of `target` is not int32.
        TypeError: If dtype of `y_grad` is not the same as `x`.
        ValueError: If length of shape of `x` is neither 1 nor 2.
        ValueError: If shape of `x` is not the same as `target`.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.
        ValueError: If shape of `y_grad` is not the same as forward output `y`.

    Supported Platforms:
        ``Ascend``
    """

    @prim_attr_register
    def __init__(self, reduction="mean"):
        """Initialize MultilabelMarginLossGrad"""
        self.reduction = validator.check_string(reduction, ['none', 'sum', 'mean'], 'reduction', self.name)
        self.init_prim_io_names(inputs=['y_grad', 'x', 'target', 'is_target'], outputs=['x_grad'])


class HShrinkGrad(Primitive):
    """
    Computes gradients for HShrinkGrad operation.

    Args:
        lambd (float): the λ value for the Hardshrink formulation. Default: 0.5

    Inputs:
        - **Gradients** (Tensor) - the gradients of loss to output of HShrink function.
          Currently gradients data type only support float16 and float32.
        - **Features** (Tensor) - Must be the input `input_x` of the forward operator HSHrink.
          Currently features data type only support float16 and float32.

    Outputs:
        backprops - Tensor, with the same shape and data type as `features`.

    Rasise:
        ValueError: If `lambd` is not a float.
        ValueError: If shape of `gradients` is not the same as `features`.
        TypeError: If dtype of `gradients` is not the same as `features`.
        TypeError: If dtype of `gradients` or `features` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self, lambd=0.5):
        validator.check_value_type("lambd", lambd, [float], self.name)
        if lambd < 0.0:
            lambd = 0.0
            self.add_prim_attr('lambd', lambd)


class Dilation2DBackpropInput(Primitive):
    """
    Computes the gradient of morphological 2-D dilation with respect to the input.

    .. warning::
        This operator is an experimental operator, which has some accuracy problems for some inputs.

    Args:
        stride (Union[int, tuple[int]]): The distance of filter moving, an int number that represents
            the height and width of movement are both strides, a tuple of two int numbers that
            represent height and width of movement respectively, or a tuple of four int numbers which
            should be :math:`(1, 1, H_{stride}, W_{stride})`.
        dilation (Union[int, tuple[int]]): The input stride for atrous morphological dilation.The data
            type is int or a tuple of 2 or 4 integers. Its value must be greater or equal to 1 and bounded
            by the height and width of the input `x`.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid".
            Default: "same". Both upper and lower case are supported.
        data_format (str): The format of input and output data. Only NCHW format is supported at present.
            Default:'NCHW'

    Inputs:
        - **x** (Tensor) - Input data. A four dimension tensor with float16 or float32 data type. The shape must be
          :math:`(N, C_{in}, H_{in}, W_{in})`.
        - **filter** (Tensor) - A three dimension tensor with the same type as input. The shape must be
          :math:`(C_{in}, H_{filter}, W_{filter})`.
        - **out_backprop** (Tensor) - The gradients with respect to the output of the convolution.
          A four dimension tensor with float16 or float32 data type. The shape must be
          :math:`(N, C_{in}, H_{out}, W_{out})`.

    outputs:
        Tensor, the gradients with respect to the input of convolution. It has the same shape and type as the input `x`.

    Raises:
        TypeError: If type of `x` or `filter` is not the tpye in [uint8, uint16, uint32, uint64, int8, int16,
                                  int32, int64, float16, float32, float64].
        TypeError: If type of `out_backprop` is not the tpye in [uint8, uint16, uint32, uint64, int8, int16,
                                  int32, int64, float16, float32, float64].
        TypeError: If `stride` or `dilation` is not an int number or a tuple of two or four int numbers.
        ValueError: If the length of `stride` or `dilation` is neither two nor four when they are tuples.
        ValueError: If `stride` or `dilation` is not (1, 1, height, width) when it is a tuple of four int numbers.
        ValueError: If `stride` is not in the range of [1, 255].
        ValueError: If `dilation` is less than 1.
        ValueError: If `pad_mode` is not a str of 'same', 'valid', 'SAME' or 'VALID'.
        ValueError: If `data_format` is not the str of 'NCHW'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        (pad_mode="SAME", data_format="NCHW")
        >>> out_backprop = Tensor(np.ones([1, 3, 4, 4]), mstype.float32)
        >>> filter = Tensor(np.ones([3 , 2 , 2]), mstype.float32)
        >>> x = Tensor(np.ones([1, 3, 4, 4]), mstype.float32)
        >>> dilation_backprop_input = G.Dilation2DBackpropInput(stride=1, dilation=1)
        >>> output = dilation_backprop_input(x, filter, out_backprop)
        >>> print(output)
        [[[[1. 1. 1. 1.]
           [1. 1. 1. 1.]
           [1. 1. 1. 1.]
           [1. 1. 1. 1.]]
          [[1. 1. 1. 1.]
           [1. 1. 1. 1.]
           [1. 1. 1. 1.]
           [1. 1. 1. 1.]]
          [[1. 1. 1. 1.]
           [1. 1. 1. 1.]
           [1. 1. 1. 1.]
           [1. 1. 1. 1.]]]]
    """

    @prim_attr_register
    def __init__(self, stride, dilation, pad_mode="SAME", data_format="NCHW"):
        """Initialize Dilation2DBackpropInput"""

        def _check_format_stride_or_dilation(arg_name, arg_value, prim_name, data_format):
            validator.check_value_type(arg_name, arg_value, (int, tuple), prim_name)
            if isinstance(arg_value, int):
                ret_value = (1, arg_value, arg_value, 1) if data_format == "NHWC" else (1, 1, arg_value, arg_value)
            elif len(arg_value) == 2:
                ret_value = (1, arg_value[0], arg_value[1], 1) if data_format == "NHWC" else \
                    (1, 1, arg_value[0], arg_value[1])
            elif len(arg_value) == 4:
                if data_format == "NHWC" and (arg_value[0] != 1 or arg_value[3] != 1):
                    raise ValueError(f"For '{prim_name}' attr '{arg_name}' should be "
                                     f"[1, {arg_name}_height, {arg_name}_weigth, 1] when data_format is 'NHWC', "
                                     f"but got {arg_value}")
                if data_format == "NCHW" and (arg_value[0] != 1 or arg_value[1] != 1):
                    raise ValueError(
                        f"For '{prim_name}' attr '{arg_name}' should be [1, 1, {arg_name}_height, {arg_name}_weigth]"
                        f"when data_format is 'NCHW', but got {arg_value}")
                ret_value = arg_value
            else:
                raise ValueError(
                    f"For '{prim_name}' attr '{arg_name}' should be an positive int number or a tuple of two "
                    f"or four positive int numbers, but got {arg_value}")
            for item in ret_value:
                if isinstance(item, int) and not isinstance(item, bool) and item > 0:
                    continue
                raise ValueError(
                    f"For '{prim_name}' attr '{arg_name}' should be an positive int number or a tuple of two "
                    f"or four positive int numbers, but got {arg_value}")
            return ret_value

        if data_format == 'NHWC':
            raise ValueError(f"For '{self.name}', NHWC format is not supported at present.")
        self.data_format = validator.check_string(self.data_format, ['NCHW', 'NHWC'], 'data_format', self.name)
        self.add_prim_attr("data_format", self.data_format)
        self.pad_mode = validator.check_string(self.pad_mode, ["SAME", "VALID", 'same', "valid"], "pad_mode", self.name)
        self.add_prim_attr("pad_mode", self.pad_mode.upper())
        self.stride = _check_format_stride_or_dilation("stride", stride, self.name, self.data_format)
        self.add_prim_attr("stride", self.stride)
        self.dilation = _check_format_stride_or_dilation("dilation", dilation, self.name, self.data_format)
        self.add_prim_attr("dilation", self.dilation)


class Dilation2DBackpropFilter(Primitive):
    """
    Computes the gradient of morphological 2-D dilation with respect to the filter.

    .. warning::
        This operator is an experimental operator, which has some accuracy problems for some inputs.

    Args:
        stride (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, a tuple of two int numbers that
            represent height and width of movement respectively, or a tuple of four int numbers which
            should be :math:`(1, 1, H_{stride}, W_{stride})`.
        dilation (Union(int, tuple[int])): The data type is int or a tuple of 2 integers or a tuple of 4 integers.
            Specifies the dilation rate to use for dilated convolution.
            If set to be :math:`k > 1`, there will be :math:`k - 1` pixels skipped for each sampling location.
            Its value must be greater or equal to 1 and bounded by the height and width of the input `x`.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid".
            Default: "same". Both upper and lower case are supported.
        data_format (str): The format of input and output data. Only NCHW format is supported at present.
            Default:'NCHW'

    Inputs:
        - **x** (Tensor) - Input data. A four dimension tensor with float16 or float32 data type. The shape must be
          :math:`(N, C_{in}, H_{in}, W_{in})`.
        - **filter** (Tensor) - A three dimension tensor with the same type as input. The shape must be
          :math:`(C_{in}, H_{filter}, W_{filter})`.
        - **out_backprop** (Tensor) - The gradients with respect to the output of the convolution.
          A four dimension tensor with float16 or float32 data type. The shape must be
          :math:`(N, C_{in}, H_{out}, W_{out})`.

    outputs:
        Tensor, the gradients with respect to the input of convolution. It has the same shape and type as the input `x`.

    Raises:
        TypeError: If type of `x` or `filter` is not the tpye in [uint8, uint16, uint32, uint64, int8, int16,
                                  int32, int64, float16, float32, float64].
        TypeError: If type of `out_backprop` is not the tpye in [uint8, uint16, uint32, uint64, int8, int16,
                                  int32, int64, float16, float32, float64].
        TypeError: If `stride` or `dilation` is not an int number or a tuple of two or four int numbers.
        ValueError: If the length of `stride` or `dilation` is neither two nor four when they are tuples.
        ValueError: If `stride` or `dilation` is not (1, 1, height, width) when it is a tuple of four int numbers.
        ValueError: If `stride` is not in the range of [1, 255].
        ValueError: If `dilation` is less than 1.
        ValueError: If `pad_mode` is not a str of 'same', 'valid', 'SAME' or 'VALID'.
        ValueError: If `data_format` is not the str of 'NCHW'.


    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        (pad_mode="SAME", data_format="NCHW")
        >>> x = Tensor(np.ones([2, 3, 4, 4]), mstype.float32)
        >>> filter = Tensor(np.ones([3,2,2]), mstype.float32)
        >>> out_backprop = Tensor(np.ones([2,3,2,2]), mstype.float32)
        >>> dilation_backprop_filter = G.Dilation2DBackpropFilter(stride=2, dilation=1)
        >>> output = dilation_backprop_filter(x, filter, out_backprop)
        >>> print(output)
        [[[8. 8. 8.]
          [0. 0. 0.]]
         [[0. 0. 0.]
          [0. 0. 0.]]]
    """

    @prim_attr_register
    def __init__(self, stride, dilation, pad_mode="SAME", data_format="NCHW"):
        """Initialize Dilation2DBackpropFilter"""

        def _check_format_stride_or_dilation(arg_name, arg_value, prim_name, data_format):
            validator.check_value_type(arg_name, arg_value, (int, tuple), prim_name)
            if isinstance(arg_value, int):
                ret_value = (1, arg_value, arg_value, 1) if data_format == "NHWC" else (1, 1, arg_value, arg_value)
            elif len(arg_value) == 2:
                ret_value = (1, arg_value[0], arg_value[1], 1) if data_format == "NHWC" else \
                    (1, 1, arg_value[0], arg_value[1])
            elif len(arg_value) == 4:
                if data_format == "NHWC" and (arg_value[0] != 1 or arg_value[3] != 1):
                    raise ValueError(
                        f"For '{prim_name}' attr '{arg_name}' should be [1, {arg_name}_height, {arg_name}_weigth, 1]"
                        f"when data_format is 'NHWC', but got {arg_value}")
                if data_format == "NCHW" and (arg_value[0] != 1 or arg_value[1] != 1):
                    raise ValueError(
                        f"For '{prim_name}' attr '{arg_name}' should be [1, 1, {arg_name}_height, {arg_name}_weigth]"
                        f"when data_format is 'NCHW', but got {arg_value}")
                ret_value = arg_value
            else:
                raise ValueError(
                    f"For '{prim_name}' attr '{arg_name}' should be an positive int number or a tuple of two "
                    f"or four positive int numbers, but got {arg_value}")
            for item in ret_value:
                if isinstance(item, int) and not isinstance(item, bool) and item > 0:
                    continue
                raise ValueError(
                    f"For '{prim_name}' attr '{arg_name}' should be an positive int number or a tuple of two "
                    f"or four positive int numbers, but got {arg_value}")
            return ret_value

        if data_format == 'NHWC':
            raise ValueError(f"For '{self.name}', NHWC format is not supported at present.")
        self.data_format = validator.check_string(self.data_format, ['NCHW', 'NHWC'], 'data_format', self.name)
        self.add_prim_attr("data_format", self.data_format)
        self.pad_mode = validator.check_string(self.pad_mode, ["SAME", "VALID", 'same', "valid"], "pad_mode", self.name)
        self.add_prim_attr("pad_mode", self.pad_mode.upper())
        self.stride = _check_format_stride_or_dilation("stride", stride, self.name, self.data_format)
        if self.stride[2] < 1 or self.stride[2] > 255 or self.stride[3] < 1 or self.stride[3] > 255:
            raise ValueError(f"For '{self.name}', size of stride is not supported, "
                             f'stride should be in the range of [1, 255], '
                             f'but got stride_h: `{self.stride[2]}`, stride_w: `{self.stride[3]}`.')
        self.add_prim_attr("stride", self.stride)
        self.dilation = _check_format_stride_or_dilation("dilation", dilation, self.name, self.data_format)
        self.add_prim_attr("dilation", self.dilation)


class ParallelResizeBilinearGrad(PrimitiveWithInfer):
    """ParallelResizeBilinearGrad ops"""

    @prim_attr_register
    def __init__(self, ori_image_size, src_start_w, dst_start_w, align_corners):
        """Initialize ParallelResizeBilinearGrad."""
        self.init_prim_io_names(inputs=["grad", "x", "size"], outputs=['y'])
        validator.check_value_type("ori_image_size", ori_image_size, [tuple, list], self.name)
        validator.check_value_type("src_start_w", src_start_w, [int], self.name)
        validator.check_value_type("dst_start_w", dst_start_w, [int], self.name)
        validator.check_value_type("align_corners", align_corners, [bool], self.name)
        self.ori_image_size = list(ori_image_size)
        self.src_start_w = src_start_w
        self.dst_start_w = dst_start_w
        self.align_corners = align_corners
        self.half_pixel_centers = False
        self.add_prim_attr('ori_image_size', self.ori_image_size)
        self.add_prim_attr('src_start_w', self.src_start_w)
        self.add_prim_attr('dst_start_w', self.dst_start_w)
        self.add_prim_attr('align_corners', self.align_corners)
        self.add_prim_attr('half_pixel_centers', self.half_pixel_centers)

    def __infer__(self, grad, x, size):
        size_val = size['value']
        grad_shape = grad['shape']
        grad_dtype = grad['dtype']
        x_shape = x['shape']
        x_dtype = x['dtype']
        validator.check_tensor_dtype_valid("grad_dtype", grad_dtype, [mstype.float16, mstype.float32], self.name)
        validator.check_tensor_dtype_valid("x_dtype", x_dtype, [mstype.float16, mstype.float32], self.name)
        if size_val is None:
            raise ValueError("size must be const input")
        output_shape = [grad_shape[0], grad_shape[1], x_shape[2], x_shape[3]]

        return {'shape': output_shape,
                'dtype': x_dtype,
                'value': None}


class MultiMarginLossGrad(Primitive):
    """
    Compute the gradients of MultiMarginLoss operation

    Args:
        p (int): Optional. The norm degree for pairwise distance.Should be 1 or 2. Default: 1.
        margin (float): Optional. A parameter to change pairwise distance. Default: 1.0.
        reduction (str): Apply specific reduction method to the output: 'none', 'mean', 'sum'. Default: "mean".

    Inputs:
        - **y_grad** (Tensor) - If it's not a scalar, the shape of 'y_grad' :math:`(N, C)`.
          Data type only support float32 or float16,float64.
        - **x** (Tensor) - Input x, with shape :math:`(N, C)`. Data type only support float32, float16 or float64.
        - **target** (Tensor) - Ground truth labels, with shape :math:`(N,)`. Data type only support int64. The
          value of target should be non-negative, less than C.
        - **weight** (Tensor, optional) - The rescaling weight to each class with shape :math:`(C,)`. Data type only
          support float32, float16 or float64. Default: None.

    Outputs:
        The shape of output :math:`(N, C)`. Data type only support float32 or float16, float64.
        Has the same data type with 'x'.

    Raises:
        TypeError: If dtype of `p` and `target` is not int.
        TypeError: If dtype of `margin` is not float.
        TypeError: If dtype of `reduction` is not str.
        TypeError: If dtype of `x` is not float16, float or float64.
        TypeError: If dtype of `weight` and `x` is not the same.
        ValueError: If 'p' is not 1 or 2.
        ValueError: If 'reduction' is not one of {'none','sum','mean'}.
        ValueError: If shape[0] of `x` is not equal to shape[0] of `target`.
        ValueError: If shape[1] of `x` is not equal to shape[0] of `weight`.
        ValueError: IF rank of `weight` is not 1.
        ValueError: If rank of `x` is not 2 or rank of 'target' is not 1.

    Supported Platforms:
        ``Ascend``  ``CPU``
    """

    @prim_attr_register
    def __init__(self, p=1, margin=1.0, reduction="mean"):
        """Initialize MultiMarginLossGrad"""
        self.p = validator.check_value_type('p', p, [int], self.name)
        validator.check_int(p, {1, 2}, Rel.IN, 'p', self.name)
        self.margin = validator.check_value_type('margin', margin, [float], self.name)
        self.reduction = validator.check_string(reduction, ['none', 'sum', 'mean'], 'reduction', self.name)
        self.init_prim_io_names(inputs=['y_grad', 'x', 'target', 'weight'], outputs=['x_grad'])


class UpsampleTrilinear3DGrad(Primitive):
    r"""
    Upsample the 3-D gradient data with trilinear interpolation algorithm.

    Args:
        input_size (Union[tuple[int], list[int]]): An required listInt.
            contain 5 elements: [batch, channels, depth, height, width]. Must:
            input_size[0] == grad_output_tensor_size[0]
            input_size[1] == grad_output_tensor_size[1].
        output_size (Union[tuple[int], list[int]]): An optional listInt. Defaults to none.
            contain 3 elements: depth, height, width. The number of elements of 'output_size' should
            be the same as the rank of input 'grad_output'.
            Only one of 'scales' and 'output_size' can be specified. Must:
            grad_output_tensor_size[2] == floor(input_size[2] * scales[0]) == output_size[0]
            grad_output_tensor_size[3] == floor(input_size[3] * scales[1]) == output_size[1]
            grad_output_tensor_size[4] == floor(input_size[4] * scales[2]) == output_size[2].
        scales (Union[tuple[float], list[float]]): An optional listFloat. Defaults to none.
            The scale array along each dimension, contain 3 elements: scale_depth, scale_height, scale_width.
            The number of elements of 'scales' should be the same as the rank of input 'grad_output'.
            One of 'scales' and 'output_size' MUST be specified and it is an error if both are specified.
        align_corners (bool): An optional bool. Defaults to false.

    Inputs:
        - **grad_output** (Tensor) - A 5-D input tensor [N, C, D, H, W].
          Must be one of the following types: [float16, float32, float64].

    Outputs:
        backprops Tensor - A Tensor with shape depends on intput_size and output_size/scales.
            Must be one of the following types: [float16, float32, float64].

    Raises:
        TypeError: If dtype of `x` is not int [float16, float32, float64].
        TypeError: If `input_size` is not list[int] or tuple[int].
        TypeError: When `output_size` is not none and `output_size` is not list[int] or tuple[int].
        TypeError: When `scales` is not none and `scales` is not list[float] or tuple[float].
        TypeError: If type of `align_corners` is not bool.
        ValueError: If any value of `output_size` is negative or zero when `output_size` is not empty.
        ValueError: If any value of `scales` is negative or zero when `scales` is not empty.
        ValueError: If shape of `x` is not 5D.
        ValueError: If none of `scales` and `output_size` is specified or both specified.
        ValueError: If size of `scales` is not equal 3 when `scales` is specified.
        ValueError: If size of `output_size` is not equal 3 when `output_size` is specified.
        TypeError: If `input_size` is not Union[tuple[int], list[int]].
        ValueError: If any value of `input_size` is negative.
        ValueError: If elements number of `input_size` is not 5.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self, input_size, output_size=None, scales=None, align_corners=False):
        """Initialize UpsampleTrilinear3DGrad."""
        self.init_prim_io_names(inputs=['grad_output'], outputs=['y'])
        self.input_size = input_size
        self.output_size = [] if output_size is None else output_size
        self.scales = [] if scales is None else scales
        self.align_corners = align_corners

        validator.check_value_type("input_size", self.input_size, [list, tuple], self.name)
        validator.check_equal_int(len(self.input_size), 5, "the dimension of input_size", self.name)
        validator.check_value_type("output_size", self.output_size, [list, tuple], self.name)
        validator.check_value_type("scales", self.scales, [list, tuple], self.name)
        validator.check_bool(self.align_corners, self.name)
        if len(self.output_size) == 3:
            validator.check_positive_int_sequence(self.output_size, "output_size", self.name)
        if len(self.scales) == 3:
            validator.check_positive_float_sequence(self.scales, "scales", self.name)
        if self.output_size == []:
            validator.check_equal_int(len(self.scales), 3, 'size of scales', self.name)
        if self.scales == []:
            validator.check_equal_int(len(self.output_size), 3, 'size of output_size', self.name)

        self.add_prim_attr('input_size', self.input_size)
        self.add_prim_attr('output_size', self.output_size)
        self.add_prim_attr('scales', self.scales)
        self.add_prim_attr('align_corners', self.align_corners)


class GridSampler3DGrad(Primitive):
    """
    Computes gradients for GridSampler3D operation.

    Args:
        interpolation_mode (str): An optional string specifying the interpolation method. The optional values are
            "bilinear" or "nearest". Default: "bilinear".
        padding_mode (str): An optional string specifying the pad method. The optional values are "zeros", "border" or
            "reflection". Default: "zeros".
        align_corners (bool): An optional bool. If "true", the centers of the corner pixels of the input and output
            tensors are aligned. Defaults to "false".

    Inputs:
        - **grad** (Tensor) - A 5-D tensor whose dtype is float32 or float64 and whose shape is :math:`(N, C, D_{out},
          H_{out}, W_{out})`. The shape is inconsistent with the shape of the output result of forward calculation.
        - **input_x** (Tensor) - A 5-D tensor whose dtype is the same as `grad` and whose shape is :math:`(N, C,
          D_{in}, H_{in}, W_{in})`.
        - **grid** (Tensor) - A 5-D tensor whose dtype is the same as `grad` and whose shape is :math:`(N, D_{out},
          H_{out}, W_{out}, 3)`.

    Outputs:
        - **dx** (Tensor) - A 5-D tensor whose dtype and shape are the same as `input_x`.
        - **dgrid** (Tensor) - A 5-D tensor whose dtype and shape are the same as `grid`.

    Raises:
        TypeError: If `grad`, `input_x` or `grid` is not a Tensor.
        TypeError: If the dtypes of `grad`, `input_x` and `grid` are inconsistent.
        TypeError: If the dtype of `grad`, `input_x` or `grid` is not a valid type.
        TypeError: If `align_corners` is not a boolean value.
        ValueError: If the rank of `grad`, `input_x` or `grid` is not equal to 5.
        ValueError: If the first dimension of `grad`, `input_x` and `grid` are inconsistent.
        ValueError: If the last dimension of `grid` is not equal to 3.
        ValueError: If `interpolation_mode` is not "bilinear", "nearest" or a string value.
        ValueError: If `padding_mode` is not "zeros", "border", "reflection" or a string value.
        ValueError: If the shape of `grad` is inconsistent with the shape of the output result of forward calculation.

    Supported Platforms:
        ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self, interpolation_mode='bilinear', padding_mode='zeros', align_corners=False):
        """Initialize GridSampler3DGrad."""
        validator.check_string(interpolation_mode, ['bilinear', 'nearest'], 'interpolation_mode', self.name)
        validator.check_string(padding_mode, ['zeros', 'border', 'reflection'], 'padding_mode', self.name)
        validator.check_bool(align_corners, 'align_corners', self.name)
        self.init_prim_io_names(inputs=['grad', 'input_x', 'grid'], outputs=['dx', 'dgrid'])
        self.add_prim_attr('interpolation_mode', interpolation_mode)
        self.add_prim_attr('padding_mode', padding_mode)
        self.add_prim_attr('align_corners', align_corners)


class SparseSegmentMeanGrad(Primitive):
    """
    Compute gradients for SparseSegmentMeanGrad operation.

    Inputs:
        - **x** (Tensor) - A Tensor of the first input of SparseSegmentMeanGrad.
        - **indices** (Tensor) - Indices is a 1-D tensor with indices into `x`. Must be one of the following
          types: int32, int64. Has same rank as `segment_ids`. The shape should be :math:`(N,)`.
        - **segment_ids** (Tensor) - Segment_ids is a 1-D tensor with indices into the output `y`. Must be one of the
          following types: int32, int64. Values should be sorted and can be repeated. The shape should be :math:`(N,)`.
        - **output_dim0** (Tensor) - Output_dim0 is a 0-D tensor. Dimension 0 of `x` passed to SparseSegmentMean op.

    Outputs:
        A Tensor. Has the same type as `x` .
        Has same shape as `x`, except for dimension 0 which is the value of `output_dim0`.

    Raises:
        TypeError: If `x` or `indices` or `segment_ids` is not a tensor.
        TypeError: If the dtype of `x` is not any of the following data types: {float32, float64}.
        TypeError: If the dtype of `indices` is not int32.
        TypeError: If the dtype of `segment_ids` is not int32.
        TypeError: If the dtype of `output_dim0` is not int32.
        ValueError: If dimension size of `x` is less than 1.
        ValueError: If rank of `indices` or `segment_ids` is not 1.
        ValueError: If dimension size of `output_dim0` is not 0.
        ValueError: If the first dimension of `indices` is not equal to the first dimension of `segment_ids`.
        ValueError: If `segment_ids` is not sorted.
        ValueError: If `indices` is out of range of `output_dim0`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseSegmentMeanGrad"""
        self.init_prim_io_names(inputs=['x', 'indices', 'segment_ids', 'output_dim0'], outputs=['y'])


class FractionalMaxPoolGrad(Primitive):
    """Computes gradients for FractionalMaxPool operation."""

    @prim_attr_register
    def __init__(self, overlapping=False):
        self.init_prim_io_names(inputs=["orig_input", "orig_output", "out_backprop",
                                        "row_pooling_sequence", "col_pooling_sequence"],
                                outputs=["y"])
        validator.check_value_type("overlapping", overlapping, [bool], self.name)


class FractionalMaxPool3DGradWithFixedKsize(Primitive):
    """Computes gradients for FractionalMaxPool3DWithFixedKsize operation."""

    @prim_attr_register
    def __init__(self, data_format="NCDHW"):
        self.init_prim_io_names(inputs=["origin_input", "out_backprop", "argmax"], outputs=["y"])
        self.data_format = validator.check_string(data_format, ['NCDHW', "NDHWC"], 'data_format', self.name)


class MaxUnpool2DGrad(Primitive):
    r"""
    Gradients for MaxUnpool2D operation.
    """

    @prim_attr_register
    def __init__(self, ksize, strides=0, pads=0, output_shape=(), data_format="NCHW"):
        """Initialize MaxUnpool2DGrad."""
        self.init_prim_io_names(inputs=['x', 'grads', 'argmax'], outputs=['y'])
        validator.check_value_type("ksize", ksize, [int, tuple], self.name)
        validator.check_value_type("strides", strides, [int, tuple], self.name)
        validator.check_value_type("pads", pads, [int, tuple], self.name)
        validator.check_value_type("output_shape", output_shape, [tuple], self.name)
        validator.check_string(data_format, ['NCHW', 'NHWC'], 'data_format', self.name)
        validator.check_int(len(ksize), 4, Rel.EQ, "ksize rank", self.name)
        validator.check_int(len(strides), 4, Rel.EQ, "strides rank", self.name)
        validator.check_int(len(pads), 4, Rel.EQ, "pads rank", self.name)


class MaxUnpool3DGrad(Primitive):
    r"""
    Gradients for MaxUnpool3D operation.
    """

    @prim_attr_register
    def __init__(self, ksize, strides=0, pads=0, output_shape=(), data_format="NCDHW"):
        """Initialize MaxUnpool3DGrad."""
        self.init_prim_io_names(inputs=['x', 'grads', 'argmax'], outputs=['y'])
        validator.check_value_type("ksize", ksize, [int, tuple], self.name)
        validator.check_value_type("strides", strides, [int, tuple], self.name)
        validator.check_value_type("pads", pads, [int, tuple], self.name)
        validator.check_value_type("output_shape", output_shape, [tuple], self.name)
        validator.check_string(data_format, ['NCDHW', 'NDHWC'], 'data_format', self.name)
        validator.check_int(len(ksize), 5, Rel.EQ, "ksize rank", self.name)
        validator.check_int(len(strides), 5, Rel.EQ, "strides rank", self.name)
        validator.check_int(len(pads), 5, Rel.EQ, "pads rank", self.name)


class FractionalAvgPoolGrad(Primitive):
    """Computes gradients for FractionalAvgPool operation."""

    @prim_attr_register
    def __init__(self, overlapping=False):
        self.add_prim_attr("max_length", 1000000)
        self.init_prim_io_names(inputs=["orig_input_tensor_shape", "out_backprop", "row_pooling_sequence",
                                        "col_pooling_sequence"],
                                outputs=["y"])
        validator.check_value_type("overlapping", overlapping, [bool], self.name)


class PSROIPoolingGrad(Primitive):
    """Computes gradients for PSROIPooling operation."""

    @prim_attr_register
    def __init__(self, input_size, spatial_scale, group_size, output_dim):
        """Initialize PSROIPoolingGrad."""
        self.init_prim_io_names(inputs=["x", "rois"], outputs=['y'])
        validator.check_value_type("input_size", input_size, [int, tuple], self.name)
        validator.check_positive_float(spatial_scale, "spatial_scale", self.name)
        validator.check_positive_int(group_size, "group_size", self.name)
        validator.check_positive_int(output_dim, "output_dim", self.name)

        if isinstance(input_size, int):
            self.input_size = [input_size, input_size]
        else:
            self.input_size = list(input_size)

        validator.check_positive_int_sequence(self.input_size, "input_size", self.name)
        self.spatial_scale = spatial_scale
        self.group_size = group_size
        self.output_dim = output_dim

        self.add_prim_attr('input_size', self.input_size)
        self.add_prim_attr('spatial_scale', self.spatial_scale)
        self.add_prim_attr('group_size', self.group_size)
        self.add_prim_attr('output_dim', self.output_dim)


class AdaptiveMaxPool3DGrad(Primitive):
    """Computes gradients for AdaptiveMaxPool3D operation."""

    @prim_attr_register
    def __init__(self):
        """Initialize AdaptiveMaxPool3DGrad"""
        self.init_prim_io_names(inputs=['input_grad', 'x', 'argmax'], outputs=['output_grad'])


class TraceGrad(Primitive):
    """
    Computes grad for Trace operation.

    Inputs:
        - **y_grad** (Tensor) - the grad of trace to output of Trace function.
          Currently grad data type support float16, float32, int8, int16, int32, int64,
          uint8, uint16, uint32, uint64, float64.
        - **x_shape** (Tensor) - the shape of trace to output of Trace function.
          Currently shape data type support int32, int64.

    Outputs:
        x_grad - Tensor, with the same data type as 'y_grad' and shape is x_shape.

    Raises:
        TypeError: If `x_shape` is not a Tensor.
        TypeError: If the dtype of `x_shape` is neither int32 nor int64.
        ValueError: If `x_shape` is not a 1D Tensor.
        ValueError: If length of shape of `x_shape` is not equal to 2.

    Support Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['y_grad', 'x_shape'], outputs=['x_grad'])


class IgammaGradA(Primitive):
    r"""
    Computes the gradient of igamma(a, x) wrt a.

    Inputs:
        - **a** (Tensor) - The input tensor. With float32 or float64 data type.
        - **x** (Tensor) - The input tensor. With float32 data or float64 type. `x` should have
          the same dtype with `a`.

    Outputs:
        Tensor, has the same dtype as `a` and `x`.

    Raises:
        TypeError: If a or grad is not a Tensor.
        TypeError: If dtype of input x and a is not float32 nor float64.
        TypeError: If x has different dtype with a.
        ValueError: If `a` could not be broadcast to a tensor with shape of `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = Tensor(np.array([2.0, 4.0, 6.0, 8.0]).astype(np.float32))
        >>> x = Tensor(np.array([2.0, 3.0, 4.0, 5.0]).astype(np.float32))
        >>> igammagrada = G.IgammaGradA()
        >>> output = igammagrada(a, x)
        >>> print (output)
        [-0.2940046  -0.20153049 -0.13028376 -0.08352186]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize IgammaGradA"""
        self.init_prim_io_names(inputs=['a', 'x'], outputs=['z'])


class DeformableOffsetsGrad(Primitive):
    r"""
    Computes gradients of DeformableOffsets operation.
    Args:
        strides (tuple[int, int ,int ,int]): A tuple of 4 integers. The stride of sliding windows for height
            and width for H/W dimension.
        pads (tuple[int, int ,int ,int]): A tuple of 4 integers.Padding added to H/W dimension of the input.The number
            of pixels to add to each (top, bottom, left,right) side of the input
        kernel_size (tuple[int, int]): Kernel size, a tuple of 2 integers.
        dilations (tuple[int, int, int, int]): A tuple of 4 integers. The dilation factor for each dimension of
            input. Default:(1, 1, 1, 1)
        data_format (str): An optional string from:"NCHW", "NHWC".Specify the data format of the input x. Default:
            "NCHW".
        deformable_groups (int): Specify the C-axis grouping number of input x. Default: 1.
        modulated (bool): Specify version of DeformableOffsetsGrad, true means v2, false means v1. Default: true.

    Inputs:
        - **grad** (Tensor) - The input grad tensor. With float16 or float32 data type.
        - **x** (Tensor) - The input `x` of DeformableOffsets with data type of float16 or float32.
        - **offsets** (Tensor) - The input 'offsets' of DeformableOffsets with data type of float16 or float32.

    Outputs:
        - **grad_x** (Tensor) - The output grad of input `x`. With same dtype and shape of input `x`.
        - ""grad_offsets** (Tensor) - The output grad of input `offsets`. With same dtype and shape of input `offsets`.

    Supported Platforms:
        ``Ascend````GPU````CPU``
    """

    @prim_attr_register
    def __init__(self,
                 strides,
                 pads,
                 kernel_size,
                 dilations=(1, 1, 1, 1),
                 data_format="NCHW",
                 deformable_groups=1,
                 modulated=True):
        """Initialize DeformableOffsetsGrad"""
        self.init_prim_io_names(inputs=['out_backprop', 'input', 'offsets'], outputs=['out_grad'])

        self.strides = _check_positive_int_or_tuple('strides', strides, self.name, allow_four=True, ret_four=True)
        self.add_prim_attr('strides', self.strides)

        self.pads = pads
        self.add_prim_attr('pads', self.pads)

        self.kernel_size = _check_positive_int_or_tuple('kernel_size', kernel_size, self.name, allow_four=True,
                                                        ret_four=False)
        self.add_prim_attr('ksize', self.kernel_size)

        self.dilations = _check_positive_int_or_tuple('dilations', dilations, self.name, allow_four=True, ret_four=True)
        self.add_prim_attr('dilations', dilations)

        self.data_format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.name)
        self.add_prim_attr('data_format', self.data_format)

        self.deformable_groups = validator.check_positive_int(deformable_groups, 'deformable_groups', self.name)
        self.add_prim_attr('deformable_groups', self.deformable_groups)

        self.modulated = validator.check_bool(modulated, 'modulated', self.name)
        self.add_prim_attr('modulated', self.modulated)


class MedianGrad(Primitive):
    """
    Computes gradient for Median operation.

    .. warning::
        When attr `global_median` is True, the value of Median's second output Tensor `indices` value is meaningless.

    Args:
        global_median (bool): Whether the output tensor is the global median of all input tensor elements
            or not in Median operation.
        axis (int): The dimension need to reduce in Median operation.
        keep_dims (bool): Whether the output tensor need to retain `axis` dimension or not in Median operation.

    Inputs:
        - **y_grad** (Tensor) - The gradients of loss to output of Median function.
        - **x** (Tensor) - The first input is a tensor whose data type is number.
          The dtype is one of the following: int16, int32, int64, float32, double.
        - **y** (Tensor) - The first output of Median function, which datatype is same as `x`.
        - **indices** (Tensor) - The second output of Median function, which datatype is int64.

    Outputs:
        x_grad - Tensor, has the same shape as the `x`, dtype is double only when dtype of `x` is double.
        Otherwise, dtype of `x_grad` is float32.

    Raises:
        TypeError: If dtype of `y_grad` is not the same as `x`.
        ValueError: If shape of `y_grad` is not the same as `y`.

    Supported Platforms:
        ``Ascend`` ``CPU``
    """

    @prim_attr_register
    def __init__(self, global_median=False, axis=0, keep_dims=False):
        validator.check_value_type("global_median", global_median, [bool], self.name)
        self.global_median = global_median
        if global_median is False:
            validator.check_value_type("axis", axis, [int], self.name)
            validator.check_value_type("keep_dims", keep_dims, [bool], self.name)
        self.init_prim_io_names(inputs=['y_grad', 'x', 'y', 'indices'], outputs=['x_grad'])


class SparseSegmentSumGrad(Primitive):
    """
    Computes gradients for SparseSegmentSumGrad operation.

    Inputs:
        - **grad** (Tensor) - A tensor.
        - **indices** (Tensor) - Indices is a 1-D tensor. Must be one of the following types: int32, int64.
          Has same rank as segment_ids. The shape should be :math:`(N,)`.
        - **segment_ids** (Tensor) - Segment_ids is a 1-D tensor. Must be one of the following types: int32, int64.
          Values should be sorted and can be repeated. The shape should be :math:`(N,)`.
        - **output_dim0** (Tensor) - Output_dim0 is a 0-D tensor. Dimension 0 of `x` passed to SparseSegmentSum op.

    Outputs:
        A Tensor. Has the same type as `grad` .
        Has same shape as `grad`, except for dimension 0 which is the value of `output_dim0`.

    Raises:
        TypeError: If `grad` or `indices` or `segment_ids` or `output_dim0` is not a tensor.
        TypeError: If the dtype of `grad` is not any of the following data types: {float16, float32, float64}.
        TypeError: If the dtype of `indices` and `segment_ids` and `output_dim0` is not int32 or int64.
        ValueError: If dimension size of `grad` less than 1.
        ValueError: If rank of `indices` or `segment_ids` is not 1.
        ValueError: If dimension size of `output_dim0` is not 0.
        ValueError: If shape[0] of `indices` is not corresponding to shape[0] of `segment_ids`.
        ValueError: If `segment_ids` is not sorted.
        ValueError: If the last number of `segment_ids` is out of range of grad's first shape.
        ValueError: If `indices` is bigger than or equal to `output_dim0`.

    Supported Platforms:
        ``GPU``
    """
    __mindspore_signature__ = (
        sig.make_sig('grad', dtype=sig.sig_dtype.T1),
        sig.make_sig('indices', dtype=sig.sig_dtype.T),
        sig.make_sig('segment_ids', dtype=sig.sig_dtype.T),
        sig.make_sig('output_dim0', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize SparseSegmentSumGrad"""
        self.init_prim_io_names(inputs=['grad', 'indices', 'segment_ids', 'output_dim0'], outputs=['y'])


class SparseSegmentSqrtNGrad(Primitive):
    """
    Computes gradients for SparseSegmentSqrtNGrad operation.

    Inputs:
        - **x** (Tensor) - A tensor. It's rank must be more than or equal to one.
        - **indices** (Tensor) - Indices is a 1-D tensor with indices into `x`. Must be one of the following
          types: int32, int64. Has same rank as segment_ids. The shape should be :math:`(N,)`.
        - **segment_ids** (Tensor) - Segment_ids is a 1-D tensor with indices into the output `y`. Must be one
          of the following types: int32, int64. Values should be sorted and can be repeated. The shape should
          be :math:`(N,)`.
        - **output_dim0** (Tensor) - Output_dim0 is a 0-D tensor. Dimension 0 of `x` passed to SparseSegmentSqrtN op.

    Outputs:
        A Tensor. Has the same type as `x` .
        Has same shape as `x`, except for dimension 0 which is the value of `output_dim0`.

    Raises:
        TypeError: If `x` or `indices` or `segment_ids` or `output_dim0` is not a tensor.
        TypeError: If the dtype of `x` is not any of the following data types: {float16, float32, float64}.
        TypeError: If the dtype of `indices` is not int32.
        TypeError: If the dtype of `segment_ids` is not int32.
        TypeError: If the dtype of `output_dim0` is not int32.
        ValueError: If dimension size of `x` is less than 1.
        ValueError: If rank of `indices` or `segment_ids` is not 1.
        ValueError: If dimension size of `output_dim0` is not 0.
        ValueError: If shape[0] of `indices` is not corresponding to shape[0] of `segment_ids`.
        ValueError: If `segment_ids` is not sorted.
        ValueError: If the last number of `segment_ids` is out of range of x's first shape.
        ValueError: If `indices` is bigger than or equal to `output_dim0`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseSegmentSqrtNGrad"""
        self.init_prim_io_names(inputs=['x', 'indices', 'segment_ids', 'output_dim0'], outputs=['y'])


class GridSampler2DGrad(Primitive):
    """
    Computes gradients for GridSampler2D operation.

      Args:
        interpolation_mode (str): An optional string specifying the interpolation method. The optional values are
            "bilinear" or "nearest". Default: "bilinear".
        padding_mode (str): An optional string specifying the pad method. The optional values are "zeros", "border" or
            "reflection". Default: "zeros".
        align_corners (bool): An optional bool. If "true", the centers of the corner pixels of the input and output
            tensors are aligned. Defaults to "false".

      Inputs:
        - **grad** (Tensor) - A 4-D tensor whose dtype is float16 or float32 and whose shape is :math:`(N, C,
          H_{out}, W_{out})`. The shape is inconsistent with the shape of the output result of forward calculation.
        - **input_x** (Tensor) - A 4-D tensor whose dtype is the same as `grad` and whose shape is :math:`(N, C,
          H_{in}, W_{in})`.
        - **grid** (Tensor) - A 4-D tensor whose dtype is the same as `grad` and whose
          shape is :math:`(N, H_{out}, W_{out}, 2)`.

    Outputs:
        - **dx** (Tensor) - A 4-D tensor whose dtype and shape are the same as `input_x`.
        - **dgrid** (Tensor) - A 4-D tensor whose dtype and shape are the same as `grid`.

    Raises:
        TypeError: If `grad`, `input_x` or `grid` is not a Tensor.
        TypeError: If the dtypes of `grad`, `input_x` and `grid` are inconsistent.
        TypeError: If the dtype of `grad`, `input_x` or `grid` is not a valid type.
        TypeError: If `align_corners` is not a boolean value.
        ValueError: If the rank of `grad`, `input_x` or `grid` is not equal to 4.
        ValueError: If the first dimension of `grad`, `input_x` and `grid` are inconsistent.
        ValueError: If the last dimension of `grid` is not equal to 2.
        ValueError: If `interpolation_mode` is not "bilinear", "nearest" or a string value.
        ValueError: If `padding_mode` is not "zeros", "border", "reflection" or a string value.
        ValueError: If the shape of `grad` is inconsistent with the shape of the output result of forward calculation.

    Supported Platforms:
        ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self, interpolation_mode='bilinear', padding_mode='zeros', align_corners=False):
        """Initialize GridSampler2DGrad."""
        validator.check_string(interpolation_mode, ['bilinear', 'nearest'], 'interpolation_mode', self.name)
        validator.check_string(padding_mode, ['zeros', 'border', 'reflection'], 'padding_mode', self.name)
        validator.check_bool(align_corners, 'align_corners', self.name)
        self.init_prim_io_names(inputs=['grad', 'input_x', 'grid'], outputs=['dx', 'dgrid'])
        self.add_prim_attr('interpolation_mode', interpolation_mode)
        self.add_prim_attr('padding_mode', padding_mode)
        self.add_prim_attr('align_corners', align_corners)


class ResizeBicubicGrad(Primitive):
    """
    Computes gradients for ResizeBicubicGrad operation.

    Args:
        align_corners (bool):If true, the centers of the 4 corner pixels of the input
    and output tensors are aligned, preserving the values at the corner pixels.Default: False.
        half_pixel_centers (bool): An optional bool. Default: False.

    Inputs:
        - **grads** (Tensor) - A Tensor of type float. 4-D with shape
          [batch, height, width,channels]. The format must be NHWC.
        - **original_image** (Tensor) - A Tensor. Must be one of the following types: float,double.
          4-D with shape [batch, orig_height, orig_width, channels], The image tensor that was resized.
          The format must be NHWC.

    Outputs:
        A 4-D Tensor , with the same shape and data type as `original_image`.

    Rasise:
        TypeError: If `grads` is not allowed.
        TypeError: If `original_image` is not allowed.
        ValueError: If `images` dim is not 4.
        ValueError: If `size` dim is not 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self, align_corners=False, half_pixel_centers=False):
        """Initialize CropAndResize"""
        validator.check_value_type('align_corners', align_corners, bool, self.name)
        validator.check_value_type('half_pixel_centers', half_pixel_centers, bool, self.name)
        self.init_prim_io_names(inputs=['grads', 'original_image'], outputs=['y'])

    def __infer__(self, grads, original_image):
        # get shape
        grads_shape = list(grads['shape'])
        original_image_shape = list(original_image['shape'])
        # get value
        if grads['value'] is None:
            raise ValueError(f"For '{self.name}', the 'grads' cannot be None,\
                            but got {grads['value']}.")
        if original_image['value'] is None:
            raise ValueError(f"For '{self.name}', the 'original_image' cannot be None,\
                                but got {original_image['value']}.")
        # get dtype
        grads_dtype = grads['dtype']
        original_image_dtype = original_image['dtype']
        # check dytpe
        validator.check_tensor_dtype_valid("grads", grads_dtype,
                                           [mstype.float32], self.name)
        validator.check_tensor_dtype_valid("original_image", original_image_dtype,
                                           [mstype.float32, mstype.float64], self.name)
        # check input shape rank
        validator.check("grads rank", len(grads_shape), "expected", 4, Rel.EQ, self.name)
        validator.check("original_image rank", len(original_image_shape), "expected", 4, Rel.EQ, self.name)
        validator.check("batch_size equal", grads_shape[0], "expected", original_image_shape[0], Rel.EQ, self.name)
        validator.check("channel equal", grads_shape[3], "expected", original_image_shape[3], Rel.EQ, self.name)
        # check original_image_shape and grads_shape
        validator.check("original_image[0] and grads[0]", original_image_shape[0],
                        "expected", grads_shape[0], Rel.EQ, self.name)
        validator.check("original_image[3] and grads[3]", original_image_shape[3],
                        "expected", grads_shape[3], Rel.EQ, self.name)

        batch_size = grads_shape[0]
        height = original_image_shape[1]
        width = original_image_shape[2]
        channel = grads_shape[3]
        out_shape = (batch_size, height, width, channel)
        return {'shape': out_shape,
                'dtype': original_image_dtype,
                'value': None}


class SparseSliceGrad(Primitive):
    r"""
    Computes gradients for SparseSlice operation.

    Inputs:
        - **backprop_val_grad** (Tensor) - A 1D Tensor.
          The shape should be :math:`(N,)`.
        - **indices** (Tensor) - A 2D Tensor (N x R matrix) of type int64. The indices of the SparseTensor.
          Support int64, each element value should be a non-negative int number. This tensor should be sorted.
          The shape is :math:`(N, R)`.
        - **start** (Tensor) - A 1D Tensor of type int64, represents the start of the indices.
          The shape should be :math:`(R,)`.
        - **new_indices** (Tensor) - A 2D Tensor (N x C matrix) of type int64. The indices of the SparseTensor.
          Support int64, each element value should be a non-negative int number. This tensor should be sorted.
          The shape is :math:`(N, C)`.

    Outputs:
        - *y_grad_val: A Tensor. Has the same type as `backprop_val_grad`.
          Has the same number as `indices`.

    Raises:
        TypeError: If the dtype of `indices`, `start`, `new_indices` are not int64.
        ValueError: If `indices`, `new_indices` are not 2-D tensor.
        ValueError: If `backprop_val_grad`, `start` is not a 1-D tensor.
        ValueError: If the number of `backprop_val_grad` is not corresponding to the number of `new_indices`.
        ValueError: If the shape of `indices[1]` is not corresponding to `start[1]`.
        ValueError: If the shape of `indices[1]` is not corresponding to `new_indices[1]`.
        RuntimeError: If the `backprop_val_grad` is not all backpropagated, because `indices` or `new_indices`
        is not sorted.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    Examples:
        >>> backprop_val_grad = Tensor(np.array([1, 2, 3, 4]).astype(np.int64))
        >>> indices = Tensor(np.array([[0, 0], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4]]).astype(np.int64))
        >>> start = Tensor(np.array([0, 0]).astype(np.int64))
        >>> new_indices = Tensor(np.array([[0, 2], [1, 2], [1, 3], [2, 4]]).astype(np.int64))
        >>> grad = SparseSliceGrad()
        >>> output = grad(backprop_val_grad, indices, start, new_indices)
        >>> print(output)
        [0 1 2 3 0 4]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseSliceGrad."""
        self.init_prim_io_names(inputs=['backprop_val_grad', 'indices', 'start', 'new_indices'], outputs=['y_grad'])


class FractionalMaxPoolGradWithFixedKsize(Primitive):
    """
    Computes the gradients of FractionalMaxPoolWithFixedKsize.

    Args:
        data_format (str): The optional value for data format, is 'NCHW'. Default: "NCHW".

    Inputs:
        - **origin_input** (Tensor) - Tensor with data format "NCHW", data type must be int32 or int64.
        - **out_backprop** (Tensor) - The gradients with respect to the output of FractionalMaxPoolWithFixedKsize
        function. Tensor with data format "NCHW", whose data type is float16, float32, float64, int32 or int64.
        - **argmax** (Tensor) - The second output of FractionalMaxPoolWithFixedKsize function, whose data
        type is int64.

    Outputs:
        - **y** (Tensor) - Tensor, with the same shape as `origin_input`, and the same data type as
        the input `out_backprop`.

    Raises:
        TypeError: If data type of `out_backprop` is not one of the following: float16, float32, float64, int32, int64.
        TypeError: If data type of `argmax` is not int64.
        ValueError: If the shape of `out_backprop` and `argmax` is not equal.
        ValueError: If the first dimension size of `origin_input` and `out_backprop` is not equal.
        ValueError: If the second dimension size of `origin_input` and `out_backprop` is not equal.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    @prim_attr_register
    def __init__(self, data_format="NCHW"):
        self.data_format = validator.check_string(data_format, ['NCHW'], 'data_format', self.name)
        self.add_prim_attr("data_format", self.data_format)
        self.init_prim_io_names(inputs=['origin_input', 'out_backprop', 'argmax'], outputs=['y'])


class AffineGridGrad(Primitive):
    r"""
    Computes gradients for AffineGrid operation.

    Args:
        align_corners (bool): if True, consider -1 and 1 to refer to the centers
            of the corner pixels rather than the image corners. Default: False.

    Inputs:
        - **y_grad** (Tensor) - Data type must be float16 or float32.
        - **x_size** (tuple) - Data type must be int32 or int64.

    Outputs:
        Tensor, with data type same as `y_grad`.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore.ops.operations._grad_ops as _grad_ops
        >>> affinegridgrad = _grad_ops.AffineGridGrad()
        >>> y_grad = Tensor(np.ones([1, 2, 2, 2]), mindspore.float32)
        >>> x_size = (1, 2, 2, 2)
        >>> x_grad = affinegridgrad(y_grad, x_size)
        >>> print(x_grad)
        [[[0. 0. 4.]
          [0. 0. 4.]]]
    """

    @prim_attr_register
    def __init__(self, align_corners=False):
        """Initialize AffineGridGrad."""
        validator.check_value_type("align_corners", align_corners, [bool], self.name)
        self.init_prim_io_names(inputs=['y_grad', 'x_size'], outputs=['x_grad'])


class HSigmoidGrad(Primitive):
    """Gets the gradient of HSigmoid operation."""
    @prim_attr_register
    def __init__(self):
        """Initialize HSigmoidGrad"""
        self.init_prim_io_names(inputs=['grads', 'input_x'], outputs=['output'])


class GluGrad(Primitive):
    """
    Computes grad for Glu operation.
    """

    @prim_attr_register
    def __init__(self, axis):
        self.add_prim_attr("cust_aicpu", self.name)
        self.init_prim_io_names(inputs=["grads", "x"], outputs=["y"])
        validator.check_value_type("axis", axis, [int], self.name)


class CholeskyGrad(Primitive):
    r"""
    Computes the reverse mode backpropgated gradient of the Cholesky algorithm.

    Inputs:
        - **x** (Tensor) - A tensor with float32 or float64 data type.
        - **grad** (Tensor) - A tensor with float32 or float64 data type. `x` should have
          the same dtype with `a`.

    Outputs:
        Tensor, has the same dtype as `a` and `x`.

    Raises:
        TypeError: If x is not Tensor.
        TypeError: If grad is not Tensor.
        TypeError: If dtype of input x and grad is not float64 nor float32,
        TypeError: If x has different dtype with grad.
        ValueError: If input tensor's last two dims are not equal,
        ValueError: If the shape of x and grad mismatch.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.array([[4, 2],[2, 3]]), mstype.float64)
        >>> grad = Tensor(np.array([[4, 2],[2, 3]]), mstype.float64)
        >>> choleskygrad = G.CholeskyGrad()
        >>> output = choleskygrad(x, grad)
        >>> print (output)
        [[0.5 0. ]
         [0.  0.5]]

    """

    @prim_attr_register
    def __init__(self):
        """Initialize CholeskyGrad"""
        self.init_prim_io_names(inputs=['x', 'grad'], outputs=['y'])


class MapTensorGetGrad(Primitive):
    """
    Computes gradients for MapTensorGet operation.

    Inputs:
        - **map_tensor** (MapTensor) - The input `map_tensor` of the forward operator MapTensorGet.
        - **key_tensor** (Tensor) - The input `key_tensor` of the forward operator MapTensorGet.
        - **default_value** (Scalar) - The input `default_value` of the forward operator MapTensorGet.
        - **grad** (Tensor) - The grad value according the forward operator MapTensorGet.

    Outputs:
        - **output** (MapTensor) -  MapTensor with grad values.
    """
    @prim_attr_register
    def __init__(self):
        """Initialize MapTensorGetGrad"""
        self.init_prim_io_names(inputs=['map_tensor', 'key_tensor', 'default_value', 'grad'], outputs=['output'])
        self.add_prim_attr('side_effect_mem', True)


class ResizeV2Grad(Primitive):
    r"""
    Calculates the gradient of ResizeV2 operation.

    Supported Platforms:
        ``CPU``
    """

    @prim_attr_register
    def __init__(self, coordinate_transformation_mode="half_pixel", mode="nearest"):
        """Initialize ResizeV2Grad."""
        self.init_prim_io_names(inputs=["grads", "roi", "scales", "original_size"], outputs=["y"])
        self.add_prim_attr("nearest_mode", "floor")
        self.add_prim_attr("cubic_coeff_a", -0.75)
        validator.check_value_type(
            "coordinate_transformation_mode", coordinate_transformation_mode, [str], self.name)
        validator.check_string(coordinate_transformation_mode,
                               ["align_corners", "half_pixel"], "coordinate_transformation_mode", self.name)
        validator.check_value_type("mode", mode, [str], self.name)
        validator.check_string(mode, ["nearest", "linear", "cubic"], "mode", self.name)
