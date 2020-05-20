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
import math
import numpy as np
from functools import reduce

from mindspore import Tensor
from mindspore._checkparam import ParamValidator as validator
from mindspore._checkparam import Rel, check_bool, check_int_positive, twice
from mindspore.common import dtype as mstype
from mindspore.ops import prim_attr_register, PrimitiveWithInfer


class Cus_Conv2D(PrimitiveWithInfer):
    r"""
    Applies 2D convolution for the input.

    Input is typically of shape :math:`(N, C, H, W)`, where :math:`N` is batch size and :math:`C` is channel number.
    For each batch of shape :math:`(C, H, W)` the formula (given mode 1) is defined as:

    .. math::
        out_j = \sum_{i=0}^{C_{in} - 1} ccor(W_{ij}, X_i) + b_j,

    where :math:`ccor` is cross correlation operator, :math:`C_{in}` is the input channel number, :math:`j` ranges
    from :math:`0` to :math:`C_{out} - 1`, :math:`W_{ij}` corresponds to i-th channel of the j-th filter and
    :math:`out_{j}` corresponds to the j-th channel of the output.

    The first introduction can be found in paper `Gradient Based Learning Applied to Document Recognition
    <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

    More detailed introduction can be found here: http://cs231n.github.io/convolutional-networks/.

    Args:
        out_channel (int): The dimensionality of the output space.
        kernel_size (Union[int, tuple[int]]): The kernel size of the 2D convolution.
        mode (int): 0 Math convolutiuon, 1 cross-correlation convolution ,
                       2 deconvolution, 3 depthwise convolution. Default: 1.
        pad_mode (str): "valid", "same", "pad" the mode to fill padding. Default: "valid".
        pad (int): The pad value to fill. Default: 0.
        stride (int): The stride to apply conv filter. Default: 1.
        dilation (int): Specifying the dilation rate to use for dilated convolution. Default: 1.
        group (int): Split input into groups. Default: 1.

    Returns:
        Tensor, the value that applied 2D convolution.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.
    """

    @prim_attr_register
    def __init__(self,
                 out_channel,
                 kernel_size,
                 mode=1,
                 pad_mode="valid",
                 pad=0,
                 stride=1,
                 dilation=1,
                 group=1):
        """init Conv2D"""
        self.init_prim_io_names(inputs=['x', 'w'], outputs=['output'])
        self.kernel_size = kernel_size
        self.kernel_size = validator.check_type('kernel_size', kernel_size, (int, tuple))
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        validator.check_integer('length of kernel_size', len(self.kernel_size), 2, Rel.GE)
        validator.equal('type of pad', type(pad), 'not bool', not isinstance(pad, bool))
        validator.equal('type of pad', type(pad), 'int', isinstance(pad, int))
        self.pad_mode = validator.check_string('pad_mode', pad_mode, ['valid', 'same', 'pad'])
        self.pad = validator.check_pad_value_by_mode(self.__class__.__name__, pad_mode, pad)
        if self.pad_mode == 'pad':
            validator.check_integer('pad', self.pad, 0, Rel.GE)

        self.mode = validator.check_integer('mode', mode, 1, Rel.EQ)
        self.add_prim_attr('data_format', "NCHW")
        self.out_channel = validator.check_integer('out_channel', out_channel, 0, Rel.GT)
        self.group = validator.check_integer('group', group, 0, Rel.GT)
        self.dilation = validator.check_integer('dilation', dilation, 1, Rel.GE)
        validator.check_type('kernel_size', kernel_size, [int, tuple])
        if isinstance(kernel_size, int) and kernel_size < 1:
            raise ValueError('Attr \'kernel_size\' of \'Conv2D\' Op passed '
                             + str(self.kernel_size) + ', should be a int or tuple and equal to or greater than 1.')
        if isinstance(kernel_size, tuple) and (len(kernel_size) != 2 or
                                               (not isinstance(kernel_size[0], int)) or
                                               (not isinstance(kernel_size[1], int)) or
                                               kernel_size[0] < 1 or kernel_size[1] < 1):
            raise ValueError('Attr \'kernel_size\' of \'Conv2D\' Op passed '
                             + str(self.kernel_size) + ', should be a int or tuple and equal to or greater than 1.')
        self.stride = validator.check_integer('stride', stride, 1, Rel.GE)
        from conv2d_impl import Cus_Conv2D

    def infer_shape(self, x_shape, w_shape):
        validator.check_integer("weight_shape", len(w_shape), 4, Rel.EQ)
        validator.check_integer("x_shape", len(x_shape), 4, Rel.EQ)
        validator.check_param_equal("x_shape[1]", x_shape[1] // self.group, "w_shape[1]", w_shape[1])
        validator.check_param_equal('out_channel', self.out_channel, 'w_shape[0]', w_shape[0])
        validator.check_param_equal('kernel_size', self.kernel_size, 'w_shape[2:4]', tuple(w_shape[2:4]))

        kernel_size_h = w_shape[2]
        kernel_size_w = w_shape[3]

        if self.pad_mode == "valid":
            h_out = math.ceil((x_shape[2] - kernel_size_h + 1) / self.stride)
            w_out = math.ceil((x_shape[3] - kernel_size_w + 1) / self.stride)
            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        elif self.pad_mode == "same":
            h_out = math.ceil(x_shape[2] / self.stride)
            w_out = math.ceil(x_shape[3] / self.stride)

            pad_needed_h = max(0, (h_out - 1) * self.stride + kernel_size_h - x_shape[2])
            pad_top = math.floor(pad_needed_h / 2)
            pad_bottom = pad_needed_h - pad_top

            pad_needed_w = max(0, (w_out - 1) * self.stride + kernel_size_w - x_shape[3])
            pad_left = math.floor(pad_needed_w / 2)
            pad_right = pad_needed_w - pad_left
        elif self.pad_mode == 'pad':
            pad_top, pad_bottom, pad_left, pad_right = self.pad, self.pad, self.pad, self.pad

            h_out = 1 + (x_shape[2] + 2 * self.pad - kernel_size_h - (kernel_size_h - 1) * (self.dilation - 1)) \
                    / self.stride
            w_out = 1 + (x_shape[3] + 2 * self.pad - kernel_size_w - (kernel_size_w - 1) * (self.dilation - 1)) \
                    / self.stride
            h_out = math.floor(h_out)
            w_out = math.floor(w_out)

        self.pad_list = [pad_top, pad_bottom, pad_left, pad_right]
        self.add_prim_attr('pad_list', (pad_top, pad_bottom, pad_left, pad_right))

        out_channel = self.out_channel
        out_shape = [x_shape[0], out_channel, h_out, w_out]
        return out_shape

    def infer_dtype(self, x_dtype, w_dtype):
        args = {'x_dtype': x_dtype, 'w_dtype': w_dtype}
        validator.check_type_same(args, [mstype.int8, mstype.int32, mstype.float16, mstype.float32])
        return x_dtype
