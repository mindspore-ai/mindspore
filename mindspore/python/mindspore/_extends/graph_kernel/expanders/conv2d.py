# Copyright 2021 Huawei Technologies Co., Ltd
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
# ===========================================================================
"""generate json desc for Conv2D"""
from mindspore._extends.graph_kernel.model.op_infer import check_format_any, check_nd, conv_had_pad
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from mindspore._extends.graph_kernel.model.model import GraphKernelUnsupportedException as GKException
from ._utils import Expander, ExpanderInfoValidator as VLD

M_ALIGN = 32
N_ALIGN = 32
K_ALIGN = 16
K_LIMIT = 800
MNK_LIMIT = 3 * (10 ** 10)
N0_CHANNEL_ALIGN = 32
N1_CHANNEL_ALIGN = 32
C_CHANNEL_ALIGN = 16
OUT_NHW_ALIGN = 128


@VLD.add_format(DF.DEFAULT, DF.DEFAULT)
@VLD.add_format(DF.NHWC, DF.NHWC)
@VLD.check_attrs('format', 'pad_list', 'pad_mode', 'groups', 'group', 'kernel_size', 'stride', 'dilation')
class Conv2D(Expander):
    """
    Conv2D expander

    Currently, only Conv2D that meets several conditions can be expanded, other cases will be skipped.
    Conditions to expand:
      inputs are NHWC format and float16.
      attr groups and group are 1.
      attr dilation are all 1.
      N channel of inputs > 16.
      C channel of inputs > 8.
      output N*H*W are multiplies of 128.
    """

    def __init__(self, expand_info):
        super().__init__(expand_info)
        self.dst_type = self.outputs[0]['data_type']
        self.dst_format = self.outputs[0]['format']
        self.has_pad = False
        self.can_optimize_to_matmul = False
        self.shape_0_pad = self.inputs[0]['shape']
        self.shape_1_pad = self.inputs[1]['shape']
        self.m = 0
        self.n = 0
        self.k = 0

    def _optimize_to_matmul(self):
        stride = self.attrs['stride']
        dilation = self.attrs['dilation']
        _, h, w, _ = self.inputs[1]['shape']
        if h == 1 and w == 1 and stride == [1, 1, 1, 1] and dilation == [1, 1, 1, 1] and \
                self.m % M_ALIGN == 0 and self.n % N_ALIGN == 0 and self.k % K_ALIGN == 0:
            return True
        return False

    def _check(self):
        type_0 = self.inputs[0]['data_type']
        type_1 = self.inputs[1]['data_type']
        if type_0 != "float16" or type_1 != "float16":
            raise GKException(
                "inputs type should be float16, but got {} and {}".format(type_0, type_1))

        formats = [self.inputs[0]['format'], self.inputs[1]['format'], self.attrs['format']]
        check_format_any(formats, DF.NHWC)

        groups = self.attrs['groups']
        group = self.attrs['group']
        if groups != 1 or group != 1:
            raise GKException(
                "groups and group should be both 1, but got {} and {}.".format(groups, group))

        dilation = self.attrs['dilation']
        check_nd(dilation, 4)
        if dilation != [1, 1, 1, 1]:
            raise GKException(
                "dilation should be all 1, but got {}".format(dilation))

        pad_list = self.attrs['pad_list']
        pad_mode = self.attrs['pad_mode']
        check_nd(pad_list, 4)
        self.has_pad = conv_had_pad(pad_list, pad_mode)

        shape_0 = self.inputs[0]['shape']
        shape_1 = self.inputs[1]['shape']
        stride = self.attrs['stride']
        check_nd(shape_0, 4)
        check_nd(shape_1, 4)
        check_nd(stride, 4)
        n0, h0, w0, c0 = shape_0
        n1, h1, w1, c1 = shape_1
        if (n0 % N0_CHANNEL_ALIGN) != 0:
            raise GKException("N({}) channel of first input should be multiples of {}".format(n0, N0_CHANNEL_ALIGN))
        if (n1 % N1_CHANNEL_ALIGN) != 0:
            raise GKException("O({}) channel of second input should be multiples of {}".format(n1, N1_CHANNEL_ALIGN))
        if c0 != c1 or (c0 % C_CHANNEL_ALIGN) != 0:
            raise GKException("C channel of inputs({}, {}) should be same and also be multiples of {}".format(
                c0, c1, C_CHANNEL_ALIGN))
        # n0 pad
        n0 = ((n0 + N0_CHANNEL_ALIGN - 1) //
              N0_CHANNEL_ALIGN) * N0_CHANNEL_ALIGN
        # h0, w0 pad
        if self.has_pad:
            h0 = h0 + pad_list[0] + pad_list[1]
            w0 = w0 + pad_list[2] + pad_list[3]
        # c0, c1 pad
        c0 = ((c0 + C_CHANNEL_ALIGN - 1) // C_CHANNEL_ALIGN) * C_CHANNEL_ALIGN
        c1 = c0
        # n1 pad
        n1 = ((n1 + N1_CHANNEL_ALIGN - 1) //
              N1_CHANNEL_ALIGN) * N1_CHANNEL_ALIGN

        # check if can optimize to matmul
        self.m, self.n, self.k = n0 * h0 * w0, n1, c1
        self.can_optimize_to_matmul = self._optimize_to_matmul()

        # requirements
        if self.can_optimize_to_matmul:
            if self.k > K_LIMIT:
                raise GKException(
                    "If transformed to MatMul, C0({}) should not be larger than {}".format(self.k, K_LIMIT))
            if self.m * self.n * self.k >= MNK_LIMIT:
                raise GKException("If transformed to MatMul, The total size({}) should not be larger than {}".format(
                    self.m * self.n * self.k, MNK_LIMIT))
        else:
            out_h, out_w = (h0 - h1) // stride[-2] + 1, (w0 - w1) // stride[-1] + 1
            if ((n0 * out_h * out_w) % OUT_NHW_ALIGN) != 0:
                raise GKException("N({}) * H({}) * W({}) of output should be multiplies of {}".format(
                    n0, out_h, out_w, OUT_NHW_ALIGN))
            if stride != [1, 1, 2, 2]:
                raise GKException("Stride H and W should be [2, 2] but got [{}, {}]".format(stride[2], stride[3]))

        self.shape_0_pad = [n0, h0, w0, c0]
        self.shape_1_pad = [n1, h1, w1, c1]

    def _expand(self, graph_builder):
        input_0 = self.inputs[0]
        input_1 = self.inputs[1]
        n0, _, _, c0 = input_0.shape
        n1, _, _, c1 = input_1.shape
        n0_p, h0_p, w0_p, c0_p = self.shape_0_pad
        n1_p, _, _, c1_p = self.shape_1_pad

        pad_value = 0
        # input0 pad
        input_0_pad_before = [0, 0, 0, 0]
        input_0_pad_after = [0, 0, 0, 0]
        if self.has_pad:
            pad_list = self.attrs['pad_list']
            input_0_pad_before = [0, pad_list[0], pad_list[2], 0]
            input_0_pad_after = [0, pad_list[1], pad_list[3], 0]
        input_0_pad_after[0] = n0_p - n0
        input_0_pad_after[3] = c0_p - c0
        if input_0_pad_before != [0, 0, 0, 0] or input_0_pad_after != [0, 0, 0, 0]:
            input_0 = graph_builder.emit('PadAkg', [input_0], attrs={'head': input_0_pad_before,
                                                                     'tail': input_0_pad_after,
                                                                     'pad_val': pad_value})
        # input1 pad
        input_1_pad_after = [n1_p - n1, 0, 0, c1_p - c1]
        if input_1_pad_after != [0, 0, 0, 0]:
            input_1 = graph_builder.emit('PadAkg', [input_1], attrs={'head': [0, 0, 0, 0],
                                                                     'tail': input_1_pad_after,
                                                                     'pad_val': pad_value})
        if self.can_optimize_to_matmul:
            a = graph_builder.emit('Reshape', [input_0], attrs={'shape': [self.m, self.k]})
            b = graph_builder.emit('Reshape', [input_1], attrs={'shape': [self.n, self.k]})
            c = graph_builder.emit('MatMul', [a, b], attrs={'transpose_a': False,
                                                            'transpose_b': True,
                                                            'dst_type': self.dst_type})
            result = graph_builder.emit('Reshape', [c], attrs={'shape': [n0_p, h0_p, w0_p, n1_p],
                                                               'format': self.dst_format})
        else:
            attrs = self.attrs
            attrs['pad_list'] = [0, 0, 0, 0]
            attrs['dst_type'] = self.dst_type
            result = graph_builder.emit('Conv2D', [input_0, input_1], attrs=attrs)
        # unpad
        unpad_after = [input_0_pad_after[0], 0, 0, input_1_pad_after[0]]
        if unpad_after != [0, 0, 0, 0]:
            result = graph_builder.emit('UnPadAkg', [result], attrs={'tail': unpad_after})

        return result
