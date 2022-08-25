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
""" ops_test """
import numpy as np

import mindspore
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops.vm_impl_registry import vm_impl_registry as vm_impl_getters


def im2col(img, filter_h, filter_w, stride=1, pad=0, dilation=1):
    """Rearranges an image to row vector"""
    if isinstance(pad, int):
        pad_top = pad
        pad_bottom = pad
        pad_left = pad
        pad_right = pad
    elif isinstance(pad, tuple) and len(pad) == 4:
        pad_top, pad_bottom, pad_left, pad_right = pad
    else:
        raise ValueError(f"The \'pad\' should be an int number or "
                         f"a tuple of two or four int numbers, but got {pad}")

    batch_num, channel, height, width = img.shape
    out_h = (height + pad_top + pad_bottom - filter_h - (filter_h - 1) * (dilation[2] - 1)) // stride[2] + 1
    out_w = (width + pad_left + pad_right - filter_w - (filter_w - 1) * (dilation[3] - 1)) // stride[3] + 1

    img = np.pad(img, [(0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)], 'constant')
    col = np.zeros((batch_num, channel, filter_h, filter_w, out_h, out_w)).astype(img.dtype)

    for y in range(filter_h):
        y_max = y + stride[2] * out_h
        for x in range(filter_w):
            x_max = x + stride[2] * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride[2], x:x_max:stride[2]]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_num * out_h * out_w, -1)
    return col


# pylint: disable=unused-argument
def conv2d(x, weight, bias=None, stride=1, pad=0, dilation=1):
    """Convolution 2D"""
    if isinstance(pad, int):
        pad_top = pad
        pad_bottom = pad
        pad_left = pad
        pad_right = pad
    elif isinstance(pad, tuple) and len(pad) == 4:
        pad_top, pad_bottom, pad_left, pad_right = pad
    else:
        raise ValueError(f"The \'pad\' should be an int number or "
                         f"a tuple of two or four int numbers, but got {pad}")

    batch_num, _, x_h, x_w = x.shape
    filter_num, _, filter_h, filter_w = weight.shape
    out_h = 1 + int((x_h + pad_top + pad_bottom - filter_h - (filter_h - 1) * (dilation[2] - 1)) / stride[2])
    out_w = 1 + int((x_w + pad_left + pad_right - filter_w - (filter_w - 1) * (dilation[3] - 1)) / stride[3])
    col = im2col(x, filter_h, filter_w, stride, pad, dilation)
    col_w = np.reshape(weight, (filter_num, -1)).T
    out = np.dot(col, col_w)
    out = out.reshape((batch_num, out_h, out_w, -1)).transpose(0, 3, 1, 2)
    if bias is not None:
        out += bias
    return out


@vm_impl_getters.register(P.Conv2D)
def vm_impl_conv2d(self):
    """Generate vm_impl function for Conv2D"""

    def vm_impl(x, w):
        x = x.asnumpy()
        weight = w.asnumpy()
        bias = None
        out = conv2d(x, weight, bias, self.stride, self.pad, self.dilation)
        return Tensor(out)

    return vm_impl


matmul = P.MatMul()
tensor1 = Tensor(np.ones([1, 3]), dtype=mindspore.float32)
tensor2 = Tensor(np.ones([3, 1]), dtype=mindspore.float32)
conv2d_prim = P.Conv2D(64, (3, 3), pad_mode='pad', pad=1, stride=2)
