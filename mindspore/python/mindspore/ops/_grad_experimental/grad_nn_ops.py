# Copyright 2021-2022 Huawei Technologies Co., Ltd
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


"""Define the grad rules of neural network related operations."""
from __future__ import absolute_import

from mindspore.ops._grad.grad_base import bprop_getters
from mindspore.ops import operations as P
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.operations.nn_ops import FractionalMaxPool
from mindspore.ops.operations._grad_ops import FractionalMaxPoolGrad
from mindspore.ops.operations.nn_ops import FractionalMaxPool3DWithFixedKsize
from mindspore.ops.operations._grad_ops import FractionalMaxPool3DGradWithFixedKsize
from mindspore.ops.operations.nn_ops import AvgPoolV1
from mindspore.ops.operations._grad_ops import AvgPoolGradV1
from mindspore.ops.operations.nn_ops import MaxPoolWithArgmaxV2
from mindspore.ops.operations.nn_ops import FractionalMaxPoolWithFixedKsize
from mindspore.ops.operations._grad_ops import FractionalMaxPoolGradWithFixedKsize
from mindspore.ops.operations.nn_ops import GLU
from mindspore.ops.operations.nn_ops import AdaptiveMaxPool3D


@bprop_getters.register(FractionalMaxPool)
def get_bprop_fractional_max_pool(self):
    """Grad definition for `FractionalMaxPool` operation."""
    fractional_max_pool_grad = FractionalMaxPoolGrad(self.overlapping)

    def bprop(x, out, dout):
        dx = fractional_max_pool_grad(x, out[0], dout[0], out[1], out[2])
        return (dx,)

    return bprop


@bprop_getters.register(FractionalMaxPool3DWithFixedKsize)
def get_bprop_fractional_max_pool3d_with_fixed_ksize(self):
    """Grad definition for `FractionalMaxPool3DWithFixedKsize` operation."""
    fractional_max_pool3d_grad_with_fixed_ksize = FractionalMaxPool3DGradWithFixedKsize(data_format=self.data_format)

    def bprop(x, random_samples, out, dout):
        dx = fractional_max_pool3d_grad_with_fixed_ksize(x, dout[0], out[1])
        return (dx, zeros_like(random_samples))

    return bprop


@bprop_getters.register(AdaptiveMaxPool3D)
def get_bprop_adaptive_max_pool_3d(self):
    """Grad definition for `AdaptiveMaxPool3D` operation."""
    grad = G.AdaptiveMaxPool3DGrad()

    def bprop(x, output_size, out, dout):
        dx = grad(dout[0], x, out[1])
        return (dx, P.ZerosLike()(output_size))

    return bprop


@bprop_getters.register(AvgPoolV1)
def get_bprop_avg_pool_v1_grad(self):
    """Grad definition for `AvgPoolV1` operation."""
    avgpool_grad_v1 = AvgPoolGradV1(
        kernel_size=self.kernel_size,
        strides=self.strides,
        pad_mode=self.pad_mode,
        data_format=self.format)
    to_arr = P.TupleToArray()
    get_shape = P.Shape()

    def bprop(x, out, dout):
        orig_input_shape = to_arr(get_shape(x))
        dx = avgpool_grad_v1(orig_input_shape, dout)
        return (dx,)

    return bprop


@bprop_getters.register(GLU)
def get_bprop_glu(self):
    """Grad definition for `Glu` operation."""
    input_grad = G.GluGrad(self.axis)

    def bprop(x, out, dout):
        dx = input_grad(dout, x)
        return (dx,)

    return bprop


@bprop_getters.register(MaxPoolWithArgmaxV2)
def get_bprop_maxpoolwithargmaxv2(self):
    """Grad definition for `MaxPoolWithArgmaxV2` operation."""
    maxpoolwithargmaxv2_grad = G.MaxPoolGradWithArgmaxV2(
        kernel_size=self.kernel_size,
        strides=self.strides,
        pads=self.pads,
        dilation=self.dilation,
        ceil_mode=self.ceil_mode,
        argmax_type=self.argmax_type)

    def bprop(x, out, dout):
        dx = maxpoolwithargmaxv2_grad(x, dout[0], out[1])
        return (dx,)

    return bprop


@bprop_getters.register(FractionalMaxPoolWithFixedKsize)
def get_bprop_fractional_max_pool_with_fixed_ksize(self):
    """Grad definition for 'FractionalMaxPoolWithFixedKsize' operation."""
    input_grad = FractionalMaxPoolGradWithFixedKsize(data_format=self.data_format)

    def bprop(x, random_samples, out, dout):
        dx = input_grad(x, dout[0], out[1])
        return (dx, zeros_like(random_samples))

    return bprop
