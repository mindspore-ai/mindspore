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
# ============================================================================

"""array_ops"""

from ...common import dtype as mstype
from .._grad.grad_math_ops import binop_grad_common
from .._grad.grad_base import bprop_getters
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from .. import functional as F
from .. import operations as P


@bprop_getters.register(P.MaskedFill)
def get_bprop_masked_select(self):
    """Generate bprop for MaskedFill"""
    mul_op = P.Mul()
    sum_op = P.ReduceSum()
    is_instance_op = P.IsInstance()

    def bprop(input_data, mask, value, out, dout):
        mask = F.cast(mask, mstype.float32)
        dinput = mul_op(dout, (1 - mask))
        dvalue = mul_op(dout, mask)
        dinput, dvalue = binop_grad_common(input_data, mask, dinput, dvalue)
        dvalue = sum_op(dvalue)
        dinput = F.cast(dinput, F.dtype(input_data))
        if is_instance_op(value, mstype.number) is True:
            dvalue = 0
        else:
            dvalue = F.cast(dvalue, F.dtype(value))
        return dinput, zeros_like(mask), dvalue

    return bprop


@bprop_getters.register(P.TensorScatterSub)
def get_bprop_tensor_scatter_sub(self):
    """Generate bprop for TensorScatterSub"""
    gather_nd = P.GatherNd()
    neg = P.Neg()

    def bprop(x, indices, update, out, dout):
        update_grad = neg(gather_nd(dout, indices))
        return dout, zeros_like(indices), update_grad

    return bprop


def tensor_scatter_possible_replacement(x, indices, updates, out, dout):
    """bpropr for any TensorScatter* op that possibly replaces values in the input tensor"""
    gather_nd = P.GatherNd()
    scatter_nd = P.ScatterNd()
    equal = P.Equal()
    shape = P.Shape()

    x_indicators = F.cast(equal(x, out), mstype.int32)
    possibly_updated = gather_nd(out, indices)
    out_indicators = F.cast(equal(updates, possibly_updated), mstype.int32)
    scattered_out_indicators = scatter_nd(indices, out_indicators, shape(x))
    indicators = x_indicators + scattered_out_indicators
    dx = dout * F.cast(x_indicators, F.dtype(dout)) / F.cast(indicators, F.dtype(dout))
    dupdates = gather_nd(dout / F.cast(indicators, F.dtype(dout)), indices) * F.cast(out_indicators, F.dtype(dout))

    return F.cast(dx, F.dtype(x)), zeros_like(indices), F.cast(dupdates, F.dtype(updates))


@bprop_getters.register(P.TensorScatterMax)
def get_bprop_tensor_scatter_max(self):
    """Generate bprop for TensorScatterMax"""
    def bprop(x, indices, updates, out, dout):
        return tensor_scatter_possible_replacement(x, indices, updates, out, dout)

    return bprop


@bprop_getters.register(P.TensorScatterMin)
def get_bprop_tensor_scatter_min(self):
    """Generate bprop for TensorScatterMin"""
    def bprop(x, indices, updates, out, dout):
        return tensor_scatter_possible_replacement(x, indices, updates, out, dout)

    return bprop


@bprop_getters.register(P.SplitV)
def get_bprop_split_v(self):
    """Generate bprop for SplitV"""
    split_dim = self.split_dim
    concat_op = P.Concat(split_dim)

    def bprop(x_input, output, dout):
        dx = concat_op(dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.ExtractVolumePatches)
def get_bprop_extract_volume_patches(self):
    """Generate bprop for ExtractVolumePatches"""
    extract_volume_patches = P.ExtractVolumePatches(kernel_size=self.kernel_size,
                                                    strides=self.strides, padding=self.padding)
    concat = P.Concat(axis=-1)
    expend_dims = P.ExpandDims()
    scatter_nd = P.ScatterNd()
    slice_op = P.Slice()
    fill = P.Fill()
    dtype = P.DType()
    cast = P.Cast()
    matmul = P.MatMul()
    _, _, ksize_d, ksize_h, ksize_w = self.kernel_size

    def bprop(x, out, dout):
        x_shape = P.Shape()(x)
        x_n, x_c, x_d, x_h, x_w = x_shape
        x_indices_num = 1 + x_d * x_h * x_w
        x_idx = cast(F.tuple_to_array(range(1, x_indices_num)), mstype.float16)
        x_idx = P.Reshape()(x_idx, (1, 1, x_d, x_h, x_w))
        x_idx_patched = extract_volume_patches(x_idx)
        x_idx_patched = P.Transpose()(x_idx_patched, (0, 2, 3, 4, 1))
        x_idx_patched = cast(x_idx_patched, mstype.int32)

        out_shape = P.Shape()(out)
        _, _, out_d, out_h, out_w = out_shape
        out_indices_num = out_d * out_h * out_w * ksize_d * ksize_h * ksize_w
        out_idx = F.tuple_to_array(range(0, out_indices_num))
        out_idx = P.Reshape()(out_idx, (1, out_d, out_h, out_w, ksize_d * ksize_h * ksize_w))

        idx_tensor = concat((expend_dims(x_idx_patched, -1), expend_dims(out_idx, -1)))
        idx_map = P.Reshape()(idx_tensor, (-1, 2))
        sp_shape = (x_indices_num, out_indices_num)
        sp_mat_full = scatter_nd(idx_map, fill(dtype(dout), (out_indices_num,), 1), sp_shape)
        sp_tensor = slice_op(sp_mat_full, (1, 0), (x_indices_num - 1, out_indices_num))

        grad = P.Transpose()(dout, (0, 2, 3, 4, 1))
        grad = P.Reshape()(grad, (x_n, out_d, out_h, out_w, ksize_d,
                                  ksize_h, ksize_w, x_c))
        grad_expended = P.Transpose()(grad, (1, 2, 3, 4, 5, 6, 0, 7))
        grad_flat = P.Reshape()(grad_expended, (-1, x_n * x_c))

        jac = matmul(sp_tensor, grad_flat)
        dx = P.Reshape()(jac, (x_d, x_h, x_w, x_n, x_c))
        dx = P.Transpose()(dx, (3, 4, 0, 1, 2))
        return (dx,)

    return bprop
