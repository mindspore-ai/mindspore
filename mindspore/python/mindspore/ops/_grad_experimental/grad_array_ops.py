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

"""array_ops"""
from __future__ import absolute_import

from mindspore import Tensor
from mindspore.ops.primitive import constexpr
from mindspore.common import dtype as mstype
from mindspore.numpy.array_ops import where
from mindspore.ops._grad.grad_math_ops import binop_grad_common
from mindspore.ops._grad.grad_base import bprop_getters, dyn_rank, dyn_fill, dyn_ones, create_tensor_by_element
from mindspore.ops._grad.grad_base import convert_to_tensor
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.operations.array_ops import Tril
from mindspore.ops.operations.array_ops import MatrixDiagV3
from mindspore.ops.operations.array_ops import MatrixDiagPartV3
from mindspore.ops.operations.array_ops import ResizeNearestNeighborV2
from mindspore.ops.operations.array_ops import MatrixSetDiagV3
from mindspore.ops.operations.array_ops import Mvlgamma
from mindspore.ops.operations.array_ops import Triu
from mindspore.ops.operations.array_ops import IdentityN
from mindspore.ops.operations.array_ops import IndexFill
from mindspore.ops.operations.array_ops import IndexPut
from mindspore.ops.operations.array_ops import CheckNumerics
from mindspore.ops.operations.array_ops import ConjugateTranspose
from mindspore.ops.operations.array_ops import SegmentMax
from mindspore.ops.operations.array_ops import SegmentMin
from mindspore.ops.operations.array_ops import SegmentSum
from mindspore.ops.operations.array_ops import TensorScatterElements
from mindspore.ops.operations.array_ops import ScatterAddWithAxis
from mindspore.ops.operations.array_ops import Expand
from mindspore.ops.operations.array_ops import SegmentMean
from mindspore.ops.operations.array_ops import AffineGrid
from mindspore.ops.operations.array_ops import Im2Col
from mindspore.ops.operations.array_ops import Col2Im
from mindspore.ops.operations.array_ops import StridedSliceV2
from mindspore.ops.operations.array_ops import MaskedScatter
from mindspore.ops.operations.array_ops import MaskedSelect
from mindspore.ops.operations.array_ops import CountNonZero
from mindspore.ops.operations._grad_ops import StridedSliceV2Grad
from mindspore.ops.operations.random_ops import LogNormalReverse
from mindspore.ops.operations.random_ops import ParameterizedTruncatedNormal
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G
from mindspore import context


@bprop_getters.register(P.FillV2)
def get_bprop_fill_v2(self):
    """Generate bprop for FillV2"""
    sum_op = P.ReduceSum()
    cast_op = P.Cast()

    def bprop(shape, value, out, dout):
        dout_type = F.dtype(dout)
        type_list = [mstype.int8, mstype.int16, mstype.int32, mstype.int64,
                     mstype.uint8, mstype.uint16, mstype.uint32, mstype.uint64, mstype.float16]
        if dout_type in type_list:
            dout = cast_op(dout, mstype.float32)
        if dout_type == mstype.float64:
            dout = cast_op(dout, mstype.float32)
        dvalue = sum_op(dout)
        return zeros_like(shape), cast_op(dvalue, dout_type)

    return bprop


@bprop_getters.register(StridedSliceV2)
def get_bprop_strided_slice_v2(self):
    """Generate bprop for StridedSliceV2"""
    shape_op = P.Shape()
    dyn_shape_op = P.TensorShape()
    input_grad = StridedSliceV2Grad(self.begin_mask,
                                    self.end_mask,
                                    self.ellipsis_mask,
                                    self.new_axis_mask,
                                    self.shrink_axis_mask)

    def bprop(x, begin, end, strides, out, dout):
        x_shape = shape_op(x)
        if F.is_sequence_value_unknown(x_shape):
            x_shape = dyn_shape_op(x)
        dx = input_grad(x_shape, begin, end, strides, dout)
        dx_all = (dx, zeros_like(begin), zeros_like(end), zeros_like(strides))
        return dx_all

    return bprop


@constexpr
def _create_tensor(data, dtype):
    return Tensor(data, dtype=dtype)


def _segment_min_or_max_grad(segment_sum_op, input_x, segment_ids, output, dout):
    """Calculate the gradient of SegmentMax or SegmentMin"""
    gather = P.Gather()
    equal = P.Equal()
    cast = P.Cast()
    divide = P.Div()
    input_x_type = F.dtype(input_x)
    input_x = cast(input_x, mstype.float32)
    output = cast(output, mstype.float32)
    dout = cast(dout, mstype.float32)
    zeros = zeros_like(input_x)
    gathered_outputs = gather(output, segment_ids, 0)
    is_selected = equal(input_x, gathered_outputs)
    num_selected = segment_sum_op(cast(is_selected, F.dtype(dout)), segment_ids)
    weighted_grads = divide(dout, num_selected)
    gathered_grads = gather(weighted_grads, segment_ids, 0)
    return cast(where(is_selected, gathered_grads, zeros), input_x_type), zeros_like(segment_ids)


@bprop_getters.register(P.MaskedFill)
def get_bprop_masked_select(self):
    """Generate bprop for MaskedFill"""
    mul_op = P.Mul()
    sum_op = P.ReduceSum()
    is_instance_op = inner.IsInstance()

    def bprop(input_data, mask, value, out, dout):
        mask = F.cast(mask, mstype.float32)
        dinput = mul_op(dout, (1 - mask))
        dvalue = mul_op(dout, mask)
        dinput, dvalue = binop_grad_common(input_data, mask, dinput, dvalue)
        dvalue = sum_op(dvalue)
        dinput = F.cast(dinput, F.dtype(input_data))
        if is_instance_op(value, mstype.number):
            dvalue = 0
        else:
            dvalue = F.cast(dvalue, F.dtype(value))
        return dinput, zeros_like(mask), dvalue

    return bprop


@bprop_getters.register(MaskedScatter)
def get_bprop_masked_scatter(self):
    """Generate bprop for MaskedScatter"""
    sort_ = P.Sort(descending=True)
    masked_scatter = MaskedScatter()
    masked_fill = P.MaskedFill()
    masked_select = P.MaskedSelect()
    size = P.Size()
    zeros = P.Zeros()
    concat = P.Concat(axis=0)
    reshape = P.Reshape()
    shape = P.Shape()

    def bprop(x, mask, updates, out, dout):
        dx = masked_fill(F.cast(dout, mstype.float32), mask, 0.0)
        mask_selected = masked_select(F.cast(dout, mstype.float32), mask)
        mask_broad = mask
        if shape(mask) != shape(x):
            broad_cast = P.BroadcastTo(shape(x))
            mask_broad = broad_cast(mask)
        mask_broad_vec = mask_broad.reshape(-1)
        mask_sorted = F.cast(sort_(F.cast(mask_broad_vec, mstype.float32))[0], F.dtype(mask))
        diff_num = size(updates) - size(mask_broad)
        if diff_num > 0:
            zeros_pad = zeros(diff_num, F.dtype(mask))
            mask_sorted = concat((mask_sorted, zeros_pad))
        zeros_tensor = zeros(size(updates), mstype.float32)
        dupdates = masked_scatter(zeros_tensor, mask_sorted, mask_selected)
        if shape(updates) != ():
            dupdates = reshape(dupdates, shape(updates))
        else:
            zeros_tensor = zeros(shape(updates), mstype.float32)
            dupdates = masked_scatter(zeros_tensor, mask, mask_selected)
        return F.cast(dx, F.dtype(x)), zeros_like(mask), F.cast(dupdates, F.dtype(updates))

    return bprop


@bprop_getters.register(CountNonZero)
def get_bprop_countnonzero(self):
    """Grad definition for CountNonZero"""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(Mvlgamma)
def get_bprop_mvlgamma(self):
    """Grad definition for Mvlgamma"""
    input_grad = G.MvlgammaGrad(p=self.p)

    def bprop(x, out, dout):
        dx = input_grad(dout, x)
        return (dx,)

    return bprop


@bprop_getters.register(P.TensorScatterDiv)
def get_bprop_tensor_scatter_div(self):
    """Generate bprop for TensorScatterDiv"""
    gather_nd = P.GatherNd()
    tensor_scatter_div = P.TensorScatterDiv()
    neg = P.Neg()
    div = P.Div()
    mul = P.Mul()

    def bprop(x, indices, update, out, dout):
        # (input)' / update
        in_grad = tensor_scatter_div(dout, indices, update)

        # - (input * (update)') / (update * update)
        gather_update = gather_nd(dout, indices)
        gather_x = gather_nd(x, indices)
        mul_result = mul(update, update)
        neg_result = neg(mul_result)
        update_grad = gather_update * div(gather_x, neg_result)

        return in_grad, zeros_like(indices), update_grad

    return bprop


@bprop_getters.register(IndexFill)
def get_bprop_index_fill(self):
    """Generate bprop for IndexFill"""
    gather = P.Gather()
    index_fill = IndexFill()
    shape = P.Shape()

    def bprop(x, dim, indices, value, out, dout):
        zero_value = zeros_like(value)
        x_grad = index_fill(dout, dim, indices, zero_value)
        if F.is_sequence_value_unknown(shape(x)):
            if dyn_rank(x) == 0:
                value_grad = dout
            else:
                value_grad = gather(dout, indices, dim).sum()
        else:
            if shape(x) == ():
                value_grad = dout
            else:
                value_grad = gather(dout, indices, dim).sum()
        result = (x_grad, zeros_like(dim), zeros_like(indices), value_grad)
        return result

    return bprop


@bprop_getters.register(IndexPut)
def get_bprop_index_put(self):
    """Generate bprop for IndexPut"""
    gather_nd = P.GatherNd()
    stack = P.Stack()
    tile = P.Tile()
    masked_select = MaskedSelect()
    masked_scatter = MaskedScatter()
    accumulate_grad = self.accumulate
    index_put = IndexPut(accumulate=accumulate_grad)
    is_ascend = context.get_context("device_target") == 'Ascend'

    # Negative value are not supported for GatherNd indices when Ascend, so convert it to positive value.
    def convert_idx_positive(indices_i, x_shape_i):
        mask = indices_i < 0
        idx_pos = masked_select(indices_i + x_shape_i, mask)
        idx = masked_scatter(indices_i, mask, idx_pos)
        return idx

    def bprop(x1, x2, indices, out, dout):
        maxsize = max(x.shape[0] for x in indices)
        indices_ms = [tile(x, (maxsize,)) if x.shape[0] == 1 else x for x in indices]
        if is_ascend:
            indices_ms = [convert_idx_positive(indices_ms[i], x1.shape[i]) for i in range(len(indices_ms))]
        indices_grad = stack(indices_ms).T
        values_grad = gather_nd(dout, indices_grad)
        if x2.shape[0] == 1:
            values_grad = values_grad.sum().reshape(1)
        if values_grad.shape != x2.shape and len(indices) < len(x1.shape):
            _, values_grad = binop_grad_common(x1, x2, dout, values_grad)
        if accumulate_grad == 0:
            dout = index_put(dout, zeros_like(x2), indices)
        return dout, values_grad, [zeros_like(item) for item in indices]

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


@bprop_getters.register(P.TensorScatterMul)
def get_bprop_tensor_scatter_mul(self):
    """Generate bprop for TensorScatterMul"""
    gather_nd = P.GatherNd()
    mul_func = P.TensorScatterMul()

    def bprop(x, indices, update, out, dout):
        gather_update = gather_nd(dout, indices)
        gather_x = gather_nd(x, indices)
        dx = mul_func(dout, indices, update)
        d_update = gather_x * gather_update
        return dx, zeros_like(indices), d_update

    return bprop


@bprop_getters.register(MatrixDiagV3)
def get_bprop_matrix_diag_v3(self):
    """Generate bprop for MatrixDiagV3"""
    align = self.align
    matrix_diag_part_v3 = MatrixDiagPartV3(align=align)
    zeros = P.Zeros()

    def bprop(x, k, num_rows, num_cols, padding_value, out, dout):
        result = (matrix_diag_part_v3(dout, k, zeros((), dout.dtype)), zeros_like(k), zeros_like(num_rows),
                  zeros_like(num_cols), zeros_like(padding_value))
        return result

    return bprop


@bprop_getters.register(MatrixDiagPartV3)
def get_bprop_matrix_diag_part_v3(self):
    """Generate bprop for MatrixDiagPartV3"""
    align = self.align
    matrix_diag_v3 = MatrixDiagV3(align=align)
    matrix_set_diag_v3 = MatrixSetDiagV3(align=align)
    zeros = P.Zeros()

    def bprop(x, k, padding_value, out, dout):
        shape_this = P.Shape()(x)[-2:]
        if not F.is_sequence_value_unknown(shape_this):
            row = shape_this[0]
            col = shape_this[1]
            result = (matrix_diag_v3(dout, k, Tensor(row, dtype=mstype.int32), Tensor(col, dtype=mstype.int32),
                                     zeros((), dout.dtype)), zeros_like(k), zeros_like(padding_value))
        else:
            result = (matrix_set_diag_v3(zeros_like(x), dout, k), zeros_like(k), zeros_like(padding_value))
        return result

    return bprop


@bprop_getters.register(MatrixSetDiagV3)
def get_bprop_matrix_set_diag_v3(self):
    """Generate bprop for MatrixSetDiagV3"""
    align = self.align
    matrix_diag_part_v3 = MatrixDiagPartV3(align=align)
    matrix_set_diag_v3 = MatrixSetDiagV3(align=align)
    zeros = P.Zeros()

    def bprop(x, diagonal, k, out, dout):
        diagonal_cal = matrix_diag_part_v3(dout, k, zeros((), dout.dtype))

        diagonal_shape = P.Shape()(diagonal)
        if F.is_sequence_value_unknown(diagonal_shape):
            diagonal = F.cast(diagonal, dout.dtype)
            x_cal = matrix_set_diag_v3(dout, zeros_like(diagonal), k)
        else:
            x_cal = matrix_set_diag_v3(dout, zeros(diagonal_shape, dout.dtype), k)

        return x_cal, diagonal_cal, zeros_like(k)

    return bprop


def tensor_scatter_possible_replacement(x, indices, updates, out, dout):
    """bpropr for any TensorScatter* op that possibly replaces values in the input tensor"""
    gather_nd = P.GatherNd()
    scatter_nd = P.ScatterNd()
    equal = P.Equal()
    shape = P.Shape()
    dyn_shape_op = P.TensorShape()

    x_indicators = F.cast(equal(x, out), mstype.int32)
    possibly_updated = gather_nd(out, indices)
    out_indicators = F.cast(equal(updates, possibly_updated), mstype.int32)
    input_shape = shape(x)
    if F.is_sequence_value_unknown(input_shape):
        input_shape = dyn_shape_op(x)

    scattered_out_indicators = scatter_nd(indices, out_indicators, input_shape)
    indicators = x_indicators + scattered_out_indicators
    dx = dout * F.cast(x_indicators, F.dtype(dout)) / F.cast(indicators, F.dtype(dout))
    dupdates = gather_nd(dout / F.cast(indicators, F.dtype(dout)), indices) * F.cast(out_indicators, F.dtype(dout))

    return F.cast(dx, F.dtype(x)), zeros_like(indices), F.cast(dupdates, F.dtype(updates))


@bprop_getters.register(LogNormalReverse)
def get_bprop_log_normal_reverse(self):
    """Grad definition for `LogNormalReverse` operation."""
    def bprop(input_data, out, dout):
        return (zeros_like(input_data),)

    return bprop


@bprop_getters.register(ParameterizedTruncatedNormal)
def get_bprop_parameterized_truncated_normal(self):
    """Grad definition for `ParameterizedTruncatedNormal` operation."""
    def bprop(shape, mean, stdevs, min_val, max_val, out, dout):
        return (zeros_like(shape), zeros_like(mean), zeros_like(stdevs), zeros_like(min_val), zeros_like(max_val))

    return bprop


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


@bprop_getters.register(P.Coalesce)
def get_bprop_coalesce(self):
    """Grad definition for `Coalesce` operation."""

    def bprop(x_indices, x_values, x_shape, out, dout):
        return dout

    return bprop


@bprop_getters.register(ConjugateTranspose)
def get_bprop_conjugate_transpose(self):
    """Generate bprop for ConjugateTranspose"""
    conjugate_transpose = ConjugateTranspose()
    invert_permutation = P.InvertPermutation()

    def bprop(x, perm, out, dout):
        return conjugate_transpose(dout, invert_permutation(perm)), zeros_like(perm)

    return bprop


@bprop_getters.register(Triu)
def get_bprop_triu(self):
    """Grad definition for 'Triu' operation"""
    diagonal = self.diagonal
    triu = Triu(diagonal)

    def bprop(x, out, dout):
        dx = triu(dout)
        return (dx,)

    return bprop


@bprop_getters.register(CheckNumerics)
def get_bprop_check_numerics(self):
    """Generate bprop for CheckNumerics"""
    check_numerics = CheckNumerics()

    def bprop(x_input, out, dout):
        return (check_numerics(dout),)

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


@bprop_getters.register(IdentityN)
def get_bprop_identity_n(self):
    """Generate bprop for IdentityN"""

    def bprop(x, out, dout):
        return (dout,)

    return bprop


@bprop_getters.register(ResizeNearestNeighborV2)
def get_bprop_resize_nearest_neighbor_v2(self):
    """Generate bprop for ResizeNearestNeighborV2"""
    align_corners = self.align_corners
    half_pixel_centers = self.half_pixel_centers
    data_format = self.data_format
    grad_op = G.ResizeNearestNeighborV2Grad(align_corners, half_pixel_centers, data_format)

    def bprop(x, size, output, dout):
        x_shape = P.Shape()(x)
        if F.is_sequence_value_unknown(x_shape):
            x_shape = P.TensorShape()(x)
        grad_in_size = x_shape[1:3]
        if data_format == 'NCHW':
            grad_in_size = x_shape[2:4]

        if F.is_sequence_value_unknown(P.Shape()(x)):
            dx = grad_op(dout, grad_in_size)
            return dx, zeros_like(grad_in_size)

        dx = grad_op(dout, _create_tensor(grad_in_size, mstype.int32))
        return dx, zeros_like(grad_in_size)

    return bprop


@bprop_getters.register(Col2Im)
def get_bprop_col2im(self):
    """Generate bprop for Col2Im"""
    ksizes = self.kernel_size
    dilations = self.dilation
    strides = self.stride
    pads = self.padding
    im2col = Im2Col(ksizes=ksizes, dilations=dilations, strides=strides, padding_mode="CALCULATED", pads=pads)

    def bprop(x, output_size, out, dout):
        dx = im2col(dout)
        return dx, zeros_like(output_size)

    return bprop


@bprop_getters.register(Im2Col)
def get_bprop_im2col(self):
    """
    Generate bprop for Im2Col

    Im2Col, corresponding to torch's UnFold operator.
    The Unfold operator has no `padding_mode` attribute,
    and it's implementation corresponds to the mindspore
    implementation when `padding_mode=CALCULATED` .
    So, currently the bprop function of Im2Col only supports
    the CALCULATED mode.
    """
    kernel_size = self.ksizes
    dilation = self.dilations
    stride = self.strides
    padding = self.pads
    col2im = Col2Im(kernel_size=kernel_size,
                    dilation=dilation,
                    stride=stride,
                    padding=padding)

    def bprop(x, out, dout):
        x_shape = Tensor(x.shape, dtype=mstype.int32)
        dx = col2im(dout, x_shape)
        return dx

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
    range_ = P.Range()
    dyn_shape_op = P.TensorShape()
    ones_like = P.OnesLike()

    def _dyn_extract_volume_patches(x, out, dout):
        x_shape = dyn_shape_op(x)
        out_shape = dyn_shape_op(out)
        x_n, x_c, x_d, x_h, x_w = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]
        x_indices_num = 1 + x_d * x_h * x_w
        x_idx = range_(cast(1, mstype.float32), cast(x_indices_num, mstype.float32), cast(1, mstype.float32))
        x_idx = cast(x_idx, mstype.float16)
        x_idx = P.Reshape()(x_idx, create_tensor_by_element((1, 1, x_d, x_h, x_w)))
        x_idx_patched = extract_volume_patches(x_idx)
        x_idx_patched = P.Transpose()(x_idx_patched, (0, 2, 3, 4, 1))
        x_idx_patched = cast(x_idx_patched, mstype.int32)

        out_d, out_h, out_w = out_shape[2], out_shape[3], out_shape[4]
        out_indices_num = out_d * out_h * out_w * ksize_d * ksize_h * ksize_w
        out_idx_ori = range_(cast(0, mstype.int32), cast(out_indices_num, mstype.int32), cast(1, mstype.int32))
        out_idx = P.Reshape()(out_idx_ori,
                              create_tensor_by_element((1, out_d, out_h, out_w, ksize_d * ksize_h * ksize_w)))

        idx_tensor = concat((expend_dims(x_idx_patched, -1), expend_dims(out_idx, -1)))
        idx_map = P.Reshape()(idx_tensor, (-1, 2))
        sp_shape = create_tensor_by_element((x_indices_num, out_indices_num))
        update = cast(ones_like(out_idx_ori), dtype(dout))
        sp_mat_full = scatter_nd(idx_map, update, sp_shape)
        begin = create_tensor_by_element((1, 0))
        size = create_tensor_by_element((x_indices_num - 1, out_indices_num))
        sp_tensor = slice_op(sp_mat_full, begin, size)

        grad = P.Transpose()(dout, (0, 2, 3, 4, 1))
        grad = P.Reshape()(grad, create_tensor_by_element((x_n, out_d, out_h, out_w, ksize_d,
                                                           ksize_h, ksize_w, x_c)))
        grad_expended = P.Transpose()(grad, (1, 2, 3, 4, 5, 6, 0, 7))
        grad_flat = P.Reshape()(grad_expended,
                                create_tensor_by_element((out_d * out_h * out_w * ksize_d * ksize_h * ksize_w,
                                                          x_n * x_c)))
        jac = matmul(sp_tensor, grad_flat)
        dx = P.Reshape()(jac, create_tensor_by_element((x_d, x_h, x_w, x_n, x_c)))
        dx = P.Transpose()(dx, (3, 4, 0, 1, 2))
        return (dx,)

    def bprop(x, out, dout):
        x_shape = P.Shape()(x)
        out_shape = P.Shape()(out)
        if F.is_sequence_value_unknown(x_shape) or F.is_sequence_value_unknown(out_shape):
            return _dyn_extract_volume_patches(x, out, dout)
        x_n, x_c, x_d, x_h, x_w = x_shape
        x_indices_num = 1 + x_d * x_h * x_w
        x_idx = cast(F.tuple_to_array(range(1, x_indices_num)), mstype.float16)
        x_idx = P.Reshape()(x_idx, (1, 1, x_d, x_h, x_w))
        x_idx_patched = extract_volume_patches(x_idx)
        x_idx_patched = P.Transpose()(x_idx_patched, (0, 2, 3, 4, 1))
        x_idx_patched = cast(x_idx_patched, mstype.int32)

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


@bprop_getters.register(Tril)
def get_bprop_tril(self):
    """Grad definition for 'Tril' operation"""
    diagonal = self.diagonal
    tril = Tril(diagonal)

    def bprop(x, out, dout):
        dx = tril(dout)
        return (dx,)

    return bprop


@bprop_getters.register(SegmentSum)
def get_bprop_segment_sum(self):
    """Generate bprop for SegmentSum"""
    gather = P.Gather()
    cast = P.Cast()

    def bprop(input_x, segment_ids, output, dout):
        dout_type = F.dtype(dout)
        type_list = [mstype.int8, mstype.int16, mstype.int64, mstype.uint8, mstype.uint16, mstype.uint32, mstype.uint64]
        if dout_type in type_list:
            dout = cast(dout, mstype.int32)
        if dout_type == mstype.float64:
            dout = cast(dout, mstype.float32)
        return cast(gather(dout, segment_ids, 0), dout_type), zeros_like(segment_ids)

    return bprop


@bprop_getters.register(AffineGrid)
def get_bprop_affinegrid(self):
    """Generate bprop for AffineGrid"""

    align_corners = self.align_corners
    input_grad = G.AffineGridGrad(align_corners)
    ones = P.Ones()
    transpose = P.Transpose()
    concat = P.Concat(1)
    concat0 = P.Concat(0)
    tile = P.Tile()
    div = P.Div()
    reshape = P.Reshape()
    linspace = P.LinSpace()
    batmatmul = P.BatchMatMul()
    expend_dims = P.ExpandDims()
    dyn_shape = P.TensorShape()
    reducesum = P.ReduceSum(keep_dims=False)

    def get_linspace(num):
        start = Tensor(-1, mstype.float32)
        stop = Tensor(1, mstype.float32)
        lins_tensor = Tensor([0], dtype=mstype.float32)
        if num != 1:
            lins_tensor = linspace(start, stop, num)
        return lins_tensor

    def dyn_bprop_five(theta, output_size, out, dout, len_output_size):
        perm1 = (1, 0)
        perm2 = (0, 2, 1)
        one_tensor = create_tensor_by_element((1,), mstype.int32)
        n_value = reducesum(output_size[0])
        d_value = reducesum(output_size[2])
        h_value = reducesum(output_size[3])
        w_value = reducesum(output_size[len_output_size - 1])
        vecx = get_linspace(w_value.astype("int64"))
        vecy = get_linspace(h_value.astype("int64"))
        vecz = get_linspace(d_value.astype("int64"))
        if align_corners is False:
            vecx = div(vecx * (w_value - 1), w_value)
            vecy = div(vecy * (h_value - 1), h_value)
            vecz = div(vecz * (d_value - 1), d_value)
        out = vecx
        if h_value * d_value != 1:
            multiples = concat0((expend_dims(h_value * d_value, -1), one_tensor))
            out = tile(vecx, multiples)
        hwd_value = h_value * w_value * d_value
        hwd_shape = concat0((expend_dims(hwd_value, -1), one_tensor))
        one = reshape(out, hwd_shape)
        if w_value == 1:
            out = expend_dims(vecy, 0)
        elif w_value != 1:
            multiples = concat0((expend_dims(w_value, -1), one_tensor))
            out = tile(vecy, multiples)
        out = transpose(out, perm1)
        if d_value != 1:
            multiples = concat0((expend_dims(d_value, -1), one_tensor))
            out = tile(out, multiples)
        two = reshape(out, hwd_shape)
        out = expend_dims(vecz, 0)
        if w_value * h_value != 1:
            multiples = concat0((expend_dims(w_value * h_value, -1), one_tensor))
            out = tile(vecz, multiples)
        out = transpose(out, perm1)
        four = dyn_ones(hwd_shape, mstype.float32)
        output = concat((one, two, reshape(out, hwd_shape), four))
        output = transpose(output, perm1)
        if n_value != 1:
            multiples = concat0((expend_dims(n_value, -1), one_tensor))
            output = tile(output, multiples)
        three_tensor = create_tensor_by_element((3,), mstype.int32)
        four_tensor = create_tensor_by_element((4,), mstype.int32)
        output_shape = concat0((expend_dims(n_value, -1), four_tensor, expend_dims(hwd_value, -1)))
        dout_shape = concat0((expend_dims(n_value, -1), expend_dims(hwd_value, -1), three_tensor))
        dtheta = batmatmul(reshape(output, output_shape), reshape(dout, dout_shape).astype("float32"))
        return transpose(dtheta, perm2), four

    def dyn_bprop_four(theta, output_size, out, dout):
        perm1 = (1, 0)
        perm2 = (0, 2, 1)
        one_tensor = create_tensor_by_element((1,), mstype.int32)
        n_value = reducesum(output_size[0])
        h_value = reducesum(output_size[2])
        w_value = reducesum(output_size[3])
        vecx = get_linspace(w_value.astype("int64"))
        vecy = get_linspace(h_value.astype("int64"))
        if align_corners is False:
            vecx = div(vecx * (w_value - 1), w_value)
            vecy = div(vecy * (h_value - 1), h_value)
        out = vecx
        if h_value != 1:
            multiples = concat0((expend_dims(h_value, -1), one_tensor))
            out = tile(vecx, multiples)
        hw_shape = concat0((expend_dims(h_value * w_value, -1), one_tensor))
        one = reshape(out, hw_shape)
        if w_value == 1:
            out = expend_dims(vecy, 0)
        elif w_value != 1:
            multiples = concat0((expend_dims(w_value, -1), one_tensor))
            out = tile(vecy, multiples)
        out = transpose(out, perm1)
        two = reshape(out, hw_shape)
        tre = dyn_ones(hw_shape, mstype.float32)
        output = concat((one, two, tre))
        multiples = concat0((expend_dims(n_value, -1), one_tensor))
        output = transpose(output, perm1)
        output = tile(output, multiples)
        two_tensor = create_tensor_by_element((2,), mstype.int32)
        three_tensor = create_tensor_by_element((3,), mstype.int32)
        output_shape = concat0((expend_dims(n_value, -1), three_tensor, expend_dims(h_value * w_value, -1)))
        dout_shape = concat0((expend_dims(n_value, -1), expend_dims(h_value * w_value, -1), two_tensor))
        dtheta = batmatmul(reshape(output, output_shape), reshape(dout, dout_shape).astype("float32"))
        return transpose(dtheta, perm2), tre

    def dyn_bprop(theta, output_size, out, dout):
        len_output_size = reducesum(dyn_shape(output_size))
        dtheta = dyn_ones(Tensor([1, 3, 2], mstype.int32), mstype.float32)
        ret = dyn_ones(Tensor([1, 6], mstype.int32), mstype.float32)
        if len_output_size == 5:
            dtheta, ret = dyn_bprop_five(theta, output_size, out, dout, len_output_size)
        elif len_output_size == 4:
            dtheta, ret = dyn_bprop_four(theta, output_size, out, dout)
        return dtheta, ret

    def static_bprop(theta, output_size, out, dout):
        x_shape = P.Shape()(dout)
        n_value = x_shape[0]
        h_value = x_shape[1]
        w_value = x_shape[2]
        d_value = output_size[2]
        perm1 = (1, 0)
        perm2 = (0, 2, 1)
        output_size_shape = len(output_size)
        len_output_size = output_size_shape
        vecx = vecy = vecz = dtheta = dout_ = two = one = tre = []
        if len_output_size == 5:
            n_value = output_size[0]
            d_value = output_size[2]
            h_value = output_size[3]
            w_value = output_size[4]
            start = Tensor(-1, mstype.float32)
            stop = Tensor(1, mstype.float32)
            vecx = Tensor([0], dtype=mstype.float32)
            if w_value != 1:
                vecx = linspace(start, stop, w_value)
            vecy = Tensor([0], dtype=mstype.float32)
            if h_value != 1:
                vecy = linspace(start, stop, h_value)
            vecz = Tensor([0], dtype=mstype.float32)
            if d_value != 1:
                vecz = linspace(start, stop, d_value)
            if align_corners is False:
                vecx = vecx * (w_value - 1) / w_value
                vecy = vecy * (h_value - 1) / h_value
                vecz = vecz * (d_value - 1) / d_value
            out = vecx
            if h_value * d_value != 1:
                multiples = (h_value * d_value, 1)
                out = tile(vecx, multiples)
            one = reshape(out, (h_value * w_value * d_value, 1))
            if w_value == 1:
                out = expend_dims(vecy, 0)
            elif w_value != 1:
                multiples = (w_value, 1)
                out = tile(vecy, multiples)
            out = transpose(out, perm1)
            if d_value != 1:
                multiples = (d_value, 1)
                out = tile(out, multiples)
            two = reshape(out, (h_value * w_value * d_value, 1))
            out = expend_dims(vecz, 0)
            if w_value * h_value != 1:
                multiples = (w_value * h_value, 1)
                out = tile(vecz, multiples)
            out = transpose(out, perm1)
            tre = reshape(out, (h_value * w_value * d_value, 1))
            fou = ones((h_value * w_value * d_value, 1), mstype.float32)
            output = concat((one, two, tre, fou))
            output = transpose(output, perm1)
            if n_value != 1:
                multiples = (n_value, 1)
                output = tile(output, multiples)
            output = output.view(n_value, 4, h_value * w_value * d_value)
            dout_ = dout.view(n_value, d_value * h_value * w_value, 3).astype("float32")
            dtheta = batmatmul(output, dout_)
            dtheta = transpose(dtheta, perm2)
        elif len_output_size == 4:
            start = Tensor(-1, mstype.float32)
            stop = Tensor(1, mstype.float32)
            vecx = Tensor([0], dtype=mstype.float32)
            if w_value != 1:
                vecx = linspace(start, stop, w_value)
            vecy = Tensor([0], dtype=mstype.float32)
            if h_value != 1:
                vecy = linspace(start, stop, h_value)
            if align_corners is False:
                vecx = vecx * (w_value - 1) / w_value
                vecy = vecy * (h_value - 1) / h_value
            out = vecx
            if h_value != 1:
                multiples = (h_value, 1)
                out = tile(vecx, multiples)
            one = reshape(out, (h_value * w_value, 1))
            if w_value == 1:
                out = expend_dims(vecy, 0)
            elif w_value != 1:
                multiples = (w_value, 1)
                out = tile(vecy, multiples)
            out = transpose(out, perm1)
            two = reshape(out, (h_value * w_value, 1))
            tre = ones((h_value * w_value, 1), mstype.float32)
            output = concat((one, two, tre))
            multiples = (n_value, 1)
            output = transpose(output, perm1)
            output = tile(output, multiples)
            output = output.view(n_value, 3, h_value * w_value)
            dout_ = dout.view(n_value, h_value * w_value, 2).astype("float32")
            dtheta = batmatmul(output, dout_)
            dtheta = transpose(dtheta, perm2)
        return dtheta, tre

    def bprop_gpu(theta, output_size, out, dout):
        is_tensor, _ = convert_to_tensor(output_size)
        if is_tensor:
            return dyn_bprop(theta, output_size, out, dout)
        return static_bprop(theta, output_size, out, dout)

    def bprop(theta, output_size, out, dout):
        dx = input_grad(dout, output_size)
        return dx, zeros_like(output_size)

    if context.get_context('device_target') == "GPU":
        return bprop_gpu

    return bprop


@bprop_getters.register(SegmentMax)
def get_bprop_segment_max(self):
    """Generate bprop for SegmentMax"""
    segment_sum = SegmentSum()

    def bprop(input_x, segment_ids, output, dout):
        return _segment_min_or_max_grad(segment_sum, input_x, segment_ids, output, dout)

    return bprop


@bprop_getters.register(SegmentMin)
def get_bprop_segment_min(self):
    """Generate bprop for SegmentMin"""
    segment_sum = SegmentSum()

    def bprop(input_x, segment_ids, output, dout):
        return _segment_min_or_max_grad(segment_sum, input_x, segment_ids, output, dout)

    return bprop


@bprop_getters.register(TensorScatterElements)
def get_bprop_tensor_scatter_elements(self):
    """Generate bprop for TensorScatterElements"""
    gather_d = P.GatherD()
    axis = self.axis
    reduction = self.reduction
    tensor_scatter_elements = TensorScatterElements(axis, reduction)

    def bprop(x, indices, update, out, dout):
        x_grad = tensor_scatter_elements(dout, indices, zeros_like(update))
        update_grad = gather_d(dout, axis, indices)
        return x_grad, zeros_like(indices), update_grad

    return bprop


@bprop_getters.register(ScatterAddWithAxis)
def get_bprop_scatter_add_with_axis(self):
    """Generate bprop for ScatterAddWithAxis"""
    gather_d = P.GatherD()
    slice_op = P.Slice()
    axis = self.axis

    def bprop(x, indices, update, out, dout):
        dout_shape = dout.shape
        index_shape = indices.shape
        if dout_shape != index_shape:
            pad_list = []
            slice_list = []
            for i, pos in enumerate(dout_shape):
                pad_list.append((0, pos - index_shape[i]))
                slice_list.append(0)
            pad_tuple = tuple(pad_list)
            out_index = P.Pad(pad_tuple)(indices)
            out_gather = gather_d(dout, axis, out_index)
            update_grad = slice_op(out_gather, slice_list, index_shape)
        else:
            update_grad = gather_d(dout, axis, indices)
        return dout, zeros_like(indices), update_grad

    return bprop


@bprop_getters.register(Expand)
def get_bprop_expand(self):
    """Generate bprop for Expand"""

    reducesum = P.ReduceSum(keep_dims=True)
    zeroslike = P.ZerosLike()

    def bprop(x, shape, out, dout):
        reduce_dims = []
        dshape = zeroslike(shape)
        dx_shape = dout.shape
        if dx_shape is None:
            return dout.sum(), dshape
        x_shape = x.shape
        leading_dims = len(dx_shape) - len(x_shape)
        for i in range(leading_dims):
            reduce_dims.append(i)
        for j in range(leading_dims, len(dx_shape)):
            if x_shape[j - leading_dims] == 1 and dx_shape[j] != 1:
                reduce_dims.append(j)
        if reduce_dims:
            dout = reducesum(dout, reduce_dims)
        dx = dout.reshape(x_shape) if leading_dims > 0 else dout
        return dx, dshape

    return bprop


@bprop_getters.register(SegmentMean)
def get_bprop_segment_mean(self):
    """Generate bprop for SegmentMean"""
    rank = P.Rank()
    shape = P.Shape()
    dyn_shape = P.TensorShape()
    fill = P.Fill()
    divide = P.Div()
    segment_sum = SegmentSum()
    gather = P.Gather()
    cast = P.Cast()
    concat = P.Concat()
    expand_dims = P.ExpandDims()

    def bprop(input_x, segment_ids, output, dout):
        input_x_type = F.dtype(input_x)
        input_x = cast(input_x, mstype.float32)
        dout = cast(dout, mstype.float32)
        dout_type = F.dtype(dout)

        ones_shape = shape(segment_ids)
        if F.is_sequence_value_unknown(ones_shape):
            ones_shape = dyn_shape(segment_ids)

        ones = ()
        inputx_shape = shape(input_x)
        if F.is_sequence_value_unknown(inputx_shape):
            input_rank = dyn_rank(input_x)
            if input_rank > cast(1, mstype.float32):
                ones_shape = concat([ones_shape, dyn_ones(expand_dims(input_rank - 1, 0), mstype.int64)])
            ones = dyn_fill(dout_type, ones_shape, 1)
        else:
            input_rank = rank(input_x)
            ones_shape = ones_shape + (1,) * (input_rank - 1)
            ones = fill(dout_type, ones_shape, 1)

        scaled_grad = divide(dout, segment_sum(ones, segment_ids))
        return cast(gather(scaled_grad, segment_ids, 0), input_x_type), zeros_like(segment_ids)

    return bprop
