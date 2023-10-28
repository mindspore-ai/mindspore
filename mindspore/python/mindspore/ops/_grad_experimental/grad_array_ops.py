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
from mindspore.ops._grad_experimental.grad_math_ops import binop_grad_common
from mindspore.ops._grad_experimental.grad_base import bprop_getters, dyn_ones
from mindspore.ops._grad_experimental.grad_base import convert_to_tensor, create_tensor_by_element
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.operations.array_ops import MatrixDiagV3
from mindspore.ops.operations.array_ops import MatrixDiagPartV3
from mindspore.ops.operations.array_ops import ResizeNearestNeighborV2
from mindspore.ops.operations.array_ops import MatrixSetDiagV3
from mindspore.ops.operations.array_ops import MatrixBandPart
from mindspore.ops.operations.array_ops import Mvlgamma
from mindspore.ops.operations.array_ops import IndexFill
from mindspore.ops.operations.array_ops import IndexPut
from mindspore.ops.operations.array_ops import SegmentSum
from mindspore.ops.operations.array_ops import ScatterAddWithAxis
from mindspore.ops.operations.array_ops import Expand
from mindspore.ops.operations.array_ops import SegmentMean
from mindspore.ops.operations.array_ops import AffineGrid
from mindspore.ops.operations.array_ops import MaskedScatter
from mindspore.ops.operations.array_ops import MaskedSelect
from mindspore.ops.operations.array_ops import CountNonZero
from mindspore.ops.operations.random_ops import LogNormalReverse
from mindspore.ops.operations.random_ops import ParameterizedTruncatedNormal
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G
from mindspore import context
from mindspore.ops.primitive import _primexpr
from mindspore.common.sparse_tensor import RowTensorInner
from mindspore.ops._utils.utils import generate_shape_index

reduce_sum = P.ReduceSum()
unsorted_segment_sum = P.UnsortedSegmentSum()
transpose = P.Transpose()
shape_op = P.Shape()
reshape = P.Reshape()
size_op = P.Size()
invert_permutation = P.InvertPermutation()
logical_and = P.LogicalAnd()


@constexpr
def _raise_value_error(*info):
    info_str = ""
    for obj in info:
        info_str = info_str + f"{obj}"
    raise ValueError(info_str)


@constexpr
def _create_tensor(data, dtype):
    return Tensor(data, dtype=dtype)


@bprop_getters.register(P.MaskedFill)
def get_bprop_masked_select(self):
    """Generate bprop for MaskedFill"""
    mul_op = P.Mul()
    sum_op = P.ReduceSum()
    is_instance_op = inner.IsInstance()
    rank = P.Rank()

    def bprop(input_data, mask, value, out, dout):
        mask = F.cast(mask, mstype.float32)
        dout = F.cast(dout, mstype.float32)
        dinput = mul_op(dout, (1 - mask))
        dvalue = mul_op(dout, mask)
        dinput, dvalue = binop_grad_common(input_data, mask, dinput, dvalue)
        # for dynamic rank, reduce axis should be calc
        if F.is_sequence_shape_unknown(P.Shape()(dvalue)):
            axis = range(0, rank(dvalue), 1)
            dvalue = sum_op(dvalue, axis)
        else:
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
    masked_fill = P.MaskedFill()
    masked_select = P.MaskedSelect()
    shape = P.TensorShape()
    range_ = P.Range()
    scatter_update = P.TensorScatterElements()
    def bprop(x, mask, updates, out, dout):
        dout = F.cast(dout, mstype.float32)
        dx = masked_fill(dout, mask, F.cast(0, mstype.float32))
        dupdates = F.cast(zeros_like(updates).reshape(-1), mstype.float32)
        dupdates_val = F.cast(masked_select(dout, mask), mstype.float32)
        length = F.cast(shape(dupdates_val)[0], mstype.int32)
        scatter_indices = range_(F.cast(0, mstype.int32), length, F.cast(1, mstype.int32))
        dupdates = scatter_update(dupdates, scatter_indices, dupdates_val)
        dupdates = reshape(dupdates, shape(updates))
        return F.cast(dx, x.dtype), zeros_like(mask), F.cast(dupdates, updates.dtype)

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


@bprop_getters.register(IndexFill)
def get_bprop_index_fill(self):
    """Generate bprop for IndexFill"""
    gather = P.Gather()
    index_fill = IndexFill()
    shape = P.Shape()
    rank = P.Rank()

    def bprop(x, dim, indices, value, out, dout):
        zero_value = zeros_like(value)
        x_grad = index_fill(dout, dim, indices, zero_value)
        if F.is_sequence_value_unknown(shape(x)):
            if rank(x) == 0:
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
    equal = P.Equal()
    cast = P.Cast()
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
        indices_me = stack(indices_ms)
        indices_grad = F.transpose(indices_me, F.make_range(F.rank(indices_me)-1, -1, -1))
        values_grad = gather_nd(dout, indices_grad)
        if equal(cast(x2.shape[0], mstype.int32), Tensor(1)):
            values_grad = values_grad.sum().reshape(1)
        if values_grad.shape != x2.shape and len(indices) < len(x1.shape):
            _, values_grad = binop_grad_common(x1, x2, dout, values_grad)
        if accumulate_grad == 0:
            dout = index_put(dout, zeros_like(x2), indices)
        return dout, values_grad, [zeros_like(item) for item in indices]

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


@bprop_getters.register(MatrixBandPart)
def get_bprop_matrix_band_part(self):
    """Grad definition for `MatrixBandPart` operation."""
    matrix_band_part = MatrixBandPart()

    def bprop(x, lower, upper, out, dout):
        return matrix_band_part(dout, lower, upper), zeros_like(lower), zeros_like(upper)

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

    x_indicators = F.cast(equal(x, out), mstype.int32)
    possibly_updated = gather_nd(out, indices)
    out_indicators = F.cast(equal(updates, possibly_updated), mstype.int32)
    input_shape = shape(x)
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


@bprop_getters.register(ResizeNearestNeighborV2)
def get_bprop_resize_nearest_neighbor_v2(self):
    """Generate bprop for ResizeNearestNeighborV2"""
    align_corners = self.align_corners
    half_pixel_centers = self.half_pixel_centers
    grad_op = G.ResizeNearestNeighborV2Grad(align_corners, half_pixel_centers)

    def bprop(x, size, output, dout):
        x_shape = P.Shape()(x)
        grad_in_size = x_shape[2:4]

        if F.is_sequence_value_unknown(P.Shape()(x)):
            dx = grad_op(dout, grad_in_size)
            return dx, zeros_like(grad_in_size)

        dx = grad_op(dout, _create_tensor(grad_in_size, mstype.int32))
        return dx, zeros_like(grad_in_size)

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
    dtype = P.DType()
    cast = P.Cast()
    matmul = P.MatMul()
    _, _, ksize_d, ksize_h, ksize_w = self.kernel_size
    range_ = P.Range()
    ones_like = P.OnesLike()

    def _dyn_extract_volume_patches(x, out, dout):
        x_shape = shape_op(x)
        out_shape = shape_op(out)
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
        sp_mat_full = scatter_nd(idx_map, F.fill(dtype(dout), (out_indices_num,), 1), sp_shape)
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
    concat = P.Concat(1)
    concat0 = P.Concat(0)
    tile = P.Tile()
    div = P.Div()
    linspace = P.LinSpace()
    batmatmul = P.BatchMatMul()
    expend_dims = P.ExpandDims()
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
        len_output_size = reducesum(shape_op(output_size))
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
    fill = P.FillV2()
    divide = P.Div()
    segment_sum = SegmentSum()
    gather = P.Gather()
    cast = P.Cast()

    def bprop(input_x, segment_ids, output, dout):
        input_x_type = F.dtype(input_x)
        input_x = cast(input_x, mstype.float32)
        dout = cast(dout, mstype.float32)
        dout_type = F.dtype(dout)
        ones_shape = shape(segment_ids)
        input_rank = rank(input_x)
        ones_shape = ones_shape + (1,) * (input_rank - 1)
        ones = fill(ones_shape, Tensor(1, dout_type))
        scaled_grad = divide(dout, segment_sum(ones, segment_ids))
        return cast(gather(scaled_grad, segment_ids, 0), input_x_type), zeros_like(segment_ids)

    return bprop


@bprop_getters.register(P.EmbeddingLookup)
def get_bprop_embedding_lookup(self):
    """Generate bprop for EmbeddingLookup"""
    sub_op = P.Sub()
    reshape_op = P.Reshape()

    def bprop_sparse(x, indices, offset, out, dout):
        x_shp = shape_op(x)
        if F.is_sequence_value_unknown(x_shp):
            raise RuntimeError("Now, EmbeddingLookup op's grad don't support Dynamic Sense!")
        new_indices = sub_op(indices, offset)
        indices_size = size_op(new_indices)
        if indices_size > 0:
            # Reshape the 'new_indices'
            new_indices_shape_changed = (indices_size,)
            new_indices = reshape_op(new_indices, new_indices_shape_changed)
        else:
            new_indices_shape_changed = ()
        x_shp_tail = x_shp[1:]
        actual_dout_shape_changed = new_indices_shape_changed + x_shp_tail
        # Reshape the 'actual_dout' on device
        actual_dout = reshape_op(dout, actual_dout_shape_changed)
        return RowTensorInner(new_indices, actual_dout, x_shp), zeros_like(indices), zeros_like(offset)

    return bprop_sparse


@_primexpr
def _generate_inverse_index(x_shape, axis, batch_dims=0):
    x_rank = len(x_shape)
    index = tuple(range(x_rank))
    if axis < 0:
        axis += x_rank
    perm = index[:batch_dims] + index[batch_dims + 1:1 + axis] + (index[batch_dims],) + index[1 + axis:]
    return perm


@bprop_getters.register(P.SparseGatherV2)
def get_bprop_sparse_gather_v2(self):
    """Generate bprop for SparseGatherV2"""

    def bprop(x, indices, axis, out, dout):
        x_shp = shape_op(x)
        if axis == 0:
            indices_size = (size_op(indices),)
            if len(x_shp) <= 1:
                x_tail_shp = ()
            else:
                x_tail_shp = x_shp[1:]
            values_shape = indices_size + x_tail_shp
            values = reshape(dout, values_shape)
            indices_new = reshape(indices, indices_size)
            return RowTensorInner(indices_new, values, x_shp), zeros_like(indices), zeros_like(axis)
        if F.rank(dout) == 0:
            dout = P.ExpandDims()(dout, -1)
        if F.rank(indices) == 0:
            indices = P.ExpandDims()(indices, -1)
        out_shp = shape_op(dout)
        ind_shp = shape_op(indices)
        # Example: out_shape:(3,2,3) axis 1 -> (1,0,2)
        perm_1 = generate_shape_index(out_shp, ind_shp, axis)
        values_transpose = transpose(dout, perm_1)
        params_grad = unsorted_segment_sum(values_transpose, indices, shape_op(x)[axis])
        # Example: out_shape:(3,2,3) axis 2 -> (1,2,0)
        perm_2 = _generate_inverse_index(x_shp, axis)
        params_grad = transpose(params_grad, perm_2)
        return params_grad, zeros_like(indices), zeros_like(axis)

    return bprop


@bprop_getters.register(P.Unstack)
def get_bprop_unstack(self):
    """Generate bprop for Unstack"""
    axis = self.axis

    def bprop(x, out, dout):
        unstack_grad = P.Stack(axis)
        out = unstack_grad(dout)
        return (out,)

    return bprop


@bprop_getters.register(P.Eye)
def get_bprop_eye(self):
    """Generate bprop for Eye"""

    def bprop(n, m, t, out, dout):
        return zeros_like(n), zeros_like(m), zeros_like(t)

    return bprop


@bprop_getters.register(P.ScatterNdUpdate)
def get_bprop_scatter_nd_update(self):
    """Generate bprop for ScatterNdUpdate"""
    op = P.GatherNd()

    def bprop(x, indices, update, out, dout):
        return dout, zeros_like(indices), op(dout, indices)

    return bprop


@bprop_getters.register(P.ScatterNonAliasingAdd)
def get_bprop_scatter_non_aliasing_add_update(self):
    """Generate bprop for ScatterNonAliasingAdd"""
    op = P.GatherNd()

    def bprop(x, indices, update, out, dout):
        return dout, zeros_like(indices), op(dout, indices)

    return bprop


@bprop_getters.register(P.ScatterUpdate)
def get_bprop_scatter_update(self):
    """Generate bprop for ScatterUpdate"""
    gather = P.Gather()

    def bprop(x, indices, update, out, dout):
        return dout, zeros_like(indices), gather(dout, indices, 0)

    return bprop


@bprop_getters.register(P.TransShape)
def get_bprop_trans_shape(self):
    """Generate bprop for TransShape"""
    op = P.TransShape()

    def bprop(x, shape, out, dout):
        dx = op(dout, shape_op(x))
        return (dx, zeros_like(shape))

    return bprop


@bprop_getters.register(P.Unique)
def get_bprop_unique(self):
    """Generate bprop for Unique"""
    op = G.UniqueGrad()

    def bprop(x, out, dout):
        dx = op(dout, out)
        return (dx,)

    return bprop
