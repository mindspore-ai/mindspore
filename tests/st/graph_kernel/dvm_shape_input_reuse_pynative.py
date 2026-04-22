# Copyright 2026 Huawei Technologies Co., Ltd
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

"""DVM shape-input cache regression cases in pynative mode."""

import numpy as np
import mindspore
from mindspore import Tensor
from mindspore import context
from mindspore import mint
from mindspore import ops
from tests.st.graph_kernel.gk_utils import compare_outputs, gen_flag, gen_input as _gen_input, trans_data_type

np.random.seed(1)
context.set_context(mode=context.PYNATIVE_MODE)


def gen_input(*args, **kwargs):
    return _gen_input(*args, **kwargs).to("Ascend")


def _get_cmp_precision(data_type):
    return 4e-3 if data_type == "bfloat16" else 1e-3


def _transpose(x, axis):
    return ops.Transpose()(x, axis)


def _permute_shape(shape, axis):
    return tuple(shape[i] for i in axis)


def _make_elemwise_input(shape, data_type, data_range=0.01, is_positive=False):
    lhs = gen_input(shape, data_type, data_range=data_range, is_positive=is_positive)
    rhs = gen_input(shape, data_type, data_range=data_range, is_positive=is_positive)
    return ops.add(lhs, rhs)


def test_matmul_family_transpose_other_shape_input_reuse():
    """
    Feature: matmul-family transpose(other) shape-input reuse
    Description: later fused add should keep using the original contiguous tensor after matmul-like ops
    Expectation: the result matches the baseline result
    """

    data_type = "float16"

    def _run_matmul(input_shape, other_shape):
        flag = gen_flag("matmul_transpose_other_shape_input_reuse", input_shape, other_shape, data_type)
        x0 = gen_input(input_shape, data_type, data_range=0.01)
        x1 = _make_elemwise_input(other_shape, data_type, data_range=0.01)
        y0 = ops.auto_generate.MatMul(False, False)(x0, _transpose(x1, (1, 0)))
        y1 = ops.add(y0, x1)
        compare_outputs(flag, [y0, y1], cmp_precision=_get_cmp_precision(data_type))

    def _run_batch_matmul(input_shape, other_shape):
        flag = gen_flag("batch_matmul_transpose_other_shape_input_reuse", input_shape, other_shape, data_type)
        x0 = gen_input(input_shape, data_type, data_range=0.01)
        x1 = _make_elemwise_input(other_shape, data_type, data_range=0.01)
        y0 = ops.auto_generate.BatchMatMul(False, False)(x0, _transpose(x1, (0, 2, 1)))
        y1 = ops.add(y0, x1)
        compare_outputs(flag, [y0, y1], cmp_precision=_get_cmp_precision(data_type))

    def _run_matmul_broadcast(input_shape, other_shape, delta_shape):
        flag = gen_flag("matmul_transpose_other_broadcast_input_reuse", input_shape, other_shape, delta_shape,
                        data_type)
        x0 = gen_input(input_shape, data_type, data_range=0.01)
        x1 = _make_elemwise_input(other_shape, data_type, data_range=0.01)
        delta = gen_input(delta_shape, data_type, data_range=0.01)
        y0 = ops.auto_generate.MatMul(False, False)(x0, _transpose(x1, (1, 0)))
        y1 = ops.add(x1, delta)
        compare_outputs(flag, [y0, y1], cmp_precision=_get_cmp_precision(data_type))

    def _run_batch_matmul_broadcast(input_shape, other_shape, delta_shape):
        flag = gen_flag("batch_matmul_transpose_other_broadcast_input_reuse",
                        input_shape, other_shape, delta_shape, data_type)
        x0 = gen_input(input_shape, data_type, data_range=0.01)
        x1 = _make_elemwise_input(other_shape, data_type, data_range=0.01)
        delta = gen_input(delta_shape, data_type, data_range=0.01)
        y0 = ops.auto_generate.BatchMatMul(False, False)(x0, _transpose(x1, (0, 2, 1)))
        y1 = ops.add(x1, delta)
        compare_outputs(flag, [y0, y1], cmp_precision=_get_cmp_precision(data_type))

    # transpose result may still be contiguous-like because one dim is 1
    _run_matmul((1, 2048), (1, 2048))
    _run_batch_matmul((2, 1, 2048), (2, 1, 2048))

    # transpose result is a non-contiguous square view, while the logical shape stays the same
    _run_matmul((1024, 1024), (1024, 1024))
    _run_batch_matmul((2, 1024, 1024), (2, 1024, 1024))

    # non-square inputs without dim-1, and later elemwise uses broadcast on the reused tensor
    _run_matmul_broadcast((512, 768), (640, 768), (1, 768))
    _run_batch_matmul_broadcast((2, 512, 768), (2, 640, 768), (1, 1, 768))


def test_matmul_family_transposed_tensor_default_input_reuse():
    """
    Feature: matmul-family transposed-tensor default input reuse
    Description: later fused elemwise ops reuse the same transposed tensor that was already consumed by matmul-like ops
    Expectation: the result matches the baseline result
    """

    data_type = "float16"

    def _run_matmul_transpose_other(input_shape, other_shape, delta_shape=None):
        if delta_shape is None:
            delta_shape = _permute_shape(other_shape, (1, 0))
        flag = gen_flag("matmul_transposed_tensor_default_input_reuse", input_shape, other_shape, delta_shape,
                        data_type)
        axis = (1, 0)
        x0 = gen_input(input_shape, data_type, data_range=0.01)
        x1 = _make_elemwise_input(other_shape, data_type, data_range=0.01)
        x1_t = _transpose(x1, axis)
        delta = gen_input(delta_shape, data_type, data_range=0.01)
        y0 = ops.auto_generate.MatMul(False, False)(x0, x1_t)
        y1 = ops.add(x1_t, delta)
        compare_outputs(flag, [y0, y1], cmp_precision=_get_cmp_precision(data_type))

    def _run_batch_matmul_transpose_other(input_shape, other_shape, delta_shape=None):
        if delta_shape is None:
            delta_shape = _permute_shape(other_shape, (0, 2, 1))
        flag = gen_flag("batch_matmul_transposed_tensor_default_input_reuse",
                        input_shape, other_shape, delta_shape, data_type)
        axis = (0, 2, 1)
        x0 = gen_input(input_shape, data_type, data_range=0.01)
        x1 = _make_elemwise_input(other_shape, data_type, data_range=0.01)
        x1_t = _transpose(x1, axis)
        delta = gen_input(delta_shape, data_type, data_range=0.01)
        y0 = ops.auto_generate.BatchMatMul(False, False)(x0, x1_t)
        y1 = ops.add(x1_t, delta)
        compare_outputs(flag, [y0, y1], cmp_precision=_get_cmp_precision(data_type))

    def _run_matmul_transpose_input(input_shape, other_shape, delta_shape=None):
        if delta_shape is None:
            delta_shape = _permute_shape(input_shape, (1, 0))
        flag = gen_flag("matmul_transposed_input_default_input_reuse", input_shape, other_shape, delta_shape, data_type)
        axis = (1, 0)
        x0 = _make_elemwise_input(input_shape, data_type, data_range=0.01)
        x1 = gen_input(other_shape, data_type, data_range=0.01)
        x0_t = _transpose(x0, axis)
        delta = gen_input(delta_shape, data_type, data_range=0.01)
        y0 = ops.auto_generate.MatMul(False, False)(x0_t, x1)
        y1 = ops.add(x0_t, delta)
        compare_outputs(flag, [y0, y1], cmp_precision=_get_cmp_precision(data_type))

    def _run_batch_matmul_transpose_input(input_shape, other_shape, delta_shape=None):
        if delta_shape is None:
            delta_shape = _permute_shape(input_shape, (0, 2, 1))
        flag = gen_flag("batch_matmul_transposed_input_default_input_reuse",
                        input_shape, other_shape, delta_shape, data_type)
        axis = (0, 2, 1)
        x0 = _make_elemwise_input(input_shape, data_type, data_range=0.01)
        x1 = gen_input(other_shape, data_type, data_range=0.01)
        x0_t = _transpose(x0, axis)
        delta = gen_input(delta_shape, data_type, data_range=0.01)
        y0 = ops.auto_generate.BatchMatMul(False, False)(x0_t, x1)
        y1 = ops.add(x0_t, delta)
        compare_outputs(flag, [y0, y1], cmp_precision=_get_cmp_precision(data_type))

    _run_matmul_transpose_other((1, 2048), (1, 2048))
    _run_batch_matmul_transpose_other((2, 1, 2048), (2, 1, 2048))

    _run_matmul_transpose_other((1024, 1024), (1024, 1024))
    _run_batch_matmul_transpose_other((2, 1024, 1024), (2, 1024, 1024))

    _run_matmul_transpose_other((512, 768), (640, 768), (1, 640))
    _run_batch_matmul_transpose_other((2, 512, 768), (2, 640, 768), (1, 1, 640))

    _run_matmul_transpose_input((2048, 1), (2048, 1))
    _run_batch_matmul_transpose_input((2, 2048, 1), (2, 2048, 1))

    _run_matmul_transpose_input((1024, 1024), (1024, 1024))
    _run_batch_matmul_transpose_input((2, 1024, 1024), (2, 1024, 1024))

    _run_matmul_transpose_input((640, 768), (640, 512), (1, 640))
    _run_batch_matmul_transpose_input((2, 640, 768), (2, 640, 512), (1, 1, 640))


def test_matmul_family_transpose_input_shape_input_reuse():
    """
    Feature: matmul-family transpose(input) shape-input reuse
    Description: later fused add should keep using the original contiguous tensor after matmul-like ops
    Expectation: the result matches the baseline result
    """

    data_type = "float16"

    def _run_matmul(input_shape, other_shape):
        flag = gen_flag("matmul_transpose_input_shape_input_reuse", input_shape, other_shape, data_type)
        x0 = _make_elemwise_input(input_shape, data_type, data_range=0.01)
        x1 = gen_input(other_shape, data_type, data_range=0.01)
        y0 = ops.auto_generate.MatMul(False, False)(_transpose(x0, (1, 0)), x1)
        y1 = ops.add(y0, x0)
        compare_outputs(flag, [y0, y1], cmp_precision=_get_cmp_precision(data_type))

    def _run_batch_matmul(input_shape, other_shape):
        flag = gen_flag("batch_matmul_transpose_input_shape_input_reuse", input_shape, other_shape, data_type)
        x0 = _make_elemwise_input(input_shape, data_type, data_range=0.01)
        x1 = gen_input(other_shape, data_type, data_range=0.01)
        y0 = ops.auto_generate.BatchMatMul(False, False)(_transpose(x0, (0, 2, 1)), x1)
        y1 = ops.add(y0, x0)
        compare_outputs(flag, [y0, y1], cmp_precision=_get_cmp_precision(data_type))

    def _run_matmul_broadcast(input_shape, other_shape, delta_shape):
        flag = gen_flag("matmul_transpose_input_broadcast_input_reuse", input_shape, other_shape, delta_shape,
                        data_type)
        x0 = _make_elemwise_input(input_shape, data_type, data_range=0.01)
        x1 = gen_input(other_shape, data_type, data_range=0.01)
        delta = gen_input(delta_shape, data_type, data_range=0.01)
        y0 = ops.auto_generate.MatMul(False, False)(_transpose(x0, (1, 0)), x1)
        y1 = ops.add(x0, delta)
        compare_outputs(flag, [y0, y1], cmp_precision=_get_cmp_precision(data_type))

    def _run_batch_matmul_broadcast(input_shape, other_shape, delta_shape):
        flag = gen_flag("batch_matmul_transpose_input_broadcast_input_reuse",
                        input_shape, other_shape, delta_shape, data_type)
        x0 = _make_elemwise_input(input_shape, data_type, data_range=0.01)
        x1 = gen_input(other_shape, data_type, data_range=0.01)
        delta = gen_input(delta_shape, data_type, data_range=0.01)
        y0 = ops.auto_generate.BatchMatMul(False, False)(_transpose(x0, (0, 2, 1)), x1)
        y1 = ops.add(x0, delta)
        compare_outputs(flag, [y0, y1], cmp_precision=_get_cmp_precision(data_type))

    def _run_matmul_ext(input_shape, other_shape):
        flag = gen_flag("matmul_ext_transpose_input_shape_input_reuse", input_shape, other_shape, data_type)
        x0 = _make_elemwise_input(input_shape, data_type, data_range=0.01)
        x1 = gen_input(other_shape, data_type, data_range=0.01)
        y0 = mint.matmul(_transpose(x0, (1, 0)), x1)
        y1 = ops.add(y0, x0)
        compare_outputs(flag, [y0, y1], cmp_precision=_get_cmp_precision(data_type))

    def _run_batch_matmul_ext(input_shape, other_shape):
        flag = gen_flag("batch_matmul_ext_transpose_input_shape_input_reuse", input_shape, other_shape, data_type)
        x0 = _make_elemwise_input(input_shape, data_type, data_range=0.01)
        x1 = gen_input(other_shape, data_type, data_range=0.01)
        y0 = ops.auto_generate.BatchMatMulExt()(_transpose(x0, (0, 2, 1)), x1)
        y1 = ops.add(y0, x0)
        compare_outputs(flag, [y0, y1], cmp_precision=_get_cmp_precision(data_type))

    def _run_dense(input_shape, weight_shape, bias_shape):
        flag = gen_flag("dense_transpose_input_shape_input_reuse", input_shape, weight_shape, bias_shape, data_type)
        x0 = _make_elemwise_input(input_shape, data_type, data_range=0.01)
        weight = gen_input(weight_shape, data_type, data_range=0.01)
        bias = gen_input(bias_shape, data_type, data_range=0.01)
        y0 = ops.auto_generate.Dense()(_transpose(x0, (1, 0)), weight, bias)
        y1 = ops.add(y0, x0)
        compare_outputs(flag, [y0, y1], cmp_precision=_get_cmp_precision(data_type))

    _run_matmul((2048, 1), (2048, 1))
    _run_batch_matmul((2, 2048, 1), (2, 2048, 1))
    _run_matmul_ext((2048, 1), (2048, 1))
    _run_batch_matmul_ext((2, 2048, 1), (2, 2048, 1))
    _run_dense((2048, 1), (1, 2048), (1,))

    _run_matmul((1024, 1024), (1024, 1024))
    _run_batch_matmul((2, 1024, 1024), (2, 1024, 1024))
    _run_matmul_ext((1024, 1024), (1024, 1024))
    _run_batch_matmul_ext((2, 1024, 1024), (2, 1024, 1024))
    _run_dense((1024, 1024), (1024, 1024), (1024,))

    _run_matmul_broadcast((640, 768), (640, 512), (1, 768))
    _run_batch_matmul_broadcast((2, 640, 768), (2, 640, 512), (1, 1, 768))


def test_matmul_interleaved_shape_and_default_input_reuse():
    """
    Feature: matmul interleaved shape/default input reuse
    Description: the same test interleaves shape-input matmul ops with default-input elemwise ops on reused tensors
    Expectation: the result matches the baseline result
    """

    data_type = "float16"

    def _run_transpose_other(lhs_shape, reused_shape):
        flag = gen_flag("matmul_interleaved_transpose_other_shape_and_default_input_reuse",
                        lhs_shape, reused_shape, data_type)
        x0 = gen_input(lhs_shape, data_type, data_range=0.01)
        x1 = _make_elemwise_input(reused_shape, data_type, data_range=0.01)
        x2 = gen_input(reused_shape, data_type, data_range=0.01)
        x3 = gen_input(lhs_shape, data_type, data_range=0.01)
        x4 = gen_input(reused_shape, data_type, data_range=0.01)
        y0 = ops.auto_generate.MatMul(False, False)(x0, _transpose(x1, (1, 0)))
        y1 = ops.add(x2, x1)
        y2 = ops.auto_generate.MatMul(False, False)(x3, _transpose(x1, (1, 0)))
        y3 = ops.sub(x4, x1)
        compare_outputs(flag, [y0, y1, y2, y3], cmp_precision=_get_cmp_precision(data_type))

    def _run_transpose_input(reused_shape, rhs_shape):
        flag = gen_flag("matmul_interleaved_transpose_input_shape_and_default_input_reuse",
                        reused_shape, rhs_shape, data_type)
        x0 = _make_elemwise_input(reused_shape, data_type, data_range=0.01)
        x1 = gen_input(rhs_shape, data_type, data_range=0.01)
        x2 = gen_input(reused_shape, data_type, data_range=0.01)
        x3 = gen_input(rhs_shape, data_type, data_range=0.01)
        x4 = gen_input(reused_shape, data_type, data_range=0.01)
        y0 = ops.auto_generate.MatMul(False, False)(_transpose(x0, (1, 0)), x1)
        y1 = ops.add(x2, x0)
        y2 = ops.auto_generate.MatMul(False, False)(_transpose(x0, (1, 0)), x3)
        y3 = ops.sub(x4, x0)
        compare_outputs(flag, [y0, y1, y2, y3], cmp_precision=_get_cmp_precision(data_type))

    _run_transpose_other((1024, 2048), (1, 2048))
    _run_transpose_other((1024, 1024), (1024, 1024))
    _run_transpose_input((2048, 1), (2048, 1024))
    _run_transpose_input((1024, 1024), (1024, 1024))


def test_batch_norm_gather_stats_with_counts_boundary():
    """
    Feature: BatchNormGatherStatsWithCounts shape-input boundary
    Description: later ops should see the original count shape after the op flushes
    Expectation: the result matches the baseline result
    """

    def _run(rank, input_shape, data_type):
        flag = gen_flag("batch_norm_gather_stats_with_counts_boundary", rank, input_shape, data_type)
        n, c, h, w = input_shape
        np_type, ms_type = trans_data_type(data_type)
        mean_np = np.random.normal(0, 1, (rank, c)).astype(np_type)
        invstd_np = np.abs(np.random.normal(0, 1, (rank, c)).astype(np_type)) + 1e-5
        counts_np = np.array([[(n + i) * h * w] for i in range(rank)]).astype(np_type)
        global_stats = Tensor(np.concatenate((mean_np, invstd_np, counts_np), axis=-1), ms_type).to("Ascend")
        x = gen_input(input_shape, data_type)
        count_delta = gen_input((rank,), data_type)
        count_bias = Tensor(np.ones((rank,), dtype=np_type), ms_type).to("Ascend")

        mean_all, invstd_all, count_all = mint.split(global_stats, c, 1)
        count_input = ops.add(count_all.view(-1).contiguous(), count_bias)
        global_mean, global_invstd = ops.auto_generate.BatchNormGatherStatsWithCounts()(
            x, mean_all, invstd_all, None, None, 0.1, 1e-5, count_input
        )
        count_reuse = ops.add(count_input, count_delta)
        compare_outputs(flag, [global_mean, global_invstd, count_reuse], cmp_precision=1e-4)

    _run(4, (2, 8, 3, 4), "float32")


def test_batch_norm_elemt_shape_input_reuse():
    """
    Feature: BatchNormElemt shape-input reuse
    Description: later fused ops should read the original 1D tensors instead of the broadcast DVM shapes
    Expectation: the result matches the baseline result
    """

    def _run(input_shape, data_type):
        flag = gen_flag("batch_norm_elemt_shape_input_reuse", input_shape, data_type)
        c = input_shape[1]
        x = gen_input(input_shape, data_type)
        weight = _make_elemwise_input((c,), data_type)
        bias = _make_elemwise_input((c,), data_type)
        mean = _make_elemwise_input((c,), data_type)
        invstd = _make_elemwise_input((c,), data_type, is_positive=True)
        mean_delta = gen_input((c,), data_type)
        invstd_delta = gen_input((c,), data_type)
        weight_delta = gen_input((c,), data_type)
        bias_delta = gen_input((c,), data_type)

        out = ops.auto_generate.BatchNormElemt()(x, weight, bias, mean, invstd, 1e-5)
        mean_reuse = ops.add(mean, mean_delta)
        invstd_reuse = ops.add(invstd, invstd_delta)
        weight_reuse = ops.add(weight, weight_delta)
        bias_reuse = ops.add(bias, bias_delta)
        compare_outputs(flag, [out, mean_reuse, invstd_reuse, weight_reuse, bias_reuse], cmp_precision=1e-4)

    _run((2, 8, 3, 4), "float32")
    _run((4, 16, 2, 2), "float32")


def test_batch_norm_elemt_grad_shape_input_reuse():
    """
    Feature: BatchNormElemtGrad shape-input reuse
    Description: later fused ops should read the original reduction tensors instead of the broadcast DVM shapes
    Expectation: the result matches the baseline result
    """

    def _run(rank, input_shape, data_type):
        flag = gen_flag("batch_norm_elemt_grad_shape_input_reuse", rank, input_shape, data_type)
        n, c, h, w = input_shape
        np_type, ms_type = trans_data_type(data_type)
        dout = gen_input(input_shape, data_type)
        x = gen_input(input_shape, data_type)
        mean = _make_elemwise_input((c,), data_type)
        invstd = _make_elemwise_input((c,), data_type, is_positive=True)
        weight = _make_elemwise_input((c,), data_type)
        sumd_dy = _make_elemwise_input((c,), data_type)
        sum_dy_xmu = _make_elemwise_input((c,), data_type)
        counts_np = np.array([[(n + i) * h * w] for i in range(rank)]).astype(np_type)
        count = Tensor(counts_np, ms_type).to("Ascend")
        mean_delta = gen_input((c,), data_type)
        invstd_delta = gen_input((c,), data_type)
        weight_delta = gen_input((c,), data_type)
        sumd_dy_delta = gen_input((c,), data_type)
        sum_dy_xmu_delta = gen_input((c,), data_type)

        out = ops.auto_generate.BatchNormElemtGrad()(dout, x, mean, invstd, weight, sumd_dy, sum_dy_xmu, count)
        mean_reuse = ops.add(mean, mean_delta)
        invstd_reuse = ops.add(invstd, invstd_delta)
        weight_reuse = ops.add(weight, weight_delta)
        sumd_dy_reuse = ops.add(sumd_dy, sumd_dy_delta)
        sum_dy_xmu_reuse = ops.add(sum_dy_xmu, sum_dy_xmu_delta)
        compare_outputs(flag, [out, mean_reuse, invstd_reuse, weight_reuse,
                               sumd_dy_reuse, sum_dy_xmu_reuse], cmp_precision=1e-4)

    _run(4, (2, 8, 3, 4), "float32")


def test_input_cast_cache_transition():
    """
    Feature: Input cache cast transition
    Description: a later Input(false) should still match the baseline after a BF16 elemwise op cached the tensor
    Expectation: the result matches the baseline result
    """

    flag = gen_flag("input_cast_cache_transition", "bfloat16")
    x = gen_input((16, 32), "bfloat16")
    other = gen_input((16, 32), "bfloat16")

    out = ops.add(x, other)
    cast_out = ops.cast(x, mindspore.float16)
    compare_outputs(flag, [out, cast_out], cmp_precision=_get_cmp_precision("bfloat16"))
