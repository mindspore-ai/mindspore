# Copyright 2025 Huawei Technologies Co., Ltd
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
"""DVM matmul operator tests in pynative mode."""

import numpy as np
import mindspore
from mindspore import ops
from mindspore import context
from mindspore import mint
from tests.st.graph_kernel.gk_utils import gen_flag, gen_input, get_func_name, compare_outputs

np.random.seed(1)
context.set_context(mode=context.PYNATIVE_MODE)


def test_dense():
    """
    Feature: Dense
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(shape1, shape2, shape_bias, data_type):
        flag = gen_flag("dense", shape1, shape2, shape_bias, data_type)
        x0 = gen_input(shape1, data_type, data_range=0.01)
        x1 = gen_input(shape2, data_type, data_range=0.01)
        x2 = gen_input(shape_bias, data_type, data_range=0.01) if shape_bias else None
        y0 = ops.auto_generate.Dense()(x0, x1, x2)
        compare_outputs(flag, y0, cmp_precision=4e-3 if data_type == "bfloat16" else 1e-3)

    for d in ["float16", "bfloat16"]:
        _run((4096, 7168), (2112, 7168), (2112,), d)
        _run((4096, 7168), (2112, 7168), None, d)


def test_matmul():
    """
    Feature: MatMul, BatchMatMul
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(func, shape1, shape2, data_type, transpose_a, transpose_b):
        flag = gen_flag(get_func_name(func), shape1, shape2, data_type, transpose_a, transpose_b)
        x0 = gen_input(shape1, data_type, data_range=0.01)
        x1 = gen_input(shape2, data_type, data_range=0.01)
        y0 = func(transpose_a, transpose_b)(x0, x1)
        compare_outputs(flag, y0, cmp_precision=4e-3 if data_type == "bfloat16" else 1e-3)

    _run(ops.auto_generate.MatMul, (4096, 1536), (16384, 1536), "float16", False, True)
    _run(ops.auto_generate.MatMul, (4096, 1536), (1536, 8192), "bfloat16", False, False)
    _run(ops.auto_generate.BatchMatMul, (4, 4096, 1536), (1, 8192, 1536), "float16", False, True)
    _run(ops.auto_generate.BatchMatMul, (4096, 1024), (4, 8, 1024, 1024), "bfloat16", False, True)


def test_matmul_ext():
    """
    Feature: MatMulExt, BatchMatMulExt
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(func, shape1, shape2, data_type):
        flag = gen_flag(get_func_name(func), shape1, shape2, data_type)
        x0 = gen_input(shape1, data_type, data_range=0.01)
        x1 = gen_input(shape2, data_type, data_range=0.01)
        y0 = func(x0, x1)
        compare_outputs(flag, y0, cmp_precision=4e-3 if data_type == "bfloat16" else 1e-3)

    _run(mint.matmul, (2, 4096, 1024), (1024, 2048), "float16")
    _run(mint.matmul, (4096, 1024), (4, 1024, 2048), "bfloat16")
    _run(ops.auto_generate.BatchMatMulExt(), (4, 4096, 7168), (4, 7168, 2112), "float16")
    _run(ops.auto_generate.BatchMatMulExt(), (4, 4096, 7168), (4, 7168, 2112), "bfloat16")


def test_matmul_ext_view():
    """
    Feature: view + matmul
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run_transpose(func, shape1, shape2, data_type, trans_axis):
        flag = gen_flag(get_func_name(func), "transpose", shape1, shape2, data_type, trans_axis)
        x0 = gen_input(shape1, data_type, data_range=0.01)
        x1 = gen_input(shape2, data_type, data_range=0.01)
        y0 = func(x0, ops.Transpose()(x1, trans_axis))
        compare_outputs(flag, y0, cmp_precision=4e-3 if data_type == "bfloat16" else 1e-3)

    _run_transpose(mint.matmul, (2, 4096, 1024), (2048, 1024), "float16", (1, 0))
    _run_transpose(mint.matmul, (4096, 1024), (4, 2048, 1024), "float16", (0, 2, 1))
    _run_transpose(ops.auto_generate.BatchMatMulExt(), (4, 4096, 7168), (7168, 4, 2112), "float16", (1, 0, 2))


def test_matmul_to_contiguous_fallback():
    """
    Feature: MatMul, BatchMatMul ToContiguous fallback
    Description: non-contiguous non-transpose views should fall back to ToContiguous before entering DVM matmul
    Expectation: the result match with the expected result
    """

    def _run_matmul_check_x(data_type):
        flag = gen_flag("matmul_to_contiguous_check_x_fallback", data_type)
        x_base = gen_input((512, 1536), data_type, data_range=0.01)
        x_view = x_base[:, ::2]
        y = gen_input((768, 640), data_type, data_range=0.01)
        out = ops.auto_generate.MatMul(False, False)(x_view, y)
        compare_outputs(flag, out, cmp_precision=4e-3 if data_type == "bfloat16" else 1e-3)

    def _run_matmul_check_y(data_type):
        flag = gen_flag("matmul_to_contiguous_check_y_fallback", data_type)
        x = gen_input((512, 768), data_type, data_range=0.01)
        y_base = gen_input((1536, 640), data_type, data_range=0.01)
        y_view = y_base[::2, :]
        out = ops.auto_generate.MatMul(False, False)(x, y_view)
        compare_outputs(flag, out, cmp_precision=4e-3 if data_type == "bfloat16" else 1e-3)

    def _run_batch_matmul_check_x(data_type):
        flag = gen_flag("batch_matmul_to_contiguous_check_x_fallback", data_type)
        x_base = gen_input((2, 512, 1536), data_type, data_range=0.01)
        x_view = x_base[:, :, ::2]
        y = gen_input((2, 768, 640), data_type, data_range=0.01)
        out = ops.auto_generate.BatchMatMul(False, False)(x_view, y)
        compare_outputs(flag, out, cmp_precision=4e-3 if data_type == "bfloat16" else 1e-3)

    def _run_batch_matmul_check_y(data_type):
        flag = gen_flag("batch_matmul_to_contiguous_check_y_fallback", data_type)
        x = gen_input((2, 512, 768), data_type, data_range=0.01)
        y_base = gen_input((2, 1536, 640), data_type, data_range=0.01)
        y_view = y_base[:, ::2, :]
        out = ops.auto_generate.BatchMatMul(False, False)(x, y_view)
        compare_outputs(flag, out, cmp_precision=4e-3 if data_type == "bfloat16" else 1e-3)

    _run_matmul_check_x("float16")
    _run_matmul_check_y("float16")
    _run_batch_matmul_check_x("float16")
    _run_batch_matmul_check_y("float16")


def test_grouped_matmul_splititem3_grouptype0():
    """
    Feature: GroupedMatmul split_item=3, group_type=0
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(m, k, n, has_bias, group_list, transpose_a, transpose_b, data_type):
        flag = gen_flag("GroupedMatmul_3_0", m, k, n, has_bias, group_list, transpose_a, transpose_b, data_type)
        e = len(group_list)
        shape1 = (k, m) if transpose_a else (m, k)
        shape2 = (e, n, k) if transpose_b else (e, k, n)
        x = gen_input(shape1, data_type, data_range=0.01)
        weight = gen_input(shape2, data_type, data_range=0.01)
        bias = [gen_input((e, n), data_type, data_range=0.01)] if has_bias else None
        group_list_ms = mindspore.Tensor(group_list, dtype=mindspore.int64)
        if transpose_a:
            x = mint.transpose(x, -1, -2)
        if transpose_b:
            weight = mint.transpose(weight, -1, -2)
        op = ops.auto_generate.GroupedMatmul(3, 0, False, False)
        y0 = op([x], [weight], bias, None, None, None, None, group_list_ms)
        compare_outputs(flag, y0, cmp_precision=4e-3 if data_type == "bfloat16" else 1e-3)

    # does not support x transposed
    # transpose_a/transpose_b must be False
    # does not support bfloat16
    _run(320, 256, 128, True, [0, 10, 30, 100, 140, 180, 220, 220, 240, 320], False, False, "float16")
    _run(320, 256, 128, True, [0, 10, 30, 100, 140, 180, 220, 220, 240, 320], False, True, "float16")
    _run(320, 256, 128, False, [0, 10, 30, 100, 140, 180, 220, 220, 240, 320], False, False, "float16")
    _run(320, 256, 128, True, [320], False, False, "float16")
    _run(320, 256, 128, False, [320], False, False, "float16")
    _run(320, 256, 128, True, [256, 320], False, False, "float16")
    _run(320, 256, 128, False, [256, 320], False, False, "float16")


def test_grouped_matmul_splititem3_grouptype2():
    """
    Feature: GroupedMatmul split_item=3, group_type=2
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(m, k, n, group_list, transpose_a, transpose_b, data_type):
        flag = gen_flag("GroupedMatmul_3_2", m, k, n, group_list, transpose_a, transpose_b, data_type)
        shape1 = (k, m) if transpose_a else (m, k)
        shape2 = (n, k) if transpose_b else (k, n)
        x = gen_input(shape1, data_type, data_range=0.01)
        weight = gen_input(shape2, data_type, data_range=0.01)
        group_list_ms = mindspore.Tensor(group_list, dtype=mindspore.int64)
        if transpose_a:
            x = mint.transpose(x, -1, -2)
        if transpose_b:
            weight = mint.transpose(weight, -1, -2)
        op = ops.auto_generate.GroupedMatmul(3, 2, False, False)
        y0 = op([x], [weight], None, None, None, None, None, group_list_ms)
        compare_outputs(flag, y0, cmp_precision=4e-3 if data_type == "bfloat16" else 1e-3)

    # does not support bias
    # if x is not separated, x must be transposed, weight can not be transposed
    # transpose_a/transpose_b must be False
    _run(1024, 2560, 1024, [0, 10, 300, 1024, 1024, 1400, 1800, 2048, 2048, 2200, 2560], True, False, "float16")
    _run(1024, 2560, 1024, [0, 10, 300, 1024, 1100, 1400, 1800, 2048, 2100, 2200, 2560], True, False, "float16")
    _run(1024, 2560, 1024, [0, 10, 300, 1024, 1024, 1400, 1800, 2048, 2048, 2200, 2560], True, False, "bfloat16")
    _run(1024, 2560, 1024, [0, 10, 300, 1024, 1100, 1400, 1800, 2048, 2100, 2200, 2560], True, False, "bfloat16")
