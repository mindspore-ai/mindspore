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

"""
dvm op test cases in pynative mode
"""

import numpy as np
from mindspore import ops
from mindspore import context
from mindspore import mint
from mindspore import Tensor, Parameter
from tests.st.graph_kernel.gk_utils import gen_flag, trans_data_type, gen_input, get_func_name, compare_outputs

np.random.seed(1)
context.set_context(mode=context.PYNATIVE_MODE)


def test_concat():
    """
    Feature: concat
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(split_axis, split_num, concat_axis):
        flag = gen_flag("concat", split_axis, split_num, concat_axis)
        y0 = ops.auto_generate.Split(split_axis, split_num)(x0)
        y1 = ops.auto_generate.Concat(concat_axis)(y0)
        compare_outputs(flag, y1)

    x = np.random.normal(0, 1, (4, 124, 80)).astype(np.float32)
    x0 = Tensor(x)

    _run(2, 8, 0)
    _run(2, 8, -3)
    _run(2, 8, 1)


def test_cast():
    """
    Feature: cast
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(src_type, dst_type):
        flag = gen_flag("cast", src_type, dst_type)
        _, ms_dst_type = trans_data_type(dst_type)
        x0 = gen_input((10, 80), src_type)
        y0 = ops.cast(x0, ms_dst_type)
        compare_outputs(flag, y0)

    for d1 in ["float32", "float16", "bfloat16", "int32", "bool"]:
        for d2 in ["float32", "float16", "bfloat16", "int32", "bool"]:
            _run(d1, d2)


def test_unary():
    """
    Feature: unary op
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(func, data_type, cmp_precision=0.0):
        flag = gen_flag(get_func_name(func), data_type)
        x0 = gen_input((10, 80), data_type)
        y0 = func(x0)
        compare_outputs(flag, y0, cmp_precision=cmp_precision)

    for item in [[ops.abs, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.neg, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.exp, ["float32", "float16", "bfloat16"]],
                 [ops.sqrt, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.reciprocal, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.round, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.ceil, ["float32", "float16", "bfloat16"]],
                 [ops.floor, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.trunc, ["float32", "float16", "bfloat16"]],
                 [ops.logical_not, ["float32", "float16", "bfloat16", "int32", "bool"]],
                 [ops.sigmoid, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.silu, ["float32", "float16", "bfloat16"]],
                 [ops.GeLU(), ["float32", "float16", "bfloat16"], 1e-4],
                 [ops.relu, ["float32", "float16", "bfloat16", "int32"]]]:
        for d in item[1]:
            if len(item) == 3:
                _run(item[0], d, item[2])
            else:
                _run(item[0], d)


def test_isfinite():
    """
    Feature: isfinite
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(data_type):
        flag = gen_flag("isfinite", data_type)
        _, ms_type = trans_data_type(data_type)
        x0 = Tensor(x, ms_type)
        y0 = ops.isfinite(x0)
        compare_outputs(flag, y0)

    x = np.array([1.0, 2.0, np.inf, 2.1, -np.inf, 0.0, np.nan]).astype(np.float32)

    _run("float32")
    _run("float16")
    _run("bfloat16")


def test_binary():
    """
    Feature: binary op
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(func, data_type):
        flag = gen_flag(get_func_name(func), data_type)
        x0 = gen_input((10, 80), data_type)
        x1 = gen_input((10, 1), data_type)
        y0 = func(x0, x1)
        compare_outputs(flag, y0)

    for item in [[ops.equal, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.not_equal, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.greater, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.greater_equal, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.less, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.less_equal, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.add, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.mul, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.sub, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.div, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.pow, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.maximum, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.minimum, ["float32", "float16", "bfloat16", "int32"]],
                 [ops.logical_and, ["float32", "float16", "bfloat16", "int32", "bool"]],
                 [ops.logical_or, ["float32", "float16", "bfloat16", "int32", "bool"]]]:
        for d in item[-1]:
            _run(item[0], d)


def test_grad_op():
    """
    Feature: grad op
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(func, input_num, data_type):
        flag = gen_flag(get_func_name(func), data_type)
        inputs = [gen_input((10, 80), data_type) for _ in range(input_num)]
        y0 = func(*inputs)
        compare_outputs(flag, y0, cmp_precision=1e-4)

    for item in [[ops.auto_generate.SigmoidGrad(), 2, ["float32", "float16", "bfloat16"]],
                 [ops.auto_generate.SiLUGrad(), 2, ["float32", "float16", "bfloat16"]],
                 [ops.auto_generate.GeLUGrad(), 3, ["float32", "float16", "bfloat16"]]]:
        for d in item[-1]:
            _run(item[0], item[1], d)


def test_tile():
    """
    Feature: tile
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(data_type, dims):
        flag = gen_flag("tile", data_type, dims)
        x0 = gen_input((1, 512), data_type)
        y0 = ops.Tile()(x0, dims)
        compare_outputs(flag, y0)

    for t in ["float32", "float16", "bfloat16", "int32"]:
        for d in [(1, 1), (1, 2), (2, 1), (2, 4)]:
            _run(t, d)


def test_sum_ext():
    """
    Feature: sum_ext
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(dims, keepdim):
        flag = gen_flag("sum_ext", dims, keepdim)
        y0 = ops.auto_generate.SumExt()(x0, dims, keepdim)
        compare_outputs(flag, y0, cmp_precision=1e-4)

    x0 = gen_input((32, 1024), "float32")
    _run((0,), False)
    _run((1,), True)
    _run((0, 1), False)


def test_linalg_vector_norm():
    """
    Feature: linalg_vector_norm
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(ord_value, dims, keepdim):
        flag = gen_flag("linalg_vector_norm", ord_value, dims, keepdim)
        y0 = ops.auto_generate.LinalgVectorNorm()(x0, ord_value, dims, keepdim)
        compare_outputs(flag, y0, cmp_precision=1e-4)

    x0 = gen_input((100, 10), "float32")
    for o in [0, 1, 2, 3]:
        _run(o, (0,), True)
        _run(o, (1,), True)
        _run(o, None, False)


def test_inplace_unary():
    """
    Feature: inplace unary op
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(func, data_type):
        flag = gen_flag(get_func_name(func), data_type)
        x0 = Parameter(gen_input((10, 80), data_type), name=flag)
        y0 = func(x0)
        compare_outputs(flag, [y0, x0])

    for item in [[ops.auto_generate.InplaceExp(), ["float32", "float16", "bfloat16"]],
                 [ops.auto_generate.InplaceReLU(), ["float32", "float16", "bfloat16", "int32"]]]:
        for d in item[-1]:
            _run(item[0], d)


def test_inplace_binary():
    """
    Feature: inplace binary op
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(func, data_type1, data_type2):
        flag = gen_flag(get_func_name(func), data_type1, data_type2)
        x0 = Parameter(gen_input((10, 80), data_type1), name=flag)
        x1 = gen_input((10, 80), data_type2)
        y0 = func(x0, x1)
        compare_outputs(flag, [y0, x0])

    for f in [ops.auto_generate.InplaceDiv()]:
        for d1 in ["float32", "float16", "bfloat16", "int32"]:
            for d2 in ["float32", "float16", "bfloat16", "int32"]:
                if d1 == "int32" and d2 != d1:
                    continue
                _run(f, d1, d2)


def test_muls():
    """
    Feature: muls
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(data_type, scalar):
        flag = gen_flag("muls", data_type, scalar)
        x0 = gen_input((10, 80), data_type)
        y0 = ops.auto_generate.Muls()(x0, scalar)
        compare_outputs(flag, y0)

    for t in ["float32", "float16", "bfloat16", "int32"]:
        for s in [2, 2.3, 2.176532]:
            _run(t, s)


def test_binary_ext():
    """
    Feature: AddExt, SubExt
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(func, data_type1, data_type2, scalar):
        if data_type1 == "int32" and data_type2 == "int32" and isinstance(scalar, float):
            return
        if data_type1 == "bfloat16" and data_type2 == "float32" and isinstance(scalar, int):
            return
        flag = gen_flag(get_func_name(func), data_type1, data_type2, scalar)
        x0 = gen_input((4, 2, 7, 2, 6, 4), data_type1)
        x1 = gen_input((1, 2, 7, 2, 6, 4), data_type2)
        y0 = func(x0, x1, scalar)
        compare_outputs(flag, y0)

    for f in [ops.auto_generate.AddExt(), ops.auto_generate.SubExt()]:
        for d1 in ["float32", "float16", "bfloat16", "int32"]:
            for d2 in ["float32", "float16", "bfloat16", "int32"]:
                for s in [2, 1, 2.324561]:
                    _run(f, d1, d2, s)


def test_inplace_binary_ext():
    """
    Feature: InplaceAddExt, InplaceSubExt
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(func, data_type1, data_type2, scalar):
        if data_type1 == "int32" and data_type2 in ["float32", "float16", "bfloat16"]:
            return
        if data_type1 == "int32" and data_type2 == "int32" and isinstance(scalar, float):
            return
        if data_type2 == "int32" and data_type1 in ["float32", "float16", "bfloat16"] and isinstance(scalar, float):
            return
        flag = gen_flag(get_func_name(func), data_type1, data_type2, scalar)
        x0 = gen_input((4, 2, 7, 2, 6, 4), data_type1)
        x1 = gen_input((1, 2, 7, 2, 6, 4), data_type2)
        y0 = func(x0, x1, scalar)
        compare_outputs(flag, y0)

    for f in [ops.auto_generate.InplaceAddExt(), ops.auto_generate.InplaceSubExt()]:
        for d1 in ["float32", "float16", "bfloat16", "int32"]:
            for d2 in ["float32", "float16", "bfloat16", "int32"]:
                for s in [2, 1, 2.324561]:
                    _run(f, d1, d2, s)


def test_batch_norm_stats():
    """
    Feature: BatchNormStats
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(shape, eps, data_type):
        flag = gen_flag("batch_norm_stats", shape)
        x0 = gen_input(shape, data_type)
        outputs = ops.auto_generate.BatchNormStats()(x0, eps)
        compare_outputs(flag, outputs, cmp_precision=1e-4)

    _run((1, 32, 2, 4), 1e-5, "float32")
    _run((30, 32, 176, 320), 1e-5, "float32")


def test_batch_gather_stats_with_counts():
    """
    Feature: BatchNormGatherStatsWithCounts
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(rank, input_shape, running_mean_shape, running_var_shape, momentum, eps, data_type):
        flag = gen_flag("batch_gather_stats_with_counts", rank, input_shape, running_mean_shape, running_var_shape,
                        momentum, eps, data_type)
        n, c, h, w = input_shape
        np_type, ms_type = trans_data_type(data_type)
        mean_np = np.random.normal(0, 1, (rank, c)).astype(np_type)
        invstd_np = np.abs(np.random.normal(0, 1, (rank, c)).astype(np_type)) + 1e-5
        counts = [[(n + i) * h * w] for i in range(rank)]
        counts_np = np.array(counts).astype(np_type)
        global_stats_np = np.concatenate((mean_np, invstd_np, counts_np), axis=-1)
        x0 = gen_input(input_shape, data_type)
        global_stats = Tensor(global_stats_np, ms_type)
        running_mean = Parameter(gen_input(running_mean_shape, data_type),
                                 name="{}_{}".format(flag, "running_mean")) if running_mean_shape else None
        running_var = Parameter(gen_input(running_var_shape, data_type),
                                name="{}_{}".format(flag, "running_var")) if running_var_shape else None
        mean_all, invstd_all, count_all = mint.split(global_stats, c, 1)
        outputs = ops.auto_generate.BatchNormGatherStatsWithCounts()(x0, mean_all, invstd_all, running_mean,
                                                                     running_var, momentum, eps, count_all.view(-1))
        outputs = list(outputs)
        outputs.append(running_mean)
        outputs.append(running_var)
        compare_outputs(flag, outputs, cmp_precision=1e-4)

    for s in [(1, 32, 2, 4), (30, 32, 176, 320)]:
        _run(4, s, (s[1],), (s[1],), 0.1, 1e-5, "float32")
        _run(4, s, None, None, 0.1, 1e-5, "float32")


def test_batch_norm_element():
    """
    Feature: BatchNormElemt
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(input_shape, weight_shape, bias_shape, mean_shape, invstd_shape, eps, data_type):
        flag = gen_flag("batch_norm_element", input_shape, weight_shape, bias_shape, mean_shape, invstd_shape, eps,
                        data_type)
        x0 = gen_input(input_shape, data_type)
        weight = gen_input(weight_shape, data_type) if weight_shape else None
        bias = gen_input(bias_shape, data_type) if bias_shape else None
        mean = gen_input(mean_shape, data_type) if mean_shape else None
        invstd = gen_input(invstd_shape, data_type, is_positive=True) if invstd_shape else None
        outputs = ops.auto_generate.BatchNormElemt()(x0, weight, bias, mean, invstd, eps)
        compare_outputs(flag, outputs, cmp_precision=1e-4)

    for s in [(1, 32, 2, 4), (30, 32, 176, 320)]:
        c = s[1]
        _run(s, (c,), (c,), (c,), (c,), 1e-5, "float32")
        _run(s, None, (c,), (c,), (c,), 1e-5, "float32")
        _run(s, (c,), None, (c,), (c,), 1e-5, "float32")
        _run(s, None, None, (c,), (c,), 1e-5, "float32")


def test_batch_norm_element_grad():
    """
    Feature: BatchNormElemtGrad
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(rank, input_shape, data_type):
        flag = gen_flag("batch_norm_element_grad", input_shape, data_type)
        n, c, h, w = input_shape
        np_type, ms_type = trans_data_type(data_type)
        dout = gen_input(input_shape, data_type)
        x = gen_input(input_shape, data_type)
        mean = gen_input((c,), data_type)
        invstd = gen_input((c,), data_type, is_positive=True)
        weight = gen_input((c,), data_type)
        sumd_dy = gen_input((c,), data_type)
        sum_dy_xmu = gen_input((c,), data_type)
        counts = [[(n + i) * h * w] for i in range(rank)]
        counts_np = np.array(counts).astype(np_type)
        count = Tensor(counts_np, ms_type)
        outputs = ops.auto_generate.BatchNormElemtGrad()(dout, x, mean, invstd, weight, sumd_dy, sum_dy_xmu, count)
        compare_outputs(flag, outputs, cmp_precision=1e-4)

    _run(4, (1, 32, 2, 4), "float32")
    _run(4, (30, 32, 176, 320), "float32")
