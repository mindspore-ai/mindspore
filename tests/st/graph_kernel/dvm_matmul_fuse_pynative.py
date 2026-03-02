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

"""
DVM CV fusion test cases in pynative mode.
"""
import numpy as np
import mindspore
from mindspore import ops
from mindspore import context
from mindspore import mint
from tests.st.graph_kernel.gk_utils import gen_flag, gen_input, compare_outputs

np.random.seed(1)
context.set_context(mode=context.PYNATIVE_MODE)


def _cmp_precision(data_type):
    return 4e-3 if data_type == "bfloat16" else 1e-3


def _run_matmul_elemwise(data_type):
    flag = gen_flag("_run_matmul_elemwise", data_type)
    x0 = gen_input((2048, 1024), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((1024, 512), data_type, data_range=0.01).to("Ascend")
    x2 = gen_input((2048, 512), data_type, data_range=0.01).to("Ascend")
    y0 = ops.add(ops.auto_generate.MatMul(False, False)(x0, x1), x2)
    compare_outputs(flag, y0, cmp_precision=_cmp_precision(data_type))


def _run_matmul_same_input_elemwise(data_type):
    flag = gen_flag("_run_matmul_same_input_elemwise", data_type)
    x0 = gen_input((1024, 1024), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((1024, 1024), data_type, data_range=0.01).to("Ascend")
    x2 = gen_input((1024, 1024), data_type, data_range=0.01).to("Ascend")
    y0 = mint.transpose(x0, -1, -2)
    output = ops.auto_generate.MatMul(False, False)(y0, y0)
    output = ops.add(output, x1)
    output = ops.add(output, x2)
    compare_outputs(flag, output, cmp_precision=_cmp_precision(data_type))


def _run_matmul_elemwise_same_input(data_type):
    flag = gen_flag("_run_matmul_elemwise_same_input", data_type)
    x0 = gen_input((1024, 1024), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((1024, 1024), data_type, data_range=0.01).to("Ascend")
    output = ops.auto_generate.MatMul(False, False)(x0, x1)
    output = ops.add(output, x1)
    output = ops.add(output, x1)
    compare_outputs(flag, output, cmp_precision=_cmp_precision(data_type))


def _run_matmul_inplace1(data_type):
    flag = gen_flag("_run_matmul_inplace1", data_type)
    x0 = gen_input((512, 1024), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((512, 448), data_type, data_range=0.01).to("Ascend")
    x2 = gen_input((1024, 448), data_type, data_range=0.01).to("Ascend")
    y0 = ops.auto_generate.MatMul(True, False)(x0, x1)
    y0 = ops.auto_generate.InplaceAddExt()(y0, x2)
    compare_outputs(flag, y0, cmp_precision=_cmp_precision(data_type))


def _run_matmul_inplace2(data_type):
    flag = gen_flag("_run_matmul_inplace2", data_type)
    x0 = gen_input((512, 1024), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((512, 448), data_type, data_range=0.01).to("Ascend")
    x2 = gen_input((1024, 448), data_type, data_range=0.01).to("Ascend")
    y0 = ops.auto_generate.InplaceAddExt()(x2, ops.auto_generate.MatMul(True, False)(x0, x1))
    compare_outputs(flag, y0, cmp_precision=_cmp_precision(data_type))


def _run_inplace_matmul(data_type):
    flag = gen_flag("_run_inplace_matmul", data_type)
    x0 = gen_input((512, 1024), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((512, 448), data_type, data_range=0.01).to("Ascend")
    x0 = ops.auto_generate.InplaceReLU()(x0)
    y0 = ops.auto_generate.MatMul(True, False)(x0, x1)
    compare_outputs(flag, y0, cmp_precision=_cmp_precision(data_type))


def _run_matmul_elemwise_inplace(data_type):
    flag = gen_flag("_run_matmul_elemwise_inplace", data_type)
    x0 = gen_input((512, 1792), data_type, data_range=0.1).to("Ascend")
    x1 = gen_input((512, 448), data_type, data_range=0.1).to("Ascend")
    x2 = gen_input((1792, 448), "float32", data_range=0.1).to("Ascend")
    x2 += ops.cast(ops.auto_generate.MatMul(True, False)(x0, x1), mindspore.float32)
    compare_outputs(flag, x2, cmp_precision=_cmp_precision(data_type))


def _run_matmul_elemwise_broadcast1(data_type):
    flag = gen_flag("_run_matmul_elemwise_broadcast1", data_type)
    x0 = gen_input((1024, 512), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((512, 256), data_type, data_range=0.01).to("Ascend")
    x2 = gen_input((1, 256), data_type, data_range=0.01).to("Ascend")
    y0 = ops.add(ops.auto_generate.MatMul(False, False)(x0, x1), ops.abs(x2))
    compare_outputs(flag, y0, cmp_precision=_cmp_precision(data_type))


def _run_matmul_elemwise_broadcast2(data_type):
    flag = gen_flag("_run_matmul_elemwise_broadcast2", data_type)
    x0 = gen_input((1024, 512), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((512, 256), data_type, data_range=0.01).to("Ascend")
    x2 = gen_input((1, 256), data_type, data_range=0.01).to("Ascend")
    y0 = ops.auto_generate.MatMul(False, False)(x0, x1)
    y1 = ops.abs(x2)
    y0 = ops.add(y0, y1)
    compare_outputs(flag, [y0, y1], cmp_precision=_cmp_precision(data_type))


def _run_matmul_elemwise_reduce(data_type):
    flag = gen_flag("_run_matmul_elemwise_reduce", data_type)
    x0 = gen_input((512, 1024), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((1024, 512), data_type, data_range=0.01).to("Ascend")
    y0 = mint.matmul(x0, x1)
    x0 = gen_input((1024, 512), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((512, 512), data_type, data_range=0.01).to("Ascend")
    y1 = mint.matmul(x0, x1)
    x0 = gen_input((1024, 512), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((512, 512), data_type, data_range=0.01).to("Ascend")
    y2 = mint.matmul(x0, x1)
    x0 = gen_input((1024, 512), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((1024, 512), data_type, data_range=0.01).to("Ascend")
    y3 = ops.auto_generate.MatMul(True, False)(x0, x1)
    y3 = ops.cast(y3, mindspore.float32)
    y3 = ops.auto_generate.SumExt()(y3, (0,), True)
    compare_outputs(flag, [y0, y1, y2, y3], cmp_precision=_cmp_precision(data_type))


def _run_matmul_matmul(data_type):
    flag = gen_flag("_run_matmul_matmul", data_type)
    x0 = gen_input((512, 1024), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((1024, 512), data_type, data_range=0.01).to("Ascend")
    x2 = gen_input((512, 256), data_type, data_range=0.01).to("Ascend")
    y0 = ops.auto_generate.MatMul(False, False)(x0, x1)
    y1 = ops.auto_generate.MatMul(False, False)(y0, x2)
    compare_outputs(flag, y1, cmp_precision=_cmp_precision(data_type))


def _run_matmul_elemwise_matmul(data_type):
    flag = gen_flag("_run_matmul_elemwise_matmul", data_type)
    x0 = gen_input((512, 1024), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((1024, 512), data_type, data_range=0.01).to("Ascend")
    x2 = gen_input((1, 512), data_type, data_range=0.01).to("Ascend")
    x3 = gen_input((512, 256), data_type, data_range=0.01).to("Ascend")
    y1 = ops.mul(ops.auto_generate.MatMul(False, False)(x0, x1), x2)
    y2 = ops.auto_generate.MatMul(False, False)(y1, x3)
    compare_outputs(flag, y2, cmp_precision=_cmp_precision(data_type))


def _run_batch_matmul_elemwise(data_type):
    flag = gen_flag("_run_batch_matmul_elemwise", data_type)
    x0 = gen_input((4, 2048, 1024), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((4, 1024, 512), data_type, data_range=0.01).to("Ascend")
    x2 = gen_input((512,), "float32", data_range=0.01).to("Ascend")
    y0 = ops.add(ops.cast(ops.auto_generate.BatchMatMul(False, False)(x0, x1), mindspore.float32), ops.abs(x2))
    compare_outputs(flag, y0, cmp_precision=_cmp_precision(data_type))


def _run_matmul_ext_elemwise(data_type):
    flag = gen_flag("_run_matmul_ext_elemwise", data_type)
    x0 = gen_input((2, 4096, 1024), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((1024, 2048), data_type, data_range=0.01).to("Ascend")
    y0 = ops.neg(ops.sigmoid(mint.matmul(x0, x1)))
    compare_outputs(flag, y0, cmp_precision=_cmp_precision(data_type))


def _run_grouped_matmul_elemwise_group_type0(data_type):
    flag = gen_flag("_run_grouped_matmul_elemwise_group_type0", data_type)
    m, k, n = 320, 256, 128
    group_list = [0, 10, 30, 100, 140, 180, 220, 220, 240, 320]
    x0 = gen_input((m, k), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((len(group_list), k, n), data_type, data_range=0.01).to("Ascend")
    x2 = gen_input((n,), data_type, data_range=0.01).to("Ascend")
    group_list_ms = mindspore.Tensor(group_list, dtype=mindspore.int64)
    op = ops.auto_generate.GroupedMatmul(3, 0, False, False)
    y0 = ops.add(op([x0], [x1], None, None, None, None, None, group_list_ms)[0], x2)
    compare_outputs(flag, y0, cmp_precision=_cmp_precision(data_type))


def _run_grouped_matmul_elemwise_group_type2(data_type):
    flag = gen_flag("_run_grouped_matmul_elemwise_group_type2", data_type)
    m, k, n = 1024, 2560, 1024
    group_list = [0, 10, 300, 1024, 1024, 1400, 1800, 2048, 2048, 2200, 2560]
    x0 = gen_input((k, m), data_type, data_range=0.01).to("Ascend")
    x1 = gen_input((k, n), data_type, data_range=0.01).to("Ascend")
    x2 = gen_input((1, n), data_type, data_range=0.01).to("Ascend")
    group_list_ms = mindspore.Tensor(group_list, dtype=mindspore.int64).to("Ascend")
    x0 = mint.transpose(x0, -1, -2)
    op = ops.auto_generate.GroupedMatmul(3, 2, False, False)
    y0 = ops.mul(op([x0], [x1], None, None, None, None, None, group_list_ms)[0], x2)
    compare_outputs(flag, y0, cmp_precision=_cmp_precision(data_type))


def test_cv_fusion_cases():
    """
    Feature: DVM CV fusion
    Description: pynative mode
    Expectation: the result match with the expected result
    """
    _run_matmul_elemwise("float16")
    _run_matmul_same_input_elemwise("float16")
    _run_matmul_elemwise_same_input("float16")
    _run_matmul_inplace1("float16")
    _run_matmul_inplace2("float16")
    _run_inplace_matmul("float16")
    _run_matmul_elemwise_inplace("float16")
    _run_matmul_elemwise_broadcast1("float16")
    _run_matmul_elemwise_broadcast2("float16")
    _run_matmul_elemwise_reduce("float16")
    _run_matmul_matmul("float16")
    _run_matmul_elemwise_matmul("bfloat16")
    _run_batch_matmul_elemwise("bfloat16")
    _run_matmul_ext_elemwise("bfloat16")
    _run_grouped_matmul_elemwise_group_type0("float16")
    _run_grouped_matmul_elemwise_group_type2("float16")
