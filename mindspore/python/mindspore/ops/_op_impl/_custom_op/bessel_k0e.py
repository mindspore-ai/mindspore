# Copyright 2022 Huawei Technologies Co., Ltd
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
"""BesseK0e op"""

from tbe import dsl
from te import tvm
from te.platform.fusion_manager import fusion_manager
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType
from .bessel_k0 import A, AA, B, BB, MAXNUM, TWO

bessel_k0e_op_info = TBERegOp("BesselK0e") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("bessel_k0e.so") \
    .compute_cost(10) \
    .kernel_name("bessel_k0e") \
    .partial_flag(True) \
    .op_pattern("formatAgnostic") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_None, DataType.F16_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None) \
    .get_op_info()


@op_info_register(bessel_k0e_op_info)
def _bessel_k0e_tbe():
    """BesselK0e TBE register"""
    return


def chebevl(x, n, coef, shape, dtype):
    """chebevl"""
    broad_coef = dsl.broadcast(coef[0], shape, dtype)
    broad_zero = dsl.broadcast(0, shape, dtype)
    none_signal = None
    for i in range(1, n):
        none_signal = broad_zero
        broad_zero = broad_coef
        coef_i = dsl.broadcast(coef[i], shape, dtype)
        broad_coef = dsl.vsub(dsl.vadd(dsl.vmul(x, broad_zero), coef_i), none_signal)
    return dsl.vmuls(dsl.vsub(broad_coef, none_signal), 0.5)


def bessel_i0_compute(input_x):
    """bessel_i0_compute"""
    dtype = input_x.dtype
    shape = input_x.shape

    k0e_has_improve_precision = False
    if dtype != "float32":
        input_x = dsl.cast_to(input_x, "float32")
        dtype = "float32"
        k0e_has_improve_precision = True

    y = dsl.vabs(input_x)

    y_le_eight_in = dsl.vmuls(y, 0.5)
    y_le_eight_in = dsl.vadds(y_le_eight_in, -2.0)
    y_le_eight = chebevl(y_le_eight_in, 30, AA, shape, dtype)

    y_gt_eight_in = dsl.vadds(dsl.vmuls(dsl.vrec(y), 32.0), -2.0)
    y_gt_eight = chebevl(y_gt_eight_in, 25, BB, shape, dtype)
    y_gt_eight = dsl.vmul(y_gt_eight, dsl.vrsqrt(y))

    res = dsl.vcmpsel(y, 8.0, 'le', y_le_eight, y_gt_eight)
    res = dsl.vmul(res, dsl.vexp(y))

    if k0e_has_improve_precision:
        res = dsl.cast_to(res, "float16")

    return res


@fusion_manager.register("bessel_k0e")
def bessel_k0e_compute(input_x, output_y, kernel_name="bessel_k0e"):
    """bessel_k0e_compute"""
    shape = input_x.shape
    dtype = input_x.dtype

    has_improve_precision = False
    if dtype != "float32":
        input_x = dsl.cast_to(input_x, "float32")
        dtype = "float32"
        has_improve_precision = True

    x_le_two = chebevl(dsl.vadds(dsl.vmul(input_x, input_x), -2.0), 10, A, shape, dtype)
    x_le_two = dsl.vadd(dsl.vmul(bessel_i0_compute(input_x), dsl.vmuls(dsl.vlog(dsl.vmuls(input_x, 0.5)), -1.0)),
                        x_le_two)
    x_le_two = dsl.vmul(dsl.vexp(input_x), x_le_two)
    x_le_two = dsl.vcmpsel(input_x, 0.0, 'le', MAXNUM, x_le_two)
    x_gt_two = dsl.vmul(dsl.vmul(dsl.vexp(dsl.vmuls(input_x, -1.0)), chebevl(dsl.vadds(dsl.vmuls(dsl.vrec(input_x),
                                                                                                 8.0), -2.0), 25, B,
                                                                             shape, dtype)), (dsl.vrsqrt(input_x)))

    res = dsl.vcmpsel(input_x, TWO, 'le', x_le_two, x_gt_two)
    if has_improve_precision:
        res = dsl.cast_to(res, "float16")

    return res


def bessel_k0e(x, output, kernel_name="bessel_k0e"):
    """bessel_k0e"""
    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")

    res = bessel_k0e_compute(data_x, output, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = dsl.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x, res]}
    dsl.build(schedule, config)
