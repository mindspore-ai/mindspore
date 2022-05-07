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
"""BesselJ0 op"""

import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.platform.fusion_manager import fusion_manager
from tbe import dsl
from tbe.common.utils import shape_util

from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

bessel_j0_op_info = TBERegOp("BesselJ0") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("bessel_j0.so") \
    .compute_cost(10) \
    .kernel_name("bessel_j0") \
    .partial_flag(True) \
    .op_pattern("formatAgnostic") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_None, DataType.F16_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None) \
    .get_op_info()


@op_info_register(bessel_j0_op_info)
def _bessel_j0_tbe():
    """BesselJ0 TBE register"""
    return

FLOAT_16 = "float16"
FLOAT_32 = "float32"

PP = [7.96936729297347051624E-4, 8.28352392107440799803E-2,
      1.23953371646414299388E0, 5.44725003058768775090E0,
      8.74716500199817011941E0, 5.30324038235394892183E0,
      9.99999999999999997821E-1]
PQ = [9.24408810558863637013E-4, 8.56288474354474431428E-2,
      1.25352743901058953537E0, 5.47097740330417105182E0,
      8.76190883237069594232E0, 5.30605288235394617618E0,
      1.00000000000000000218E0]
QP = [-1.13663838898469149931E-2, -1.28252718670509318512E0,
      -1.95539544257735972385E1, -9.32060152123768231369E1,
      -1.77681167980488050595E2, -1.47077505154951170175E2,
      -5.14105326766599330220E1, -6.05014350600728481186E0]
QQ = [1.00000000000000000000E0, 6.43178256118178023184E1,
      8.56430025976980587198E2, 3.88240183605401609683E3,
      7.24046774195652478189E3, 5.93072701187316984827E3,
      2.06209331660327847417E3, 2.42005740240291393179E2]
RP = [-4.79443220978201773821E9, 1.95617491946556577543E12,
      -2.49248344360967716204E14, 9.70862251047306323952E15]
RQ = [1.00000000000000000000E0, 4.99563147152651017219E2,
      1.73785401676374683123E5, 4.84409658339962045305E7,
      1.11855537045356834862E10, 2.11277520115489217587E12,
      3.10518229857422583814E14, 3.18121955943204943306E16,
      1.71086294081043136091E18]

DR1 = 5.78318596294678452118E0
DR2 = 3.04712623436620863991E1
SQ2OPI = 7.9788456080286535587989E-1
NEG_PIO4 = -0.7853981633974483096

PI = 3.14159265358979
FIRST_ORDER = 5
LAST_ORDER = 13
FIRST_FACTOR = -1.0 / 6.0


def besselj0_cos(x):
    """cos"""
    dtype = x.dtype
    shape = shape_util.shape_to_list(x.shape)
    # cast to type float32 when type is float16
    has_improve_precision = False
    if dtype.lower() == FLOAT_16 and tbe_platform.cce_conf.api_check_support("te.lang.cce.vmul", "float32"):
        x = dsl.cast_to(x, FLOAT_32)
        dtype = FLOAT_32
        has_improve_precision = True
    # round the input
    round_fp16 = dsl.round(dsl.vmuls(x, 1.0 / (2 * PI)))
    round_fp32 = dsl.cast_to(round_fp16, dtype)
    input_x_round = dsl.vsub(x, dsl.vmuls(round_fp32, 2 * PI))
    # the initial value one
    const_res = tvm.const(1.0, dtype=dtype)
    res = dsl.broadcast(const_res, shape)
    # compute the rank 2
    input_x_power = dsl.vmul(input_x_round, input_x_round)
    iter_value = dsl.vmuls(input_x_power, -1.0/2.0)
    res = dsl.vadd(res, iter_value)
    # compute the rank 4~14
    iter_list = (4, 6, 8, 10, 12, 14)
    for i in iter_list:
        iter_value = dsl.vmuls(dsl.vmul(input_x_power, iter_value), -1.0/(i*(i-1)))
        res = dsl.vadd(res, iter_value)
    # cast the dtype to float16
    if has_improve_precision:
        res = dsl.cast_to(res, "float16")
    return res


def _besselj0_sin(x):
    """_besselj0_sin"""
    input_x_power = dsl.vmul(x, x)
    iter_value = dsl.vmul(dsl.vmuls(input_x_power, FIRST_FACTOR), x)
    res = dsl.vadd(x, iter_value)
    signal = FIRST_ORDER
    while signal < LAST_ORDER:
        iter_value = dsl.vmuls(dsl.vmul(input_x_power, iter_value),
                               -1.0 / (signal*(signal - 1)))
        res = dsl.vadd(res, iter_value)
        signal = signal + 2

    return res


def besselj0_sin(x):
    """sin"""
    dtype = x.dtype
    shape = shape_util.shape_to_list(x.shape)
    has_improve_precision = False
    cast_dtype = FLOAT_16
    if tbe_platform.api_check_support("te.lang.cce.vmul", "float32"):
        has_improve_precision = True
        cast_dtype = FLOAT_32

    # cast to type float32 when type is float16
    if dtype == FLOAT_16 and has_improve_precision:
        x = tbe.cast_to(x, FLOAT_32)

    pai_multiple = tbe.vmuls(x, 1 / PI)
    round_float = tbe.cast_to(tbe.round(pai_multiple), cast_dtype)
    # to adjust x to [-pai/2,pai/2]
    x = tbe.vsub(x, tbe.vmuls(round_float, PI))
    res = _besselj0_sin(x)
    # if round is odd, the final result need to multiply -1.Need to multiply 1/2 to get the ceil value
    ceil_value = tbe.ceil(tbe.vmuls(round_float, 1 / 2))
    # if odd, ceil*2-round is 1,if even, the value is 0
    sub_value = tbe.vsub(tbe.vmuls(ceil_value, tvm.const(2, dtype)), round_float)
    tensor_one = tbe.broadcast(tvm.const(1, cast_dtype), shape)
    odd_tensor = tbe.vsub(tensor_one, sub_value)
    even_tensor = tbe.vsub(odd_tensor, tensor_one)
    odd_even_tensor = tbe.vadd(odd_tensor, even_tensor)
    res = tbe.vmul(res, odd_even_tensor)

    # cast the dtype to float16
    if dtype == FLOAT_16 and has_improve_precision:
        res = tbe.cast_to(res, FLOAT_16)

    return res


def besselj0_polevl(x, n, coef, shape):
    """polevl"""
    dtype = 'float32'
    x = dsl.cast_to(x, dtype)

    if n == 0:
        coef_0 = dsl.broadcast(coef[0], shape, output_dtype=dtype)
        return dsl.cast_to(coef_0, dtype)

    coef_n = dsl.broadcast(coef[n], shape, output_dtype=dtype)
    res = dsl.vadd(dsl.vmul(besselj0_polevl(x, n-1, coef, shape), x), coef_n)
    return dsl.cast_to(res, 'float32')


@fusion_manager.register("bessel_j0")
def bessel_j0_compute(x, kernel_name="bessel_j0"):
    """bessel_j0_compute"""
    dtype = x.dtype
    shape = shape_util.shape_to_list(x.shape)

    # cast to type float32 when type is float16
    has_improve_precision = False
    if dtype.lower() == FLOAT_16 and tbe_platform.cce_conf.api_check_support("te.lang.cce.vmul", "float32"):
        x = dsl.cast_to(x, FLOAT_32)
        dtype = FLOAT_32
        has_improve_precision = True

    y = dsl.vabs(x)
    z = dsl.vmul(y, y)
    y_le_five = dsl.vcmpsel(y, 1.0e-5, 'lt', dsl.vadds(dsl.vmuls(z, -0.25), 1),
                            dsl.vmul(dsl.vmul(dsl.vadds(z, -DR1), dsl.vadds(z, -DR2)),
                                     dsl.vdiv(besselj0_polevl(z, 3, RP, shape), besselj0_polevl(z, 8, RQ, shape))))
    s = dsl.vmuls(dsl.vrec(z), 25)
    p = dsl.vdiv(besselj0_polevl(s, 6, PP, shape), besselj0_polevl(s, 6, PQ, shape))
    q = dsl.vdiv(besselj0_polevl(s, 7, QP, shape), besselj0_polevl(s, 6, PQ, shape))
    yn = dsl.vadds(y, NEG_PIO4)
    w = dsl.vmuls(dsl.vrec(y), -5.0)
    p = dsl.vadd(dsl.vmul(p, besselj0_cos(yn)), dsl.vmul(w, dsl.vmul(q, besselj0_sin(yn))))
    y_gt_five = dsl.vmul(dsl.vmuls(p, SQ2OPI), dsl.vrsqrt(y))
    res = dsl.vcmpsel(y, 5.0, 'le', y_le_five, y_gt_five)

    if has_improve_precision:
        res = dsl.cast_to(res, "float16")

    return res


def bessel_j0(x, y, kernel_name="bessel_j0"):
    """bessel_j0"""
    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")
    res = bessel_j0_compute(data_x, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = dsl.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x, res]}
    dsl.build(schedule, config)
