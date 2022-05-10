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
"""BesselJ1 op"""


import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.platform.fusion_manager import fusion_manager
from tbe import dsl
from tbe.common.utils import shape_util


from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

bessel_j1_op_info = TBERegOp("BesselJ1") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("bessel_j1.so") \
    .compute_cost(10) \
    .kernel_name("bessel_j1") \
    .partial_flag(True) \
    .op_pattern("formatAgnostic") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_None, DataType.F16_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None) \
    .get_op_info()


@op_info_register(bessel_j1_op_info)
def _bessel_j1_tbe():
    """BesselJ1 TBE register"""
    return


FLOAT_16 = "float16"
FLOAT_32 = "float32"

PI = 3.14159265358979
FIRST_ORDER = 5
LAST_ORDER = 13
FIRST_FACTOR = -1.0 / 6.0


def besselj1_cos(x):
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
    besselj1_input_x_round = dsl.vsub(x, dsl.vmuls(round_fp32, 2 * PI))
    # the initial value one
    const_res = tvm.const(1.0, dtype=dtype)
    res = dsl.broadcast(const_res, shape)
    # compute the rank 2
    input_x_power = dsl.vmul(besselj1_input_x_round, besselj1_input_x_round)
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


def _besselj1_sin(x):
    """_sin"""
    input_x_power = dsl.vmul(x, x)
    iter_value = dsl.vmul(dsl.vmuls(input_x_power, FIRST_FACTOR), x)
    besselj1_res = dsl.vadd(x, iter_value)

    signal = FIRST_ORDER
    while signal < LAST_ORDER:
        iter_value = dsl.vmuls(dsl.vmul(input_x_power, iter_value),
                               -1.0 / (signal*(signal - 1)))
        besselj1_res = dsl.vadd(besselj1_res, iter_value)
        signal = signal + 2

    return besselj1_res


def besselj1_sin(besselj1_x):
    """sin"""
    dtype = besselj1_x.dtype
    shape = shape_util.shape_to_list(besselj1_x.shape)

    has_improve_precision = False
    cast_dtype = FLOAT_16
    if tbe_platform.api_check_support("te.lang.cce.vmul", "float32"):
        has_improve_precision = True
        cast_dtype = FLOAT_32

    # cast to type float32 when type is float16
    if dtype == FLOAT_16 and has_improve_precision:
        besselj1_x = tbe.cast_to(besselj1_x, FLOAT_32)

    pai_multiple = tbe.vmuls(besselj1_x, 1 / PI)
    round_float = tbe.cast_to(tbe.round(pai_multiple), cast_dtype)
    # to adjust x to [-pai/2,pai/2]
    besselj1_x = tbe.vsub(besselj1_x, tbe.vmuls(round_float, PI))

    besselj1_res = _besselj1_sin(besselj1_x)

    # if round is odd, the final result need to multiply -1.Need to multiply 1/2 to get the ceil value
    ceil_value = tbe.ceil(tbe.vmuls(round_float, 1 / 2))
    # if odd, ceil*2-round is 1,if even, the value is 0
    sub_value = tbe.vsub(tbe.vmuls(ceil_value, tvm.const(2, dtype)), round_float)
    tensor_one = tbe.broadcast(tvm.const(1, cast_dtype), shape)
    odd_tensor = tbe.vsub(tensor_one, sub_value)
    even_tensor = tbe.vsub(odd_tensor, tensor_one)
    odd_even_tensor = tbe.vadd(odd_tensor, even_tensor)
    besselj1_res = tbe.vmul(besselj1_res, odd_even_tensor)

    # cast the dtype to float16
    if dtype == FLOAT_16 and has_improve_precision:
        besselj1_res = tbe.cast_to(besselj1_res, FLOAT_16)

    return besselj1_res


PP = [7.62125616208173112003E-4, 7.31397056940917570436E-2,
      1.12719608129684925192E0, 5.11207951146807644818E0,
      8.42404590141772420927E0, 5.21451598682361504063E0,
      1.00000000000000000254E0]

PQ = [5.71323128072548699714E-4, 6.88455908754495404082E-2,
      1.10514232634061696926E0, 5.07386386128601488557E0,
      8.39985554327604159757E0, 5.20982848682361821619E0,
      9.99999999999999997461E-1]

QP = [5.10862594750176621635E-2, 4.98213872951233449420E0,
      7.58238284132545283818E1, 3.66779609360150777800E2,
      7.10856304998926107277E2, 5.97489612400613639965E2,
      2.11688757100572135698E2, 2.52070205858023719784E1]

QQ = [1.00000000000000000000E0, 6.43178256118178023184E1,
      8.56430025976980587198E2, 3.88240183605401609683E3,
      7.24046774195652478189E3, 5.93072701187316984827E3,
      2.06209331660327847417E3, 2.42005740240291393179E2]

RP = [-8.99971225705559398224E8, 4.52228297998194034323E11,
      -7.27494245221818276015E13, 3.68295732863852883286E15]

RQ = [1.00000000000000000000E0, 6.20836478118054335476E2,
      2.56987256757748830383E5, 8.35146791431949253037E7,
      2.21511595479792499675E10, 4.74914122079991414898E12,
      7.84369607876235854894E14, 8.95222336184627338078E16,
      5.32278620332680085395E18]

Z1 = 1.46819706421238932572E1
Z2 = 4.92184563216946036703E1
NEG_THPIO4 = -2.35619449019234492885
SQ2OPI = 7.9788456080286535587989E-1


def polevl(x, n, coef, shape):
    """polevl"""
    dtype = 'float32'
    x = dsl.cast_to(x, dtype)

    if n == 0:
        coef_0 = dsl.broadcast(coef[0], shape, output_dtype=dtype)
        return dsl.cast_to(coef_0, dtype)

    coef_n = dsl.broadcast(coef[n], shape, output_dtype=dtype)
    res = dsl.vadd(dsl.vmul(polevl(x, n-1, coef, shape), x), coef_n)
    return dsl.cast_to(res, 'float32')


@fusion_manager.register("bessel_j1")
def bessel_j1_compute(x, kernel_name="bessel_j1"):
    """bessel_j1_compute"""
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
    y_le_five = dsl.vdiv(polevl(z, 3, RP, shape), polevl(z, 8, RQ, shape))
    y_le_five = dsl.vmul(y_le_five, dsl.vmul(x, dsl.vmul(dsl.vadds(z, -Z1), dsl.vadds(z, -Z2))))
    s = dsl.vmuls(dsl.vrec(z), 25)
    p = dsl.vdiv(polevl(s, 6, PP, shape), polevl(s, 6, PQ, shape))
    q = dsl.vdiv(polevl(s, 7, QP, shape), polevl(s, 7, QQ, shape))
    yn = dsl.vadds(y, NEG_THPIO4)
    w = dsl.vmuls(dsl.vrec(y), -5.0)
    p = dsl.vadd(dsl.vmul(p, besselj1_cos(yn)), dsl.vmul(w, dsl.vmul(q, besselj1_sin(yn))))

    y_gt_five = dsl.vmul(dsl.vmuls(p, SQ2OPI), dsl.vrsqrt(y))
    y_gt_five = dsl.vcmpsel(x, 0.0, 'lt', dsl.vmuls(y_gt_five, -1.0), y_gt_five)

    res = dsl.vcmpsel(y, 5.0, 'le', y_le_five, y_gt_five)

    if has_improve_precision:
        res = dsl.cast_to(res, "float16")

    return res


def bessel_j1(x, y, kernel_name="bessel_j1"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    """
    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")

    res = bessel_j1_compute(data_x, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = dsl.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x, res]}
    dsl.build(schedule, config)
