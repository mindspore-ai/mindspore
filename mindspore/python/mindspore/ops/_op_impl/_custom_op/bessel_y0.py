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
"""BesselY0 op"""

import te.lang.cce as tbe
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from tbe.common.platform import api_check_support
from tbe.common.register import register_op_compute

from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

bessel_y0_op_info = TBERegOp("BesselY0") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("bessel_y0.so") \
    .compute_cost(10) \
    .kernel_name("bessel_y0") \
    .partial_flag(True) \
    .op_pattern("formatAgnostic") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_None, DataType.F16_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None) \
    .get_op_info()


@op_info_register(bessel_y0_op_info)
def _bessel_y0_tbe():
    """BesselY0 TBE register"""
    return


ITR_A1 = (57568490574.0, -13362590354.0, 651619640.7, -11214424.18, 77392.33017, -184.9052456)
ITR_A2 = (57568490411.0, 1029532985.0, 9494680.718, 59272.64853, 267.8532712, 1.0)
ITR_A33 = (1.0, -0.1098628627e-2, 0.2734510407e-4, -0.2073370639e-5, 0.2093887211e-6)
ITR_A44 = (-0.1562499995e-1, 0.1430488765e-3, -0.6911147651e-5, 0.7621095161e-6, 0.934935152e-7)
ITR_A5 = (-2957821389.0, 7062834065.0, -512359803.6, 10879881.29, -86327.92757, 228.4622733)
ITR_A6 = (40076544269.0, 745249964.8, 7189466.438, 47447.26470, 226.1030244, 1.0)
ITR_A = (-0.785398164, 0.636619772)
LEN_A1256 = 6
LEN_A34 = 5
EIGHT = 8.0
ZERO = 0
NEG_ONE = -1.0
ONE = 1.0
PI2 = 6.2831853071796
HALF_PI = 1.5707963267948966192313216916398
BOUNDARY_1 = 0.70710678118654752440084436210485
# Taylor coefficient
COEF = (1.0,
        0.16666666666666666666666666666667,
        0.075,
        0.04464285714285714285714285714286,
        0.03038194444444444444444444444444,
        0.02237215909090909090909090909091,
        0.01735276442307692307692307692308,
        0.01396484375)
# TAYLOR COUNT
TAYLOR_COUNT = 7
# negative min float16 value
NEG_MIN_FP16 = -2 ** (-24)
# min float16 * 2
TWO_MIN_FP16 = 2 ** (-23)


def _taylor_compute(data_x, x_square=None):
    """_taylor_compute"""
    if x_square is None:
        x_square = tbe.vmul(data_x, data_x)

    res = tbe.vmuls(x_square, tvm.const(COEF[TAYLOR_COUNT], "float32"))
    for temp in reversed(range(TAYLOR_COUNT)):
        res = tbe.vadds(res, tvm.const(COEF[temp], "float32"))
        if temp == 0:
            res = tbe.vmul(res, data_x)
        else:
            res = tbe.vmul(x_square, res)
    return res


def acos_compute(x):
    """
    do element-wise acos compute using asin op
    acos(x) = HALF_PI - asin(x)

    asin(x) = | arcsin(sqrt(1-x^2)) - HALF_PI, x belongs to (-1, -2^(-0.5))
              | the 15th order taylor expansion, x belongs to (-2^(-0.5),
              | 2^(-0.5))
              | HALF_PI - arcsin(sqrt(1-x^2)), x belongs to (2^(-0.5), 1)

    Parameters:
    ----------
    x: the placeholder of data input

    y : the dict of output

    Returns : A Tensor. Has the same type as x.
    -------
    """
    shape = x.shape
    dtype = x.dtype
    # Change dtype to float32
    if (dtype in ('float16', 'double')) and \
            api_check_support("te.lang.cce.vadd", "float32"):
        x = tbe.cast_to(x, "float32")

    # to fix bug for input data is 1.0
    x = tbe.vadds(x, NEG_MIN_FP16)
    # Sign mask
    sign = tbe.vcmpsel(x, 0, 'lt', NEG_ONE, ONE)
    # All positive
    x = tbe.vmul(x, sign)

    # x belongs to (0, 2^(-0.5))
    if  api_check_support("te.lang.cce.vmins", x.dtype):
        choice_1 = tbe.vmins(x, tvm.const(BOUNDARY_1, x.dtype))
    else:
        boundary_mask1 = tbe.broadcast(tvm.const(BOUNDARY_1, x.dtype), shape)
        choice_1 = tbe.vmin(x, boundary_mask1)

    if  api_check_support("te.lang.cce.vsubs", choice_1.dtype):
        choice_1 = tbe.vsubs(choice_1, tvm.const(BOUNDARY_1, choice_1.dtype))
    else:
        boundary_mask1 = tbe.broadcast(tvm.const(BOUNDARY_1, choice_1.dtype), shape)
        choice_1 = tbe.vsub(choice_1, boundary_mask1)

    choice_1 = tbe.vmuls(tbe.floor(choice_1), NEG_ONE)

    res_1 = _taylor_compute(x)
    res_1 = tbe.vmul(res_1, choice_1)

    # x belongs to (2^(-0.5), 1)
    choice_2 = tbe.vmuls(choice_1, tvm.const(NEG_ONE, x.dtype))
    choice_2 = tbe.vadds(choice_2, tvm.const(ONE, x.dtype))

    # to fix bug for input data is 1.0
    x = tbe.vadds(x, TWO_MIN_FP16)
    res_2 = tbe.vmul(x, x)
    res_2 = tbe.vmuls(res_2, tvm.const(NEG_ONE, x.dtype))
    res_2 = tbe.vadds(res_2, tvm.const(ONE, x.dtype))
    res_2_sqrt = tbe.vsqrt(res_2, 1)

    res_2 = _taylor_compute(res_2_sqrt, res_2)
    res_2 = tbe.vmuls(res_2, tvm.const(NEG_ONE, x.dtype))
    res_2 = tbe.vadds(res_2, tvm.const(HALF_PI, x.dtype))
    res_2 = tbe.vmul(res_2, choice_2)

    # Restore sign of asin
    res_1 = tbe.vadd(res_1, res_2)
    res_1 = tbe.vmul(res_1, sign)
    res_1 = tbe.vmuls(res_1, tvm.const(NEG_ONE, x.dtype))
    res_1 = tbe.vadds(res_1, tvm.const(HALF_PI, x.dtype))

    return res_1


def asin_compute(x):
    """
    do element-wise asin compute
    asin(x) = | arcsin(sqrt(1-x^2)) - HALF_PI, x belongs to (-1, -2^(-0.5))
              | the 15th order taylor expansion, x belongs to (-2^(-0.5), 2^(-0.5))
              | HALF_PI - arcsin(sqrt(1-x^2)), x belongs to (2^(-0.5), 1)

    Parameters:
    ----------
    x: the placeholder of data input

    y : the dict of output

    Returns : A Tensor. Has the same type as data_input.
    -------
    """
    shape = x.shape
    dtype = x.dtype
    # Change dtype to float32
    if (dtype in ('float16', 'double')) and api_check_support("te.lang.cce.vadd", "float32"):
        x = tbe.cast_to(x, "float32")

    # Sign mask
    bessely0_sign = tbe.vcmpsel(x, 0, 'lt', NEG_ONE, ONE)

    # All positive
    x = tbe.vmul(x, bessely0_sign)

    # x belongs to (0, 2^(-0.5))
    if api_check_support("te.lang.cce.vmins", x.dtype):
        y0_choice_1 = tbe.vmins(x, tvm.const(BOUNDARY_1, x.dtype))
    else:
        boundary_mask1 = tbe.broadcast(tvm.const(BOUNDARY_1, x.dtype), shape)
        y0_choice_1 = tbe.vmin(x, boundary_mask1)

    if api_check_support("te.lang.cce.vsubs", y0_choice_1.dtype):
        y0_choice_1 = tbe.vsubs(y0_choice_1, tvm.const(BOUNDARY_1, y0_choice_1.dtype))
    else:
        boundary_mask1 = tbe.broadcast(tvm.const(BOUNDARY_1, y0_choice_1.dtype), shape)
        y0_choice_1 = tbe.vsub(y0_choice_1, boundary_mask1)

    y0_choice_1 = tbe.vmuls(tbe.floor(y0_choice_1), NEG_ONE)

    res_1 = _taylor_compute(x)
    res_1 = tbe.vmul(res_1, y0_choice_1)

    # x belongs to (2^(-0.5), 1)
    choice_2 = tbe.vmuls(y0_choice_1, tvm.const(NEG_ONE, x.dtype))
    choice_2 = tbe.vadds(choice_2, tvm.const(ONE, x.dtype))

    res_2 = tbe.vmul(x, x)
    res_2 = tbe.vmuls(res_2, tvm.const(NEG_ONE, x.dtype))
    res_2 = tbe.vadds(res_2, tvm.const(ONE, x.dtype))
    res_2_sqrt = tbe.vsqrt(res_2)

    res_2 = _taylor_compute(res_2_sqrt, res_2)

    res_2 = tbe.vmuls(res_2, tvm.const(NEG_ONE, x.dtype))
    res_2 = tbe.vadds(res_2, tvm.const(HALF_PI, x.dtype))
    res_2 = tbe.vmul(res_2, choice_2)

    # Restore sign
    res_1 = tbe.vadd(res_1, res_2)
    res_1 = tbe.vmul(res_1, bessely0_sign)

    return res_1


def _besselj0(x):
    """
    Algrithm:
        y = x * x;
        ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y *
        (77392.33017 + y * (-184.9052456)))))
        ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))))
        ans = ans1 / ans2  (x < 8.0)
        z = 8.0 / x
        y = z * z
        xx = ax - 0.785398164;
        ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)))
        ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 +
        y * (0.7621095161e-6 - y * 0.934935152e-7)));
        ans = sqrt(0.636619772 / ax) * (cos(xx) * ans1 - z * sin(xx) * ans2), (x >= 8.0)

    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    Returns
    -------
    A tensor. Has the same type as x.

    """
    jax = tbe.vabs(x)
    tensor_eight = tbe.broadcast(tvm.const(EIGHT, jax.dtype), jax.shape)
    first_res = tbe.vmax(jax, tensor_eight)
    jz = tbe.vdiv(tensor_eight, first_res)
    jy = tbe.vmul(jz, jz)
    jxx = tbe.vadds(first_res, ITR_A[0])
    jans1 = tbe.vmuls(jy, tvm.const(ITR_A33[LEN_A34 - 1]))
    jans1 = tbe.vadds(jans1, ITR_A33[LEN_A34 - 2])
    for index in reversed(range(LEN_A34 - 2)):
        jans1 = tbe.vmul(jans1, jy)
        jans1 = tbe.vadds(jans1, ITR_A33[index])
    jans2 = tbe.vmuls(jy, tvm.const(ITR_A44[LEN_A34 - 1]))
    jans2 = tbe.vadds(jans2, ITR_A44[LEN_A34 - 2])
    for index in reversed(range(LEN_A34 - 2)):
        jans2 = tbe.vmul(jans2, jy)
        jans2 = tbe.vadds(jans2, ITR_A44[index])
    jansres1 = tbe.vmul(tbe.vsqrt(tbe.vmuls(tbe.vrec(first_res), ITR_A[1])),
                        tbe.vsub(tbe.vmul(acos_compute(jxx), jans1), tbe.vmul(jz, tbe.vmul(asin_compute(jxx), jans2))))

    first_res = tbe.vmin(jax, tensor_eight)
    jy = tbe.vmul(first_res, first_res)
    jans1 = tbe.vmuls(jy, tvm.const(ITR_A1[LEN_A1256 - 1]))
    jans1 = tbe.vadds(jans1, ITR_A1[LEN_A1256 - 2])
    for index in reversed(range(LEN_A1256 - 2)):
        jans1 = tbe.vmul(jans1, jy)
        jans1 = tbe.vadds(jans1, ITR_A1[index])
    jans2 = tbe.vmuls(jy, tvm.const(ITR_A2[LEN_A1256 - 1]))
    jans2 = tbe.vadds(jans2, ITR_A2[LEN_A1256 - 2])
    for index in reversed(range(LEN_A1256 - 2)):
        jans2 = tbe.vmul(jans2, jy)
        jans2 = tbe.vadds(jans2, ITR_A2[index])
    jansres2 = tbe.vdiv(jans1, jans2)
    res = tbe.vcmpsel(jax, tensor_eight, 'lt', jansres2, jansres1)

    return res


@register_op_compute("bessel_y0")
def bessel_y0_compute(x, y, kernel_name="bessel_y0"):
    """
    Algrithm:
        y = x * x;
        ans1 = -2957821389.0 + y * (7062834065.0 + y * (-512359803.6 + y * (10879881.29 +
        y * (-86327.92757 + y * 228.4622733))))
        ans2 = 40076544269.0 + y * (745249964.8 + y * (7189466.438 + y * (47447.26470 + y * (226.1030244 + y * 1.0))))
        ans = (ans1 / ans2) + 0.636619772 * bessj0(x) * math.log(x), (x < 8.0)

        z = 8.0 / x
        y = z * z
        xx = x - 0.785398164
        ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)))
        ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 +
        y * (0.7621095161e-6 + y * (-0.934945152e-7))))
        ans = math.sqrt(0.636619772 / x) * (math.asin(xx) * ans1 + z * math.acos(xx) * ans2), (x >= 8.0)

    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "bessel_y0"

    Returns
    -------
    A tensor. Has the same type as x.

    """
    dtype_input = x.dtype

    # chose the type of data in begin
    if (dtype_input in ('float16', 'double', 'float32')) \
            and api_check_support("te.lang.cce.vadd", "float32"):
        x = tbe.cast_to(x, "float32")
    else:
        raise RuntimeError("BesselY0 kernel data type [%s] not support." % dtype_input)
    x = tbe.vabs(x)
    tensor_eight = tbe.broadcast(tvm.const(EIGHT, x.dtype), x.shape)
    first_res = tbe.vmin(x, tensor_eight)
    yy = tbe.vmul(first_res, first_res)
    ans1 = tbe.vmuls(yy, tvm.const(ITR_A5[LEN_A1256 - 1]))
    ans1 = tbe.vadds(ans1, ITR_A5[LEN_A1256 - 2])
    for index in reversed(range(LEN_A1256 - 2)):
        ans1 = tbe.vmul(ans1, yy)
        ans1 = tbe.vadds(ans1, ITR_A5[index])
    ans2 = tbe.vmuls(yy, tvm.const(ITR_A6[LEN_A1256 - 1]))
    ans2 = tbe.vadds(ans2, ITR_A6[LEN_A1256 - 2])
    for index in reversed(range(LEN_A1256 - 2)):
        ans2 = tbe.vmul(ans2, yy)
        ans2 = tbe.vadds(ans2, ITR_A6[index])
    res1 = tbe.vadd(tbe.vdiv(ans1, ans2), tbe.vmuls(tbe.vmul(_besselj0(first_res), tbe.vlog(first_res)), ITR_A[1]))

    first_res = tbe.vmax(x, tensor_eight)
    z = tbe.vdiv(tensor_eight, first_res)
    y = tbe.vmul(z, z)
    xx = tbe.vadds(first_res, ITR_A[0])
    ans1 = tbe.vmuls(y, tvm.const(ITR_A33[LEN_A34 - 1]))
    ans1 = tbe.vadds(ans1, ITR_A33[LEN_A34 - 2])
    for index in reversed(range(LEN_A34 - 2)):
        ans1 = tbe.vmul(ans1, y)
        ans1 = tbe.vadds(ans1, ITR_A33[index])
    ans2 = tbe.vmuls(y, tvm.const(ITR_A44[LEN_A34 - 1]))
    ans2 = tbe.vadds(ans2, ITR_A44[LEN_A34 - 2])
    for index in reversed(range(LEN_A34 - 2)):
        ans2 = tbe.vmul(ans2, y)
        ans2 = tbe.vadds(ans2, ITR_A44[index])
    res2 = tbe.vmul(tbe.vsqrt(tbe.vmuls(tbe.vrec(first_res), ITR_A[1])),
                    tbe.vadd(tbe.vmul(asin_compute(xx), ans1),
                             tbe.vmul(z, tbe.vmul(acos_compute(xx), ans2))))
    res = tbe.vcmpsel(x, tensor_eight, 'lt', res1, res2)

    # Restore dtype
    if dtype_input == "float16":
        res = tbe.cast_to(res, "float16")
    if dtype_input == "double":
        res = tbe.cast_to(res, "double")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def bessel_y0(x, y, kernel_name="bessel_y0"):
    """
    Computes the Bessel y0 function of x element-wise.

    Parameters
    ----------
    x: only support float16, float32, double

    y : output

    kernel_name : cce kernel name, default value is "bessel_y0"

    Returns
    -------
    None
    """
    shape_input = x.get("shape")
    dtype_input = x.get("dtype")

    para_check.check_shape(shape_input, param_name="x")
    shape_input, _ = shape_util.refine_shape_axes(shape_input, [])

    check_list = ("float16", "float32", "double")
    para_check.check_dtype(dtype_input, check_list, param_name="x")

    input_dtype = dtype_input.lower()
    data = tvm.placeholder(shape_input, dtype=input_dtype, name="data_input")

    res = bessel_y0_compute(data, y, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": (data, res),
              "bool_storage_as_1bit": False}
    tbe.cce_build_code(sch, config)
