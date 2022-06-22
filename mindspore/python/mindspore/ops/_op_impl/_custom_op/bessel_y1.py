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
"""BesselY1 op"""

import tbe.dsl as dsl
from tbe.common.platform import api_check_support
from tbe.common.register import register_op_compute
from tbe.common.utils import para_check
from tbe.common.utils import shape_util
from te import tvm

from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

bessel_y1_op_info = TBERegOp("BesselY1") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("bessel_y1.so") \
    .compute_cost(10) \
    .kernel_name("bessel_y1") \
    .partial_flag(True) \
    .op_pattern("formatAgnostic") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_None, DataType.F16_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None) \
    .get_op_info()


@op_info_register(bessel_y1_op_info)
def _bessel_y1_tbe():
    """BesselY1 TBE register"""
    return


CONST_LIMIT = 8.0
ITR_BEFORE1_J = [72362614232.0, -7895059235.0, 242396853.1, -2972611.439, 15704.48260, -30.16036606]
ITR_BEFORE2_J = [144725228442.0, 2300535178.0, 18583304.74, 99447.43394, 376.9991397, 1.0]
ITR_AFTER1_J = [1.0, 0.183105e-2, -0.3516396496e-4, 0.2457520174e-5, -0.240337019e-6]
ITR_AFTER2_J = [0.04687499995, -0.2002690873e-3, 0.8449199096e-5, -0.88228987e-6, 0.105787412e-6]
ITR_BEFORE1 = [-0.4900604943e13, 0.1275274390e13, -0.5153438139e11, 0.7349264551e9,
               -0.4237922726e7, 0.8511937935e4]
ITR_BEFORE2 = [0.2499580570e14, 0.4244419664e12, 0.3733650367e10, 0.2245904002e8,
               0.1020426050e6, 0.3549632885e3, 1.0]
THREE_QUARTERS_PI = 2.356194490192345
ITR_AFTER1 = [1.0, 0.183105e-2, -0.3516396496e-4, 0.2457520174e-5, -0.240337019e-6]
ITR_AFTER2 = [0.04687499995, -0.2002690873e-3, 0.8449199096e-5, -0.88228987e-6, 0.105787412e-6]
TOW_OVER_PI = 0.636619772367581
PI = 3.141592653589793


def angle_trans_cal(x):
    """angle_trans_cal"""
    consult = dsl.vdiv(x, dsl.broadcast(tvm.const(PI * 2), x.shape))
    floor_consult = dsl.cast_to(dsl.floor(consult), 'float32')
    fixed_x = dsl.vsub(x, dsl.vmuls(floor_consult, PI * 2))

    coe = -0.707106781186548 # -sqrt(2)/2
    quarter_x = dsl.vmuls(x, 0.25)
    sin_quar_x, cos_quar_x = cordic(quarter_x)
    cos_quar_x2 = dsl.vmul(cos_quar_x, cos_quar_x)
    sin_quar_x2 = dsl.vmul(sin_quar_x, sin_quar_x)
    cos_quar_x4 = dsl.vmul(cos_quar_x2, cos_quar_x2)

    temp_res1 = dsl.vadds(dsl.vadd(dsl.vmuls(cos_quar_x2, -8.0), dsl.vmuls(cos_quar_x4, 8.0)), 1)
    temp_res2 = dsl.vmuls(sin_quar_x2, 2.0)
    temp_res2 = dsl.vadds(temp_res2, -1.0)
    temp_res2 = dsl.vmul(temp_res2, sin_quar_x)
    temp_res2 = dsl.vmul(temp_res2, cos_quar_x)
    temp_res2 = dsl.vmuls(temp_res2, -4.0)

    sin_res = dsl.vmuls(dsl.vadd(temp_res1, temp_res2), coe)
    cos_res = dsl.vmuls(dsl.vsub(temp_res2, temp_res1), coe)
    return sin_res, cos_res


def cordic(angle):
    """cordic"""
    shape = angle.shape
    dtype = angle.dtype
    ceof = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256,
            1 / 512, 1 / 1024, 1 / 2048, 1 / 4096, 1 / 8192, 1 / 16384, 1 / 32768,
            1 / 65536, 1 / 131072, 1 / 262144, 1 / 524288, 1 / 1048576]

    dangle = [45, 26.565051177078, 14.0362434679265, 7.1250163489018,
              3.57633437499735, 1.78991060824607, 0.8951737102111,
              0.4476141708606, 0.2238105003685, 0.1119056770662,
              0.0559528918938, 0.027976452617, 0.01398822714227,
              0.006994113675353, 0.003497056950704, 0.001748528426980,
              0.000874264213694, 0.000437132106872, 0.000218566053439,
              0.000109283026720, 0.000054641513360]

    k = 0.60725293500888
    x = dsl.broadcast(1.0, shape, dtype)
    y = dsl.broadcast(0.0, shape, dtype)
    z = angle

    for i in range(10):
        ones = dsl.broadcast(1.0, shape, dtype)
        nones = dsl.broadcast(-1.0, shape, dtype)
        cmp = dsl.vcmp(z, dsl.broadcast(0.0, shape, dtype))
        d = dsl.vsel(cmp, ones, nones)

        xn = x

        d_ceof = dsl.vmuls(d, ceof[i])
        x = dsl.vsub(xn, dsl.vmul(y, d_ceof))
        y = dsl.vadd(y, dsl.vmul(xn, d_ceof))
        z = dsl.vsub(z, dsl.vmuls(d, dangle[i]))

    return dsl.vmuls(y, k), dsl.vmuls(x, k)


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name,too-many-locals,
def prod(data, iter_arr):
    """prod"""
    input_shape = data.shape
    input_dtype = data.dtype
    res = dsl.broadcast(tvm.const(iter_arr[-1], input_dtype), input_shape)
    for addition in reversed(iter_arr[:-1]):
        res = dsl.vmul(res, data)
        res = dsl.vadd(res, dsl.broadcast(tvm.const(addition, input_dtype), input_shape))
    return res


def bessel_j1(x):
    """bessel_j1"""
    shape_input = x.shape
    dtype_input = x.dtype
    # 1. chose the type of data in begin
    if (dtype_input in ('float16', 'float32')) and api_check_support("te.lang.cce.vadd", "float32"):
        x = dsl.cast_to(x, "float32")
    else:
        raise RuntimeError("BesselY0 kernel data type [%s] not support." % dtype_input)
    abs_data = dsl.vabs(x)

    # 2. compute bessel_j1 for data in (-8, 8)
    broad_const_limit = dsl.broadcast(tvm.const(CONST_LIMIT, x.dtype), shape_input)
    before_abs_data = dsl.vmin(abs_data, broad_const_limit)
    square_data = dsl.vmul(before_abs_data, before_abs_data)  # x * x
    before_res1 = prod(square_data, ITR_BEFORE1_J)
    before_res1 = dsl.vmul(before_res1, before_abs_data)

    before_res2 = prod(square_data, ITR_BEFORE2_J)
    before_final_res = dsl.vdiv(before_res1, before_res2)

    # 3. compute bessel_j1 for data in (-inf, -8) or (8, inf)
    div_data = dsl.vdiv(dsl.broadcast(tvm.const(8.0), shape_input), abs_data)
    square_div_data = dsl.vmul(div_data, div_data)
    minus_pi_data = dsl.vsub(abs_data, dsl.broadcast(tvm.const(THREE_QUARTERS_PI), shape_input))
    after_res1 = prod(square_div_data, ITR_AFTER1_J)
    after_res2 = prod(square_div_data, ITR_AFTER2_J)
    # 3.1 sqrt(0.636619772/ax)
    tmp_res1 = dsl.vsqrt(dsl.vdiv(dsl.broadcast(tvm.const(TOW_OVER_PI), shape_input), abs_data),
                         impl_mode='high_precision')
    # 3.2 cos(xx)*ans1
    sinv, cosv = angle_trans_cal(abs_data)
    tmp_res2 = dsl.vmul(cosv, after_res1)
    # 3.3 z*math.sin(xx)*ans2

    tmp_res3 = dsl.vmul(dsl.vmul(div_data, sinv), after_res2)
    after_final_res = dsl.vmul(tmp_res1, dsl.vsub(tmp_res2, tmp_res3))

    zero = dsl.broadcast(0.0, shape_input, 'float32')
    neg_cond = dsl.vcmp(after_final_res, zero, operation='lt', mode='bool')
    neg_after_res = dsl.vmuls(after_final_res, -1.0)
    after_final_res = dsl.vsel(neg_cond, neg_after_res, after_final_res)

    # 5. select res
    # 5.1 compare with limit
    select_condition = dsl.vcmp(abs_data, broad_const_limit, operation='lt', mode='bool')

    # 5.2 select
    res = dsl.vsel(select_condition, before_final_res, after_final_res)

    # 6. chose the type of data in end
    if dtype_input == "float16":
        res = dsl.cast_to(res, "float16")
    return res


@register_op_compute("bessel_y1")
def bessel_y1_compute(x, y, kernel_name="bessel_y1"):
    """bessel_y1_compute"""
    shape_input = x.shape
    dtype_input = x.dtype
    # chose the type of data in begin
    if (dtype_input in ('float16', 'float32')) and api_check_support("te.lang.cce.vadd", "float32"):
        x = dsl.cast_to(x, "float32")
    else:
        raise RuntimeError("BesselY0 kernel data type [%s] not support." % dtype_input)
    x = dsl.vabs(x)

    # compute bessel_y1 for data in (0, 8)
    broad_const_limit = dsl.broadcast(tvm.const(CONST_LIMIT, x.dtype), shape_input)
    square_data = dsl.vmul(x, x) # y = x * x
    before_res1 = prod(square_data, ITR_BEFORE1)
    before_res1 = dsl.vmul(before_res1, x)
    before_res2 = prod(square_data, ITR_BEFORE2)
    before_final_res = dsl.vmul(bessel_j1(x), dsl.vlog(x, impl_mode="high_precision"))

    before_final_res = dsl.vsub(before_final_res, dsl.vrec(x, impl_mode="high_precision"))
    before_final_res = dsl.vmuls(before_final_res, TOW_OVER_PI)
    before_final_res = dsl.vadd(before_final_res, dsl.vdiv(before_res1, before_res2))


    # compute bessel_y1 for data in (8, inf)
    div_data = dsl.vdiv(broad_const_limit, x) # z = 8.0 / x
    square_div_data = dsl.vmul(div_data, div_data) # y = z * z
    minus_pi_data = dsl.vsub(x, dsl.broadcast(tvm.const(THREE_QUARTERS_PI), shape_input)) # xx
    after_res1 = prod(square_div_data, ITR_AFTER1)
    after_res2 = prod(square_div_data, ITR_AFTER2)
    tmp_res1 = dsl.vsqrt(dsl.vdiv(dsl.broadcast(tvm.const(TOW_OVER_PI), shape_input), x), impl_mode="high_precision")
    sinv, cosv = angle_trans_cal(x)
    tmp_res2 = dsl.vmul(sinv, after_res1)
    tmp_res3 = dsl.vmul(dsl.vmul(div_data, cosv), after_res2)
    after_final_res = dsl.vmul(tmp_res1, dsl.vadd(tmp_res2, tmp_res3))

    # select res
    # compare with limit
    select_condition = dsl.vcmp(x, broad_const_limit, operation='lt', mode='bool')

    # select
    res = dsl.vsel(select_condition, before_final_res, after_final_res)

    # chose the type of data in end
    if dtype_input == "float16":
        res = dsl.cast_to(res, "float16")
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def bessel_y1(x, y, kernel_name="bessel_y1"):
    """
    Computes the Bessel j1 function of x element-wise.

    Parameters
    ----------
    x: the dict of input, only support float16, float32

    y : the dict of output

    kernel_name : cce kernel name, default value is "bessel_y1"

    Returns
    -------
    None
    """

    shape_input = x.get("shape")
    dtype_input = x.get("dtype")

    para_check.check_shape(shape_input, param_name="x")
    shape_input, _ = shape_util.refine_shape_axes(shape_input, [])

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_input, check_list, param_name="x")

    input_dtype = dtype_input.lower()
    data = tvm.placeholder(shape_input, dtype=input_dtype, name="data_input")

    res = bessel_y1_compute(data, y, kernel_name)

    with tvm.target.cce():
        sch = dsl.auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": (data, res)}
    dsl.build(sch, config)
