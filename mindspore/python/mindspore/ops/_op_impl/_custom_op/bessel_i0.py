# Copyright 2021 Huawei Technologies Co., Ltd
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
"""BesselI0 op"""

from tbe import dsl
from te import tvm
from te.platform.fusion_manager import fusion_manager

from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

bessel_i0_op_info = TBERegOp("BesselI0") \
                        .fusion_type("ELEMWISE") \
                        .async_flag(False) \
                        .binfile_name("bessel_i0.so") \
                        .compute_cost(10) \
                        .kernel_name("bessel_i0") \
                        .partial_flag(True) \
                        .op_pattern("formatAgnostic") \
                        .input(0, "x", False, "required", "all") \
                        .output(0, "y", False, "required", "all") \
                        .dtype_format(DataType.F16_None, DataType.F16_None) \
                        .dtype_format(DataType.F32_None, DataType.F32_None) \
                        .get_op_info()


@op_info_register(bessel_i0_op_info)
def _bessel_i0_tbe():
    """BesselI0 TBE register"""
    return


A = [-1.30002500998624804212E-8, 6.04699502254191894932E-8,
     -2.67079385394061173391E-7, 1.11738753912010371815E-6,
     -4.41673835845875056359E-6, 1.64484480707288970893E-5,
     -5.75419501008210370398E-5, 1.88502885095841655729E-4,
     -5.76375574538582365885E-4, 1.63947561694133579842E-3,
     -4.32430999505057594430E-3, 1.05464603945949983183E-2,
     -2.37374148058994688156E-2, 4.93052842396707084878E-2,
     -9.49010970480476444210E-2, 1.71620901522208775349E-1,
     -3.04682672343198398683E-1, 6.76795274409476084995E-1]

B = [3.39623202570838634515E-9, 2.26666899049817806459E-8,
     2.04891858946906374183E-7, 2.89137052083475648297E-6,
     6.88975834691682398426E-5, 3.36911647825569408990E-3,
     8.04490411014108831608E-1]


def chebevl(x, num, coef, shape, dtype):
    """chebevl"""
    broad_coef = dsl.broadcast(coef[0], shape, dtype)
    broad_zero = dsl.broadcast(0, shape, dtype)
    none_signal = None
    for i in range(1, num):
        none_signal = broad_zero
        broad_zero = broad_coef
        coef_i = dsl.broadcast(coef[i], shape, dtype)
        broad_coef = dsl.vsub(dsl.vadd(dsl.vmul(x, broad_zero), coef_i), none_signal)
    return dsl.vmuls(dsl.vsub(broad_coef, none_signal), 0.5)


@fusion_manager.register("bessel_i0")
def bessel_i0_compute(input_x, output, kernel_name="bessel_i0"):
    """bessel_i0_compute"""
    dtype = input_x.dtype
    shape = input_x.shape

    has_improve_precision = False
    if dtype != "float32":
        input_x = dsl.cast_to(input_x, "float32")
        dtype = "float32"
        has_improve_precision = True

    y = dsl.vabs(input_x)

    y_le_eight_in = dsl.vmuls(y, 0.5)
    y_le_eight_in = dsl.vadds(y_le_eight_in, -2.0)
    y_le_eight = chebevl(y_le_eight_in, 18, A, shape, dtype)

    y_gt_eight_in = dsl.vadds(dsl.vmuls(dsl.vrec(y), 32.0), -2.0)
    y_gt_eight = chebevl(y_gt_eight_in, 7, B, shape, dtype)
    y_gt_eight = dsl.vmul(y_gt_eight, dsl.vrsqrt(y))

    res = dsl.vcmpsel(y, 8.0, 'le', y_le_eight, y_gt_eight)
    res = dsl.vmul(res, dsl.vexp(y))

    if has_improve_precision:
        res = dsl.cast_to(res, "float16")

    return res


def bessel_i0(x, output, kernel_name="bessel_i0"):
    """bessel_i0"""
    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")

    res = bessel_i0_compute(data_x, output, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = dsl.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x, res]}
    dsl.build(schedule, config)
