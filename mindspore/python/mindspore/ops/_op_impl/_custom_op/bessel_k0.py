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
"""BesseK0 op"""

from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType
from tbe import dsl
from te import tvm
from te.platform.fusion_manager import fusion_manager

bessel_k0_op_info = TBERegOp("BesselK0") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("bessel_k0.so") \
    .compute_cost(10) \
    .kernel_name("bessel_k0") \
    .partial_flag(True) \
    .op_pattern("formatAgnostic") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_None, DataType.F16_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None) \
    .get_op_info()


@op_info_register(bessel_k0_op_info)
def _bessel_k0_tbe():
    """BesselK0 TBE register"""
    return


A = [1.37446543561352307156E-16,
     4.25981614279661018399E-14,
     1.03496952576338420167E-11,
     1.90451637722020886025E-9,
     2.53479107902614945675E-7,
     2.28621210311945178607E-5,
     1.26461541144692592338E-3,
     3.59799365153615016266E-2,
     3.44289899924628486886E-1,
     -5.35327393233902768720E-1]
B = [
    5.30043377268626276149E-18, -1.64758043015242134646E-17,
    5.21039150503902756861E-17, -1.67823109680541210385E-16,
    5.51205597852431940784E-16, -1.84859337734377901440E-15,
    6.34007647740507060557E-15, -2.22751332699166985548E-14,
    8.03289077536357521100E-14, -2.98009692317273043925E-13,
    1.14034058820847496303E-12, -4.51459788337394416547E-12,
    1.85594911495471785253E-11, -7.95748924447710747776E-11,
    3.57739728140030116597E-10, -1.69753450938905987466E-9,
    8.57403401741422608519E-9, -4.66048989768794782956E-8,
    2.76681363944501510342E-7, -1.83175552271911948767E-6,
    1.39498137188764993662E-5, -1.28495495816278026384E-4,
    1.56988388573005337491E-3, -3.14481013119645005427E-2,
    2.44030308206595545468E0]

AA = [-4.41534164647933937950E-18, 3.33079451882223809783E-17,
      -2.43127984654795469359E-16, 1.71539128555513303061E-15,
      -1.16853328779934516808E-14, 7.67618549860493561688E-14,
      -4.85644678311192946090E-13, 2.95505266312963983461E-12,
      -1.72682629144155570723E-11, 9.67580903537323691224E-11,
      -5.18979560163526290666E-10, 2.65982372468238665035E-9,
      -1.30002500998624804212E-8, 6.04699502254191894932E-8,
      -2.67079385394061173391E-7, 1.11738753912010371815E-6,
      -4.41673835845875056359E-6, 1.64484480707288970893E-5,
      -5.75419501008210370398E-5, 1.88502885095841655729E-4,
      -5.76375574538582365885E-4, 1.63947561694133579842E-3,
      -4.32430999505057594430E-3, 1.05464603945949983183E-2,
      -2.37374148058994688156E-2, 4.93052842396707084878E-2,
      -9.49010970480476444210E-2, 1.71620901522208775349E-1,
      -3.04682672343198398683E-1, 6.76795274409476084995E-1
      ]
BB = [
    -7.23318048787475395456E-18, -4.83050448594418207126E-18,
    4.46562142029675999901E-17, 3.46122286769746109310E-17,
    -2.82762398051658348494E-16, -3.42548561967721913462E-16,
    1.77256013305652638360E-15, 3.81168066935262242075E-15,
    -9.55484669882830764870E-15, -4.15056934728722208663E-14,
    1.54008621752140982691E-14, 3.85277838274214270114E-13,
    7.18012445138366623367E-13, -1.79417853150680611778E-12,
    -1.32158118404477131188E-11, -3.14991652796324136454E-11,
    1.18891471078464383424E-11, 4.94060238822496958910E-10,
    3.39623202570838634515E-9, 2.26666899049817806459E-8,
    2.04891858946906374183E-7, 2.89137052083475648297E-6,
    6.88975834691682398426E-5, 3.36911647825569408990E-3,
    8.04490411014108831608E-1]

MAXNUM = 4294967295.0
TWO = 2.0


def chebevl(x, n, coef, shape, dtype):
    """chebevl"""
    k0_broad_coef = dsl.broadcast(coef[0], shape, dtype)
    k0_broad_zero = dsl.broadcast(0, shape, dtype)
    k0_none_signal = None
    for i in range(1, n):
        k0_none_signal = k0_broad_zero
        k0_broad_zero = k0_broad_coef
        coef_i = dsl.broadcast(coef[i], shape, dtype)
        k0_broad_coef = dsl.vsub(dsl.vadd(dsl.vmul(x, k0_broad_zero), coef_i), k0_none_signal)
    return dsl.vmuls(dsl.vsub(k0_broad_coef, k0_none_signal), 0.5)


def bessel_i0_compute(input_x):
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
    y_le_eight = chebevl(y_le_eight_in, 30, AA, shape, dtype)

    y_gt_eight_in = dsl.vadds(dsl.vmuls(dsl.vrec(y), 32.0), -2.0)
    y_gt_eight = chebevl(y_gt_eight_in, 25, BB, shape, dtype)
    y_gt_eight = dsl.vmul(y_gt_eight, dsl.vrsqrt(y))

    res = dsl.vcmpsel(y, 8.0, 'le', y_le_eight, y_gt_eight)
    res = dsl.vmul(res, dsl.vexp(y))

    if has_improve_precision:
        res = dsl.cast_to(res, "float16")

    return res


@fusion_manager.register("bessel_k0")
def bessel_k0_compute(input_x, output_y, kernel_name="bessel_k0"):
    """bessel_k0_compute"""
    shape = input_x.shape
    dtype = input_x.dtype

    has_improve_precision = False
    if dtype != "float32":
        input_x = dsl.cast_to(input_x, "float32")
        dtype = "float32"
        has_improve_precision = True

    x_le_two = chebevl(dsl.vadds(dsl.vmul(input_x, input_x), -2.0), 10, A, shape, dtype)
    x_le_two = dsl.vadd(dsl.vmul(bessel_i0_compute(input_x),
                                 dsl.vmuls(dsl.vlog(dsl.vmuls(input_x, 0.5)), -1.0)), x_le_two)
    x_le_two = dsl.vcmpsel(input_x, 0.0, 'le', MAXNUM, x_le_two)
    x_gt_two = dsl.vmul(dsl.vmul(dsl.vexp(dsl.vmuls(input_x, -1.0)),
                                 chebevl(dsl.vadds(dsl.vmuls(dsl.vrec(input_x), 8.0), -2.0), 25, B, shape, dtype)),
                        (dsl.vrsqrt(input_x)))

    res = dsl.vcmpsel(input_x, TWO, 'le', x_le_two, x_gt_two)
    if has_improve_precision:
        res = dsl.cast_to(res, "float16")

    return res


def bessel_k0(x, output, kernel_name="bessel_k0"):
    """bessel_k0"""
    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")

    res = bessel_k0_compute(data_x, output, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = dsl.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x, res]}
    dsl.build(schedule, config)
