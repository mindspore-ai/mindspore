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
"""BesseK1 op"""

from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType
from tbe import dsl
from te import tvm
from te.platform.fusion_manager import fusion_manager

bessel_k1_op_info = TBERegOp("BesselK1") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("bessel_k1.so") \
    .compute_cost(10) \
    .kernel_name("bessel_k1") \
    .partial_flag(True) \
    .op_pattern("formatAgnostic") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_None, DataType.F16_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None) \
    .get_op_info()


@op_info_register(bessel_k1_op_info)
def _bessel_k1_tbe():
    """BesselK1 TBE register"""
    return


AA = [2.77791411276104639959E-18, -2.11142121435816608115E-17,
      1.55363195773620046921E-16, -1.10559694773538630805E-15,
      7.60068429473540693410E-15, -5.04218550472791168711E-14,
      3.22379336594557470981E-13, -1.98397439776494371520E-12,
      1.17361862988909016308E-11, -6.66348972350202774223E-11,
      3.62559028155211703701E-10, -1.88724975172282928790E-9,
      9.38153738649577178388E-9, -4.44505912879632808065E-8,
      2.00329475355213526229E-7, -8.56872026469545474066E-7,
      3.47025130813767847674E-6, -1.32731636560394358279E-5,
      4.78156510755005422638E-5, -1.61760815825896745588E-4,
      5.12285956168575772895E-4, -1.51357245063125314899E-3,
      4.15642294431288815669E-3, -1.05640848946261981558E-2,
      2.47264490306265168283E-2, -5.29459812080949914269E-2,
      1.02643658689847095384E-1, -1.76416518357834055153E-1,
      2.52587186443633654823E-1]

BB = [
    7.51729631084210481353E-18, 4.41434832307170791151E-18,
    -4.65030536848935832153E-17, -3.20952592199342395980E-17,
    2.96262899764595013876E-16, 3.30820231092092828324E-16,
    -1.88035477551078244854E-15, -3.81440307243700780478E-15,
    1.04202769841288027642E-14, 4.27244001671195135429E-14,
    -2.10154184277266431302E-14, -4.08355111109219731823E-13,
    -7.19855177624590851209E-13, 2.03562854414708950722E-12,
    1.41258074366137813316E-11, 3.25260358301548823856E-11,
    -1.89749581235054123450E-11, -5.58974346219658380687E-10,
    -3.83538038596423702205E-9, -2.63146884688951950684E-8,
    -2.51223623787020892529E-7, -3.88256480887769039346E-6,
    -1.10588938762623716291E-4, -9.76109749136146840777E-3,
    7.78576235018280120474E-1]

A = [-7.02386347938628759343E-18, -2.42744985051936593393E-15,
     -6.66690169419932900609E-13, -1.41148839263352776110E-10,
     -2.21338763073472585583E-8, -2.43340614156596823496E-6,
     -1.73028895751305206302E-4, -6.97572385963986435018E-3,
     -1.22611180822657148235E-1, -3.53155960776544875667E-1,
     1.52530022733894777053E0]

B = [-5.75674448366501715755E-18, 1.79405087314755922667E-17,
     -5.68946255844285935196E-17, 1.83809354436663880070E-16,
     -6.05704724837331885336E-16, 2.03870316562433424052E-15,
     -7.01983709041831346144E-15, 2.47715442448130437068E-14,
     -8.97670518232499435011E-14, 3.34841966607842919884E-13,
     -1.28917396095102890680E-12, 5.13963967348173025100E-12,
     -2.12996783842756842877E-11, 9.21831518760500529508E-11,
     -4.19035475934189648750E-10, 2.01504975519703286596E-9,
     -1.03457624656780970260E-8, 5.74108412545004946722E-8,
     -3.50196060308781257119E-7, 2.40648494783721712015E-6,
     -1.93619797416608296024E-5, 1.95215518471351631108E-4,
     -2.85781685962277938680E-3, 1.03923736576817238437E-1,
     2.72062619048444266945E0]

MAX_NUM = 4294967295.0
NUM_TWO = 2.0


def bessel_k1_chebevl(x, n, coef, shape, dtype):
    """chebevl"""
    k1_broad_coef = dsl.broadcast(coef[0], shape, dtype)
    k1_broad_zero = dsl.broadcast(0, shape, dtype)
    k1_none_signal = None
    for i in range(1, n):
        k1_none_signal = k1_broad_zero
        k1_broad_zero = k1_broad_coef
        coef_i = dsl.broadcast(coef[i], shape, dtype)
        k1_broad_coef = dsl.vsub(dsl.vadd(dsl.vmul(x, k1_broad_zero), coef_i), k1_none_signal)
    return dsl.vmuls(dsl.vsub(k1_broad_coef, k1_none_signal), 0.5)


def bessel_i1_compute(input_x):
    """bessel_i1_compute"""
    dtype = input_x.dtype
    shape = input_x.shape

    k1_has_improve_precision = False
    if dtype != "float32":
        input_x = dsl.cast_to(input_x, "float32")
        dtype = "float32"
        k1_has_improve_precision = True

    y = dsl.vabs(input_x)

    y_le_eight = dsl.vmul(y, bessel_k1_chebevl(dsl.vadds(dsl.vmuls(y, 0.5), -2), 29, AA, shape, dtype))
    y_gt_eight = dsl.vmul(bessel_k1_chebevl(dsl.vadds(dsl.vmuls(dsl.vrec(y), 32.0), -2.0), 25, BB,
                                            shape, dtype), dsl.vrsqrt(y))

    y = dsl.vcmpsel(y, 8.0, 'le', y_le_eight, y_gt_eight)
    res = dsl.vcmpsel(input_x, 0, 'lt', dsl.vmuls(y, -1.0), y)
    res = dsl.vmul(res, dsl.vexp(dsl.vabs(input_x)))

    if k1_has_improve_precision:
        res = dsl.cast_to(res, "float16")

    return res


@fusion_manager.register("bessel_k1")
def bessel_k1_compute(input_x, output_y, kernel_name="bessel_k1"):
    """bessel_k1_compute"""
    shape = input_x.shape
    dtype = input_x.dtype

    has_improve_precision = False
    if dtype != "float32":
        input_x = dsl.cast_to(input_x, "float32")
        dtype = "float32"
        has_improve_precision = True

    x_le_two = dsl.vdiv(bessel_k1_chebevl(dsl.vadds(dsl.vmul(input_x, input_x), -2.0), 11, A, shape, dtype), input_x)
    x_le_two = dsl.vadd(dsl.vmul(bessel_i1_compute(input_x), dsl.vlog(dsl.vmuls(input_x, 0.5))), x_le_two)
    x_le_two = dsl.vcmpsel(input_x, 0.0, 'le', MAX_NUM, x_le_two)
    x_gt_two = dsl.vmul(dsl.vmul(dsl.vexp(dsl.vmuls(input_x, -1.0)),
                                 bessel_k1_chebevl(dsl.vadds(dsl.vmuls(dsl.vrec(input_x), 8.0), -2.0), 25,
                                                   B, shape, dtype)), (dsl.vrsqrt(input_x)))

    res = dsl.vcmpsel(input_x, NUM_TWO, 'le', x_le_two, x_gt_two)
    if has_improve_precision:
        res = dsl.cast_to(res, "float16")

    return res


def bessel_k1(x, output, kernel_name="bessel_k1"):
    """bessel_k1"""
    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")

    res = bessel_k1_compute(data_x, output, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = dsl.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x, res]}
    dsl.build(schedule, config)
