# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Utitly functions to help distribution class."""
import numpy as np
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

def exp_generic(input_x):
    """
    Log op on Ascend doesn't supprot int types.
    Fix this with casting the type.
    """
    exp = P.Exp()
    cast = P.Cast()

    input_x = cast(input_x, mstype.float32)
    return exp(input_x)


def expm1_generic(input_x):
    """
    Expm1 ops under GPU context.
    """
    return exp_generic(input_x) - 1.0


def log_generic(input_x):
    """
    Log op on Ascend is calculated as log(abs(x)).
    Fix this with putting negative values as nan.
    And log op on Ascend doesn't supprot int types.
    Fix this with casting the type.
    """
    log = P.Log()
    less = P.Less()
    lessequal = P.LessEqual()
    fill = P.Fill()
    cast = P.Cast()
    dtype = P.DType()
    shape = P.Shape()
    select = P.Select()

    input_x = cast(input_x, mstype.float32)
    nan = fill(dtype(input_x), shape(input_x), np.nan)
    inf = fill(dtype(input_x), shape(input_x), np.inf)
    neg_x = less(input_x, 0.0)
    nonpos_x = lessequal(input_x, 0.0)
    log_x = log(input_x)
    result = select(nonpos_x, -inf, log_x)
    return select(neg_x, nan, result)


def log1p_generic(x):
    """
    Log1p ops on GPU device or when device_target == GPU.
    """
    return log_generic(x + 1.0)


def _evaluate_polynomial(x, coefficients):
    poly = 0
    for co in coefficients:
        poly = poly * x + co
    return  poly


def erf_f32_generic(x):
    """
    Calculate erf for dtype of f32
    """
    k_erf_tcoefficient = [+7.853861353153693e-5,
                          -8.010193625184903e-4,
                          +5.188327685732524e-3,
                          -2.685381193529856e-2,
                          +1.128358514861418e-1,
                          -3.761262582423300e-1,
                          +1.128379165726710e+0]

    poly = _evaluate_polynomial(x * x, k_erf_tcoefficient)
    return x * poly


def erf_f64_generic(x):
    """
    Calculate erf for dtype of f64
    """
    k_erf_tcoefficient = [9.60497373987051638749e0,
                          9.00260197203842689217e1,
                          2.23200534594684319226e3,
                          7.00332514112805075473e3,
                          5.55923013010394962768e4]
    k_erf_ucoefficient = [1.00000000000000000000e0,
                          3.35617141647503099647e1,
                          5.21357949780152679795e2,
                          4.59432382970980127987e3,
                          2.26290000613890934246e4,
                          4.92673942608635921086e4]
    z = x * x
    poly1 = _evaluate_polynomial(z, k_erf_tcoefficient)
    poly2 = _evaluate_polynomial(z, k_erf_ucoefficient)
    return x * poly1 / poly2


def erfc_f32_generic(x):
    """
    Calculate erfc for dtype of f32
    """
    k_maxlog = 88.72283905206835
    k_erfc_pcoefficient = [+2.326819970068386e-2,
                           -1.387039388740657e-1,
                           +3.687424674597105e-1,
                           -5.824733027278666e-1,
                           +6.210004621745983e-1,
                           -4.944515323274145e-1,
                           +3.404879937665872e-1,
                           -2.741127028184656e-1,
                           +5.638259427386472e-1]
    k_erfc_rcoefficient = [-1.047766399936249e+1,
                           +1.297719955372516e+1,
                           -7.495518717768503e+0,
                           +2.921019019210786e+0,
                           -1.015265279202700e+0,
                           +4.218463358204948e-1,
                           -2.820767439740514e-1,
                           +5.641895067754075e-1]
    abs_cal = P.Abs()
    select = P.Select()
    less = P.Less()
    fill = P.Fill()
    dtype = P.DType()
    shape = P.Shape()

    abs_x = abs_cal(x)
    z = exp_generic(-x * x)
    q = 1 / abs_x
    y = q * q
    poly1 = _evaluate_polynomial(y, k_erfc_pcoefficient)
    poly2 = _evaluate_polynomial(y, k_erfc_rcoefficient)
    p = select(less(abs_x, 2.0), poly1, poly2)
    y = z * q * p
    zeros = fill(dtype(x), shape(x), 0)
    y_clamp = select(less(z, -k_maxlog), zeros, y)
    return select(less(x, 0), 2.0 - y_clamp, y_clamp)


def erfc_f64_generic(x):
    """
    Calculate erfc for dtype of f64
    """
    k_maxlog = 7.09782712893383996843e2
    k_erfc_pcoefficient = [2.46196981473530512524e-10,
                           5.64189564831068821977e-1,
                           7.46321056442269912687e0,
                           4.86371970985681366614e1,
                           1.96520832956077098242e2,
                           5.26445194995477358631e2,
                           9.34528527171957607540e2,
                           1.02755188689515710272e3,
                           5.57535335369399327526e2]
    k_erfc_qcoefficient = [1.00000000000000000000e0,
                           1.32281951154744992508e1,
                           8.67072140885989742329e1,
                           3.54937778887819891062e2,
                           9.75708501743205489753e2,
                           1.82390916687909736289e3,
                           2.24633760818710981792e3,
                           1.65666309194161350182e3,
                           5.57535340817727675546e2]
    k_erfc_rcoefficient = [5.64189583547755073984e-1,
                           1.27536670759978104416e0,
                           5.01905042251180477414e0,
                           6.16021097993053585195e0,
                           7.40974269950448939160e0,
                           2.97886665372100240670e0]
    k_erfc_scoefficient = [1.00000000000000000000e0,
                           2.26052863220117276590e0,
                           9.39603524938001434673e0,
                           1.20489539808096656605e1,
                           1.70814450747565897222e1,
                           9.60896809063285878198e0,
                           3.36907645100081516050e02]
    abs_cal = P.Abs()
    select = P.Select()
    less = P.Less()
    fill = P.Fill()
    dtype = P.DType()
    shape = P.Shape()

    abs_x = abs_cal(x)
    z = -x * x
    exp_z = exp_generic(z)

    temp1 = exp_z * _evaluate_polynomial(abs_x, k_erfc_pcoefficient) / _evaluate_polynomial(abs_x, k_erfc_qcoefficient)
    temp2 = exp_z * _evaluate_polynomial(abs_x, k_erfc_rcoefficient) / _evaluate_polynomial(abs_x, k_erfc_scoefficient)
    y = select(less(abs_x, 8.0), temp1, temp2)
    zeros = fill(dtype(x), shape(x), 0)
    y_clamp = select(less(z, k_maxlog), zeros, y)

    poly2 = _evaluate_polynomial(y, k_erfc_rcoefficient)
    p = select(less(abs_x, 2.0), poly1, poly2)
    y = z * q * p
    zeros = fill(dtype(x), shape(x), 0)
    y_clamp = select(less(z, -k_maxlog), zeros, y)
    return select(less(x, 0), 2.0 - y_clamp, y_clamp)


def erfc_generic(x):
    select = P.Select()
    greater = P.Greater()
    abs_cal = P.Abs()

    return select(greater(abs_cal(x), 1), erfc_f32_generic(x), 1 - erf_f32_generic(x))


def erf_generic(x):
    select = P.Select()
    less = P.Less()
    abs_cal = P.Abs()

    return select(less(abs_cal(x), 1), erf_f32_generic(x), 1 - erfc_f32_generic(x))
