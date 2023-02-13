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

"""_BatchNormFold op"""
from __future__ import absolute_import

import te
from te import tvm
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

batch_norm_op_info = TBERegOp("BatchNormFoldD") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("batchnorm_fold.so") \
    .compute_cost(10) \
    .kernel_name("batchnorm_fold") \
    .partial_flag(True) \
    .attr("momentum", "optional", "float", "all", "0.9") \
    .attr("epsilon", "optional", "float", "all", "0.00001") \
    .attr("is_training", "optional", "bool", "all", "true") \
    .attr("freeze_bn", "optional", "int", "all", "0") \
    .attr("format", "optional", "str", "all", "NCHW") \
    .input(0, "x", False, "required", "all") \
    .input(1, "x_sum", False, "required", "all") \
    .input(2, "x_square_sum", False, "required", "all") \
    .input(3, "mean", False, "required", "all") \
    .input(4, "variance", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .output(1, "batch_mean", False, "required", "all") \
    .output(2, "batch_std", False, "required", "all") \
    .output(3, "running_mean", False, "required", "all") \
    .output(4, "running_std", False, "required", "all") \
    .output(5, "mean_updated", False, "required", "all") \
    .output(6, "variance_updated", False, "required", "all") \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(batch_norm_op_info)
def _batchnorm_fold_tbe():
    """_BatchNormFold TBE register"""
    return


def _batchnorm_fold_compute(x_input, x_sum, x_square_sum, mean, variance, momentum, epsilon):
    """_batchnorm_fold_compute"""
    shape_x = te.lang.cce.util.shape_to_list(x_input.shape)
    num = shape_x[0] * shape_x[2] * shape_x[3]
    if num == 0.0:
        raise ValueError('`num` is zero, which leads to divide zero error.')
    num_rec = 1.0 / num

    # compute the mean of x
    batch_mean = te.lang.cce.vmuls(x_sum, num_rec)

    # compute the variance of x
    variance_div = te.lang.cce.vmuls(x_square_sum, num_rec)
    mean_square = te.lang.cce.vmul(batch_mean, batch_mean)
    batch_var_biased = te.lang.cce.vsub(variance_div, mean_square)
    batch_std = te.lang.cce.vsqrt(te.lang.cce.vadds(batch_var_biased, epsilon))
    if num == 1:
        batch_var_scaler = 0.0
    else:
        batch_var_scaler = float(num) / (num - 1)
    batch_var_unbiased = te.lang.cce.vmuls(batch_var_biased, batch_var_scaler)

    factor = 1.0 - momentum
    factor_reverse = momentum
    mean_mul = te.lang.cce.vmuls(batch_mean, factor)
    mean_mul_rev = te.lang.cce.vmuls(mean, factor_reverse)
    mean_updated = te.lang.cce.vadd(mean_mul, mean_mul_rev)

    var_mul = te.lang.cce.vmuls(batch_var_unbiased, factor)
    var_mul_rev = te.lang.cce.vmuls(variance, factor_reverse)
    variance_updated = te.lang.cce.vadd(var_mul, var_mul_rev)

    y = te.lang.cce.vadds(x_input, 0.0)
    running_mean = te.lang.cce.vadds(mean, 0.0)
    running_std = te.lang.cce.vsqrt(te.lang.cce.vadds(variance, epsilon))
    res = [y, batch_mean, batch_std, running_mean, running_std, mean_updated, variance_updated]
    return res


@util.check_input_type(dict, dict, dict, dict, dict,
                       dict, dict, dict, dict, dict, dict, dict,
                       float, float, bool, int, str, str)
def batchnorm_fold(x, x_sum, x_square_sum, mean, variance,
                   y, batch_mean, batch_std, running_mean, running_std, mean_updated, variance_updated,
                   momentum=0.9, epsilon=1e-5, is_training=True, freeze_bn=0, data_format="NCHW",
                   kernel_name="batchnorm_fold"):
    """batchnorm_fold TBE op"""
    util.check_kernel_name(kernel_name)
    data_format = data_format.upper()
    if data_format != "NCHW":
        raise RuntimeError("The data_format only support NCHW")

    shape_x = x.get("shape")
    shape_mean = mean.get("shape")
    shape_variance = variance.get("shape")
    dtype_x = x.get("dtype")
    dtype_mean = mean.get("dtype")
    dtype_variance = variance.get("dtype")
    for shape in (shape_x, shape_mean, shape_variance):
        util.check_shape_rule(shape)
        util.check_tensor_shape_size(shape)
    check_tuple = ("float16", "float32")
    for dtype in (dtype_x, dtype_mean, dtype_variance):
        util.check_dtype_rule(dtype.lower(), check_tuple)

    format_data = x.get("format").upper()
    if format_data not in ("NCHW", "NC1HWC0"):
        raise RuntimeError("Format of input only support 4D and 5HD")

    if format_data == "NC1HWC0":
        if len(shape_x) != 5:
            raise RuntimeError("batchnorm_fold only support shape 5D"
                               "when input format is NC1HWC0")
        shape_mean = (1, shape_x[1], 1, 1, shape_x[4])
    elif format_data == "NCHW":
        if len(shape_x) < 2 or len(shape_x) > 4:
            raise RuntimeError("batchnorm_fold only support shape 2D to 4D")
        if shape_x[1] != shape_mean[0]:
            raise RuntimeError("data_format is NCHW, shape_bias must"
                               "be equal to the second axis of shape_x")
        shape_mean = (1, shape_x[1],)
        for _ in range(2, len(shape_x)):
            shape_mean = shape_mean + (1,)

    x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x.lower())
    x_sum = tvm.placeholder(shape_mean, name="x_sum", dtype=dtype_x.lower())
    x_square_sum = tvm.placeholder(shape_mean, name="x_square_sum", dtype=dtype_x.lower())
    mean = tvm.placeholder(shape_mean, name="mean", dtype=dtype_mean.lower())
    variance = tvm.placeholder(shape_mean, name="variance", dtype=dtype_variance.lower())

    res = _batchnorm_fold_compute(x_input, x_sum, x_square_sum, mean, variance, momentum, epsilon)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)
    config = {"name": kernel_name,
              "tensor_list": [x_input, x_sum, x_square_sum, mean, variance] + res}
    te.lang.cce.cce_build_code(sch, config)
