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

"""_BatchNormFoldGrad op"""

from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType
import te.lang.cce
from te import tvm
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util

batch_norm_op_info = TBERegOp("BatchNormFoldGradD") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("batchnorm_fold_grad.so") \
    .compute_cost(10) \
    .kernel_name("batchnorm_fold_grad") \
    .partial_flag(True) \
    .attr("epsilon", "optional", "float", "all") \
    .attr("is_training", "optional", "bool", "all") \
    .attr("freeze_bn", "optional", "int", "all") \
    .input(0, "d_batch_mean", False, "required", "all") \
    .input(1, "d_batch_std", False, "required", "all") \
    .input(2, "x", False, "required", "all") \
    .input(3, "batch_mean", False, "required", "all") \
    .input(4, "batch_std", False, "required", "all") \
    .output(0, "dx", False, "required", "all") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F16_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F16_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(batch_norm_op_info)
def _batchnorm_fold_grad_tbe():
    """_BatchNormFoldGrad TBE register"""
    return


def _batchnorm_fold_grad_compute(d_batch_mean, d_batch_std, data_x, batch_mean, batch_std):
    """_batchnorm_fold_grad_compute """
    shape_x = te.lang.cce.util.shape_to_list(data_x.shape)
    normal_size = shape_x[0] * shape_x[2] * shape_x[3]

    d_batch_mean_broad = te.lang.cce.broadcast(d_batch_mean, shape_x)
    d_batch_std_broad = te.lang.cce.broadcast(d_batch_std, shape_x)
    batch_mean_broad = te.lang.cce.broadcast(batch_mean, shape_x)
    batch_std_broad = te.lang.cce.broadcast(batch_std, shape_x)

    dx = te.lang.cce.vsub(data_x, batch_mean_broad)
    dx = te.lang.cce.vmul(dx, d_batch_std_broad)
    dx = te.lang.cce.vdiv(dx, batch_std_broad)
    dx = te.lang.cce.vadd(dx, d_batch_mean_broad)
    dx = te.lang.cce.vmuls(dx, tvm.const(1. / normal_size, dtype=dx.dtype))
    return [dx]


@util.check_input_type(dict, dict, dict, dict, dict, dict,
                       float, bool, int, str)
def batchnorm_fold_grad(d_batch_mean, d_batch_std, x, batch_mean, batch_std, dx,
                        epsilon=1e-5, is_training=True, freeze_bn=0, kernel_name="batchnorm_fold_grad"):
    """batchnorm_fold_grad op """
    util.check_kernel_name(kernel_name)
    for iv in (d_batch_mean, d_batch_std, x, batch_mean, batch_std):
        util.check_shape_rule(iv.get("shape"))
        util.check_tensor_shape_size(iv.get("shape"))
    check_tuple = ("float16", "float32")
    for iv in (d_batch_mean, d_batch_std, x, batch_mean, batch_std):
        util.check_dtype_rule(iv.get("dtype").lower(), check_tuple)

    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    format_data = x.get("format").upper()
    if format_data not in ("NCHW", "NC1HWC0"):
        raise RuntimeError("Format of input only support 4D and 5HD")

    shape_mean = d_batch_mean.get("shape")
    dtype_mean = d_batch_mean.get("dtype").lower()
    if format_data == "NC1HWC0":
        if len(shape_x) != 5:
            raise RuntimeError("batchnorm_fold grad only support shape 5D"
                               "when input format is NC1HWC0")
        shape_mean = (1, shape_x[1], 1, 1, shape_x[4])
    elif format_data == "NCHW":
        if len(shape_x) < 2 or len(shape_x) > 4:
            raise RuntimeError("batchnorm_fold grad only support shape 2D to 4D")
        if shape_x[1] != shape_mean[0]:
            raise RuntimeError("data_format is NCHW, shape_bias must"
                               "be equal to the second axis of shape_x")
        shape_mean = (1, shape_x[1],)
        for _ in range(2, len(shape_x)):
            shape_mean = shape_mean + (1,)

    d_batch_mean = tvm.placeholder(shape_mean, name="d_batch_mean", dtype=dtype_mean)
    d_batch_std = tvm.placeholder(shape_mean, name="d_batch_std", dtype=dtype_mean)
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x.lower())
    batch_mean = tvm.placeholder(shape_mean, name="batch_mean", dtype=dtype_mean)
    batch_std = tvm.placeholder(shape_mean, name="batch_std", dtype=dtype_mean)

    res = _batchnorm_fold_grad_compute(d_batch_mean, d_batch_std, data_x, batch_mean, batch_std)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    tensor_list = [d_batch_mean, d_batch_std, data_x, batch_mean, batch_std] + res
    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
