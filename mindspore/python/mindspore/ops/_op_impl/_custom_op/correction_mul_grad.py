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

"""CorrectionMul op"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

SHAPE_SIZE_LIMIT = 2147483648

correction_mul_grad_op_info = TBERegOp("CorrectionMulGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("correction_mul_grad.so") \
    .compute_cost(10) \
    .kernel_name("correction_mul_grad") \
    .partial_flag(True) \
    .attr("channel_axis", "optional", "int", "all") \
    .input(0, "dout", None, "required", None) \
    .input(1, "x", None, "required", None) \
    .input(2, "batch_std", None, "required", None) \
    .input(3, "running_std", None, "required", None) \
    .output(0, "dx", True, "required", "all") \
    .output(1, "mul_dx", True, "required", "all") \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(correction_mul_grad_op_info)
def _correction_mul_grad_tbe():
    """CorrectionMulGrad TBE register"""
    return


@fusion_manager.register("correction_mul_grad")
def correction_mul_grad_compute(dout, x, batch_std, running_std, channel, data_format, kernel_name="correction_mul"):
    """CorrectionMulGrad compute"""
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    factor = te.lang.cce.vdiv(batch_std, running_std)
    factor_b = te.lang.cce.broadcast(factor, shape_x)
    dx = te.lang.cce.vmul(dout, factor_b)
    mul_dx = te.lang.cce.vmul(dout, x)
    running_std_b = te.lang.cce.broadcast(running_std, shape_x)
    mul_dx = te.lang.cce.vdiv(mul_dx, running_std_b)
    return [dx, mul_dx]


@util.check_input_type(dict, dict, dict, dict, dict, dict, int, str)
def correction_mul_grad(dout, x, batch_std, running_std, dx, mul_dx, channel, kernel_name="correction_mul_grad"):
    """CorrectionMulGrad op"""
    shape_dout = dout.get("shape")
    shape_x = dout.get("shape")

    dtype_dout = dout.get("dtype")
    dtype_x = x.get("dtype")
    dtype_batch_std = batch_std.get("dtype")
    dtype_running_std = running_std.get("dtype")

    inp_dtype_dout = dtype_dout.lower()
    inp_dtype_x = dtype_x.lower()
    inp_dtype_batch_std = dtype_batch_std.lower()
    inp_dtype_running_std = dtype_running_std.lower()

    util.check_dtype_rule(inp_dtype_dout, ("float16", "float32"))
    util.check_dtype_rule(inp_dtype_x, ("float16", "float32"))
    util.check_dtype_rule(inp_dtype_batch_std, ("float16", "float32"))
    util.check_dtype_rule(inp_dtype_running_std, ("float16", "float32"))
    util.compare_tensor_dict_key(dout, x, "dtype")
    util.compare_tensor_dict_key(dout, x, "shape")
    util.compare_tensor_dict_key(dx, x, "shape")
    util.compare_tensor_dict_key(batch_std, running_std, "shape")
    util.compare_tensor_dict_key(dx, mul_dx, "shape")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)

    data_format = dout.get("format")
    ori_format = dout.get("format")
    if data_format.upper() not in ("NC1HWC0", "NCHW"):
        raise RuntimeError("Un supported data format {}".format(data_format))
    if data_format.upper() == "NCHW" and ori_format != "NCHW":
        raise RuntimeError("data_format(NCHW) must same as ori_format")

    shape_c = [1] * len(shape_x)
    shape_c[channel] = batch_std.get("ori_shape")[0]
    if data_format == "NC1HWC0" and channel == 1:
        shape_c = batch_std.get("shape")

    dout_t = tvm.placeholder(shape_dout, name="dout", dtype=inp_dtype_dout)
    x_t = tvm.placeholder(shape_x, name="x", dtype=inp_dtype_x)
    batch_std_t = tvm.placeholder(shape_c, name="batch_std", dtype=inp_dtype_batch_std)
    running_std_t = tvm.placeholder(shape_c, name="running_std", dtype=inp_dtype_running_std)
    res_list = correction_mul_grad_compute(dout_t, x_t, batch_std_t, running_std_t, channel, data_format, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res_list)

    tensor_list = [dout_t, x_t, batch_std_t, running_std_t] + res_list
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(sch, config)


correction_mul_grad_reduce_op_info = TBERegOp("CorrectionMulGradReduce") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("correction_mul_grad_reduce.so") \
    .compute_cost(10) \
    .kernel_name("correction_mul_grad_reduce") \
    .partial_flag(True) \
    .attr("channel_axis", "optional", "int", "all") \
    .input(0, "dout", None, "required", None) \
    .output(0, "d_batch_std", True, "required", "all") \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(correction_mul_grad_reduce_op_info)
def _correction_mul_grad_reduce_tbe():
    """CorrectionMulGradReduce TBE register"""
    return


@fusion_manager.register("correction_mul_grad_reduce")
def correction_mul_grad_reduce_compute(mul_dx, channel, data_format, kernel_name="correction_mul"):
    """CorrectionMulGradReduce compute"""
    if channel == 0:
        if data_format == "NCHW":
            axis = [1, 2, 3]
        else:
            axis = [1, 2, 3, 4]
    else:
        axis = [2, 3]
    d_batch_std = te.lang.cce.sum(mul_dx, axis, keepdims=True)
    return d_batch_std


@util.check_input_type(dict, dict, int, str)
def correction_mul_grad_reduce(mul_dx, d_batch_std, channel, kernel_name="correction_mul_grad_reduce"):
    """CorrectionMulGradReduce op"""
    shape_dout = mul_dx.get("shape")
    shape_x = mul_dx.get("shape")

    dtype_dout = mul_dx.get("dtype")

    inp_dtype_dout = dtype_dout.lower()

    util.check_dtype_rule(inp_dtype_dout, ("float16", "float32"))

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_x)
    util.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)

    data_format = mul_dx.get("format")
    ori_format = mul_dx.get("format")
    if data_format.upper() not in ("NC1HWC0", "NCHW"):
        raise RuntimeError("Un supported data format {}".format(data_format))
    if data_format.upper() == "NCHW" and ori_format != "NCHW":
        raise RuntimeError("data_format(NCHW) must same as ori_format")

    shape_c = [1] * len(shape_x)
    shape_c[channel] = d_batch_std.get("ori_shape")[0]
    if data_format == "NC1HWC0" and channel == 1:
        shape_c = d_batch_std.get("shape")

    dout_t = tvm.placeholder(shape_dout, name="dout", dtype=inp_dtype_dout)
    res = correction_mul_grad_reduce_compute(dout_t, channel, data_format, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    tensor_list = [dout_t, res]
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(sch, config)
