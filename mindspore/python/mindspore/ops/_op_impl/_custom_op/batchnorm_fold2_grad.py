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

"""_BatchNormFold2Grad op"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

SHAPE_SIZE_LIMIT = 2147483648

batchnorm_fold2_grad_op_info = TBERegOp("BatchNormFold2GradD") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("batchnorm_fold2_grad.so") \
    .compute_cost(10) \
    .kernel_name("batchnorm_fold2_grad") \
    .partial_flag(True) \
    .input(0, "dout", None, "required", None) \
    .input(1, "dout_reduce", None, "required", None) \
    .input(2, "dout_x_reduce", None, "required", None) \
    .input(3, "gamma", None, "required", None) \
    .input(4, "batch_std", None, "required", None) \
    .input(5, "batch_mean", None, "required", None) \
    .input(6, "running_std", None, "required", None) \
    .output(0, "d_batch_std", True, "required", "all") \
    .output(1, "d_batch_mean", True, "required", "all") \
    .output(2, "d_gamma", True, "required", "all") \
    .output(3, "dx", True, "required", "all") \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD) \
    .get_op_info()


@op_info_register(batchnorm_fold2_grad_op_info)
def _batchnorm_fold2_grad_tbe():
    """_BatchNormFold2Grad TBE register"""
    return


@fusion_manager.register("batchnorm_fold2_grad")
def batchnorm_fold2_grad_compute(dout, dout_reduce, dout_x_reduce, gamma, batch_std, batch_mean, running_std,
                                 kernel_name="batchnorm_fold2_grad"):
    """_BatchNormFold2Grad"""
    shape_x = te.lang.cce.util.shape_to_list(dout.shape)

    d_batch_std_1 = te.lang.cce.vmul(dout_reduce, batch_mean)
    d_batch_std_1 = te.lang.cce.vmul(d_batch_std_1, gamma)
    d_batch_std_2 = te.lang.cce.vmul(dout_x_reduce, running_std)
    d_batch_std = te.lang.cce.vsub(d_batch_std_1, d_batch_std_2)
    d_batch_std = te.lang.cce.vdiv(d_batch_std, batch_std)
    d_batch_std = te.lang.cce.vdiv(d_batch_std, batch_std)

    d_batch_mean = te.lang.cce.vmul(dout_reduce, gamma)
    d_batch_mean = te.lang.cce.vdiv(d_batch_mean, batch_std)
    d_batch_mean = te.lang.cce.vmuls(d_batch_mean, -1.)

    d_gamma = te.lang.cce.vmul(dout_reduce, batch_mean)
    d_gamma = te.lang.cce.vdiv(d_gamma, batch_std)
    d_gamma = te.lang.cce.vmuls(d_gamma, -1.)

    dx = te.lang.cce.vdiv(running_std, batch_std)
    dx = te.lang.cce.broadcast(dx, shape_x)
    dx = te.lang.cce.vmul(dx, dout)
    return [d_batch_std, d_batch_mean, d_gamma, dx]


@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, dict, dict, dict, dict, str)
def batchnorm_fold2_grad(dout, dout_reduce, dout_x_reduce, gamma, batch_std, batch_mean, running_std, d_batch_std,
                         d_batch_mean, d_gamma, dx, kernel_name="batchnorm_fold2_grad"):
    """_BatchNormFold2Grad op """
    shape = dout.get("shape")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
    check_list = ["float16", "float32"]
    inp_dtype = dout.get("dtype").lower()
    if inp_dtype not in check_list:
        raise RuntimeError("Dtype of input only support float16, float32")
    data_format = dout.get("format")
    ori_format = dout.get("ori_format")
    if data_format.upper() not in ("NC1HWC0", "NCHW"):
        raise RuntimeError("Un supported data format {}".format(data_format))
    if data_format.upper() == "NCHW" and ori_format != "NCHW":
        raise RuntimeError("data_format(NCHW) must same as ori_format")
    shape_c = gamma.get("shape")
    if gamma.get("format").upper() == "NCHW":
        shape_c = 1, gamma.get("shape")[0], 1, 1

    dout_t = tvm.placeholder(shape, name="dout", dtype=inp_dtype)
    dout_reduce_t = tvm.placeholder(shape_c, name="dout_reduce", dtype=inp_dtype)
    dout_x_reduce_t = tvm.placeholder(shape_c, name="dout_x_reduce", dtype=inp_dtype)
    gamma_t = tvm.placeholder(shape_c, name="gamma", dtype=inp_dtype)
    batch_std_t = tvm.placeholder(shape_c, name="batch_std", dtype=inp_dtype)
    batch_mean_t = tvm.placeholder(shape_c, name="batch_mean", dtype=inp_dtype)
    running_std_t = tvm.placeholder(shape_c, name="running_std", dtype=inp_dtype)

    res_list = batchnorm_fold2_grad_compute(dout_t, dout_reduce_t, dout_x_reduce_t, gamma_t, batch_std_t, batch_mean_t,
                                            running_std_t, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res_list)

    tensor_list = [dout_t, dout_reduce_t, dout_x_reduce_t, gamma_t, batch_std_t, batch_mean_t, running_std_t] + list(
        res_list)
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(sch, config)
