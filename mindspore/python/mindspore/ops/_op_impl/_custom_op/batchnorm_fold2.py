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

"""_BatchNormFold2 op"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

SHAPE_SIZE_LIMIT = 2147483648

batchnorm_fold2_op_info = TBERegOp("BatchNormFold2D") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("batchnorm_fold2.so") \
    .compute_cost(10) \
    .kernel_name("batchnorm_fold2") \
    .partial_flag(True) \
    .input(0, "x", None, "required", None) \
    .input(1, "beta", None, "required", None) \
    .input(2, "gamma", None, "required", None) \
    .input(3, "batch_std", None, "required", None) \
    .input(4, "batch_mean", None, "required", None) \
    .input(5, "running_std", None, "required", None) \
    .output(0, "y", True, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD,
                  DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(batchnorm_fold2_op_info)
def _batchnorm_fold2_tbe():
    """_BatchNormFold2 TBE register"""
    return


@fusion_manager.register("batchnorm_fold2")
def batchnorm_fold2_compute(x, beta, gamma, batch_std, batch_mean, running_std, kernel_name="batchnorm_fold2"):
    """_BatchNormFold2 compute"""
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    factor = te.lang.cce.vdiv(running_std, batch_std)
    factor_b = te.lang.cce.broadcast(factor, shape_x)
    res = te.lang.cce.vmul(x, factor_b)
    bias = te.lang.cce.vdiv(batch_mean, batch_std)
    bias = te.lang.cce.vmul(bias, gamma)
    bias = te.lang.cce.vsub(beta, bias)
    bias_b = te.lang.cce.broadcast(bias, shape_x)
    res = te.lang.cce.vadd(res, bias_b)
    return res


@util.check_input_type(dict, dict, dict, dict, dict, dict, dict, str)
def batchnorm_fold2(x, beta, gamma, batch_std, batch_mean, running_std, y, kernel_name="batchnorm_fold2"):
    """_BatchNormFold2 op"""
    shape = x.get("shape")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
    check_list = ["float16", "float32"]
    inp_dtype = x.get("dtype").lower()
    if inp_dtype not in check_list:
        raise RuntimeError("Dtype of input only support float16, float32")
    data_format = x.get("format")
    ori_format = x.get("ori_format")
    if data_format.upper() not in ("NC1HWC0", "NCHW"):
        raise RuntimeError("Un supported data format {}".format(data_format))
    if data_format.upper() == "NCHW" and ori_format != "NCHW":
        raise RuntimeError("data_format(NCHW) must same as ori_format")
    shape_c = gamma.get("shape")
    if gamma.get("format").upper() == "NCHW":
        shape_c = 1, gamma.get("shape")[0], 1, 1
    x_t = tvm.placeholder(shape, name="x", dtype=inp_dtype)
    beta_t = tvm.placeholder(shape_c, name="beta", dtype=inp_dtype)
    gamma_t = tvm.placeholder(shape_c, name="gamma", dtype=inp_dtype)
    batch_std_t = tvm.placeholder(shape_c, name="batch_std", dtype=inp_dtype)
    batch_mean_t = tvm.placeholder(shape_c, name="batch_mean", dtype=inp_dtype)
    running_std_t = tvm.placeholder(shape_c, name="running_std", dtype=inp_dtype)

    res = batchnorm_fold2_compute(x_t, beta_t, gamma_t, batch_std_t, batch_mean_t,
                                  running_std_t, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [x_t, beta_t, gamma_t, batch_std_t, batch_mean_t, running_std_t, res]}

    te.lang.cce.cce_build_code(sch, config)
