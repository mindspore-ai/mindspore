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

correction_mul_op_info = TBERegOp("CorrectionMul") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("correction_mul.so") \
    .compute_cost(10) \
    .kernel_name("correction_mul") \
    .partial_flag(True) \
    .attr("channel_axis", "optional", "int", "all") \
    .input(0, "x", None, "required", None) \
    .input(1, "batch_std", None, "required", None) \
    .input(2, "running_std", None, "required", None) \
    .output(0, "y", True, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(correction_mul_op_info)
def _correction_mul_tbe():
    """CorrectionMul TBE register"""
    return


@fusion_manager.register("correction_mul")
def correction_mul_compute(x, batch_std, running_std, kernel_name="correction_mul"):
    """CorrectionMul compute"""
    shape_x = te.lang.cce.util.shape_to_list(x.shape)
    factor = te.lang.cce.vdiv(batch_std, running_std)
    factor_b = te.lang.cce.broadcast(factor, shape_x)
    res = te.lang.cce.vmul(x, factor_b)
    return res


@util.check_input_type(dict, dict, dict, dict, int, str)
def correction_mul(x, batch_std, running_std, y, channel, kernel_name="correction_mul"):
    """CorrectionMul op"""
    shape = x.get("shape")
    data_format = x.get("format")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape)
    util.check_shape_size(shape, SHAPE_SIZE_LIMIT)
    inp_dtype = x.get("dtype").lower()
    if inp_dtype not in ["float16", "float32"]:
        raise RuntimeError("Dtype of input only support float16, float32")

    x_t = tvm.placeholder(shape, name="x", dtype=inp_dtype)
    shape_c = [1] * len(shape)
    shape_c[channel] = batch_std.get("ori_shape")[0]
    if data_format == "NC1HWC0" and channel == 1:
        shape_c = batch_std.get("shape")
    batch_std_t = tvm.placeholder(shape_c, name="batch_std", dtype=inp_dtype)
    running_std_t = tvm.placeholder(shape_c, name="running_std", dtype=inp_dtype)
    res = correction_mul_compute(x_t, batch_std_t, running_std_t, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [x_t, batch_std_t, running_std_t, res]}

    te.lang.cce.cce_build_code(sch, config)
