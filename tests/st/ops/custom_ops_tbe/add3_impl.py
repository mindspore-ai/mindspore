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
from __future__ import absolute_import
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

@fusion_manager.register("add3")
def add3_compute(input1, input2, const_bias):
    sum2 = te.lang.cce.vadd(input1, input2)
    sum3 = te.lang.cce.vadds(sum2, tvm.const(const_bias, dtype=input1.dtype))
    return sum3


cus_add3_op_info = TBERegOp("CusAdd3") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("add3.so") \
    .compute_cost(10) \
    .kernel_name("CusAdd3Impl") \
    .partial_flag(True) \
    .attr("const_bias", "required", "float", "all") \
    .input(0, "input1", False, "required", "all") \
    .input(1, "input2", False, "required", "all") \
    .output(0, "sum", False, "required", "all") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .get_op_info()


@op_info_register(cus_add3_op_info)
def CusAdd3Impl(input1, inptu2, sum1, const_bias, kernel_name="CusAdd3Impl"):
    shape = input1.get("shape")
    shape = util.shape_refine(shape)
    dtype = input1.get("dtype").lower()
    input1 = tvm.placeholder(shape, name="input1", dtype=dtype.lower())
    input2 = tvm.placeholder(shape, name="input2", dtype=dtype.lower())

    with tvm.target.cce():
        res = add3_compute(input1, input2, const_bias)
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [input1, input2, res]}

    te.lang.cce.cce_build_code(sch, config)
