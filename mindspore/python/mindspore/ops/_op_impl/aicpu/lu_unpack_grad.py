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

"""LuUnpackGrad op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

lu_unpack_grad_op_info = AiCPURegOp("LuUnpackGrad") \
    .fusion_type("OPAQUE") \
    .attr("L_grad_flag", "bool") \
    .input(0, "L_grad", "required") \
    .input(1, "U_grad", "required") \
    .input(2, "LU_data", "required") \
    .output(0, "L_data_grad", "optional") \
    .output(1, "U_data_grad", "optional") \
    .dtype_format(DataType.I8_Default, DataType.I8_Default, DataType.I8_Default,
                  DataType.I8_Default, DataType.I8_Default) \
    .dtype_format(DataType.I16_Default, DataType.I16_Default, DataType.I16_Default,
                  DataType.I16_Default, DataType.I16_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.U8_Default, DataType.U8_Default, DataType.U8_Default,
                  DataType.U8_Default, DataType.U8_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.F64_Default,
                  DataType.F64_Default, DataType.F64_Default) \
    .get_op_info()


@op_info_register(lu_unpack_grad_op_info)
def _lu_unpack_grad_aicpu():
    """LuUnpackGrad aicpu register"""
    return
