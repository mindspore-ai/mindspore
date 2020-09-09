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

"""Reshape op"""
from mindspore.ops.op_info_register import op_info_register, AkgGpuRegOp, DataType as DT

op_info = AkgGpuRegOp("Reshape") \
    .fusion_type("OPAQUE") \
    .input(0, "x") \
    .output(0, "y") \
    .attr("shape", "required", "listInt") \
    .dtype_format(DT.BOOL_Default, DT.BOOL_Default) \
    .dtype_format(DT.I8_Default, DT.I8_Default) \
    .dtype_format(DT.I16_Default, DT.I16_Default) \
    .dtype_format(DT.I32_Default, DT.I32_Default) \
    .dtype_format(DT.I64_Default, DT.I64_Default) \
    .dtype_format(DT.U8_Default, DT.U8_Default) \
    .dtype_format(DT.U16_Default, DT.U16_Default) \
    .dtype_format(DT.U32_Default, DT.U32_Default) \
    .dtype_format(DT.U64_Default, DT.U64_Default) \
    .dtype_format(DT.F16_Default, DT.F16_Default) \
    .dtype_format(DT.F32_Default, DT.F32_Default) \
    .dtype_format(DT.F64_Default, DT.F64_Default) \
    .get_op_info()

@op_info_register(op_info)
def _reshape_akg():
    """Reshape Akg register"""
    return
