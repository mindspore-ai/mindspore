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

"""Assign op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

assign_op_info = TBERegOp("Assign") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("assign.so") \
    .compute_cost(10) \
    .kernel_name("assign") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .input(0, "ref", False, "required", "all") \
    .input(1, "value", False, "required", "all") \
    .output(0, "ref", False, "required", "all") \
    .dtype_format(DataType.BOOL_Default, DataType.BOOL_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.BOOL_5HD, DataType.BOOL_5HD, DataType.BOOL_5HD) \
    .dtype_format(DataType.BOOL_C1HWNCoC0, DataType.BOOL_C1HWNCoC0, DataType.BOOL_C1HWNCoC0) \
    .dtype_format(DataType.BOOL_FracZ, DataType.BOOL_FracZ, DataType.BOOL_FracZ) \
    .dtype_format(DataType.I8_Default, DataType.I8_Default, DataType.I8_Default) \
    .dtype_format(DataType.I8_5HD, DataType.I8_5HD, DataType.I8_5HD) \
    .dtype_format(DataType.I8_C1HWNCoC0, DataType.I8_C1HWNCoC0, DataType.I8_C1HWNCoC0) \
    .dtype_format(DataType.I8_FracZ, DataType.I8_FracZ, DataType.I8_FracZ) \
    .dtype_format(DataType.U8_Default, DataType.U8_Default, DataType.U8_Default) \
    .dtype_format(DataType.U8_5HD, DataType.U8_5HD, DataType.U8_5HD) \
    .dtype_format(DataType.U8_C1HWNCoC0, DataType.U8_C1HWNCoC0, DataType.U8_C1HWNCoC0) \
    .dtype_format(DataType.U8_FracZ, DataType.U8_FracZ, DataType.U8_FracZ) \
    .dtype_format(DataType.I16_Default, DataType.I16_Default, DataType.I16_Default) \
    .dtype_format(DataType.I16_5HD, DataType.I16_5HD, DataType.I16_5HD) \
    .dtype_format(DataType.I16_C1HWNCoC0, DataType.I16_C1HWNCoC0, DataType.I16_C1HWNCoC0) \
    .dtype_format(DataType.I16_FracZ, DataType.I16_FracZ, DataType.I16_FracZ) \
    .dtype_format(DataType.U16_Default, DataType.U16_Default, DataType.U16_Default) \
    .dtype_format(DataType.U16_5HD, DataType.U16_5HD, DataType.U16_5HD) \
    .dtype_format(DataType.U16_C1HWNCoC0, DataType.U16_C1HWNCoC0, DataType.U16_C1HWNCoC0) \
    .dtype_format(DataType.U16_FracZ, DataType.U16_FracZ, DataType.U16_FracZ) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I32_5HD, DataType.I32_5HD, DataType.I32_5HD) \
    .dtype_format(DataType.I32_C1HWNCoC0, DataType.I32_C1HWNCoC0, DataType.I32_C1HWNCoC0) \
    .dtype_format(DataType.I32_FracZ, DataType.I32_FracZ, DataType.I32_FracZ) \
    .dtype_format(DataType.U32_Default, DataType.U32_Default, DataType.U32_Default) \
    .dtype_format(DataType.U32_5HD, DataType.U32_5HD, DataType.U32_5HD) \
    .dtype_format(DataType.U32_C1HWNCoC0, DataType.U32_C1HWNCoC0, DataType.U32_C1HWNCoC0) \
    .dtype_format(DataType.U32_FracZ, DataType.U32_FracZ, DataType.U32_FracZ) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.I64_5HD, DataType.I64_5HD, DataType.I64_5HD) \
    .dtype_format(DataType.I64_C1HWNCoC0, DataType.I64_C1HWNCoC0, DataType.I64_C1HWNCoC0) \
    .dtype_format(DataType.I64_FracZ, DataType.I64_FracZ, DataType.I64_FracZ) \
    .dtype_format(DataType.U64_Default, DataType.U64_Default, DataType.U64_Default) \
    .dtype_format(DataType.U64_5HD, DataType.U64_5HD, DataType.U64_5HD) \
    .dtype_format(DataType.U64_C1HWNCoC0, DataType.U64_C1HWNCoC0, DataType.U64_C1HWNCoC0) \
    .dtype_format(DataType.U64_FracZ, DataType.U64_FracZ, DataType.U64_FracZ) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F16_C1HWNCoC0, DataType.F16_C1HWNCoC0, DataType.F16_C1HWNCoC0) \
    .dtype_format(DataType.F16_FracZ, DataType.F16_FracZ, DataType.F16_FracZ) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD) \
    .dtype_format(DataType.F32_C1HWNCoC0, DataType.F32_C1HWNCoC0, DataType.F32_C1HWNCoC0) \
    .dtype_format(DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ) \
    .get_op_info()


@op_info_register(assign_op_info)
def _assign_ds_tbe():
    """Assign TBE register"""
    return
