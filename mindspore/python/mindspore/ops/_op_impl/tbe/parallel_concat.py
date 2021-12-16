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

"""ParallelConcat op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

parallel_concat_op_info = TBERegOp("ParallelConcat") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("parallel_concat.so") \
    .compute_cost(10) \
    .kernel_name("parallel_concat") \
    .partial_flag(True) \
    .attr("shape", "required", "listInt", "all") \
    .attr("N", "required", "int", "all") \
    .input(0, "values", False, "dynamic", "all") \
    .output(0, "output_data", False, "required", "all") \
    .dtype_format(DataType.BOOL_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.BOOL_5HD, DataType.BOOL_5HD) \
    .dtype_format(DataType.I8_Default, DataType.I8_Default) \
    .dtype_format(DataType.I8_5HD, DataType.I8_5HD) \
    .dtype_format(DataType.U8_Default, DataType.U8_Default) \
    .dtype_format(DataType.U8_5HD, DataType.U8_5HD) \
    .dtype_format(DataType.I16_Default, DataType.I16_Default) \
    .dtype_format(DataType.I16_5HD, DataType.I16_5HD) \
    .dtype_format(DataType.U16_Default, DataType.U16_Default) \
    .dtype_format(DataType.U16_5HD, DataType.U16_5HD) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I32_5HD, DataType.I32_5HD) \
    .dtype_format(DataType.U32_Default, DataType.U32_Default) \
    .dtype_format(DataType.U32_5HD, DataType.U32_5HD) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.I64_5HD, DataType.I64_5HD) \
    .dtype_format(DataType.U64_Default, DataType.U64_Default) \
    .dtype_format(DataType.U64_5HD, DataType.U64_5HD) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD) \
    .dtype_format(DataType.BOOL_NHWC, DataType.BOOL_NHWC) \
    .dtype_format(DataType.BOOL_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.I8_NHWC, DataType.I8_NHWC) \
    .dtype_format(DataType.U8_NHWC, DataType.U8_NHWC) \
    .dtype_format(DataType.I16_NHWC, DataType.I16_NHWC) \
    .dtype_format(DataType.U16_NHWC, DataType.U16_NHWC) \
    .dtype_format(DataType.I32_NHWC, DataType.I32_NHWC) \
    .dtype_format(DataType.U32_NHWC, DataType.U32_NHWC) \
    .dtype_format(DataType.I64_NHWC, DataType.I64_NHWC) \
    .dtype_format(DataType.U64_NHWC, DataType.U64_NHWC) \
    .dtype_format(DataType.F16_NHWC, DataType.F16_NHWC) \
    .dtype_format(DataType.F32_NHWC, DataType.F32_NHWC) \
    .get_op_info()


@op_info_register(parallel_concat_op_info)
def _parallel_concat_tbe():
    """ParallelConcat TBE register"""
    return
