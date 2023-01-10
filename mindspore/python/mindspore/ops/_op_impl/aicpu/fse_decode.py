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

"""QuantDTypeCast op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

fse_decode_op_info = AiCPURegOp("FSEDecode") \
    .fusion_type("OPAQUE") \
    .input(0, "x", "required") \
    .input(1, "states_table", "required") \
    .input(2, "bit_count_table", "required") \
    .input(3, "symbol_table", "required") \
    .input(4, "centroids", "required") \
    .input(5, "input_shape", "required") \
    .output(0, "y", "required") \
    .attr("dst_t", "int") \
    .attr("curr_chunk", "int") \
    .attr("curr_chunk_index", "int") \
    .attr("curr_bit_count", "int") \
    .attr("table_log", "int") \
    .dtype_format(DataType.I8_Default, DataType.U16_Default, DataType.U8_Default, DataType.U16_Default,
                  DataType.F32_Default, DataType.I32_Default, DataType.F32_Default) \
    .dtype_format(DataType.I8_Default, DataType.U16_Default, DataType.U8_Default, DataType.U16_Default,
                  DataType.F32_Default, DataType.I32_Default, DataType.F16_FracNZ) \
    .get_op_info()


@op_info_register(fse_decode_op_info)
def _fse_decode_aicpu():
    """QuantDTypeCast AiCPU register"""
    return
