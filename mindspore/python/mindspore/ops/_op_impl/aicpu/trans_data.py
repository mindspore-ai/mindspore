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

"""TransData op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

trans_data_op_info = AiCPURegOp("TransData") \
    .fusion_type("OPAQUE") \
    .input(0, "src", "required") \
    .output(0, "dst", "required") \
    .attr("src_format", "str") \
    .attr("dst_format", "str") \
    .attr("groups", "int") \
    .dtype_format(DataType.U16_Default, DataType.U16_5HD) \
    .dtype_format(DataType.U16_5HD, DataType.U16_Default) \
    .dtype_format(DataType.I64_5HD, DataType.I64_Default) \
    .dtype_format(DataType.I32_5HD, DataType.I32_Default) \
    .dtype_format(DataType.F16_FracZ, DataType.F16_Default) \
    .dtype_format(DataType.F32_FracZ, DataType.F32_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_FracZ) \
    .dtype_format(DataType.F32_Default, DataType.F32_FracZ) \
    .get_op_info()


@op_info_register(trans_data_op_info)
def _trans_data_aicpu():
    """TransData aicpu register"""
    return
