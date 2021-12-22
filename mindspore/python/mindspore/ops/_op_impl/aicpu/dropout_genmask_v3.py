# Copyright 2021 Huawei Technologies Co., Ltd
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

"""DropoutGenMaskV3 op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

dropout_genmask_v3_op_info = AiCPURegOp("DropoutGenMaskV3") \
    .fusion_type("OPAQUE") \
    .input(0, "x1", "required") \
    .input(1, "x2", "required") \
    .output(0, "y", "required") \
    .attr("Seed0", "int") \
    .attr("Seed1", "int") \
    .dtype_format(DataType.I32_Default, DataType.F16_Default, DataType.U8_Default) \
    .get_op_info()

@op_info_register(dropout_genmask_v3_op_info)
def _dropout_genmask_v3_aicpu():
    """DropoutGenMaskV3 AiCPU register"""
    return
