# Copyright 2023 Huawei Technologies Co., Ltd
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

"""DeformableOffsetsGrad op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

deformable_offsets_grad_op_info = AiCPURegOp("DeformableOffsetsGrad") \
    .fusion_type("OPAQUE") \
    .input(0, "grad", "required") \
    .input(1, "x", "required") \
    .input(2, "offsets", "required") \
    .output(0, "grad_x", "required") \
    .output(1, "grad_offsets", "required") \
    .attr("strides", "listInt") \
    .attr("pads", "listInt") \
    .attr("ksize", "listInt") \
    .attr("dilations", "listInt", "{1, 1, 1, 1}") \
    .attr("format", "str", "NCHW") \
    .attr("deformable_groups", "int", "1") \
    .attr("modulated", "bool", "true") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default) \
    .get_op_info()


@op_info_register(deformable_offsets_grad_op_info)
def _deformable_offsets_grad_aicpu():
    """DeformableOffsetsGrad AiCPU register"""
    return
