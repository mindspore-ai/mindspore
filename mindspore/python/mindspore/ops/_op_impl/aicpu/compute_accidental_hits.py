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

"""ComputeAccidentalHits op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType
compute_accidental_hits_op_info = AiCPURegOp("ComputeAccidentalHits") \
    .fusion_type("OPAQUE") \
    .input(0, "true_classes", "required") \
    .input(1, "sampled_candidates", "required") \
    .output(0, "indices", "required") \
    .output(1, "ids", "required") \
    .output(2, "weights", "required") \
    .attr("num_true", "int") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.F64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.F32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.F16_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.F64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.F32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.F16_Default) \
    .get_op_info()


@op_info_register(compute_accidental_hits_op_info)
def _compute_accidental_hits_aicpu():
    """ComputeAccidentalHits AiCPU register"""
    return
