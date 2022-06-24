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

"""NonMaxSuppressionWithOverlaps op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

non_max_suppression_with_overlaps_op_info = AiCPURegOp("NonMaxSuppressionWithOverlaps")\
    .fusion_type("OPAQUE")\
    .input(0, "overlaps", "required")\
    .input(1, "scores", "required")\
    .input(2, "max_output_size", "required")\
    .input(3, "overlap_threshold", "required")\
    .input(4, "score_threshold", "required")\
    .output(0, "selected_indices", "required")\
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.I32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.I32_Default)\
    .get_op_info()


@op_info_register(non_max_suppression_with_overlaps_op_info)
def _non_max_suppression_with_overlaps_aicpu():
    """NonMaxSuppressionWithOverlaps AiCPU register"""
    return
