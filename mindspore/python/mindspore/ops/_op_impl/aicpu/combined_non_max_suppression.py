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

"""CombinedNonMaxSuppression op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

combined_non_max_suppression_op_info = AiCPURegOp("CombinedNonMaxSuppression")\
    .fusion_type("OPAQUE")\
    .attr("pad_per_class", "bool")\
    .attr("clip_boxes", "bool")\
    .input(0, "boxes", "required")\
    .input(1, "scores", "required")\
    .input(2, "max_output_size_per_class", "required")\
    .input(3, "max_total_size", "required")\
    .input(4, "iou_threshold", "required")\
    .input(5, "score_threshold", "required")\
    .output(0, "nmsed_box", "required")\
    .output(1, "nmsed_scores", "required")\
    .output(2, "nmsed_classes", "required")\
    .output(3, "valid_detections", "required")\
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.I32_Default, DataType.I32_Default, \
    DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, \
    DataType.F32_Default, DataType.I32_Default)\
    .get_op_info()


@op_info_register(combined_non_max_suppression_op_info)
def _combined_non_max_suppression_aicpu():
    """CombinedNonMaxSuppression AiCPU register"""
    return
