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

"""SampleDistortedBoundingBoxV2 op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

sample_distorted_bounding_box_v2_op_info = AiCPURegOp("SampleDistortedBoundingBoxV2") \
    .fusion_type("OPAQUE") \
    .attr("seed", "int") \
    .attr("seed2", "int") \
    .attr("aspect_ratio_range", "listFloat") \
    .attr("area_range", "listFloat") \
    .attr("max_attempts", "int") \
    .attr("use_image_if_no_bounding_boxes", "bool") \
    .input(0, "image_size", "required") \
    .input(1, "bounding_boxes", "required") \
    .input(2, "min_object_covered", "required") \
    .output(0, "begin", "required") \
    .output(1, "size", "required") \
    .output(2, "bboxes", "required") \
    .dtype_format(DataType.U8_Default, DataType.F32_Default, DataType.F32_Default, \
                  DataType.U8_Default, DataType.U8_Default, DataType.F32_Default) \
    .dtype_format(DataType.I8_Default, DataType.F32_Default, DataType.F32_Default, \
                  DataType.I8_Default, DataType.I8_Default, DataType.F32_Default) \
    .dtype_format(DataType.I16_Default, DataType.F32_Default, DataType.F32_Default, \
                  DataType.I16_Default, DataType.I16_Default, DataType.F32_Default) \
    .dtype_format(DataType.I32_Default, DataType.F32_Default, DataType.F32_Default, \
                  DataType.I32_Default, DataType.I32_Default, DataType.F32_Default) \
    .dtype_format(DataType.I64_Default, DataType.F32_Default, DataType.F32_Default, \
                  DataType.I64_Default, DataType.I64_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(sample_distorted_bounding_box_v2_op_info)
def _sample_distorted_bounding_box_v2_aicpu():
    """SampleDistortedBoundingBoxV2 aicpu register"""
    return
