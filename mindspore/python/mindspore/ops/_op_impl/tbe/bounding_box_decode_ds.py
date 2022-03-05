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

"""BoundingBoxDecode op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

bounding_box_decode_op_info = TBERegOp("BoundingBoxDecode") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("bounding_box_decode.so") \
    .compute_cost(10) \
    .kernel_name("bounding_box_decode") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .attr("means", "optional", "listFloat", "all") \
    .attr("stds", "optional", "listFloat", "all") \
    .attr("max_shape", "optional", "listInt", "all", "None") \
    .attr("wh_ratio_clip", "optional", "float", "all", "0.016") \
    .input(0, "rois", False, "required", "all") \
    .input(1, "deltas", False, "required", "all") \
    .output(0, "bboxes", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(bounding_box_decode_op_info)
def _bounding_box_decode_ds_tbe():
    """BoundingBoxDecode TBE register"""
    return
