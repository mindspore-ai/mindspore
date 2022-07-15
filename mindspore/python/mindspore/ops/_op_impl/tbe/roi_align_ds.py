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

"""ROIAlign op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

roi_align_op_info = TBERegOp("ROIAlign") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("roi_align.so") \
    .compute_cost(10) \
    .kernel_name("roi_align") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .attr("spatial_scale", "required", "float", "all") \
    .attr("pooled_height", "required", "int", "all") \
    .attr("pooled_width", "required", "int", "all") \
    .attr("sample_num", "optional", "int", "all", "2") \
    .attr("roi_end_mode", "optional", "int", "0,1", "1") \
    .input(0, "features", False, "required", "all") \
    .input(1, "rois", False, "required", "all") \
    .input(2, "rois_n", False, "optional", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_Default, DataType.I32_Default, DataType.F16_5HD) \
    .dtype_format(DataType.F32_5HD, DataType.F32_Default, DataType.I32_Default, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(roi_align_op_info)
def _roi_align_ds_tbe():
    """ROIAlign TBE register"""
    return
