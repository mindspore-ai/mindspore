# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# #http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""ExtractVolumePatches op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

extract_volume_patches_op_info = TBERegOp("ExtractVolumePatches") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("extract_volume_patches.so") \
    .compute_cost(10) \
    .kernel_name("extract_volume_patches") \
    .partial_flag(True) \
    .attr("kernel_size", "required", "listInt", "all") \
    .attr("strides", "required", "listInt", "all") \
    .attr("padding", "required", "str", "all") \
    .input(0, "input_x", False, "required", "all") \
    .output(0, "output_y", False, "required", "all") \
    .dtype_format(DataType.F16_NDC1HWC0, DataType.F16_NDC1HWC0) \
    .dtype_format(DataType.I8_NDC1HWC0, DataType.I8_NDC1HWC0) \
    .dtype_format(DataType.U8_NDC1HWC0, DataType.U8_NDC1HWC0) \
    .get_op_info()

@op_info_register(extract_volume_patches_op_info)
def _extract_volume_patches_tbe():
    """ExtractVolumePatches TBE register"""
    return
