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

"""ExtractImagePatches op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

extract_image_patches_op_info = TBERegOp("ExtractImagePatches") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("extract_image_patches.so") \
    .compute_cost(10) \
    .kernel_name("extract_image_patches") \
    .partial_flag(True) \
    .attr("ksizes", "required", "listInt", "all") \
    .attr("strides", "required", "listInt", "all") \
    .attr("rates", "required", "listInt", "all") \
    .attr("padding", "required", "str", "all") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_NHWC) \
    .dtype_format(DataType.I8_5HD, DataType.I8_NHWC) \
    .dtype_format(DataType.U8_5HD, DataType.U8_NHWC) \
    .get_op_info()


@op_info_register(extract_image_patches_op_info)
def _extract_image_patches_tbe():
    """ExtractImagePatches TBE register"""
    return
