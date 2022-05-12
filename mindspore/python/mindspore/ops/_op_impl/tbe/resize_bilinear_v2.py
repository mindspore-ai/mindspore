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

"""ResizeBilinear op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

resize_bilinear_v2_op_info = TBERegOp("ResizeBilinearV2") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("resize_bilinear_v2.so") \
    .compute_cost(10) \
    .kernel_name("resize_bilinear_v2") \
    .partial_flag(True) \
    .need_check_supported(False) \
    .dynamic_compile_static(True) \
    .dynamic_shape(True) \
    .attr("align_corners", "optional", "bool", "all", "false") \
    .attr("half_pixel_centers", "optional", "bool", "all", "false") \
    .input(0, "x", False, "required", "all") \
    .input(1, "size", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.I32_Default, DataType.F32_5HD) \
    .dtype_format(DataType.F32_5HD, DataType.I32_Default, DataType.F32_5HD) \
    .dtype_format(DataType.F16_5HD, DataType.I32_Default, DataType.F16_5HD) \
    .get_op_info()


@op_info_register(resize_bilinear_v2_op_info)
def _resize_bilinear_v2_tbe():
    """ResizeBilinear TBE register"""
    return
