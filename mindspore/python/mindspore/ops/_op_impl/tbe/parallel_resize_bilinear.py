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

"""SyncResizeBilinearV2 op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

parallel_resize_bilinear_op_info = TBERegOp("ParallelResizeBilinear") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("sync_resize_bilinear_v2.so") \
    .compute_cost(10) \
    .kernel_name("sync_resize_bilinear_v2") \
    .partial_flag(True) \
    .dynamic_compile_static(True) \
    .dynamic_shape(True) \
    .attr("ori_image_size", "optional", "listInt", "all") \
    .attr("split_size", "optional", "listInt", "all") \
    .attr("src_start_w", "optional", "int", "all", "0") \
    .attr("dst_start_w", "optional", "int", "all", "0") \
    .attr("align_corners", "optional", "bool", "all", "false") \
    .attr("half_pixel_centers", "optional", "bool", "all", "false") \
    .input(0, "x", False, "required", "all") \
    .input(1, "size", False, "required", "all", "optional") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F32_5HD, DataType.I32_Default, DataType.F32_5HD) \
    .dtype_format(DataType.F32_5HD, DataType.I32_Default, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(parallel_resize_bilinear_op_info)
def _parallel_resize_bilinear_op_info_tbe():
    """ParallelResizeBilinear TBE register"""
    return
