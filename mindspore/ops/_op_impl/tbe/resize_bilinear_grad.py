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

"""ResizeBilinearGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

resize_bilinear_grad_op_info = TBERegOp("ResizeBilinearGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("resize_bilinear_v2_grad.so") \
    .compute_cost(10) \
    .kernel_name("resize_bilinear_v2_grad") \
    .partial_flag(True) \
    .need_check_supported(True) \
    .attr("align_corners", "optional", "bool", "all") \
    .attr("half_pixel_centers", "optional", "bool", "all")\
    .input(0, "grads", False, "required", "all") \
    .input(1, "original_image", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(resize_bilinear_grad_op_info)
def _resize_bilinear_grad_tbe():
    """ResizeBilinearGrad TBE register"""
    return
