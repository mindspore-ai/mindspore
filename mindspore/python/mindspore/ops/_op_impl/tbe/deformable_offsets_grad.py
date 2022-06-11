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

"""DeformableOffsetsGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

deformable_offsets_grad_op_info = TBERegOp("DeformableOffsetsGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("deformable_offsets_grad.so") \
    .compute_cost(10) \
    .kernel_name("deformable_offsets_grad") \
    .partial_flag(True) \
    .need_check_supported(True) \
    .attr("strides", "required", "listInt", "all") \
    .attr("pads", "required", "listInt", "all") \
    .attr("ksize", "required", "listInt", "all") \
    .attr("dilations", "optional", "listInt", "all", "1,1,1,1") \
    .attr("data_format", "optional", "str", "all", "NHWC") \
    .attr("deformable_groups", "optional", "int", "all", "1") \
    .attr("modulated", "optional", "bool", "all", "true") \
    .input(0, "grad", False, "required", "all") \
    .input(1, "x", False, "required", "all") \
    .input(2, "offsets", False, "required", "all") \
    .input(3, "helper", False, "optional", "all") \
    .output(0, "grad_x", False, "required", "all") \
    .output(1, "grad_offsets", False, "required", "all") \
    .dtype_format(DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC,
                  DataType.F32_NHWC, DataType.F32_NHWC) \
    .get_op_info()


@op_info_register(deformable_offsets_grad_op_info)
def _deformable_offsets_grad_tbe():
    """DeformableOffsetsGrad TBE register"""
    return
