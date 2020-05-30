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

"""StridedSliceGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

strided_slice_grad_d_op_info = TBERegOp("StridedSliceGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("strided_slice_grad_d.so") \
    .compute_cost(10) \
    .kernel_name("strided_slice_grad_d") \
    .partial_flag(True) \
    .attr("shapex", "optional", "listInt", "all") \
    .attr("begin", "optional", "listInt", "all") \
    .attr("end", "optional", "listInt", "all") \
    .attr("strides", "optional", "listInt", "all") \
    .attr("begin_mask", "optional", "int", "all") \
    .attr("end_mask", "optional", "int", "all") \
    .attr("ellipsis_mask", "optional", "int", "all") \
    .attr("new_axis_mask", "optional", "int", "all") \
    .attr("shrink_axis_mask", "optional", "int", "all") \
    .input(0, "dy", False, "required", "all") \
    .output(0, "output", False, "required", "all") \
    .dtype_format(DataType.I8_Default, DataType.I8_Default) \
    .dtype_format(DataType.U8_Default, DataType.U8_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(strided_slice_grad_d_op_info)
def _strided_slice_grad_d_tbe():
    """StridedSliceGrad TBE register"""
    return
