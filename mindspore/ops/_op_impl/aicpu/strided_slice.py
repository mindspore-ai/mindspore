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

"""StridedSlice op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

strided_slice_op_info = AiCPURegOp("StridedSliceAICPU") \
    .fusion_type("OPAQUE") \
    .input(0, "input", "required") \
    .input(1, "begin", "required") \
    .input(2, "end", "required") \
    .input(3, "stride", "required") \
    .output(0, "output", "required") \
    .attr("begin_mask", "int") \
    .attr("end_mask", "int") \
    .attr("ellipsis_mask", "int") \
    .attr("new_axis_mask", "int") \
    .attr("shrink_axis_mask", "int") \
    .dtype_format(DataType.F32_Default,
                  DataType.I32_Default,
                  DataType.I32_Default,
                  DataType.I32_Default,
                  DataType.F32_Default) \
    .get_op_info()

@op_info_register(strided_slice_op_info)
def _strided_slice_aicpu():
    """StridedSlice AiCPU register"""
    return
