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

"""ResizeBilinearGrad op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

resize_bilinear_grad_op_info = AiCPURegOp("ResizeBilinearGrad") \
    .fusion_type("OPAQUE") \
    .input(0, "output_grad", "required") \
    .input(0, "input", "required") \
    .output(1, "input_grad", "required") \
    .attr("align_corners", "bool") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(resize_bilinear_grad_op_info)
def _resize_bilinear_grad_aicpu():
    """ResizeBilinearGrad AiCPU register"""
    return
