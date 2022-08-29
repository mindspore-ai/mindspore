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

"""MinimumGradGrad op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

minimum_grad_grad_op_info = AiCPURegOp("MinimumGradGrad") \
    .fusion_type("OPAQUE") \
    .input(0, "x1", "required") \
    .input(1, "x2", "required") \
    .input(2, "grad_y1", "required") \
    .input(3, "grad_y2", "required") \
    .output(0, "sopd_x1", "required") \
    .output(1, "sopd_x2", "required") \
    .output(2, "sopd_grads", "required") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .get_op_info()


@op_info_register(minimum_grad_grad_op_info)
def _minimum_grad_grad_aicpu():
    """MinimumGradGrad AiCPU register"""
    return
