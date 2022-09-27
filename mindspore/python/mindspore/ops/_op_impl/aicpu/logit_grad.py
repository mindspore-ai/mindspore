# Copyright (c) Huawei Technologies Co., Ltd. 2022. All right reserved.
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

"""LogitGrad op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

logit_grad_op_info = AiCPURegOp("LogitGrad") \
    .fusion_type("OPAQUE") \
    .attr("eps", "float", "-1") \
    .input(0, "y_grad", "required") \
    .input(1, "x", "required") \
    .output(0, "x_grad", "required") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.F64_Default) \
    .get_op_info()


@op_info_register(logit_grad_op_info)
def _logit_grad_aicpu():
    """LogitGrad aicpu register"""
    return
