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

"""FractionalAvgPoolGrad op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

fractional_avg_pool_grad_op_info = AiCPURegOp("FractionalAvgPoolGrad") \
    .fusion_type("OPAQUE") \
    .input(0, "orig_input_tensor_shape", "required") \
    .input(1, "out_backprop", "required") \
    .input(2, "row_pooling_sequence", "required") \
    .input(3, "col_pooling_sequence", "required") \
    .output(0, "y", "required") \
    .attr("overlapping", "bool") \
    .dtype_format(DataType.I64_Default, DataType.I32_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.F32_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.F32_Default) \
    .dtype_format(DataType.I64_Default, DataType.F64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.F64_Default) \
    .get_op_info()


@op_info_register(fractional_avg_pool_grad_op_info)
def _fractional_avg_pool_grad_aicpu():
    """FractionalAvgPoolGrad AiCPU register"""
    return
