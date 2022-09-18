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

"""BatchNormGradGrad op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

batch_norm_grad_grad_op_info = AiCPURegOp("BatchNormGradGrad") \
    .fusion_type("OPAQUE") \
    .attr("epsilon", "float")\
    .attr("format", "str")\
    .attr("is_training", "bool")\
    .input(0, "x", "required") \
    .input(1, "dy", "required") \
    .input(2, "scale", "required") \
    .input(3, "reserve_space_1", "required") \
    .input(4, "reserve_space_2", "required") \
    .input(5, "ddx", "required") \
    .input(6, "ddscale", "required") \
    .input(7, "ddoffset", "required") \
    .output(0, "dx", "required") \
    .output(1, "ddy", "required") \
    .output(2, "dscale", "required") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, \
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, \
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, \
                  DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F32_Default, \
                  DataType.F32_Default, DataType.F32_Default, DataType.F16_Default, \
                  DataType.F32_Default, DataType.F32_Default, DataType.F16_Default, \
                  DataType.F16_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(batch_norm_grad_grad_op_info)
def _batch_norm_grad_grad_aicpu():
    """BatchNormGradGrad AiCPU register"""
    return
