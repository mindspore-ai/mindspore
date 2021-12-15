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

"""TopK op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

top_k_op_info = AiCPURegOp("TopK") \
    .fusion_type("OPAQUE") \
    .attr("sorted", "bool")\
    .input(0, "input", "required") \
    .input(1, "k", "required") \
    .output(0, "values", "required") \
    .output(1, "indices", "required") \
    .dtype_format(DataType.F16_Default, DataType.I32_Default, DataType.F16_Default, DataType.I32_Default) \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.F32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default) \
    .get_op_info()

@op_info_register(top_k_op_info)
def _top_k_aicpu():
    """TopK aicpu register"""
    return
