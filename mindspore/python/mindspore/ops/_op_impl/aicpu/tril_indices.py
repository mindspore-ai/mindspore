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

"""TrilIndices op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

tril_indices_op_info = AiCPURegOp("TrilIndices") \
    .fusion_type("OPAQUE") \
    .output(0, "output", "required") \
    .attr("row", "int", "required") \
    .attr("col", "int", "required") \
    .attr("offset", "int") \
    .attr("dtype", "Type") \
    .dtype_format(DataType.I32_Default) \
    .dtype_format(DataType.I64_Default) \
    .get_op_info()


@op_info_register(tril_indices_op_info)
def _tril_indices_aicpu():
    """TrilIndices aicpu register"""
    return
