# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
"""Uniform op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

uniform_op_info = AiCPURegOp("Uniform") \
    .fusion_type("OPAQUE") \
    .input(0, "x", "required") \
    .input(1, "counts", "required") \
    .input(2, "states", "required") \
    .output(0, "y", "required") \
    .attr("from", "float") \
    .attr("to", "float") \
    .attr("seed", "int") \
    .attr("offset", "int") \
    .dtype_format(DataType.F32_Default, DataType.U64_Default, DataType.U64_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.U64_Default, DataType.U64_Default, DataType.F64_Default) \
    .get_op_info()


@op_info_register(uniform_op_info)
def _uniform_aicpu():
    """Uniform aicpu register"""
    return
