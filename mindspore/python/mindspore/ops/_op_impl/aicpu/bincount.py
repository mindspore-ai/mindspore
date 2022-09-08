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
"""Bincount op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType
bincount_op_info = AiCPURegOp("Bincount") \
    .fusion_type("OPAQUE") \
    .input(0, "array", "required") \
    .input(1, "size", "required") \
    .input(2, "weights", "required") \
    .output(0, "bins", "required") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.F64_Default, DataType.F64_Default) \
    .get_op_info()


@op_info_register(bincount_op_info)
def _bincount_aicpu():
    """Bincount aicpu register"""
    return
