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

"""AdaptiveMaxPool3D op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

adaptive_max_pool_3d_op_info = AiCPURegOp("AdaptiveMaxPool3D")                                            \
    .fusion_type("OPAQUE")                                                                                \
    .input(0, "x", "required")                                                                            \
    .input(1, "output_size", "required")                                                                  \
    .output(0, "y", "required")                                                                           \
    .output(1, "argmax", "required")                                                                      \
    .dtype_format(DataType.F16_Default, DataType.I32_Default, DataType.F16_Default, DataType.I32_Default) \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.F32_Default, DataType.I32_Default) \
    .dtype_format(DataType.F64_Default, DataType.I32_Default, DataType.F64_Default, DataType.I32_Default) \
    .dtype_format(DataType.I8_Default, DataType.I32_Default, DataType.I8_Default, DataType.I32_Default)   \
    .dtype_format(DataType.I16_Default, DataType.I32_Default, DataType.I16_Default, DataType.I32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I32_Default, DataType.I64_Default, DataType.I32_Default) \
    .dtype_format(DataType.U8_Default, DataType.I32_Default, DataType.U8_Default, DataType.I32_Default)   \
    .dtype_format(DataType.U16_Default, DataType.I32_Default, DataType.U16_Default, DataType.I32_Default) \
    .dtype_format(DataType.U32_Default, DataType.I32_Default, DataType.U32_Default, DataType.I32_Default) \
    .dtype_format(DataType.U64_Default, DataType.I32_Default, DataType.U64_Default, DataType.I32_Default) \
    .get_op_info()


@op_info_register(adaptive_max_pool_3d_op_info)
def _adaptive_max_pool_3d_aicpu():
    """AdaptiveMaxPool3D aicpu register"""
    return
