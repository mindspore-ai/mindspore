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

"""FractionalMaxPool3DWithFixedKsize op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

fractional_max_pool3d_with_fixed_ksize_op_info = AiCPURegOp("FractionalMaxPool3DWithFixedKsize") \
    .fusion_type("OPAQUE") \
    .input(0, "x", "required") \
    .input(1, "random_samples", "required") \
    .output(0, "y", "required") \
    .output(1, "argmax", "optional") \
    .attr("ksize", "listInt") \
    .attr("output_shape", "listInt") \
    .attr("format", "str") \
    .dtype_format(DataType.I32_Default, DataType.F16_Default, DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.F16_Default, DataType.I64_Default, DataType.I32_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.I32_Default) \
    .dtype_format(DataType.F32_Default, DataType.F16_Default, DataType.F32_Default, DataType.I32_Default) \
    .dtype_format(DataType.F64_Default, DataType.F16_Default, DataType.F64_Default, DataType.I32_Default) \
    .dtype_format(DataType.I32_Default, DataType.F32_Default, DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.F32_Default, DataType.I64_Default, DataType.I32_Default) \
    .dtype_format(DataType.F16_Default, DataType.F32_Default, DataType.F16_Default, DataType.I32_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.I32_Default) \
    .dtype_format(DataType.F64_Default, DataType.F32_Default, DataType.F64_Default, DataType.I32_Default) \
    .dtype_format(DataType.I32_Default, DataType.F64_Default, DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.F64_Default, DataType.I64_Default, DataType.I32_Default) \
    .dtype_format(DataType.F16_Default, DataType.F64_Default, DataType.F16_Default, DataType.I32_Default) \
    .dtype_format(DataType.F32_Default, DataType.F64_Default, DataType.F32_Default, DataType.I32_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.F64_Default, DataType.I32_Default) \
    .dtype_format(DataType.I32_Default, DataType.F16_Default, DataType.I32_Default, DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.F16_Default, DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.I64_Default) \
    .dtype_format(DataType.F32_Default, DataType.F16_Default, DataType.F32_Default, DataType.I64_Default) \
    .dtype_format(DataType.F64_Default, DataType.F16_Default, DataType.F64_Default, DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.F32_Default, DataType.I32_Default, DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.F32_Default, DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.F16_Default, DataType.F32_Default, DataType.F16_Default, DataType.I64_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.I64_Default) \
    .dtype_format(DataType.F64_Default, DataType.F32_Default, DataType.F64_Default, DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.F64_Default, DataType.I32_Default, DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.F64_Default, DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.F16_Default, DataType.F64_Default, DataType.F16_Default, DataType.I64_Default) \
    .dtype_format(DataType.F32_Default, DataType.F64_Default, DataType.F32_Default, DataType.I64_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.F64_Default, DataType.I64_Default) \
    .get_op_info()


@op_info_register(fractional_max_pool3d_with_fixed_ksize_op_info)
def _fractional_max_pool3d_with_fixed_ksize_aicpu():
    """FractionalMaxPool3DWithFixedKsize aicpu register"""
    return
    