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

"""FFTWithSize op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

fft_with_size_op_info = AiCPURegOp("FFTWithSize") \
    .fusion_type("OPAQUE") \
    .input(0, "x", "required") \
    .output(0, "y", "required") \
    .attr("signal_ndim", "int") \
    .attr("inverse", "bool") \
    .attr("signal_sizes", "listInt") \
    .attr("norm", "str") \
    .attr("onesided", "bool") \
    .attr("real", "bool") \
    .dtype_format(DataType.BOOL_Default, DataType.C64_Default) \
    .dtype_format(DataType.U8_Default, DataType.C64_Default) \
    .dtype_format(DataType.I8_Default, DataType.C64_Default) \
    .dtype_format(DataType.I16_Default, DataType.C64_Default) \
    .dtype_format(DataType.I32_Default, DataType.C64_Default) \
    .dtype_format(DataType.I64_Default, DataType.C64_Default) \
    .dtype_format(DataType.C64_Default, DataType.C64_Default) \
    .dtype_format(DataType.C128_Default, DataType.C128_Default) \
    .dtype_format(DataType.F32_Default, DataType.C64_Default) \
    .dtype_format(DataType.F64_Default, DataType.C128_Default) \
    .dtype_format(DataType.C64_Default, DataType.F32_Default) \
    .dtype_format(DataType.C128_Default, DataType.F64_Default) \
    .get_op_info()


@op_info_register(fft_with_size_op_info)
def _fft_with_size_aicpu():
    """FFTWithSize aicpu register"""
    return
