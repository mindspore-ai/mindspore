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

"""STFT op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

stft_op_info = AiCPURegOp("STFT") \
    .fusion_type("OPAQUE") \
    .input(0, "x", "required") \
    .input(1, "window", "optional") \
    .output(0, "y", "required") \
    .attr("hop_length", "int") \
    .attr("win_length", "int") \
    .attr("normalized", "bool") \
    .attr("onesided", "bool") \
    .attr("return_complex", "bool") \
    .attr("n_fft", "int") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.C64_Default) \
    .dtype_format(DataType.F32_Default, DataType.F64_Default, DataType.F64_Default) \
    .dtype_format(DataType.F32_Default, DataType.F64_Default, DataType.C128_Default) \
    .dtype_format(DataType.F32_Default, DataType.C64_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_Default, DataType.C64_Default, DataType.C64_Default) \
    .dtype_format(DataType.F32_Default, DataType.C128_Default, DataType.F64_Default) \
    .dtype_format(DataType.F32_Default, DataType.C128_Default, DataType.C128_Default) \
    .dtype_format(DataType.F64_Default, DataType.F32_Default, DataType.F64_Default) \
    .dtype_format(DataType.F64_Default, DataType.F32_Default, DataType.C128_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.F64_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.C128_Default) \
    .dtype_format(DataType.F64_Default, DataType.C64_Default, DataType.F64_Default) \
    .dtype_format(DataType.F64_Default, DataType.C64_Default, DataType.C128_Default) \
    .dtype_format(DataType.F64_Default, DataType.C128_Default, DataType.F64_Default) \
    .dtype_format(DataType.F64_Default, DataType.C128_Default, DataType.C128_Default) \
    .dtype_format(DataType.C64_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.C64_Default, DataType.F32_Default, DataType.C64_Default) \
    .dtype_format(DataType.C64_Default, DataType.F64_Default, DataType.F64_Default) \
    .dtype_format(DataType.C64_Default, DataType.F64_Default, DataType.C128_Default) \
    .dtype_format(DataType.C64_Default, DataType.C64_Default, DataType.F32_Default) \
    .dtype_format(DataType.C64_Default, DataType.C64_Default, DataType.C64_Default) \
    .dtype_format(DataType.C64_Default, DataType.C128_Default, DataType.F64_Default) \
    .dtype_format(DataType.C64_Default, DataType.C128_Default, DataType.C128_Default) \
    .dtype_format(DataType.C128_Default, DataType.F32_Default, DataType.F64_Default) \
    .dtype_format(DataType.C128_Default, DataType.F32_Default, DataType.C128_Default) \
    .dtype_format(DataType.C128_Default, DataType.F64_Default, DataType.F64_Default) \
    .dtype_format(DataType.C128_Default, DataType.F64_Default, DataType.C128_Default) \
    .dtype_format(DataType.C128_Default, DataType.C64_Default, DataType.F64_Default) \
    .dtype_format(DataType.C128_Default, DataType.C64_Default, DataType.C128_Default) \
    .dtype_format(DataType.C128_Default, DataType.C128_Default, DataType.F64_Default) \
    .dtype_format(DataType.C128_Default, DataType.C128_Default, DataType.C128_Default) \
    .get_op_info()




@op_info_register(stft_op_info)
def _stft_aicpu():
    """STFT aicpu register"""
    return
