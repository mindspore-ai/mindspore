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

"""TransData op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

trans_data_op_info = TBERegOp("TransData") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("trans_data.so") \
    .compute_cost(10) \
    .kernel_name("trans_data") \
    .partial_flag(True) \
    .attr("src_format", "required", "str", "DefaultFormat,NC1HWC0,FracZ,FRACTAL_NZ,HWCN,C1HWNCoC0")\
    .attr("dst_format", "required", "str", "DefaultFormat,NC1HWC0,FracZ,FRACTAL_NZ,HWCN,C1HWNCoC0")\
    .input(0, "src", False, "required", "all") \
    .output(0, "dst", False, "required", "all") \
    .dtype_format(DataType.U16_Default, DataType.U16_5HD) \
    .dtype_format(DataType.U16_Default, DataType.U16_FracZ) \
    .dtype_format(DataType.U16_Default, DataType.U16_FracNZ) \
    .dtype_format(DataType.U16_FracZ, DataType.U16_Default) \
    .dtype_format(DataType.U16_FracZ, DataType.U16_HWCN) \
    .dtype_format(DataType.U16_FracNZ, DataType.U16_Default) \
    .dtype_format(DataType.U16_5HD, DataType.U16_Default) \
    .dtype_format(DataType.U16_HWCN, DataType.U16_FracZ) \
    .dtype_format(DataType.U16_HWCN, DataType.U16_C1HWNCoC0) \
    .dtype_format(DataType.U16_C1HWNCoC0, DataType.U16_HWCN) \
    .dtype_format(DataType.BOOL_Default, DataType.BOOL_5HD) \
    .dtype_format(DataType.F16_Default, DataType.F16_5HD) \
    .dtype_format(DataType.F16_Default, DataType.F16_FracZ) \
    .dtype_format(DataType.F16_Default, DataType.F16_FracNZ) \
    .dtype_format(DataType.F16_FracZ, DataType.F16_Default) \
    .dtype_format(DataType.F16_FracZ, DataType.F16_HWCN) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_Default) \
    .dtype_format(DataType.F16_HWCN, DataType.F16_FracZ) \
    .dtype_format(DataType.F16_HWCN, DataType.F16_C1HWNCoC0) \
    .dtype_format(DataType.F16_C1HWNCoC0, DataType.F16_HWCN) \
    .dtype_format(DataType.F32_Default, DataType.F32_5HD) \
    .dtype_format(DataType.F32_Default, DataType.F32_FracZ) \
    .dtype_format(DataType.F32_Default, DataType.F32_FracNZ) \
    .dtype_format(DataType.F32_FracZ, DataType.F32_Default) \
    .dtype_format(DataType.F32_FracZ, DataType.F32_HWCN) \
    .dtype_format(DataType.F32_FracNZ, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_Default) \
    .dtype_format(DataType.F32_HWCN, DataType.F32_FracZ) \
    .dtype_format(DataType.F32_HWCN, DataType.F32_C1HWNCoC0) \
    .dtype_format(DataType.F32_C1HWNCoC0, DataType.F32_HWCN) \
    .get_op_info()


@op_info_register(trans_data_op_info)
def _trans_data_tbe():
    """TransData TBE register"""
    return
