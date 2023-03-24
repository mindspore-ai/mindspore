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
    .dynamic_shape(True) \
    .attr("src_format", "required", "str",
          "DefaultFormat, NC1HWC0, FRACTAL_Z, FRACTAL_NZ, HWCN, C1HWNCoC0, NDHWC, NHWC") \
    .attr("dst_format", "required", "str",
          "DefaultFormat, NC1HWC0, FRACTAL_Z, FRACTAL_NZ, HWCN, C1HWNCoC0, NDHWC, NHWC") \
    .attr("groups", "optional", "int", "all", "1") \
    .input(0, "src", False, "required", "all") \
    .output(0, "dst", False, "required", "all") \
    .dtype_format(DataType.F32_NHWC, DataType.F32_5HD) \
    .dtype_format(DataType.F32_Default, DataType.F32_5HD) \
    .dtype_format(DataType.F32_5HD, DataType.F32_NHWC) \
    .dtype_format(DataType.F32_5HD, DataType.F32_Default) \
    .dtype_format(DataType.F32_FracZ, DataType.F32_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_FracZ) \
    .dtype_format(DataType.F32_HWCN, DataType.F32_FracZ) \
    .dtype_format(DataType.F32_FracZ, DataType.F32_HWCN) \
    .dtype_format(DataType.F32_C1HWNCoC0, DataType.F32_HWCN) \
    .dtype_format(DataType.F32_HWCN, DataType.F32_C1HWNCoC0) \
    .dtype_format(DataType.F16_Default, DataType.F16_FracZ) \
    .dtype_format(DataType.F16_NHWC, DataType.F16_FracZ) \
    .dtype_format(DataType.F16_HWCN, DataType.F16_FracZ) \
    .dtype_format(DataType.F16_Default, DataType.F16_5HD) \
    .dtype_format(DataType.F16_NHWC, DataType.F16_5HD) \
    .dtype_format(DataType.F16_HWCN, DataType.F16_5HD) \
    .dtype_format(DataType.F16_5HD, DataType.F16_NHWC) \
    .dtype_format(DataType.F16_5HD, DataType.F16_Default) \
    .dtype_format(DataType.F16_FracZ, DataType.F16_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_FracZ) \
    .dtype_format(DataType.F16_HWCN, DataType.F16_FracZ) \
    .dtype_format(DataType.F16_FracZ, DataType.F16_HWCN) \
    .dtype_format(DataType.F16_C1HWNCoC0, DataType.F16_HWCN) \
    .dtype_format(DataType.F16_HWCN, DataType.F16_C1HWNCoC0) \
    .dtype_format(DataType.F16_Default, DataType.F16_FracNZ) \
    .dtype_format(DataType.F32_Default, DataType.F32_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_Default) \
    .dtype_format(DataType.F32_FracNZ, DataType.F32_Default) \
    .dtype_format(DataType.BOOL_NHWC, DataType.BOOL_5HD) \
    .dtype_format(DataType.BOOL_Default, DataType.BOOL_5HD) \
    .dtype_format(DataType.BOOL_5HD, DataType.BOOL_NHWC) \
    .dtype_format(DataType.BOOL_5HD, DataType.BOOL_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_NHWC) \
    .dtype_format(DataType.F16_Default, DataType.F16_HWCN) \
    .dtype_format(DataType.F16_NHWC, DataType.F16_Default) \
    .dtype_format(DataType.F16_NHWC, DataType.F16_HWCN) \
    .dtype_format(DataType.F16_HWCN, DataType.F16_Default) \
    .dtype_format(DataType.F16_HWCN, DataType.F16_NHWC) \
    .dtype_format(DataType.F32_Default, DataType.F32_NHWC) \
    .dtype_format(DataType.F32_Default, DataType.F32_HWCN) \
    .dtype_format(DataType.F32_NHWC, DataType.F32_Default) \
    .dtype_format(DataType.F32_NHWC, DataType.F32_HWCN) \
    .dtype_format(DataType.F32_HWCN, DataType.F32_Default) \
    .dtype_format(DataType.F32_HWCN, DataType.F32_NHWC) \
    .dtype_format(DataType.I8_Default, DataType.I8_FracNZ) \
    .dtype_format(DataType.I8_Default, DataType.I8_FracZ) \
    .dtype_format(DataType.I8_Default, DataType.I8_NHWC) \
    .dtype_format(DataType.I8_Default, DataType.I8_HWCN) \
    .dtype_format(DataType.I8_NHWC, DataType.I8_Default) \
    .dtype_format(DataType.I8_NHWC, DataType.I8_HWCN) \
    .dtype_format(DataType.I8_HWCN, DataType.I8_Default) \
    .dtype_format(DataType.I8_HWCN, DataType.I8_NHWC) \
    .dtype_format(DataType.I8_Default, DataType.I8_NC1HWC0) \
    .dtype_format(DataType.I8_HWCN, DataType.I8_C1HWNCoC0) \
    .dtype_format(DataType.I8_NCDHW, DataType.I8_NDC1HWC0) \
    .dtype_format(DataType.I8_NDC1HWC0, DataType.I8_NCDHW) \
    .dtype_format(DataType.I16_Default, DataType.I16_NHWC) \
    .dtype_format(DataType.I16_Default, DataType.I16_HWCN) \
    .dtype_format(DataType.I16_NHWC, DataType.I16_Default) \
    .dtype_format(DataType.I16_NHWC, DataType.I16_HWCN) \
    .dtype_format(DataType.I16_HWCN, DataType.I16_Default) \
    .dtype_format(DataType.I16_HWCN, DataType.I16_NHWC) \
    .dtype_format(DataType.I32_Default, DataType.I32_NHWC) \
    .dtype_format(DataType.I32_Default, DataType.I32_HWCN) \
    .dtype_format(DataType.I32_NHWC, DataType.I32_Default) \
    .dtype_format(DataType.I32_NHWC, DataType.I32_HWCN) \
    .dtype_format(DataType.I32_HWCN, DataType.I32_Default) \
    .dtype_format(DataType.I32_HWCN, DataType.I32_NHWC) \
    .dtype_format(DataType.I32_NDC1HWC0, DataType.I32_NCDHW) \
    .dtype_format(DataType.I32_NCDHW, DataType.I32_NDC1HWC0) \
    .dtype_format(DataType.I64_Default, DataType.I64_NHWC) \
    .dtype_format(DataType.I64_Default, DataType.I64_HWCN) \
    .dtype_format(DataType.I64_NHWC, DataType.I64_Default) \
    .dtype_format(DataType.I64_NHWC, DataType.I64_HWCN) \
    .dtype_format(DataType.I64_HWCN, DataType.I64_Default) \
    .dtype_format(DataType.I64_HWCN, DataType.I64_NHWC) \
    .dtype_format(DataType.U8_Default, DataType.U8_NHWC) \
    .dtype_format(DataType.U8_Default, DataType.U8_HWCN) \
    .dtype_format(DataType.U8_NHWC, DataType.U8_Default) \
    .dtype_format(DataType.U8_NHWC, DataType.U8_HWCN) \
    .dtype_format(DataType.U8_HWCN, DataType.U8_Default) \
    .dtype_format(DataType.U8_HWCN, DataType.U8_NHWC) \
    .dtype_format(DataType.U8_Default, DataType.U8_NC1HWC0) \
    .dtype_format(DataType.U8_NCDHW, DataType.U8_NDC1HWC0) \
    .dtype_format(DataType.U8_NDC1HWC0, DataType.U8_NCDHW) \
    .dtype_format(DataType.U16_Default, DataType.U16_NHWC) \
    .dtype_format(DataType.U16_Default, DataType.U16_HWCN) \
    .dtype_format(DataType.U16_NHWC, DataType.U16_Default) \
    .dtype_format(DataType.U16_NHWC, DataType.U16_HWCN) \
    .dtype_format(DataType.U16_HWCN, DataType.U16_Default) \
    .dtype_format(DataType.U16_HWCN, DataType.U16_NHWC) \
    .dtype_format(DataType.U32_Default, DataType.U32_NHWC) \
    .dtype_format(DataType.U32_Default, DataType.U32_HWCN) \
    .dtype_format(DataType.U32_NHWC, DataType.U32_Default) \
    .dtype_format(DataType.U32_NHWC, DataType.U32_HWCN) \
    .dtype_format(DataType.U32_HWCN, DataType.U32_Default) \
    .dtype_format(DataType.U32_HWCN, DataType.U32_NHWC) \
    .dtype_format(DataType.U64_Default, DataType.U64_NHWC) \
    .dtype_format(DataType.U64_Default, DataType.U64_HWCN) \
    .dtype_format(DataType.U64_NHWC, DataType.U64_Default) \
    .dtype_format(DataType.U64_NHWC, DataType.U64_HWCN) \
    .dtype_format(DataType.U64_HWCN, DataType.U64_Default) \
    .dtype_format(DataType.U64_HWCN, DataType.U64_NHWC) \
    .dtype_format(DataType.I32_FracNZ, DataType.I32_Default) \
    .dtype_format(DataType.F16_NDHWC, DataType.F16_5HD) \
    .dtype_format(DataType.F16_5HD, DataType.F16_NDHWC) \
    .dtype_format(DataType.F16_HWCN, DataType.F16_FracZ) \
    .dtype_format(DataType.F16_FracZ, DataType.F16_HWCN) \
    .dtype_format(DataType.F16_HWCN, DataType.F16_FracNZ) \
    .dtype_format(DataType.F32_HWCN, DataType.F16_FracNZ) \
    .dtype_format(DataType.F16_HWCN, DataType.F16_FracZNLSTM) \
    .dtype_format(DataType.F32_HWCN, DataType.F32_FracZNLSTM) \
    .dtype_format(DataType.F16_FracZNLSTM, DataType.F16_HWCN) \
    .dtype_format(DataType.F32_FracZNLSTM, DataType.F32_HWCN) \
    .dtype_format(DataType.F16_NDHWC, DataType.F16_NDC1HWC0) \
    .dtype_format(DataType.F16_NDC1HWC0, DataType.F16_NDHWC) \
    .dtype_format(DataType.F16_DHWCN, DataType.F16_FRACTAL_Z_3D) \
    .dtype_format(DataType.F16_FRACTAL_Z_3D, DataType.F16_DHWCN) \
    .dtype_format(DataType.F16_NCDHW, DataType.F16_NDC1HWC0) \
    .dtype_format(DataType.F16_NDC1HWC0, DataType.F16_NCDHW) \
    .dtype_format(DataType.F16_NCDHW, DataType.F16_FRACTAL_Z_3D) \
    .dtype_format(DataType.F32_NCDHW, DataType.F32_FRACTAL_Z_3D) \
    .dtype_format(DataType.F16_FRACTAL_Z_3D, DataType.F16_NCDHW) \
    .dtype_format(DataType.F32_FRACTAL_Z_3D, DataType.F32_NCDHW) \
    .dtype_format(DataType.F16_NDHWC, DataType.F16_FRACTAL_Z_3D) \
    .dtype_format(DataType.F32_NDHWC, DataType.F32_FRACTAL_Z_3D) \
    .dtype_format(DataType.F16_FRACTAL_Z_3D, DataType.F16_NDHWC) \
    .dtype_format(DataType.F32_FRACTAL_Z_3D, DataType.F32_NDHWC) \
    .dtype_format(DataType.F32_DHWCN, DataType.F32_FRACTAL_Z_3D) \
    .dtype_format(DataType.F32_FRACTAL_Z_3D, DataType.F32_DHWCN) \
    .dtype_format(DataType.F32_NDC1HWC0, DataType.F32_NDHWC) \
    .dtype_format(DataType.F32_NDHWC, DataType.F32_NDC1HWC0) \
    .dtype_format(DataType.F32_NDC1HWC0, DataType.F32_NCDHW) \
    .dtype_format(DataType.F32_NCDHW, DataType.F32_NDC1HWC0) \
    .dtype_format(DataType.F32_NDC1HWC0, DataType.F32_NCDHW) \
    .dtype_format(DataType.F32_NCDHW, DataType.F32_NDC1HWC0) \
    .dtype_format(DataType.I8_Default, DataType.I8_5HD) \
    .dtype_format(DataType.I8_5HD, DataType.I8_Default) \
    .dtype_format(DataType.U8_Default, DataType.U8_5HD) \
    .dtype_format(DataType.U8_5HD, DataType.U8_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_5HD) \
    .dtype_format(DataType.I32_5HD, DataType.I32_Default) \
    .get_op_info()


@op_info_register(trans_data_op_info)
def _trans_data_ds_tbe():
    """TransData TBE register"""
    return
