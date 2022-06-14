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

"""Sspaddmm op"""
from mindspore.ops.op_info_register import op_info_register, CpuRegOp, DataType

sspaddmm_op_info = CpuRegOp("Sspaddmm") \
    .input(0, "x1_indices", "required") \
    .input(1, "x1_values", "required") \
    .input(2, "x1_shape", "required") \
    .input(3, "x2_indices", "required") \
    .input(4, "x2_values", "required") \
    .input(5, "x2_shape", "required") \
    .input(6, "x3_dense", "required") \
    .input(7, "alpha", "required") \
    .input(8, "beta", "required") \
    .output(0, "y_indices", "required") \
    .output(1, "y_values", "required") \
    .output(2, "y_shape", "required")


def _reg_format(op_info, dtype, alpha, beta):
    return op_info.dtype_format(DataType.I32_Default, dtype, DataType.I32_Default, DataType.I32_Default, dtype,
                                DataType.I32_Default, dtype, alpha, beta, DataType.I64_Default, dtype,
                                DataType.I64_Default)


def _reg_format_beta(op_info, alpha, beta):
    op_info = _reg_format(op_info, DataType.U8_Default, alpha, beta)
    op_info = _reg_format(op_info, DataType.I8_Default, alpha, beta)
    op_info = _reg_format(op_info, DataType.I16_Default, alpha, beta)
    op_info = _reg_format(op_info, DataType.I32_Default, alpha, beta)
    op_info = _reg_format(op_info, DataType.I64_Default, alpha, beta)
    op_info = _reg_format(op_info, DataType.F32_Default, alpha, beta)
    op_info = _reg_format(op_info, DataType.F64_Default, alpha, beta)
    return op_info


def _reg_format_alpha(op_info, alpha):
    """alpha reg"""
    op_info = _reg_format_beta(op_info, alpha, DataType.U8_Default)
    op_info = _reg_format_beta(op_info, alpha, DataType.U16_Default)
    op_info = _reg_format_beta(op_info, alpha, DataType.U32_Default)
    op_info = _reg_format_beta(op_info, alpha, DataType.U64_Default)
    op_info = _reg_format_beta(op_info, alpha, DataType.I8_Default)
    op_info = _reg_format_beta(op_info, alpha, DataType.I16_Default)
    op_info = _reg_format_beta(op_info, alpha, DataType.I32_Default)
    op_info = _reg_format_beta(op_info, alpha, DataType.I64_Default)
    op_info = _reg_format_beta(op_info, alpha, DataType.F16_Default)
    op_info = _reg_format_beta(op_info, alpha, DataType.F32_Default)
    op_info = _reg_format_beta(op_info, alpha, DataType.F64_Default)
    return op_info


def _reg_format_indices(op_info):
    """indices reg"""
    op_info = _reg_format_alpha(op_info, DataType.U8_Default)
    op_info = _reg_format_alpha(op_info, DataType.U16_Default)
    op_info = _reg_format_alpha(op_info, DataType.U32_Default)
    op_info = _reg_format_alpha(op_info, DataType.U64_Default)
    op_info = _reg_format_alpha(op_info, DataType.I8_Default)
    op_info = _reg_format_alpha(op_info, DataType.I16_Default)
    op_info = _reg_format_alpha(op_info, DataType.I32_Default)
    op_info = _reg_format_alpha(op_info, DataType.I64_Default)
    op_info = _reg_format_alpha(op_info, DataType.F16_Default)
    op_info = _reg_format_alpha(op_info, DataType.F32_Default)
    op_info = _reg_format_alpha(op_info, DataType.F64_Default)
    return op_info


def _reg_format_indices_shape(op_info):
    """shape reg"""
    op_info = _reg_format_indices(op_info)
    return op_info.get_op_info()


sspaddmm_op_info_all = _reg_format_indices_shape(sspaddmm_op_info)


@op_info_register(sspaddmm_op_info_all)
def _sspaddmm_cpu():
    """Sspaddmm cpu register"""
    return
