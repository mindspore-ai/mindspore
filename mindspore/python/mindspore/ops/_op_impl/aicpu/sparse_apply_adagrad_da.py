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

"""SparseApplyAdagradDA op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

sparse_apply_adagrad_da_op_info = AiCPURegOp("SparseApplyAdagradDA") \
    .fusion_type("OPAQUE") \
    .attr("use_locking", "bool") \
    .input(0, "var", "required") \
    .input(1, "grad_accum", "required") \
    .input(2, "grad_square_accum", "required") \
    .input(3, "grad", "required") \
    .input(4, "indices", "required") \
    .input(5, "lr", "required") \
    .input(6, "l1", "required") \
    .input(7, "l2", "required") \
    .input(8, "global_step", "required") \
    .output(0, "var", "required") \
    .dtype_format(DataType.I8_Default, DataType.I8_Default, DataType.I8_Default, DataType.I8_Default,     \
                  DataType.I32_Default, DataType.I8_Default, DataType.I8_Default, DataType.I8_Default,    \
                  DataType.I64_Default, DataType.I8_Default,)                                             \
    .dtype_format(DataType.I16_Default, DataType.I16_Default, DataType.I16_Default, DataType.I16_Default, \
                  DataType.I32_Default, DataType.I16_Default, DataType.I16_Default, DataType.I16_Default, \
                  DataType.I64_Default, DataType.I16_Default,)                                            \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, \
                  DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, \
                  DataType.I64_Default, DataType.I32_Default,)                                            \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, \
                  DataType.I32_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, \
                  DataType.I64_Default, DataType.I64_Default,)                                            \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, \
                  DataType.I32_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, \
                  DataType.I64_Default, DataType.F16_Default,)                                            \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, \
                  DataType.I32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, \
                  DataType.I64_Default, DataType.F32_Default,)                                            \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.F64_Default, DataType.F64_Default, \
                  DataType.I32_Default, DataType.F64_Default, DataType.F64_Default, DataType.F64_Default, \
                  DataType.I64_Default, DataType.F64_Default,)                                            \
    .dtype_format(DataType.I8_Default, DataType.I8_Default, DataType.I8_Default, DataType.I8_Default,     \
                  DataType.I64_Default, DataType.I8_Default, DataType.I8_Default, DataType.I8_Default,    \
                  DataType.I64_Default, DataType.I8_Default,)                                             \
    .dtype_format(DataType.I16_Default, DataType.I16_Default, DataType.I16_Default, DataType.I16_Default, \
                  DataType.I64_Default, DataType.I16_Default, DataType.I16_Default, DataType.I16_Default, \
                  DataType.I64_Default, DataType.I16_Default,)                                            \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, \
                  DataType.I64_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, \
                  DataType.I64_Default, DataType.I32_Default,)                                            \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, \
                  DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, \
                  DataType.I64_Default, DataType.I64_Default,)                                            \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, \
                  DataType.I64_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, \
                  DataType.I64_Default, DataType.F16_Default,)                                            \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, \
                  DataType.I64_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, \
                  DataType.I64_Default, DataType.F32_Default,)                                            \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.F64_Default, DataType.F64_Default, \
                  DataType.I64_Default, DataType.F64_Default, DataType.F64_Default, DataType.F64_Default, \
                  DataType.I64_Default, DataType.F64_Default,)                                            \
    .get_op_info()


@op_info_register(sparse_apply_adagrad_da_op_info)
def _sparse_apply_adagrad_da_aicpu():
    """SparseApplyAdagradDA AiCPU register"""
    return
