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

"""MatMul op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

matmul_op_info = TBERegOp("MatMul") \
    .fusion_type("DYNAMIC") \
    .async_flag(False) \
    .binfile_name("matmul.so") \
    .compute_cost(10) \
    .kernel_name("matmul") \
    .partial_flag(True) \
    .attr("transpose_a", "required", "bool", "all") \
    .attr("transpose_b", "required", "bool", "all") \
    .input(0, "x1", False, "required", "all") \
    .input(1, "x2", False, "required", "all") \
    .input(2, "x3", False, "optional", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_Default, DataType.F16_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F32_Default, DataType.F32_FracNZ) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(matmul_op_info)
def _matmul_tbe():
    """Mul TBE register"""
    return
