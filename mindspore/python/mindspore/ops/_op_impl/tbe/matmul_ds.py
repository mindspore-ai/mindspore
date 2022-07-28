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
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("mat_mul.so") \
    .compute_cost(10) \
    .kernel_name("mat_mul") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .attr("transpose_x1", "required", "bool", "all") \
    .attr("transpose_x2", "required", "bool", "all") \
    .attr("offset_x", "optional", "int", "all", "0") \
    .input(0, "x1", False, "required", "all") \
    .input(1, "x2", False, "required", "all") \
    .input(2, "bias", False, "optional", "all") \
    .input(3, "offset_w", False, "optional", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_Default, DataType.I8_Default,
                  DataType.F16_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F32_Default, DataType.I8_Default,
                  DataType.F16_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F32_Default, DataType.I8_Default,
                  DataType.F32_FracNZ) \
    .get_op_info()


@op_info_register(matmul_op_info)
def _matmul_ds_tbe():
    """MatMul TBE register"""
    return
