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

"""BatchMatMul op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

batch_matmul_op_info = TBERegOp("BatchMatMul") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("batch_matmul.so") \
    .compute_cost(10) \
    .kernel_name("batch_matmul") \
    .attr("transpose_x1", "required", "bool", "all") \
    .attr("transpose_x2", "required", "bool", "all") \
    .partial_flag(True) \
    .need_check_supported(True) \
    .input(0, "x1", False, "required", "all") \
    .input(1, "x2", False, "required", "all") \
    .input(2, "bias", False, "optional", "all") \
    .output(0, "y", False, "required", "all") \
    .is_dynamic_format(True) \
    .dtype_format(DataType.I32_None, DataType.I32_None, DataType.I32_None, DataType.I32_None) \
    .dtype_format(DataType.F16_None, DataType.F16_None, DataType.F16_None, DataType.F16_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None, DataType.F32_None, DataType.F32_None) \
    .get_op_info()


@op_info_register(batch_matmul_op_info)
def _batch_matmul_tbe():
    """BatchMatMul TBE register"""
    return
