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

"""SparseGatherV2 op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

sparse_gather_v2_ds_op_info = TBERegOp("SparseGatherV2") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("gather_v2.so") \
    .compute_cost(10) \
    .kernel_name("gather_v2") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .attr("batch_dims", "required", "int", "all", "0") \
    .input(0, "x", False, "required", "all") \
    .input(1, "indices", False, "required", "all") \
    .input(2, "axis", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.I32_Default, DataType.I32_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.I32_Default, DataType.F32_Default) \
    .dtype_format(DataType.I8_Default, DataType.I32_Default, DataType.I32_Default, DataType.I8_Default) \
    .dtype_format(DataType.U8_Default, DataType.I32_Default, DataType.I32_Default, DataType.U8_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.U32_Default, DataType.I32_Default, DataType.I32_Default, DataType.U32_Default) \
    .dtype_format(DataType.I16_Default, DataType.I32_Default, DataType.I32_Default, DataType.I16_Default) \
    .dtype_format(DataType.U16_Default, DataType.I32_Default, DataType.I32_Default, DataType.U16_Default) \
    .dtype_format(DataType.I64_Default, DataType.I32_Default, DataType.I32_Default, DataType.I64_Default) \
    .dtype_format(DataType.U64_Default, DataType.I32_Default, DataType.I32_Default, DataType.U64_Default) \
    .dtype_format(DataType.F16_Default, DataType.I64_Default, DataType.I32_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.I64_Default, DataType.I32_Default, DataType.F32_Default) \
    .dtype_format(DataType.I8_Default, DataType.I64_Default, DataType.I32_Default, DataType.I8_Default) \
    .dtype_format(DataType.U8_Default, DataType.I64_Default, DataType.I32_Default, DataType.U8_Default) \
    .dtype_format(DataType.I32_Default, DataType.I64_Default, DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.U32_Default, DataType.I64_Default, DataType.I32_Default, DataType.U32_Default) \
    .dtype_format(DataType.I16_Default, DataType.I64_Default, DataType.I32_Default, DataType.I16_Default) \
    .dtype_format(DataType.U16_Default, DataType.I64_Default, DataType.I32_Default, DataType.U16_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I32_Default, DataType.I64_Default) \
    .dtype_format(DataType.U64_Default, DataType.I64_Default, DataType.I32_Default, DataType.U64_Default) \
    .get_op_info()


@op_info_register(sparse_gather_v2_ds_op_info)
def _sparse_gather_v2_ds_tbe():
    """SparseGatherV2 TBE register"""
    return
