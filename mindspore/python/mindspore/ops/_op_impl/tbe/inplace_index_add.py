# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

"""IndexAdd op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

inplace_index_add_op_info = TBERegOp("InplaceIndexAdd") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("inplace_index_add.so") \
    .compute_cost(10) \
    .kernel_name("inplace_index_add") \
    .partial_flag(True) \
    .dynamic_compile_static(True) \
    .dynamic_shape(True) \
    .attr("axis", "required", "int", "all") \
    .input(0, "input_x", False, "required", "all") \
    .input(1, "indices", False, "required", "all") \
    .input(2, "input_y", False, "required", "all") \
    .output(0, "input_x", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.I32_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.U8_Default, DataType.I32_Default, DataType.U8_Default, DataType.U8_Default) \
    .dtype_format(DataType.I8_Default, DataType.I32_Default, DataType.I8_Default, DataType.I8_Default) \
    .dtype_format(DataType.I16_Default, DataType.I32_Default, DataType.I16_Default, DataType.I16_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default) \
    .get_op_info()


@op_info_register(inplace_index_add_op_info)
def _inplace_index_add_tbe():
    """IndexAdd TBE register"""
    return
