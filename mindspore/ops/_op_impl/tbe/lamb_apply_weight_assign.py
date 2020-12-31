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

"""LambApplyWeightAssign op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

lamb_apply_weight_assign_op_info = TBERegOp("LambApplyWeightAssign") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("lamb_apply_weight_assign.so") \
    .compute_cost(10) \
    .kernel_name("lamb_apply_weight_assign") \
    .partial_flag(True) \
    .input(0, "input0", False, "required", "all") \
    .input(1, "input1", False, "required", "all") \
    .input(2, "input2", False, "required", "all") \
    .input(3, "input3", False, "required", "all") \
    .input(4, "input_param", False, "required", "all") \
    .output(0, "input_param", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(lamb_apply_weight_assign_op_info)
def _lamb_apply_weight_assign_tbe():
    """LambApplyWeightAssign TBE register"""
    return
