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

"""LambNextRight op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

lamb_next_right_op_info = TBERegOp("LambNextRight") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("lamb_next_right.so") \
    .compute_cost(10) \
    .kernel_name("lamb_next_right") \
    .partial_flag(True) \
    .input(0, "input_square", False, "required", "all") \
    .input(1, "input_mul2", False, "required", "all") \
    .input(2, "mul2_x", False, "required", "all") \
    .input(3, "mul3_x", False, "required", "all") \
    .input(4, "truediv1_recip", False, "required", "all") \
    .input(5, "add2_y", False, "required", "all") \
    .output(0, "y1", False, "required", "all") \
    .output(1, "y2", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(lamb_next_right_op_info)
def _lamb_next_right_tbe():
    """LambNextRight TBE register"""
    return
