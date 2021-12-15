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

"""LambNextMVWithDecay op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

lamb_next_m_v_with_decay_op_info = TBERegOp("LambNextMVWithDecay") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("lamb_next_m_v_with_decay.so") \
    .compute_cost(10) \
    .kernel_name("lamb_next_m_v_with_decay") \
    .partial_flag(True) \
    .input(0, "input_mul3", False, "required", "all") \
    .input(1, "input_mul2", False, "required", "all") \
    .input(2, "input_realdiv1", False, "required", "all") \
    .input(3, "input_mul1", False, "required", "all") \
    .input(4, "input_mul0", False, "required", "all") \
    .input(5, "input_realdiv0", False, "required", "all") \
    .input(6, "input_mul4", False, "required", "all") \
    .input(7, "mul0_x", False, "required", "all") \
    .input(8, "mul1_sub", False, "required", "all") \
    .input(9, "mul2_x", False, "required", "all") \
    .input(10, "mul3_sub1", False, "required", "all") \
    .input(11, "mul4_x", False, "required", "all") \
    .input(12, "add2_y", False, "required", "all") \
    .output(0, "y1", True, "required", "all") \
    .output(1, "y2", True, "required", "all") \
    .output(2, "y3", True, "required", "all") \
    .output(3, "y4", True, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .get_op_info()


@op_info_register(lamb_next_m_v_with_decay_op_info)
def _lamb_next_mv_with_decay_tbe():
    """LambNextMVWithDecay TBE register"""
    return
