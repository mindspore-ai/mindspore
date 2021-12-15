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

"""BasicLSTMCellCStateGradV2 op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

basic_lstm_cell_c_state_grad_op_info_v2 = TBERegOp("BasicLSTMCellCStateGradV2") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("basic_lstm_cell_c_state_grad.so") \
    .compute_cost(10) \
    .kernel_name("basic_lstm_cell_c_state_grad_v2") \
    .attr("forget_bias", "optional", "float", "all", "1") \
    .attr("activation", "optional", "str", "all", "None") \
    .partial_flag(True) \
    .input(0, "c", False, "required", "all") \
    .input(1, "dy", False, "required", "all") \
    .input(2, "dht", False, "required", "all") \
    .input(3, "dct", False, "required", "all") \
    .input(4, "it", False, "required", "all") \
    .input(5, "jt", False, "required", "all") \
    .input(6, "ft", False, "required", "all") \
    .input(7, "ot", False, "required", "all") \
    .input(8, "tanhct", False, "required", "all") \
    .output(0, "dgate", False, "required", "all") \
    .output(1, "dct_1", False, "required", "all") \
    .dtype_format(DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ, DataType.F16_FracNZ, DataType.F32_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ) \
    .get_op_info()


@op_info_register(basic_lstm_cell_c_state_grad_op_info_v2)
def _basic_lstm_cell_c_state_grad_tbe_v2():
    """BasicLSTMCellCStateGradV2 TBE register"""
    return
