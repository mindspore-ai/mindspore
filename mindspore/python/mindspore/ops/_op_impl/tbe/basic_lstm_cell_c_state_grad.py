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

"""BasicLSTMCellCStateGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

basic_lstm_cell_c_state_grad_op_info = TBERegOp("BasicLSTMCellCStateGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("basic_lstm_cell_c_state_grad.so") \
    .compute_cost(10) \
    .kernel_name("basic_lstm_cell_c_state_grad") \
    .attr("forget_bias", "optional", "float", "all", "1") \
    .attr("activation", "optional", "str", "all", "None") \
    .partial_flag(True) \
    .input(0, "c", False, "required", "all") \
    .input(1, "dht", False, "required", "all") \
    .input(2, "dct", False, "required", "all") \
    .input(3, "it", False, "required", "all") \
    .input(4, "jt", False, "required", "all") \
    .input(5, "ft", False, "required", "all") \
    .input(6, "ot", False, "required", "all") \
    .input(7, "tanhct", False, "required", "all") \
    .output(0, "dgate", False, "required", "all") \
    .output(1, "dct_1", False, "required", "all") \
    .dtype_format(DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F16_FracNZ, DataType.F32_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ) \
    .get_op_info()


@op_info_register(basic_lstm_cell_c_state_grad_op_info)
def _basic_lstm_cell_c_state_grad_tbe():
    """BasicLSTMCellCStateGrad TBE register"""
    return
