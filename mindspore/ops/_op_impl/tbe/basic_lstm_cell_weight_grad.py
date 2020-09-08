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

"""BasicLSTMCellWeightGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

basic_lstm_cell_weight_grad_op_info = TBERegOp("BasicLSTMCellWeightGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("basic_lstm_cell_weight_grad.so") \
    .compute_cost(10) \
    .kernel_name("basic_lstm_cell_weight_grad") \
    .partial_flag(True) \
    .input(0, "x", False, "required", "all") \
    .input(1, "h", False, "required", "all") \
    .input(2, "dgate", False, "required", "all") \
    .output(0, "dw", False, "required", "all", reshape_type="CN") \
    .output(1, "db", False, "required", "all") \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracZ,
                  DataType.F32_Default) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracZ,
                  DataType.F16_Default) \
    .get_op_info()


@op_info_register(basic_lstm_cell_weight_grad_op_info)
def _basic_lstm_cell_weight_grad_tbe():
    """BasicLSTMCellWeightGrad TBE register"""
    return
