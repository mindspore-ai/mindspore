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

"""BasicLSTMCell op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

basic_lstm_cell_op_info = TBERegOp("BasicLSTMCell") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("basic_lstm_cell.so") \
    .compute_cost(10) \
    .kernel_name("basic_lstm_cell") \
    .attr("keep_prob", "optional", "float", "all") \
    .attr("forget_bias", "optional", "float", "all") \
    .attr("state_is_tuple", "optional", "bool", "true") \
    .attr("activation", "optional", "str", "all") \
    .partial_flag(True) \
    .input(0, "x", False, "required", "all") \
    .input(1, "h", False, "required", "all") \
    .input(2, "c", False, "required", "all") \
    .input(3, "w", False, "required", "all", reshape_type="CN") \
    .input(4, "b", False, "required", "all") \
    .input(5, "mask", False, "optional", "all") \
    .output(0, "ct", False, "required", "all") \
    .output(1, "ht", False, "required", "all") \
    .output(2, "it", False, "optional", "all") \
    .output(3, "jt", False, "optional", "all") \
    .output(4, "ft", False, "optional", "all") \
    .output(5, "ot", False, "optional", "all") \
    .output(6, "tanhct", False, "optional", "all") \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F32_FracNZ, DataType.F16_FracZNLSTM,
                  DataType.F32_Default, DataType.U8_Default, DataType.F32_FracNZ, DataType.F16_FracNZ,
                  DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracZNLSTM,
                  DataType.F16_Default, DataType.U8_Default, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ) \
    .get_op_info()


@op_info_register(basic_lstm_cell_op_info)
def _basic_lstm_cell_tbe():
    """BasicLSTMCell TBE register"""
    return
