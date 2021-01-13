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

"""DynamicRNN op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

dynamic_rnn_op_info = TBERegOp("DynamicRNN") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("dynamic_rnn.so") \
    .compute_cost(10) \
    .kernel_name("dynamic_rnn") \
    .attr("cell_type", "optional", "str", "all", "LSTM") \
    .attr("direction", "optional", "str", "all", "UNIDIRECTIONAL") \
    .attr("cell_depth", "optional", "int", "all", "1") \
    .attr("use_peephole", "optional", "bool", "all", "false") \
    .attr("keep_prob", "optional", "float", "all", "1") \
    .attr("cell_clip", "optional", "float", "all", "-1") \
    .attr("num_proj", "optional", "int", "all", "0") \
    .attr("time_major", "optional", "bool", "all", "true") \
    .attr("activation", "optional", "str", "all", "tanh") \
    .attr("forget_bias", "optional", "float", "all", "0") \
    .attr("is_training", "optional", "bool", "all", "true") \
    .partial_flag(True) \
    .input(0, "x", False, "required", "all") \
    .input(1, "w", False, "required", "all", reshape_type="CN") \
    .input(2, "b", False, "required", "all") \
    .input(3, "seq_length", False, "optional", "all") \
    .input(4, "init_h", False, "optional", "all") \
    .input(5, "init_c", False, "optional", "all") \
    .input(6, "wci", False, "optional", "all") \
    .input(7, "wcf", False, "optional", "all") \
    .input(8, "wco", False, "optional", "all") \
    .input(9, "mask", False, "optional", "all") \
    .output(0, "y", False, "required", "all") \
    .output(1, "output_h", False, "required", "all") \
    .output(2, "output_c", False, "required", "all") \
    .output(3, "i", False, "required", "all") \
    .output(4, "j", False, "required", "all") \
    .output(5, "f", False, "required", "all") \
    .output(6, "o", False, "required", "all") \
    .output(7, "tanhc", False, "required", "all") \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracZNLSTM, DataType.F32_Default, DataType.None_Default,
                  DataType.F16_FracNZ, DataType.F32_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.U8_Default, DataType.F32_FracNZ, DataType.F16_FracNZ,
                  DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ, DataType.F32_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracZNLSTM, DataType.F16_Default, DataType.None_Default,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.U8_Default, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracZNLSTM, DataType.F32_Default, DataType.I32_Default,
                  DataType.F16_FracNZ, DataType.F32_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.U8_Default, DataType.F32_FracNZ, DataType.F16_FracNZ,
                  DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ, DataType.F32_FracNZ,
                  DataType.F32_FracNZ, DataType.F32_FracNZ) \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracZNLSTM, DataType.F16_Default, DataType.I32_Default,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.U8_Default, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ) \
    .get_op_info()


@op_info_register(dynamic_rnn_op_info)
def _dynamic_rnn_tbe():
    """DynamicRNN TBE register"""
    return
