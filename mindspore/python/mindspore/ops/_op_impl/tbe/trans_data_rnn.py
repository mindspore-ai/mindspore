# Copyright 2021 Huawei Technologies Co., Ltd
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

"""TransDataRNN op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

trans_data_rnn_op_info = TBERegOp("TransDataRNN") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("trans_data_rnn.so") \
    .compute_cost(10) \
    .kernel_name("trans_data_rnn") \
    .partial_flag(True) \
    .attr("src_format", "required", "str", "FRACTAL_ZN_RNN, ND_RNN_BIAS") \
    .attr("dst_format", "required", "str", "FRACTAL_ZN_RNN, ND_RNN_BIAS") \
    .attr("input_size", "required", "int", "all") \
    .attr("hidden_size", "required", "int", "all") \
    .input(0, "src", False, "required", "all") \
    .output(0, "dst", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_FracZNRNN) \
    .dtype_format(DataType.F16_FracZNRNN, DataType.F16_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_ND_RNNBIAS) \
    .dtype_format(DataType.F16_ND_RNNBIAS, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_ND_RNNBIAS) \
    .dtype_format(DataType.F32_ND_RNNBIAS, DataType.F32_Default) \
    .get_op_info()


@op_info_register(trans_data_rnn_op_info)
def _trans_data_rnn_tbe():
    """TransDataRNN TBE register"""
    return
