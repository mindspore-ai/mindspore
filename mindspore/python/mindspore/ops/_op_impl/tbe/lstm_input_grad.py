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

"""LSTMInputGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

lstm_input_grad_op_info = TBERegOp("LSTMInputGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("lstm_input_grad.so") \
    .compute_cost(10) \
    .kernel_name("lstm_input_grad") \
    .partial_flag(True) \
    .input(0, "w", False, "required", "all") \
    .input(1, "init_c", False, "required", "all") \
    .input(2, "c", False, "required", "all") \
    .input(3, "dy", False, "required", "all") \
    .input(4, "dh", False, "required", "all") \
    .input(5, "dc", False, "required", "all") \
    .input(6, "i", False, "required", "all") \
    .input(7, "j", False, "required", "all") \
    .input(8, "f", False, "required", "all") \
    .input(9, "o", False, "required", "all") \
    .input(10, "tanhct", False, "optional", "all") \
    .output(0, "dx", False, "required", "all") \
    .output(1, "dh_prev", False, "required", "all") \
    .output(2, "dc_prev", False, "required", "all") \
    .output(3, "dgate", False, "required", "all") \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ,
                  DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_FracNZ) \
    .get_op_info()


@op_info_register(lstm_input_grad_op_info)
def _lstm_input_grad_tbe():
    """LSTMInputGrad TBE register"""
    return
