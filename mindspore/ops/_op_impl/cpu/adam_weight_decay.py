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

"""AdamWeightDecay op"""
from mindspore.ops.op_info_register import op_info_register, CpuRegOp, DataType

adam_weight_decay_op_info = CpuRegOp("AdamWeightDecay") \
    .input(0, "var", "required") \
    .input(1, "m", "required") \
    .input(2, "v", "required") \
    .input(3, "lr", "required") \
    .input(4, "beta1", "required") \
    .input(5, "beta2", "required") \
    .input(6, "epsilon", "required") \
    .input(7, "decay", "required") \
    .input(8, "gradient", "required") \
    .output(0, "output0", "required") \
    .output(1, "output1", "required") \
    .output(2, "output2", "required") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .get_op_info()


@op_info_register(adam_weight_decay_op_info)
def _adam_weight_decay_cpu():
    """AdamWeightDecay cpu register"""
    return
