# Copyright 2020-2023 Huawei Technologies Co., Ltd
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

"""RandomCategorical op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

random_categorical_op_info = AiCPURegOp("RandomCategorical") \
    .fusion_type("OPAQUE") \
    .input(0, "logits", "required") \
    .input(1, "num_sample", "required") \
    .input(2, "seed", "required") \
    .input(3, "counts", "required") \
    .input(4, "states", "required") \
    .output(0, "output", "required") \
    .dtype_format(DataType.F16_Default, DataType.I32_Default, DataType.I32_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I16_Default) \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.I32_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I16_Default) \
    .dtype_format(DataType.F64_Default, DataType.I32_Default, DataType.I32_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I16_Default) \
    .dtype_format(DataType.F16_Default, DataType.I32_Default, DataType.I32_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I32_Default) \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.I32_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I32_Default) \
    .dtype_format(DataType.F64_Default, DataType.I32_Default, DataType.I32_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I32_Default) \
    .dtype_format(DataType.F16_Default, DataType.I32_Default, DataType.I32_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I64_Default) \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.I32_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I64_Default) \
    .dtype_format(DataType.F64_Default, DataType.I32_Default, DataType.I32_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I64_Default) \
    .dtype_format(DataType.F16_Default, DataType.I64_Default, DataType.I64_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I16_Default) \
    .dtype_format(DataType.F32_Default, DataType.I64_Default, DataType.I64_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I16_Default) \
    .dtype_format(DataType.F64_Default, DataType.I64_Default, DataType.I64_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I16_Default) \
    .dtype_format(DataType.F16_Default, DataType.I64_Default, DataType.I64_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I32_Default) \
    .dtype_format(DataType.F32_Default, DataType.I64_Default, DataType.I64_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I32_Default) \
    .dtype_format(DataType.F64_Default, DataType.I64_Default, DataType.I64_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I32_Default) \
    .dtype_format(DataType.F16_Default, DataType.I64_Default, DataType.I64_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I64_Default) \
    .dtype_format(DataType.F32_Default, DataType.I64_Default, DataType.I64_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I64_Default) \
    .dtype_format(DataType.F64_Default, DataType.I64_Default, DataType.I64_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.I64_Default) \
    .get_op_info()

@op_info_register(random_categorical_op_info)
def _random_categorical_aicpu():
    """RandomCategorical AiCPU register"""
    return
