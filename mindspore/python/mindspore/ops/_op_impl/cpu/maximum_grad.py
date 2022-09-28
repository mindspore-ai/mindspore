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

"""MaximumGrad op"""
from mindspore.ops.op_info_register import op_info_register, CpuRegOp, DataType

maximum_grad_op_info = CpuRegOp("MaximumGrad") \
    .input(0, "grads", "required") \
    .input(1, "x1", "required") \
    .input(2, "x2", "required") \
    .output(0, "y1", "required") \
    .output(1, "y2", "required") \
    .dtype_format(DataType.U16_Default, DataType.U16_Default, DataType.U16_Default,
                  DataType.U16_Default, DataType.U16_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.U32_Default, DataType.U32_Default, DataType.U32_Default,
                  DataType.U32_Default, DataType.U32_Default) \
    .dtype_format(DataType.U64_Default, DataType.U64_Default, DataType.U64_Default,
                  DataType.U64_Default, DataType.U64_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.F64_Default,
                  DataType.F64_Default, DataType.F64_Default) \
    .get_op_info()


@op_info_register(maximum_grad_op_info)
def _maximum_grad_cpu():
    """MaximumGrad cpu register"""
    return
