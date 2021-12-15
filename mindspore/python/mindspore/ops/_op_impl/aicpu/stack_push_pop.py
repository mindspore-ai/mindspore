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

"""StackPush and StackPop op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

stack_init_op_info = AiCPURegOp("StackInit") \
    .fusion_type("OPAQUE") \
    .attr("index", "int") \
    .get_op_info()

stack_push_op_info = AiCPURegOp("StackPush") \
    .fusion_type("OPAQUE") \
    .input(0, "src", "required") \
    .attr("index", "int") \
    .dtype_format(DataType.U8_Default) \
    .dtype_format(DataType.U16_Default) \
    .dtype_format(DataType.U32_Default) \
    .dtype_format(DataType.U64_Default) \
    .dtype_format(DataType.I8_Default) \
    .dtype_format(DataType.I16_Default) \
    .dtype_format(DataType.I32_Default) \
    .dtype_format(DataType.I64_Default) \
    .dtype_format(DataType.F16_Default) \
    .dtype_format(DataType.F32_Default) \
    .dtype_format(DataType.F64_Default) \
    .dtype_format(DataType.BOOL_Default) \
    .get_op_info()

stack_pop_op_info = AiCPURegOp("StackPop") \
    .fusion_type("OPAQUE") \
    .output(0, "dst", "required") \
    .attr("index", "int") \
    .dtype_format(DataType.U8_Default) \
    .dtype_format(DataType.U16_Default) \
    .dtype_format(DataType.U32_Default) \
    .dtype_format(DataType.U64_Default) \
    .dtype_format(DataType.I8_Default) \
    .dtype_format(DataType.I16_Default) \
    .dtype_format(DataType.I32_Default) \
    .dtype_format(DataType.I64_Default) \
    .dtype_format(DataType.F16_Default) \
    .dtype_format(DataType.F32_Default) \
    .dtype_format(DataType.F64_Default) \
    .dtype_format(DataType.BOOL_Default) \
    .get_op_info()

stack_destroy_op_info = AiCPURegOp("StackDestroy") \
    .fusion_type("OPAQUE") \
    .attr("index", "int") \
    .get_op_info()


@op_info_register(stack_init_op_info)
def _stack_init_aicpu():
    """StackInit aicpu register"""
    return


@op_info_register(stack_push_op_info)
def _stack_push_aicpu():
    """StackPush aicpu register"""
    return


@op_info_register(stack_pop_op_info)
def _stack_pop_aicpu():
    """StackPop aicpu register"""
    return


@op_info_register(stack_destroy_op_info)
def _stack_destroy_aicpu():
    """StackDestroy aicpu register"""
    return
