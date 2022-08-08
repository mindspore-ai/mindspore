# Copyright 2022 Huawei Technologies Co., Ltd
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

"""ReservoirReplayBuffer op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType


rrb_create_op_info = AiCPURegOp("ReservoirReplayBufferCreate") \
    .fusion_type("OPAQUE") \
    .output(0, "handle", "required") \
    .attr("capacity", "int") \
    .attr("schema", "listInt") \
    .attr("seed0", "int") \
    .attr("seed1", "int") \
    .dtype_format(DataType.I64_Default) \
    .get_op_info()


rrb_push_op_info = AiCPURegOp("ReservoirReplayBufferPush") \
    .input(0, "transition", "dynamic") \
    .output(0, "handle", "required") \
    .attr("handle", "int") \
    .dtype_format(DataType.BOOL_Default, DataType.I64_Default) \
    .dtype_format(DataType.I8_Default, DataType.I64_Default) \
    .dtype_format(DataType.I16_Default, DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.F16_Default, DataType.I64_Default) \
    .dtype_format(DataType.U8_Default, DataType.I64_Default) \
    .dtype_format(DataType.U16_Default, DataType.I64_Default) \
    .dtype_format(DataType.U32_Default, DataType.I64_Default) \
    .dtype_format(DataType.U64_Default, DataType.I64_Default) \
    .dtype_format(DataType.F32_Default, DataType.I64_Default) \
    .get_op_info()


rrb_sample_op_info = AiCPURegOp("ReservoirReplayBufferSample") \
    .output(0, "transitions", "dynamic") \
    .attr("handle", "int") \
    .attr("batch_size", "int") \
    .attr("schema", "listInt") \
    .dtype_format(DataType.BOOL_Default) \
    .dtype_format(DataType.I8_Default) \
    .dtype_format(DataType.I16_Default) \
    .dtype_format(DataType.I32_Default) \
    .dtype_format(DataType.I64_Default) \
    .dtype_format(DataType.F16_Default) \
    .dtype_format(DataType.U8_Default) \
    .dtype_format(DataType.U16_Default) \
    .dtype_format(DataType.U32_Default) \
    .dtype_format(DataType.U64_Default) \
    .dtype_format(DataType.F32_Default) \
    .get_op_info()


rrb_destroy_op_info = AiCPURegOp("ReservoirReplayBufferDestroy") \
    .output(0, "handle", "required") \
    .attr("handle", "int") \
    .dtype_format(DataType.I64_Default) \
    .get_op_info()


@op_info_register(rrb_create_op_info)
def _rrb_create_op_cpu():
    """ReservoirReplayBufferCreate AICPU register"""
    return


@op_info_register(rrb_push_op_info)
def _rrb_push_op_cpu():
    """ReservoirReplayBufferPush AICPU register"""
    return


@op_info_register(rrb_sample_op_info)
def _rrb_sample_op_cpu():
    """ReservoirReplayBufferSample AICPU register"""
    return


@op_info_register(rrb_destroy_op_info)
def _rrb_destroy_op_cpu():
    """ReservoirReplayBufferDestroy AICPU register"""
    return
