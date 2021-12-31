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
"""Thr parser for parsing framework files."""

from mindspore.profiler.common.struct_type import StructType

# Note: All keys should be named with lower camel case, which are the same as those in C++.

TASK_DESC_STRUCT = dict(
    magicNumber=StructType.UINT16,
    dataTag=StructType.UINT16,
    taskType=StructType.UINT32,
    opName=[StructType.UINT64] * 16,  # opName is a mix data
    opType=[StructType.UINT64] * 8,   # opType is a mix data
    curIterNum=StructType.UINT64,
    timeStamp=StructType.UINT64,
    shapeType=StructType.UINT32,
    blockDims=StructType.UINT32,
    modelId=StructType.UINT32,
    streamId=StructType.UINT32,
    taskId=StructType.UINT32,
    threadId=StructType.UINT32,
    reserve=[StructType.UINT8] * 16
)

STEP_INFO_STRUCT = dict(
    magicNumber=StructType.UINT16,
    dataTag=StructType.UINT16,
    modelId=StructType.UINT32,
    streamId=StructType.UINT32,
    taskId=StructType.UINT32,
    timeStamp=StructType.UINT64,
    curIterNum=StructType.UINT64,
    threadId=StructType.UINT32,
    tag=StructType.UINT8,
    reserve=[StructType.UINT8] * 27
)

TENSOR_DATA_STRUCT = dict(
    magicNumber=StructType.UINT16,
    dataTag=StructType.UINT16,
    modelId=StructType.UINT32,
    curIterNum=StructType.UINT64,
    streamId=StructType.UINT32,
    taskId=StructType.UINT32,
    tensorNum=[StructType.UINT8] * 4,  # Note: Here the memory is aligned. The actual memory usage is 1, 3 is padding.
    tensorData=[[StructType.UINT32] * 11] * 5,
    reserve=[StructType.UINT8] * 8     # Note: Here the memory is aligned. The actual memory usage is 4, 4 is padding.
)
