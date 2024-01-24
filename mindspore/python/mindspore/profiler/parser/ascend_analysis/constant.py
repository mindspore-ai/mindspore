# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Constant value for ascend profiling parser."""


class Constant:
    """Constant values"""
    INVALID_VALUE = -1
    NULL_VALUE = 0

    # dir name
    FRAMEWORK_DIR = "FRAMEWORK"
    OUTPUT_DIR = "ASCEND_PROFILER_OUTPUT"

    # file authority
    FILE_AUTHORITY = 0o640
    DIR_AUTHORITY = 0o700
    MAX_FILE_SIZE = 1024 * 1024 * 1024 * 10
    MAX_CSV_SIZE = 1024 * 1024 * 1024 * 5
    MAX_PATH_LENGTH = 4096
    MAX_WORKER_NAME_LENGTH = 226
    MAX_FILE_NAME_LENGTH = 255
    PROF_WARN_SIZE = 1024 * 1024 * 1024

    # tlv constant struct
    FIX_SIZE_BYTES = "fix_size_bytes"
    NS_TO_US = 1000

    # field name
    SEQUENCE_UNMBER = "Sequence number"
    FORWORD_THREAD_ID = "Fwd thread id"
    OP_NAME = "op_name"
    INPUT_SHAPES = "Input Dims"
    INPUT_DTYPES = "Input type"
    CALL_STACK = "Call stack"
    MODULE_HIERARCHY = "Module Hierarchy"
    FLOPS = "flops"
    NAME = "name"

    # trace constant
    PROCESS_NAME = "process_name"
    PROCESS_LABEL = "process_labels"
    PROCESS_SORT = "process_sort_index"
    THREAD_NAME = "thread_name"
    THREAD_SORT = "thread_sort_index"
    FLOW_START_PH = "s"
    FLOW_END_PH = "f"

    ACL_OP_EXE_NAME = ("AscendCL@aclopCompileAndExecute".lower(), "AscendCL@aclopCompileAndExecuteV2".lower())
    AI_CORE = "AI_CORE"

    # profiler begin info
    CANN_BEGIN_TIME = "collectionTimeBegin"
    CANN_BEGIN_MONOTONIC = "clockMonotonicRaw"
