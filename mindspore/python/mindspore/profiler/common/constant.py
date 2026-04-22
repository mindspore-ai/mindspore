# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
"""Constant values"""
from enum import Enum


class DeviceTarget(Enum):
    """The device target enum."""

    CPU = "CPU"
    GPU = "GPU"
    NPU = "Ascend"


class ProfilerLevel(Enum):
    """The profiler level enum."""

    LevelNone = "LevelNone"
    Level0 = "Level0"
    Level1 = "Level1"
    Level2 = "Level2"


class ProfilerActivity(Enum):
    """The profiler activity enum."""

    NPU = "NPU"
    GPU = "GPU"
    CPU = "CPU"


class AicoreMetrics(Enum):
    """The aicore metrics enum."""

    PipeUtilization = "PipeUtilization"
    ArithmeticUtilization = "ArithmeticUtilization"
    Memory = "Memory"
    MemoryL0 = "MemoryL0"
    MemoryUB = "MemoryUB"
    ResourceConflictRatio = "ResourceConflictRatio"
    L2Cache = "L2Cache"
    MemoryAccess = "MemoryAccess"
    AiCoreNone = "None"


class OverlapAnalysisTidName(Enum):
    """The overlap analysis tidName."""
    COMPUTING = "Computing"
    COMMUNICATION = "Communication"
    COMMUNICATION_NOT_OVERLAP = "Communication(Not Overlapped)"
    FREE = "Free"


class OpSummaryHeaders(Enum):
    """The op summary headers."""
    STEP_ID = "Step ID"
    DEVICE_ID = "Device_id"
    MODEL_NAME = "Model Name"
    MODEL_ID = "Model ID"
    TASK_ID = "Task ID"
    STREAM_ID = "Stream ID"
    OP_NAME = "Op Name"
    OP_TYPE = "OP Type"
    TASK_TYPE = "Task Type"
    TASK_START_TIME = "Task Start Time(us)"
    TASK_DURATION = "Task Duration(us)"
    TASK_WAIT_TIME = "Task Wait Time(us)"
    BLOCK_NUM = "Block Num"
    MIX_BLOCK_NUM = "Mix Block Num"
    HF32_ELIGIBLE = "HF32 Eligible"
    INPUT_SHAPES = "Input Shapes"
    INPUT_DATA_TYPES = "Input Data Types"
    INPUT_FORMATS = "Input Formats"
    OUTPUT_SHAPES = "Output Shapes"
    OUTPUT_DATA_TYPES = "Output Data Types"
    OUTPUT_FORMATS = "Output Formats"
    CONTEXT_ID = "Context ID"


class EventConstant:
    """Timeline event constant values"""

    START_FLOW = "s"
    END_FLOW = "f"
    META_EVENT = 'M'
    COMPLETE_EVENT = 'X'
    INSTANT_EVENT = 'i'
    COUNTER_EVENT = 'C'

    PROCESS_NAME = "process_name"
    PROCESS_LABEL = "process_labels"
    PROCESS_SORT = "process_sort_index"
    THREAD_NAME = "thread_name"
    THREAD_SORT = "thread_sort_index"

    CPU_LABEL = "CPU"
    ASCEND_LABEL = "NPU"

    HOST_TO_DEVICE_FLOW_CAT = "HostToDevice"
    MINDSPORE_NPU_FLOW_CAT = "async_npu"
    MINDSPORE_SELF_FLOW_CAT = "async_mindspore"
    MSTX_FLOW_CAT = "MsTx"
    MINDSPORE_SELF_FLOW_NAME = "mindspore_to_self"
    MINDSPORE_NPU_FLOW_NAME = "mindspore_to_npu"
    MSTX_FLOW_NAME = "mindspore_to_mstx"

    MINDSPORE_PID = 1
    CPU_OP_PID = 2
    SCOPE_LAYER_PID = 3

    MINDSPORE_SORT_IDX = 1
    CPU_OP_SORT_IDX = 2
    SCOPE_LAYER_SORT_IDX = 12

    # field name
    SEQUENCE_NUMBER = "Sequence number"
    FORWARD_THREAD_ID = "Fwd thread id"
    OP_NAME = "op_name"
    INPUT_SHAPES = "Input Dims"
    INPUT_DTYPES = "Input type"
    CALL_STACK = "Call stack"
    MODULE_HIERARCHY = "Module Hierarchy"
    FLOPS = "flops"
    NAME = "name"
    CUSTOM_INFO = "custom_info"
    TOP_SCOPE_NAMES = ('Default', 'Gradients', 'recompute_Default')
    KERNEL_LAUNCH_KEYWORDS = ("KernelLaunch", "LaunchTask")
    MSTX_KEYWORD = "Mstx"
    STACK_EVENT_CAT = "python_function"

    FLOW_OP = "flow"
    INVALID_FLOW_ID = 18446744073709551615


class TimeConstant:
    """Time constant values"""

    NS_TO_US = 0.001
    MS_TO_US = 1000


class FileConstant:
    """File constant values"""

    # tlv constant struct
    FIX_SIZE_DATA = "fix_size_data"
    CANN_FILE_REGEX = r"^PROF_\d+_\d+_[0-9a-zA-Z]+"
    FRAMEWORK_DIR = "FRAMEWORK"


class ProfilerStepNameConstant:
    """Profiler step name."""

    PROFILER_STEP = "ProfilerStep#"


class TimelineLayerName(Enum):
    """Timeline layer types."""
    MINDSPORE = "MindSpore"
    CPU_OP = "CPU OP"
    MSTX = ["python", "python3"]
    CANN = "CANN"
    SCOPER_LAYER = "Scope Layer"
    ASCEND_HARDWARE = "Ascend Hardware"
    HCCL = "HCCL"
    AI_CORE_FREQ = "AI Core Freq"
    HBM = "HBM"
    PCLE = "PCle"
    HCCS = "HCCS"
    LLC = "LLC"
    NPU_MEM = "NPU MEM"
    STARS_SOC_INFO = "Stars Soc Info"
    STARS_Chip_Trans = "Stars Chip Trans"
    ACC_PMU = "Acc PMU"
    SIO = "SIO"
    QOS = "QoS"
    NIC = "NIC"
    ROCE = "RoCE"
    OVERLAP_ANALYSIS = "Overlap Analysis"


class AnalysisMode(Enum):
    """analysis mode"""
    SYNC_MODE = "sync"
    ASYNC_MODE = "async"


class JitLevel:
    """jit level"""
    KBK_LEVEL = "O0"
    KBK_DVM_LEVEL = "O1"
    GRAPH_LEVEL = "O2"


class ExportType(Enum):
    Db = "db"
    Text = "text"


class CannLibName:
    """CANN lib name"""
    CANN_MSPTI = "libmspti.so"


class DynoMode:
    """dyno mode"""
    DYNO_DAEMON = "MSMONITOR_USE_DAEMON"


class HostSystem(Enum):
    """host system"""
    CPU = "cpu"
    MEM = "mem"
    DISK = "disk"
    NETWORK = "network"
    OSRT = "osrt"


class MsprofModeName:
    """msprof mode name"""
    MSPROF_DYNAMIC_ENV = "PROFILING_MODE"
