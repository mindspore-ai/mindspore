# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
"""
Name: AUTO_PASSES_OPTIMIZE_PATH
Function: Whether to do optimize the passes configure.
Value Range:
    string: The passes configure file path.
    Default: '' .empty string. Disable to do optimize the passes.
"""
AUTO_PASSES_OPTIMIZE_PATH = ''

"""
Name: COMPILE_PROFILE
Function: Whether to do profile and print profile log.
Value Range:
    1: Enable.
    Default: Disable.
"""
COMPILE_PROFILE = ''

"""
Name: COMPILE_PROFILE_FINISH_ACTION
Function: Specify the last action name to print profile log.
Value Range:
    Action name string, for example, validate.
"""
COMPILE_PROFILE_FINISH_ACTION = ''

"""
Name: DEBUG_MODE
Function: Whether to compile in debug mode.
Value Range:
    "debug": Debug mode
    "release": Release mode
    Default: "debug"
"""
COMPILE_DEBUG_MODE = ''

"""
Name: FALLBACK_SUPPORT_LIST_DICT_INPLACE
Function: Whether to support the inplace operation of list and dict.
Value Range:
    0: Disable.
    Default: Enable.
"""
FALLBACK_SUPPORT_LIST_DICT_INPLACE = ''

"""
Name: FALLBACK_FORCE_ANY
Function: Whether to force nodes with ambiguous type in JIT Fallback to be of type Any.
Value Range:
    1: Enable.
    Default: Disable. According to the actual type of node's abstract.
"""
FALLBACK_FORCE_ANY = ''

"""
Name: IF_PARALLEL_CALL
Function: Specify the structure of parallel if in control flow.
Value Range:
    0: Parallel if will be converted into a nested if structure.
    1 or Default: Parallel if remains unchanged.
    2: Whe the subgraph outputs are all scalar, the parallel if is converted into a nested if structure,
       Otherwise the parallel if structure is maintained.
"""
IF_PARALLEL_CALL = ''

"""
Name: FOR_HALF_UNROLL
Function: Whether to use the for loop half expansion.
Value Range:
    1: Convert loops calls into partial loop calls.
    Default: Parse the for loop according to the original process.
"""
FOR_HALF_UNROLL = ''

"""
Name: NOT_WAIT_BRANCH_EVAL
Function: When deriving control flow branches, whether to wait for both switch branches to complete derivation
          before continue.
Value Range:
    1: Do not wait.
    Default: Wait.
"""
NOT_WAIT_BRANCH_EVAL = ''

"""
Name: RECURSIVE_EVAL
Function: Set the method of type inference.
Value Range:
    1: Recursive.
    Default: Stack derivation.
"""
RECURSIVE_EVAL = ''

"""
Name: SINGLE_EVAL
Function: Whether to use a single thread for type inference.
Value Range:
    1: Single thread.
    Default: Multithreading.
"""
SINGLE_EVAL = ''

"""
Name: ENABLE_DDE
Function: Whether to eliminate elements in tuple/list that are not used by other nodes.
Value Range:
    0: Do not eliminate.
    Default: Eliminate.
"""
ENABLE_DDE = ''

"""
Name: DDE_ONLY_MARK
Function: Whether to eliminate elements in tuple/list that have been marked and are not used by other nodes.
Value Range:
    1: Eliminate.
    Default: Do not eliminate.
"""
DDE_ONLY_MARK = ''

"""
Name: BOOST_PARSE
Function: Whether to perform constant folding of if conditional statements in advance.
Value Range:
    0: Disable.
    Default: Enable.
"""
BOOST_PARSE = ''

"""
Name: GREED_PARSE
Function: Whether to perform syntax decomposition in advance, including getattr, etc.
Value Range:
    1: Enable.
    Default: Disable.
"""
GREED_PARSE = ''

"""
Name: AMP_ENABLE_ALL_FG
Function: Whether to enable the function of setting mixed precision markers for all subgraphs of the top graphs.
Value Range:
    1: Enable.
    Default: Disable.
"""
AMP_ENABLE_ALL_FG = ''

"""
Name: DUMP_IR_META_DSL
Function: Whether to dump IR for meta op.
Value Range:
    String. The name of the operator that needs to dump IR.
"""
DUMP_IR_META_DSL = ''

"""
Name: DUMP_IR_CONFIG
Function: Configuring the generation method of IR graphs.
Value Range:
    String. The available options are: "DISABLE_BACKEND", "ENABLE_PASS_IR", "LINE_LEVEL0", "LINE_LEVEL1",
    "LINE_LEVEL2". For example: compile_config.DUMP_IR_CONFIG = "DISABLE_BACKEND#ENABLE_PASS_IR#LINE_LEVEL1".
"""
DUMP_IR_CONFIG = ''

"""
Name: TRAVERSE_SUBSTITUTIONS_MODE
Function: Set the execution method of IR passes.
Value Range:
    1: A single IR pass processes all nodes before switching to the next IR pass.
    Default: A single node matches all IR passes in sequence.
"""
TRAVERSE_SUBSTITUTIONS_MODE = ''

"""
Name: PRE_LIFT
Function: Whether to use performance pre-accleration means to improve compilation performance.
Value Range:
    1: Use.
    Default: Do not use.
"""
PRE_LIFT = ''

"""
Name: COMPILE_PRINT
Function: When using print, whether to print the type, shape, value information obtained during compilation.
Value Range:
    1: Print.
    Default: Do not print.
"""
COMPILE_PRINT = ''

"""
Name: ENABLE_FIX_CODE_LINE
Function: Whether to retain code line information.
Value Range:
    1: No code line information is retained, which can improve some compilation performance.
    Default: Preserve code line information.
"""
ENABLE_FIX_CODE_LINE = ''

"""
Name: RECORD_MEMORY
Function: Whether each action of graph compilation records memory usage tags for detecting memory leaks.
Value Range:
    1: Record.
    Default: Do not record.
"""
RECORD_MEMORY = ''

"""
Name: TRACE_LABEL_WITH_UNIQUE_ID
Function: When generating IR files, whether to use unique IDs for naming in the IR graphs.
Value Range:
    1: Use a unique ID for naming.
    Default: Use a short name without adding ID information.
"""
TRACE_LABEL_WITH_UNIQUE_ID = ''

"""
Name: DUMP_IR_DDE_DETAIL
Function: When generating IR files, whether to print DDE node detail.
Value Range:
    1: Print DDE node detail.
    Default: Only print used flags.
"""
DUMP_IR_DDE_DETAIL = ''

"""
Name: COMBINE_LIKE_GRAPHS
Function: Whether to combine the func_graphs which have the same object key according to the @cell_attr_register.
Value Range:
    0: Disable
    Default: Enable.
"""
COMBINE_LIKE_GRAPHS = ''

"""
Name: DUMP_VALIDATE_BEFORE_RESET_ID
Function: Whether to dump validate ir before reset id.
Value Range:
    1: Enable
    Default: Disable.
"""
DUMP_VALIDATE_BEFORE_RESET_ID = ''

"""
Name: ENABLE_RECOMPUTE_BEFORE_INLINE
Function: Whether to do recomputation before fprop and bprop being inlined.
Value Range:
    1: Enable
    Default: Disable.
"""
ENABLE_RECOMPUTE_BEFORE_INLINE = ''

"""
Name: STRICT_CHECK_PARENT_CONTEXT
Function: Whether to check parent context strictly.
Value Range:
    1: Enable
    Default: Disable.
"""
STRICT_CHECK_PARENT_CONTEXT = ''

"""
Name: CHECK_BPROP
Function: Whether to check back propagation nodes. The checking ensures that the shape and dtype of
          back propagation node outputs is the same as input parameters.
Value Range:
    1: Enable
    Default: Disable.
"""
CHECK_BPROP = ''

"""
Name: GRAD_FOR_SCALAR
Function: Whether to get gradient for scalar. When enable, the function's scalar input can be derived.
          Because the back-end does not support scaling operations currently, this interface only
          supports simple operations that can be deduced by the front-end.
Value Range:
    1: Enable
    Default: Disable.
"""
GRAD_FOR_SCALAR = ''

"""
Name: DEBUG_LEVEL
Function: Whether to record more debug information in compiling process. Used for debugging when errors occur.
Value Range:
    1: Enable
    Default: Disable.
"""
DEBUG_LEVEL = ''

"""
Name: HYPER_OFFLOAD_SELECT_DISTANCE
Function: Indicate minimum usage distance for node to offload.
Value Range: Int value.
    Default: 100
"""
HYPER_OFFLOAD_SELECT_DISTANCE = '500'

"""
Name: HYPER_OFFLOAD_SELECT_NUM
Function: Indicate number of data to offload.
Value Range: Int value.
    Default: 100
"""
HYPER_OFFLOAD_SELECT_NUM = '10000'

"""
Name: HYPER_OFFLOAD_PREFETCH_DISTANCE
Function: Indicate data prefetch distance.
Value Range: Int value.
    Default: 50
"""
HYPER_OFFLOAD_PREFETCH_DISTANCE = '50'

"""
Name: HYPER_OFFLOAD_RELEASE_DISTANCE
Function: Indicate data release wait distance.
Value Range: Int value.
    Default: 0
"""
HYPER_OFFLOAD_RELEASE_DISTANCE = '50'

"""
Name: HYPER_OFFLOAD_MAX_DEVICE_MEMORY
Function: Max device memory threshold in bytes for memory-peak-based offload strategy.
          Used when MS_DEV_HYPER_OFFLOAD_PEAK_STRATEGY=1.
          Set this value according to the real available memory capacity of the target device.
          Backend subtracts HYPER_OFFLOAD_PEAK_MODELED_HEADROOM_BYTES as headroom before
          comparing with the modeled peak.
Value Range: Int value (bytes), 0 to disable peak candidate selection.
    Default: 40GB
"""
HYPER_OFFLOAD_MAX_DEVICE_MEMORY = '42949672960'

"""
Name: HYPER_OFFLOAD_SCORE_WEIGHT_SIZE
Function: Score weight for candidate tensor size (w1). Larger weight favors bigger tensors.
Value Range: Float value.
    Default: 1.0
"""
HYPER_OFFLOAD_SCORE_WEIGHT_SIZE = '1.0'

"""
Name: HYPER_OFFLOAD_SCORE_WEIGHT_COVER_START
Function: Score weight for cover-start distance (w2). Larger weight favors earlier creation.
Value Range: Float value.
    Default: 1.0
"""
HYPER_OFFLOAD_SCORE_WEIGHT_COVER_START = '1.0'

"""
Name: HYPER_OFFLOAD_SCORE_WEIGHT_COVER_END
Function: Score weight for cover-end distance (w3). Larger weight favors later usage.
Value Range: Float value.
    Default: 1.0
"""
HYPER_OFFLOAD_SCORE_WEIGHT_COVER_END = '1.0'

"""
Name: HYPER_OFFLOAD_SCORE_WEIGHT_NEIGHBOR_IMPACT
Function: Score weight for neighbor clustering penalty (w4). Avoids too many adjacent offloads.
Value Range: Float value.
    Default: 1.0
"""
HYPER_OFFLOAD_SCORE_WEIGHT_NEIGHBOR_IMPACT = '1.0'

"""
Name: HYPER_OFFLOAD_MEMORY_BLOCK_PATH
Function: Path to memory_block.csv for HyperOffload peak memory modeling.
Value Range: File path string.
    Default: Empty
"""
HYPER_OFFLOAD_MEMORY_BLOCK_PATH = ''

"""
Name: HYPER_OFFLOAD_DYNAMIC_SHAPE_PLACEHOLDER_BYTES
Function: Placeholder memory size used by HyperOffload peak modeling fallback when a dynamic-shape operator cannot be
    resolved precisely.
Value Range: Int value (bytes).
    Default: 128MB
"""
HYPER_OFFLOAD_DYNAMIC_SHAPE_PLACEHOLDER_BYTES = '134217728'

"""
Name: HYPER_OFFLOAD_PEAK_MODELED_HEADROOM_BYTES
Function: Reserved memory headroom subtracted from HYPER_OFFLOAD_MAX_DEVICE_MEMORY.
    It is used when deciding whether the modeled peak already fits without additional offload
    selection. This margin covers modeling error and runtime memory fluctuation.
    It should be determined from the observed gap between modeled peak memory and actual
    device memory usage on the target model and device.
Value Range: Int value (bytes).
    Default: 4GB
"""
HYPER_OFFLOAD_PEAK_MODELED_HEADROOM_BYTES = '4294967296'

"""
Name: HYPER_OFFLOAD_D2H_BANDWIDTH_GBPS
Function: Device-to-Host memory transfer bandwidth in GB/s.
    Used for adaptive release_distance calculation.
Value Range: Float value (GB/s).
    Default: 20.0 GB/s
Example: export HYPER_OFFLOAD_D2H_BANDWIDTH_GBPS=20.0
"""
HYPER_OFFLOAD_D2H_BANDWIDTH_GBPS = '20.0'

"""
Name: HYPER_OFFLOAD_H2D_BANDWIDTH_GBPS
Function: Host-to-Device memory transfer bandwidth in GB/s.
    Used for adaptive prefetch_distance calculation.
Value Range: Float value (GB/s).
    Default: 55.0 GB/s
Example: export HYPER_OFFLOAD_H2D_BANDWIDTH_GBPS=55.0
"""
HYPER_OFFLOAD_H2D_BANDWIDTH_GBPS = '55.0'

"""
Name: HYPER_OFFLOAD_DEFAULT_KERNEL_TIME_US
Function: Default kernel execution time in microseconds.
    Used for adaptive distance calculation.
Value Range: Float value (microseconds).
    Default: 400.0 us
Example: export HYPER_OFFLOAD_DEFAULT_KERNEL_TIME_US=400.0
"""
HYPER_OFFLOAD_DEFAULT_KERNEL_TIME_US = '400.0'

"""
Name: MS_DEV_HYPER_OFFLOAD_ENABLE_ADAPTIVE_RELEASE_DISTANCE
Function: Enable adaptive release_distance calculation for peak offload strategy.
    When enabled, release_distance is calculated based on operator size, D2H bandwidth,
    and kernel execution time, instead of using the fixed HYPER_OFFLOAD_RELEASE_DISTANCE.
Value Range:
    1: Enable adaptive calculation (only for peak strategy)
    0 or empty: Disable, use fixed HYPER_OFFLOAD_RELEASE_DISTANCE
    Default: Disable
Example: export MS_DEV_HYPER_OFFLOAD_ENABLE_ADAPTIVE_RELEASE_DISTANCE=1
"""
MS_DEV_HYPER_OFFLOAD_ENABLE_ADAPTIVE_RELEASE_DISTANCE = ''

"""
Name: MS_DEV_HYPER_OFFLOAD_ENABLE_ADAPTIVE_PREFETCH_DISTANCE
Function: Enable adaptive prefetch_distance calculation for peak offload strategy.
    When enabled, prefetch_distance is calculated based on operator size, H2D bandwidth,
    and kernel execution time, instead of using the fixed HYPER_OFFLOAD_PREFETCH_DISTANCE.
Value Range:
    1: Enable adaptive calculation (only for peak strategy)
    0 or empty: Disable, use fixed HYPER_OFFLOAD_PREFETCH_DISTANCE
    Default: Disable
Example: export MS_DEV_HYPER_OFFLOAD_ENABLE_ADAPTIVE_PREFETCH_DISTANCE=1
"""
MS_DEV_HYPER_OFFLOAD_ENABLE_ADAPTIVE_PREFETCH_DISTANCE = ''

"""
Name: PIJIT_SUBGRAPH_BREAK_OPTIMIZE
Function: Whether to enable subgraph break optimization in PIJit.
Value Range:
    0: Disable subgraph break optimization in PIJit.
    Default: Enable.
"""
PIJIT_SUBGRAPH_BREAK_OPTIMIZE = ''

"""
Name: ENABLE_ELIMINATE_UNUSED_PARAMS
Function: Whether to enable eliminate unused parameters optimization in PIJit.
Value Range:
    1: Enable, Disable if other value.
    Default: Disable.
"""
ENABLE_ELIMINATE_UNUSED_PARAMS = ''

"""
Name: PUT_ALL_CNODE_INTO_ORDER_LIST
Function: Whether to put all CNode into order list in back prop.
Value Range:
    0: Disable
    Default: Enable.
"""
PUT_ALL_CNODE_INTO_ORDER_LIST = ''

"""
Name: CHECK_PASS_NODE_SCOPE
Function: Whether to check
Value Range:
    1: Enable
    Default: Disable.
"""
CHECK_PASS_NODE_SCOPE = ''

"""
Name: JIT_ENABLE_AUGASSIGN_INPLACE
Function: Whether enable augassign inplace.
Value Range:
    0: Disable
    1: Enable
    Default: Enable
"""
JIT_ENABLE_AUGASSIGN_INPLACE = '1'

"""
Name: JIT_ENABLE_AUGASSIGN_INPLACE_FALLBACK
Function: Whether enable augassign inplace fallback.
Value Range:
    0: Disable
    1: Enable
    Default: Enable
"""
JIT_ENABLE_AUGASSIGN_INPLACE_FALLBACK = '1'

"""
Name: GRAD_JIT_FILTER
Function: Whether to filter grad jit graph.
Value Range:
    1: Enable filter grad jit output
    2: Enable filter grad jit input and output. May cause error when input gradient information changed.
    Default: Disable.
"""
GRAD_JIT_FILTER = '1'

"""
Name: ENABLE_HYPER_OFFLOAD_SLIDE
Function: Whether to enable the hyper offload with sliding window method.
Value Range:
    1: Enable
    Default: Disable
"""
ENABLE_HYPER_OFFLOAD_SLIDE = ''

__all__ = [
    "COMPILE_PROFILE",
    "COMPILE_PROFILE_FINISH_ACTION",
    "COMPILE_DEBUG_MODE",
    "FALLBACK_SUPPORT_LIST_DICT_INPLACE",
    "FALLBACK_FORCE_ANY",
    "IF_PARALLEL_CALL",
    "FOR_HALF_UNROLL",
    "NOT_WAIT_BRANCH_EVAL",
    "RECURSIVE_EVAL",
    "SINGLE_EVAL",
    "ENABLE_DDE",
    "DDE_ONLY_MARK",
    "BOOST_PARSE",
    "GREED_PARSE",
    "AMP_ENABLE_ALL_FG",
    "DUMP_IR_META_DSL",
    "DUMP_IR_CONFIG",
    "TRAVERSE_SUBSTITUTIONS_MODE",
    "PRE_LIFT",
    "COMPILE_PRINT",
    "ENABLE_FIX_CODE_LINE",
    "RECORD_MEMORY",
    "TRACE_LABEL_WITH_UNIQUE_ID",
    "DUMP_IR_DDE_DETAIL",
    "COMBINE_LIKE_GRAPHS",
    "DUMP_VALIDATE_BEFORE_RESET_ID",
    "ENABLE_RECOMPUTE_BEFORE_INLINE",
    "STRICT_CHECK_PARENT_CONTEXT",
    "AUTO_PASSES_OPTIMIZE_PATH",
    "CHECK_BPROP",
    "GRAD_FOR_SCALAR",
    "DEBUG_LEVEL",
    "HYPER_OFFLOAD_SELECT_DISTANCE",
    "HYPER_OFFLOAD_SELECT_NUM",
    "HYPER_OFFLOAD_PREFETCH_DISTANCE",
    "HYPER_OFFLOAD_RELEASE_DISTANCE",
    "HYPER_OFFLOAD_MAX_DEVICE_MEMORY",
    "HYPER_OFFLOAD_SCORE_WEIGHT_SIZE",
    "HYPER_OFFLOAD_SCORE_WEIGHT_COVER_START",
    "HYPER_OFFLOAD_SCORE_WEIGHT_COVER_END",
    "HYPER_OFFLOAD_SCORE_WEIGHT_NEIGHBOR_IMPACT",
    "HYPER_OFFLOAD_MEMORY_BLOCK_PATH",
    "HYPER_OFFLOAD_DYNAMIC_SHAPE_PLACEHOLDER_BYTES",
    "HYPER_OFFLOAD_PEAK_MODELED_HEADROOM_BYTES",
    "HYPER_OFFLOAD_D2H_BANDWIDTH_GBPS",
    "HYPER_OFFLOAD_H2D_BANDWIDTH_GBPS",
    "HYPER_OFFLOAD_DEFAULT_KERNEL_TIME_US",
    "MS_DEV_HYPER_OFFLOAD_ENABLE_ADAPTIVE_RELEASE_DISTANCE",
    "MS_DEV_HYPER_OFFLOAD_ENABLE_ADAPTIVE_PREFETCH_DISTANCE",
    "PIJIT_SUBGRAPH_BREAK_OPTIMIZE",
    "ENABLE_ELIMINATE_UNUSED_PARAMS",
    "PUT_ALL_CNODE_INTO_ORDER_LIST",
    "CHECK_PASS_NODE_SCOPE",
    "JIT_ENABLE_AUGASSIGN_INPLACE",
    "GRAD_JIT_FILTER",
    "JIT_ENABLE_AUGASSIGN_INPLACE_FALLBACK",
    "ENABLE_HYPER_OFFLOAD_SLIDE"
]
