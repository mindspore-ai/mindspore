/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_UTILS_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_UTILS_H_

#include <fcntl.h>
#include <sys/stat.h>
#ifndef _MSC_VER
#include <sys/time.h>
#endif
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <sstream>
#include <vector>
#include <tuple>

#include "base/base.h"
#include "include/utils/visible.h"
#include "include/utils/stream_util.h"
#include "ir/dtype/type_id.h"
#include "utils/log_adapter.h"

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__) && !defined(ENABLE_ANDROID)
#include "tools/profiler/mstx/mstx_impl.h"
#endif

#include "ops_utils/op_constants.h"

#ifndef MS_UNLIKELY
#ifdef _MSC_VER
#define MS_UNLIKELY(x) (x)
#else
#define MS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif
#endif

#ifndef MS_LIKELY
#ifdef _MSC_VER
#define MS_LIKELY(x) (x)
#else
#define MS_LIKELY(x) __builtin_expect(!!(x), 1)
#endif
#endif

namespace mindspore {
// FuncGraph Flags
constexpr auto kFlagIsPynativeBpropGraph = "is_pynative_bprop_graph";
constexpr auto kFlagPyNativeRunInGraph = "pynative_run_in_graph";
constexpr auto kFlagNeedRenormalize = "need_renormalize";
constexpr auto kFlagEnableZeroCopyInGraph = "enable_zero_copy_in_graph";
constexpr auto kFlagJitCallGraph = "jit_call_graph";
constexpr auto kFlagSwitchInline = "switch_inline_graph";
constexpr auto kFlagIsControlFlow = "is_control_flow";

// custom operator func type
constexpr auto kCustomTypeAOT = "aot";
constexpr auto kCustomTypeOPPlugin = "op_plugin";
constexpr auto kCustomTypePyfunc = "pyfunc";
constexpr auto kCustomTypeTbe = "tbe";
constexpr auto kCustomTypeAICPU = "aicpu";
constexpr auto kCustomTypeHybrid = "hybrid";
constexpr auto kCustomTypeCustom = "Custom";

// backend
constexpr auto kBackendMSBackend = "ms_backend";
constexpr auto kBackendGE = "GE";
constexpr auto kBackendJitConfig = "backend_jit_config";

// primal attr key name
constexpr auto kPrimalAttrForwardNodeName = "forward_node_name";
constexpr auto kPrimalAttrBackwardMicroEnd = "backward_micro_end";
constexpr auto kPrimalAttrForwardEnd = "forward_end";
constexpr auto kPrimalAttrSegmentMax = "segment_max";
constexpr auto kPrimalAttrUniqueId = "unique_id";
constexpr auto kPrimalAttrForwardUniqueId = "forward_unique_id";
constexpr auto kPrimalAttrForwardCommNodeUniqueId = "forward_comm_node_unique_id";
constexpr auto kPrimalAttrMirrorUserId = "mirror_user_id";
constexpr auto kPrimalAttr1b1fCallCall = "1b1f_call_call";
constexpr auto kCNodeAttr1f1bIndexBp = "1f1b_index_bp";
constexpr auto kCNodeAttr1f1bIndexFp = "1f1b_index_fp";
constexpr auto kCNodeAttr1f1bIndexRecv = "1f1b_index_recv";
constexpr auto kCNodeAttr1f1bIndexInterRecv = "1f1b_index_inter_recv";
constexpr auto kCNodeAttr1f1bIndexBpBegin = "1f1b_index_bp_begin";
constexpr auto kCNodeAttr1f1bLastCNode = "1f1b_last_cnode";
constexpr auto kCNodeAttr1f1bMiddleCNode = "1f1b_middle_cnode";
constexpr auto kCNodeAttrForwardAll2AllInput = "forward_all2all_input";
constexpr auto kCNodeAttrForwardAll2AllOutput = "forward_all2all_output";
constexpr auto kCNodeAttrBackwardAll2AllInput = "backward_all2all_input";
constexpr auto kCNodeAttrBackwardAll2AllOutput = "backward_all2all_output";
// attr value
constexpr auto kValueTargetSwitch = "target_switch";
constexpr auto kValueTargetOther = "target_other";
constexpr auto kValueTrue = "true";
constexpr auto kValueFalse = "false";
constexpr auto kValueTrueToupper = "True";
constexpr auto kValueFalseToupper = "False";
constexpr auto kTensorValueIsType = "tensor_value_is_type";
constexpr auto kTensorValueIsEmpty = "tensor_value_is_empty";
constexpr auto kTensorUserDataIsSensTensor = "is_sens_tensor";
constexpr auto kFakeTensorPos = "fake_tensor_pos";
constexpr auto kFakeTensorListPos = "fake_tensor_list_pos";
constexpr auto kChannelNameNpuLog = "_npu_log";

// env key
constexpr auto kCompilerCacheEnable = "MS_COMPILER_CACHE_ENABLE";
constexpr auto kCompilerCachePath = "MS_COMPILER_CACHE_PATH";
constexpr auto kSimulationLevel = "MS_SIMULATION_LEVEL";
constexpr auto kSimulationLevelCompileGraph = "0";
constexpr auto kSimulationLevelCompileKernel = "1";

// comm
constexpr auto kHCCLWorldGroup = "hccl_world_group";
constexpr auto kNCCLWorldGroup = "nccl_world_group";
constexpr auto kEnvRankSize = "RANK_SIZE";
constexpr auto kEnvRankId = "RANK_ID";
constexpr auto kEnvDeviceId = "DEVICE_ID";
constexpr auto kEnvLocalRankSize = "LOCAL_RANK_SIZE";
constexpr auto kEnvLocalRankId = "LOCAL_RANK_ID";

// define special index in special node
constexpr auto kAnfPrimitiveIndex = 0;
constexpr auto kFirstDataInputIndex = 1;
constexpr auto kRealInputNodeIndexInTupleGetItem = 1;
constexpr auto kInputNodeOutputIndexInTupleGetItem = 2;
constexpr auto kSparseGetAttrInputSize = 2;
constexpr auto kTupleGetItemInputSize = 3;

// index define of kTupleSetItem
constexpr auto kTupleSetItemTupleIndex = 1;
constexpr auto kTupleSetItemIndexIndex = 2;
constexpr auto kTupleSetItemValueIndex = 3;
constexpr auto kTupleSetItemInputSize = 4;
// index define of partial
constexpr auto kPartialMinInputSize = 2;
constexpr auto kPartialGraphIndex = 1;

// index define of switch
constexpr auto kSwitchInputSize = 4;
constexpr auto kSwitchCondIndex = 1;
constexpr auto kSwitchTrueBranchIndex = 2;
constexpr auto kSwitchFalseBranchIndex = 3;
constexpr auto kSwitchBranchesNum = 2;

// index define of switch_layer
constexpr auto kSwitchLayerInputSize = 3;
constexpr auto kSwitchLayerSelectIndex = 1;
constexpr auto kSwitchLayerBranchesIndex = 2;

// index define of depend
constexpr auto kRealInputIndexInDepend = 1;
constexpr auto kDependAttachNodeIndex = 2;
constexpr auto kDependInputSize = 3;
// index define of UpdateState
constexpr auto kUpdateStateStateInput = 1;
constexpr auto kUpdateStateRealInput = 2;
// index define of Load
constexpr auto kLoadRealInput = 1;
constexpr auto kLoadStateInput = 2;
// time transfer unit
constexpr int kBasicTimeTransferUnit = 1000;
constexpr int kMaxVectorSize = 10000;

// graph parse
constexpr auto kClassTensorObject = "class_tensor_object";

// graph type
constexpr auto kFuncGraphTypeName = "FuncGraph";
constexpr auto kKernelGraphTypeName = "KernelGraph";

// graph group
constexpr auto kDefaultGroup = "DefaultGroup";
constexpr auto kKernelGroup = "KernelGroup";
constexpr auto kGraphGroup = "GraphGroup";

// dump execute order
constexpr auto kExecuteOrderFileName = "execute_order/execute_order.csv";
constexpr auto kCommExecuteOrderFileName = "execute_order/comm_execute_order.csv";

// compile cache
constexpr auto kUniqueCacheName = "UniqueCacheName";
constexpr auto kDistributedSplit = "distribtued_split";
constexpr auto kValidate = "validate";
constexpr auto kGraphId = "graph_id";
constexpr auto kBackendFrontAnf = "backend_front_anf";
constexpr auto kInternalParameterToFrontNode = "internal_parameter_to_front_node";
constexpr auto kOutInRef = "out_in_ref";
constexpr auto kIsFeatureMap = "is_feature_map";
constexpr auto kGraphValueNodes = "graph_value_nodes";
constexpr auto kExecutionOrder = "execution_order";
constexpr auto kChildGraphOrder = "child_graph_order";
constexpr auto kRunMode = "run_mode";
constexpr auto kIsLoopCountSink = "is_loop_count_sink";
constexpr auto kIsDynamicShape = "is_dynamic_shape";
constexpr auto kInputs = "inputs";
constexpr auto kParameters = "parameters";
constexpr auto kForwardOutput = "forward_output";
constexpr auto kChildGraphResult = "child_graph_result";
constexpr auto kDeviceTarget = "device_target";
constexpr auto kRootGraphId = "root_graph_id";
constexpr auto kExecutable = "executable";
constexpr auto kValidInputs = "valid_inputs";
constexpr auto kNeedInline = "need_inline";
constexpr auto kStartLabel = "start_label";
constexpr auto kEndGoto = "end_goto";
constexpr auto kPreGraphs = "pre_graphs";
constexpr auto kPostGraphs = "post_graphs";
constexpr auto kHasRecursiveCall = "has_recursive_call";
constexpr auto kHasSubgraphMultiCall = "has_subgraph_multicall";
constexpr auto kIsNeedGil = "is_need_gil";
constexpr auto kIsFromSingleOp = "is_from_single_op";
constexpr auto kCommSubGraphIds = "comm_sub_graph_ids";
constexpr auto kNodesKernelInfo = "nodes_kernel_info";
constexpr auto kAllInputFormat = "all_input_format";
constexpr auto kAllOutputFormat = "all_output_format";
constexpr auto kAllInputDeviceType = "all_input_device_type";
constexpr auto kAllOutputDeviceType = "all_output_device_type";
constexpr auto kAllInputReshapeType = "all_input_reshape_type";
constexpr auto kAllOutputReshapeType = "all_output_reshape_type";
constexpr auto kOutputDataDesc = "output_data_desc";
constexpr auto kCoreType = "core_type";
constexpr auto kRuntimeCacheValid = "runtime_cache_valid";
constexpr auto kRuntimeCacheDeviceTarget = "runtime_cache_device_target";
constexpr auto kRuntimeCacheOutputTensorNum = "runtime_cache_output_tensor_num";
constexpr auto kRuntimeCacheIsRealKernel = "runtime_cache_is_real_kernel";
constexpr auto kRuntimeCachePrevOutputs = "runtime_cache_prev_outputs";
constexpr auto kCorrespondFrontendGraph = "correspond_frontend_graph";
constexpr auto kReturnNode = "_return_node";
constexpr auto kReturnPrimNode = "_return_prim_node";
constexpr auto kOriginDataFormat = "origin_data_format";
constexpr auto kKernelType = "kernel_type";
constexpr auto kOpType = "op_type";
constexpr auto kFusionType = "fusion_type";
constexpr auto kOpPattern = "op_pattern";
constexpr auto kProcessor = "processor";
constexpr auto kKernelBuildInfoValid = "kernel_build_info_valid";
constexpr auto kInputKernelObjectTypes = "input_kernel_object_types";
constexpr auto kOutputKernelObjectTypes = "output_kernel_object_types";
constexpr auto kOutputElementsKernelObjectTypes = "output_elements_kernel_object_types";
constexpr auto kInputSizeList = "input_size_list";
constexpr auto kOutputSizeList = "output_size_list";
constexpr auto kJsonName = "json_name";
constexpr auto kHasSelectKernelBuildInfo = "has_select_kernel_build_info";
constexpr auto kBackendParamToFrontendParamIndex = "backend_param_to_frontend_param_index_";
constexpr auto kLabelNum = "label_num";
constexpr auto kEnableMultiStream = "enable_multi_stream";
constexpr auto kParameterUniqueNameToName = "param_unique_name_to_name";
constexpr auto kRefInOutMap = "ref_in_out_map";
constexpr auto kRetryIntervalMilliSeconds = 500;
constexpr auto kSummaryNodes = "summary_nodes";
constexpr auto kSummaryNodeExist = "summary_node_exist";
constexpr auto kGeCache = "ge_cache";
constexpr auto kGeGraphKey = "ge.graph_key";
constexpr auto kGeGraphCompilerCacheDir = "ge.graph_compiler_cache_dir";
constexpr auto kIsRefGraph = "is_ref_graph";
constexpr auto kFromRefGraph = "from_ref_graph";
constexpr auto kGraphOutputToFrontNodeMap = "graph_output_to_front_node_map";
constexpr auto kFrontNodeToGraphOutputMap = "front_node_to_graph_output_map";
constexpr auto kInlineSubGraphKernelsMap = "inline_sub_graph_kernels";
constexpr auto kConditionGatherToSwitchMap = "condition_gather_to_switch";
constexpr auto kSomasOutputResult = "somas_output_result";
constexpr auto kSomasWorkspaceResult = "somas_workspace_result";
constexpr auto kSomasWholeBlockSize = "somas_whole_block_size";
constexpr auto kSomasMergedBlocksMap = "somas_merged_blocks_map";

// Backend compile cache.
constexpr auto kKernelGraphToDeviceContext = "kernel_graph_to_device_context";
constexpr auto kGrahpId = "graph_id";
constexpr auto kKernelGraphToDeviceId = "kernel_graph_to_device_id";
constexpr auto kKernelGraphToDeviceName = "kernel_graph_to_device_name";
constexpr auto kFuncGraphToKernelGraphIds = "func_graph_to_kernel_graph_ids";
constexpr auto kFuncGraphPtrId = "sub_func_graph_ptr_id";
constexpr auto kSubFuncGraphId = "sub_func_graph_ids";
constexpr auto kControlNodeId = "control_node_id";
constexpr auto kOutputNodeId = "output_node_id";
constexpr auto kDeviceName = "device_name";
constexpr auto kDeviceId = "device_id";
constexpr auto kIsRootGraph = "CompileCacheFuncGraph";
constexpr auto kMsExcutionMode = "ms_excution_mode";
constexpr auto kControlNodeCache = "ControlNodeCache";
constexpr auto kTupleBackendFrontAnfIndex = "tuple_backend_front_anf_index_map";
constexpr auto kKernelGraphNum = "kernel_graph_num";
constexpr auto kBackendFrontAnfExt = "backend_front_anf_ext";
constexpr auto kBackendAddTensorMove = "backend_add_tensor_move";
constexpr auto kIncludeNotCutAttrAnf = "include_not_cut_attr_anf";
constexpr auto kIncludeRealBackendAttrAnf = "include_real_backend_flag_anf";
constexpr auto kGraphIdsSingleCache = "graph_ids_for_single_cache";

// recompute and parallel
constexpr auto kRecomputeInsert = "recompute_insert";
constexpr auto kAddedRecomputeDependAttr = "added_recompute_depend";
constexpr auto kRecomputeSubgraphIdAttr = "recompute_subgraph_id";
constexpr auto kCondidateOverlapBlockId = "condidate_overlap_block_id";
constexpr auto kNcclWorldGroup = "nccl_world_group";
constexpr auto kHcclWorldGroup = "hccl_world_group";
constexpr auto kSyncBnGroup = "sync_bn_group";
constexpr auto kRankID = "RANK_ID";
constexpr auto kIsBp = "is_bp";
constexpr auto kSeqChunk = "seq_chunk";
constexpr auto kChunk = "chunk";
constexpr auto kMicro = "micro";

// User data key.

// pyexecute.
constexpr auto kSyncUserDataHandler = "sync_user_data_handler";
constexpr auto kGetValueByUserDataHandler = "get_value_by_user_data_handler";

constexpr auto kRealElementsSize = "real_elements_size";

// // For expander and pynative grad graph
// enum class InputType {
//   // Scala or Constant tensor, no need to grad
//   kConstant = 0,
//   // Weight parameter tensor
//   kParameter,
//   // Net input tensor
//   kInput,
//   // Other op output tensor
//   kOpOutput,
//   // Default
//   kUnkown,
// };

// MoveTo dst string
constexpr auto kToNpu = "Ascend";
constexpr auto kToCpu = "CPU";
constexpr auto kToRemote = "Remote";
constexpr auto kToDisk = "Disk";

// Return vec<filename, line number, function name>
COMMON_EXPORT std::vector<std::tuple<std::string, int, std::string>> GetPythonStack_();
COMMON_EXPORT std::string GetPythonStackStr_();

COMMON_EXPORT bool IsOneOfCustomAkgType(const std::string &name);
COMMON_EXPORT bool IsOneOfOperator(const std::string &name);
COMMON_EXPORT bool IsOneOfNotSupportedTransFormat(const std::string &format);
COMMON_EXPORT bool IsOneOfPosteriorOperator(const std::string &name);
COMMON_EXPORT bool IsOneOfCacheBlackList(const std::string &name);
COMMON_EXPORT bool IsOneOf3DFormat(const std::string &format);
COMMON_EXPORT bool IsOneOfNoPaddingFormat(const std::string &format);
COMMON_EXPORT bool IsOneOfDynamicShapeConstInputToAttrGPU(const std::string &name);
COMMON_EXPORT bool IsOneOfComputeDepend(const std::string &name);
COMMON_EXPORT bool IsOneOfHWSpecialFormat(const std::string &format);
COMMON_EXPORT bool IsOneOfFormat(const std::string &format);
COMMON_EXPORT bool IsOneOfDefaultFormat(const std::string &format);
COMMON_EXPORT bool IsOneOfServerFormatC04(const std::string &format);
COMMON_EXPORT bool IsOneOfDynRankNeedPadShape(const std::string &format);
COMMON_EXPORT bool IsOneOfUnsignedType(const TypeId &type_id);

COMMON_EXPORT size_t GetSystemMemorySize(const std::string &key);
COMMON_EXPORT size_t GetSystemFreeDiskSize(const std::string &path);

COMMON_EXPORT bool IsDisableGeKernel();

COMMON_EXPORT AnfNodeWeakPtrList SuccDeeperWithAttrGraph(const AnfNodePtr &node);

// copy once flag, and reset copy flag when step end
COMMON_EXPORT bool SkipOrResetCopyAction(bool need_reset = false);
// only sync once flag
COMMON_EXPORT bool SkipOrResetSyncAction(bool need_reset = false);
// Return vec<filename, line number, function name>
COMMON_EXPORT std::vector<std::tuple<std::string, int, std::string>> GetPythonStack();
COMMON_EXPORT std::string GetPythonStackStr();

// Return whether it is jit.
COMMON_EXPORT bool IsJit();
// Return whether it is compiling in jit compilation.
COMMON_EXPORT bool JitCompiling();
// Return whether it is compiling by jit pipeline.
COMMON_EXPORT bool JitPipelineCompiling();
// Return whether it is compiling by graph mode pipeline.
COMMON_EXPORT bool GraphPipelineCompiling();
// Return whether GraphPipelineCompiling was executed
COMMON_EXPORT bool IsGraphPipelineCompiled();
// Return whether it is running in jit compilation.
COMMON_EXPORT bool JitRunning();
// Return format mode.
COMMON_EXPORT std::string GetFormatMode();

// The map between kernel's output and input ref relationship.
// Key is the output index while the value is input index which will be used as the reference of output.
using OutputInputRefMap = std::map<size_t, size_t>;

using HashTableExportData = std::vector<std::shared_ptr<std::vector<char>>>;

static inline double GetCurrentUSec() {
  auto time_now = std::chrono::system_clock::now();
  auto tv_usec = std::chrono::duration_cast<std::chrono::microseconds>(time_now.time_since_epoch()).count();
  return static_cast<double>(tv_usec);
}

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__) && !defined(ENABLE_ANDROID)
#define PROF_START(stage)                                                                                   \
  uint64_t mstx_range_id_##stage = 0;                                                                       \
  double start_usec_##stage = mindspore::GetCurrentUSec();                                                  \
  if (mindspore::profiler::MstxImpl::GetInstance().IsEnable()) {                                            \
    MSTX_START(mstx_range_id_##stage, #stage, nullptr, mindspore::profiler::MSTX_DOMAIN_MODEL_PREPARATION); \
  }

#define PROF_END(stage)                                                                                        \
  do {                                                                                                         \
    if (mindspore::profiler::MstxImpl::GetInstance().IsEnable()) {                                             \
      MSTX_END(mstx_range_id_##stage, mindspore::profiler::MSTX_DOMAIN_MODEL_PREPARATION);                     \
    }                                                                                                          \
    double end_usec_##stage = mindspore::GetCurrentUSec();                                                     \
    std::ostringstream oss;                                                                                    \
    oss << "[PROF]" << #stage << " costs " << (end_usec_##stage - start_usec_##stage) / kBasicTimeTransferUnit \
        << " msec.";                                                                                           \
    const auto &value = common::GetConfigValue("MS_DEV_RUNTIME_CONF", "compile_statistics");                   \
    if ((value == "True") || (value == "true")) {                                                              \
      std::cout << oss.str() << std::endl;                                                                     \
    }                                                                                                          \
    MS_LOG(INFO) << oss.str();                                                                                 \
    MS_VLOG(VL_FLOW) << oss.str();                                                                             \
  } while (0)
#else
#define PROF_START(stage) double start_usec_##stage = mindspore::GetCurrentUSec()
#define PROF_END(stage)                                                                                        \
  do {                                                                                                         \
    double end_usec_##stage = mindspore::GetCurrentUSec();                                                     \
    std::ostringstream oss;                                                                                    \
    oss << "[PROF]" << #stage << " costs " << (end_usec_##stage - start_usec_##stage) / kBasicTimeTransferUnit \
        << " msec.";                                                                                           \
    const auto &value = common::GetConfigValue("MS_DEV_RUNTIME_CONF", "compile_statistics");                   \
    if ((value == "True") || (value == "true")) {                                                              \
      std::cout << oss.str() << std::endl;                                                                     \
    }                                                                                                          \
    MS_LOG(INFO) << oss.str();                                                                                 \
    MS_VLOG(VL_FLOW) << oss.str();                                                                             \
  } while (0)
#endif

#define PROF_MULTI_DEFINE(stage)       \
  do {                                 \
    static uint64_t total_##stage = 0; \
    static uint64_t count_##stage = 0; \
  } while (0)

#define PROF_LOCAL_DEFINE(stage) \
  do {                           \
    uint64_t total_##stage = 0;  \
    uint64_t count_##stage = 0;  \
  } while (0)

#define PROF_MULTI_START(stage) uint64_t start_usec_##stage = mindspore::GetCurrentUSec()

#define PROF_MULTI_END(stage)                                 \
  do {                                                        \
    ++count_##stage;                                          \
    uint64_t end_usec_##stage = mindspore::GetCurrentUSec();  \
    total_##stage += (end_usec_##stage - start_usec_##stage); \
  } while (0)

#define PROF_MULTI_PRINT(stage)                                                                             \
  do {                                                                                                      \
    MS_LOG(INFO) << #stage << " called " << count_##stage << " times, costs " << total_##stage << " usec."; \
  } while (0)

#define SET_FLAG(value, flag) ((value) = ((value) | (flag)))
#define TEST_FLAG(value, flag) (((value) & (flag)) == (flag))
#define CLEAR_FLAG(value, flag) ((value) = ((value) & (~(flag))))

#define _STRING_COMPILE_OPT(x) #x
#define STRING_COMPILE_OPT(x) _STRING_COMPILE_OPT(x)

static inline errno_t Memcpy(void *dest, size_t destMax, const void *src, size_t count) {
  if (count > SECUREC_MEM_MAX_LEN || destMax > SECUREC_MEM_MAX_LEN) {
    (void)memcpy(dest, src, count);
    return EOK;
  }
  return memcpy_s(dest, destMax, src, count);
}
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_UTILS_H_
