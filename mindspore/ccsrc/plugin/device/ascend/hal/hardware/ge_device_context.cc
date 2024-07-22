/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/hardware/ge_device_context.h"
#include <tuple>
#include <algorithm>
#include <sstream>
#include <map>
#include <set>
#include "include/transform/graph_ir/types.h"
#include "include/transform/graph_ir/utils.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/common.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/scoped_long_running.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "plugin/device/ascend/hal/hardware/ge_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/hal/device/cpu_memory_manager.h"
#include "include/backend/debug/profiler/profiling.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/common/utils/compile_cache_context.h"
#include "mindspore/core/utils/file_utils.h"
#include "plugin/device/ascend/hal/device/dump/ascend_dump.h"
#include "plugin/device/ascend/optimizer/ge_backend_optimization.h"
#include "transform/symbol/acl_base_symbol.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"
#include "transform/symbol/acl_compiler_symbol.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
constexpr auto kOpDebugConfigFile = "ge_op_debug_config.ini";
constexpr char kGeDumpMode[3][7] = {"all", "input", "output"};
constexpr auto kSaturationMode = "Saturation";
constexpr auto kINFNANMode = "INFNAN";

bool IsDynamicShapeFuncGraph(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return false;
  }
  auto nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  return std::any_of(nodes.begin(), nodes.end(), [](const AnfNodePtr &node) {
    if (node == nullptr || common::AnfAlgo::IsCallNode(node)) {
      return false;
    }
    return common::AnfAlgo::IsDynamicShape(node) || common::AnfAlgo::IsDynamicSequence(node) ||
           common::AnfAlgo::IsNodeMutableScalar(node);
  });
}

void SetAclOpDebugOption(const std::shared_ptr<MsContext> &ms_context) {
  MS_EXCEPTION_IF_NULL(ms_context);
  auto op_debug_option = ms_context->get_param<std::string>(MS_CTX_OP_DEBUG_OPTION);
  if (op_debug_option == "oom") {
    auto ret = CALL_ASCEND_API(aclSetCompileopt, aclCompileOpt::ACL_OP_DEBUG_OPTION, op_debug_option.c_str());
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Acl set op debug option: " << op_debug_option << " failed! Error flag is " << ret;
    }
  }
}
}  // namespace

bool GeDeviceContext::PartitionGraph(const FuncGraphPtr &func_graph) const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (IsDynamicShapeFuncGraph(func_graph)) {
    // dynamic shape default kernel be kernel before ge support
    if (GetRunMode(func_graph) == RunMode::kKernelMode) {
      return true;
    }
    opt::GEDynamicUnifyMindIR(func_graph);
    bool all_support = true;
    auto mng = func_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    const auto &sub_graphs = mng->func_graphs();
    for (const auto &sub_graph : sub_graphs) {
      if (sub_graph == nullptr) {
        continue;
      }
      auto nodes = TopoSort(sub_graph->get_return());
      for (const auto &node : nodes) {
        if (!node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
          continue;
        }
        if (GetCNodeTarget(node) != kAscendDevice) {
          all_support = false;
          continue;
        }
        if (GetCNodePrimitive(node) == nullptr) {
          continue;
        }
        if (!transform::ConvertCheck(node)) {
          all_support = false;
          common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue<std::string>(kCPUDevice), node);
          MS_LOG(DEBUG) << node->fullname_with_scope() << " can not find adpt, run on CPU";
          continue;
        }
        if (!transform::DynamicShapeSupportCheck(node)) {
          all_support = false;
          common::AnfAlgo::SetNodeAttr(kAttrGraphSplitGroup, MakeValue<std::string>(kKernelGroup), node);
          MS_LOG(DEBUG) << node->fullname_with_scope() << " not support dynamic shape, will run in KernelGraph";
          continue;
        }
        if (!transform::SinkGraphCheck(node)) {
          all_support = false;
          common::AnfAlgo::SetNodeAttr(kAttrGraphSplitGroup, MakeValue<std::string>(kKernelGroup), node);
          MS_LOG(DEBUG) << node->fullname_with_scope() << " have attrs is not ValueNode, will run in KernelGraph";
        }
      }
    }
    if (!all_support) {
      context_ptr->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, false);
    }
  }
  return context_ptr->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK);
}

RunMode GeDeviceContext::GetRunMode(const FuncGraphPtr &func_graph) const {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (IsDynamicShapeFuncGraph(func_graph)) {
    if (context->get_param<std::string>(MS_CTX_JIT_LEVEL) == "O2" &&
        context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
      MS_LOG(INFO) << "set dynamic shape RunMode::kGraphMode";
      return RunMode::kGraphMode;
    }
    MS_LOG(INFO) << "dynamic shape default RunMode::kKernelMode";
    // Dynamic shape runs in kbk mode, not support ge graph sink mode.
    auto set_ctx = [&context](bool task_sink, bool is_multi_graph_sink, bool enable_loop_sink) {
      context->set_param<bool>(MS_CTX_ENABLE_TASK_SINK, task_sink);
      context->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, is_multi_graph_sink);
      context->set_param<bool>(MS_CTX_ENABLE_LOOP_SINK, enable_loop_sink);
    };
    set_ctx(false, false, false);
    return RunMode::kKernelMode;
  }

  if (context->IsKByKExecutorMode()) {
    MS_LOG(INFO) << "RunMode::kKernelMode";
    return RunMode::kKernelMode;
  } else {
    MS_LOG(INFO) << "RunMode::kGraphMode";
    return RunMode::kGraphMode;
  }
}

void GeDeviceContext::Initialize() {
  GilReleaseWithCheck gil_release;
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (initialized_) {
    return;
  }

  MS_LOG(DEBUG) << "Start initialize...";

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  // set overflow mode
  const auto &soc_version = ms_context->ascend_soc_version();
  if (soc_version == "ascend910b" || soc_version == "ascend910c") {
    bool is_sat = (common::GetEnv("MS_ASCEND_CHECK_OVERFLOW_MODE") == "SATURATION_MODE");
    auto mode = (is_sat) ? aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_SATURATION
                         : aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_INFNAN;
    auto overflow_mode = (is_sat) ? kSaturationMode : kINFNANMode;
    MS_LOG(INFO) << "The current overflow detection mode is " << overflow_mode << ".";
    auto ret = CALL_ASCEND_API(aclrtSetDeviceSatMode, mode);
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Set " << overflow_mode << " mode failed.";
    }
  }

  MS_EXCEPTION_IF_NULL(device_res_manager_);
  device_res_manager_->Initialize();

  // set MS_CTX_ENABLE_GE_HETEROGENOUS true according to  heterogeneous mode
  ms_context->set_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS, false);
  InitGe(ms_context);

  MS_EXCEPTION_IF_NULL(GetKernelExecutor(false));
  GetKernelExecutor(false)->Initialize();
  // DynamicKernelExecutor and KernenlExecutor should be equal for GE
  MS_EXCEPTION_IF_CHECK_FAIL(GetKernelExecutor(true) == GetKernelExecutor(false),
                             "GE dynamic KernelExecutor and KernenlExecutor is not Equal.");
  MS_EXCEPTION_IF_NULL(GetKernelExecutor(true));
  GetKernelExecutor(true)->Initialize();

  InitDump();
  if (ms_context->EnableAoeOnline()) {
    transform::InitializeAoeUtil();
  }
  if (ms_context->EnableAoeOffline()) {
    transform::EnableAoeOffline();
  }
  initialized_ = true;
}

void GeDeviceContext::Destroy() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->EnableAoeOnline()) {
    transform::DestroyAoeUtil();
  }
  FinalizeDump();
  // Device resource manager must be destroyed before 'FinalizeGe' unless some runtime APIs will throw exception.
  device_res_manager_->Destroy();
  (void)FinalizeGe(ms_context);
  if (hccl::HcclAdapter::GetInstance().Inited()) {
    (void)hccl::HcclAdapter::GetInstance().FinalizeHccl();
  }
  if (deprecated_interface_ != nullptr) {
    (void)deprecated_interface_->CloseTsd(MsContext::GetInstance(), true);
  }
  initialized_ = false;
}

void GeDeviceContext::InitGe(const std::shared_ptr<MsContext> &inst_context) {
  MS_EXCEPTION_IF_NULL(inst_context);

  if (inst_context->get_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT)) {
    return;
  }

  if (static_cast<bool>(inst_context->get_param<uint32_t>(MS_CTX_GE_REF))) {
    inst_context->increase_param<uint32_t>(MS_CTX_GE_REF);
    return;
  }

  std::map<std::string, std::string> ge_options;
  GetGeOptions(inst_context, &ge_options);
  {
    // Release GIL before calling into (potentially long-running) C++ code
    GilReleaseWithCheck gil_release;
    if (::ge::GEInitialize(ge_options) != ::ge::GRAPH_SUCCESS) {
      MS_LOG(EXCEPTION) << "Initialize GE failed!";
    }
  }
  // should be called after ge initialize.
  SetAclOpDebugOption(inst_context);

  GeDeviceResManager::CreateSessionAndGraphRunner();
  auto graph_runner = transform::GetGraphRunner();
  MS_EXCEPTION_IF_NULL(graph_runner);
  if (IsEnableRefMode()) {
    transform::Status ret = transform::RegisterExternalAllocator(
      graph_runner, dynamic_cast<GeDeviceResManager *>(device_res_manager_.get())->GetStream(),
      dynamic_cast<GeDeviceResManager *>(device_res_manager_.get())->GetAllocator());
    if (ret != transform::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "RegisterExternalAllocator failed";
    }
    MS_LOG(INFO) << "Create session and graphrunner successful.";
  }

  inst_context->increase_param<uint32_t>(MS_CTX_GE_REF);
  MS_LOG(INFO) << "Init ge successful, ge reference = " << inst_context->get_param<uint32_t>(MS_CTX_GE_REF) << ".";
  return;
}

// ge.exec.allow_hf32 default value is "10"(enable Conv, disable Matmul) set by CANN
void SetAscendHF32Config(const std::shared_ptr<MsContext> &ms_context_ptr,
                         std::map<std::string, std::string> *ge_options) {
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  std::string allow_matmul_hf32 = ms_context_ptr->get_param<std::string>(MS_CTX_MATMUL_ALLOW_HF32);
  std::string allow_conv_hf32 = ms_context_ptr->get_param<std::string>(MS_CTX_CONV_ALLOW_HF32);
  if (allow_matmul_hf32.empty() && allow_conv_hf32.empty()) {
    MS_LOG(INFO) << "The default value of allow_matmul_hf32 and allow_conv_hf32 are set by CANN.";
  } else if (allow_matmul_hf32.empty() && !allow_conv_hf32.empty()) {
    (*ge_options)["ge.exec.allow_hf32"] = allow_conv_hf32 + std::string("0");
  } else if (!allow_matmul_hf32.empty() && allow_conv_hf32.empty()) {
    (*ge_options)["ge.exec.allow_hf32"] = std::string("1") + allow_matmul_hf32;
  } else {
    (*ge_options)["ge.exec.allow_hf32"] = allow_conv_hf32 + allow_matmul_hf32;
  }

  MS_LOG(INFO) << "allow_matmul_hf32: " << allow_matmul_hf32 << ", allow_conv_hf32: " << allow_conv_hf32;
}

void GeDeviceContext::SetAscendConfig(const std::shared_ptr<MsContext> &ms_context_ptr,
                                      std::map<std::string, std::string> *ge_options) const {
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  MS_EXCEPTION_IF_NULL(ge_options);

  std::string topo_sorting_mode = "0";
  if (ms_context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    topo_sorting_mode = "2";
  }
  (*ge_options)["ge.topoSortingMode"] = topo_sorting_mode;
  // disable RemoveSameConstPass, it will be caused the communication failed on multi-card.
  (*ge_options)["ge.disableOptimizations"] = "RemoveSameConstPass";

  (*ge_options)["ge.exec.memoryOptimizationPolicy"] = "MemoryPriority";
  MS_LOG(INFO) << "Set GE topo mode to memory-priority.";

  (*ge_options)["ge.exec.staticMemoryPolicy"] = "2";
  MS_LOG(INFO) << "Set staticMemoryPolicy to default mode 2.";

  if (ms_context_ptr->get_param<std::string>(MS_CTX_ENABLE_JIT_COMPILE) != "") {
    (*ge_options)["ge.jit_compile"] = ms_context_ptr->get_param<std::string>(MS_CTX_ENABLE_JIT_COMPILE);
    MS_LOG(INFO) << "Set jit_compile " << ms_context_ptr->get_param<std::string>(MS_CTX_ENABLE_JIT_COMPILE) << ".";
  } else {
    (*ge_options)["ge.jit_compile"] = "2";
    MS_LOG(INFO) << "The default value of jit_compile is set to 2.";
  }

  auto ge_exception_dump = ms_context_ptr->get_param<std::string>(MS_CTX_ENABLE_EXCEPTION_DUMP);
  (*ge_options)["ge.exec.enable_exception_dump"] = ge_exception_dump;

  SetAscendHF32Config(ms_context_ptr, ge_options);

  if (ms_context_ptr->get_param<std::string>(MS_CTX_OP_PRECISION_MODE) != "") {
    (*ge_options)["ge.exec.op_precision_mode"] = ms_context_ptr->get_param<std::string>(MS_CTX_OP_PRECISION_MODE);
    MS_LOG(INFO) << "Set op_precision_mode " << ms_context_ptr->get_param<std::string>(MS_CTX_OP_PRECISION_MODE) << ".";
  }
}

void GeDeviceContext::GetGeOptions(const std::shared_ptr<MsContext> &ms_context_ptr,
                                   std::map<std::string, std::string> *ge_options) {
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  MS_EXCEPTION_IF_NULL(ge_options);

  SetHcclOptions(ms_context_ptr, ge_options);
  SetDumpOptions(ge_options);

  (*ge_options)["ge.exec.jobId"] = "0";
  MS_LOG(INFO) << "Set ge.exec.jobId to default value 0";

  auto proto_lib_path = common::GetEnv("OPTION_PROTO_LIB_PATH");
  if (!proto_lib_path.empty()) {
    char real_path[PATH_MAX] = {0};
    if (realpath(proto_lib_path.c_str(), real_path)) {
      proto_lib_path = real_path;
      (*ge_options)["ge.opsProtoLibPath"] = proto_lib_path;
    }
  } else {
    MS_LOG(INFO) << "Got empty proto lib path, cannot set ge.opsProtoLibPath.";
  }

  SetAscendConfig(ms_context_ptr, ge_options);

  auto op_debug_level = common::GetEnv("MS_COMPILER_OP_LEVEL");
  if (!op_debug_level.empty()) {
    (*ge_options)["ge.opDebugLevel"] = op_debug_level;
    MS_LOG(INFO) << "Use MS_COMPILER_OP_LEVEL, op debug level:" << op_debug_level;
  }

  // Enable the global variable acc may cause accuracy problems in train+eval.
  (*ge_options)["ge.exec.variable_acc"] = "0";

  // ge heterogeneous mode
  if (ms_context_ptr->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
    (*ge_options)["ge.socVersion"] = "Ascend310P3";
  }

  // enable overflow detection
  (*ge_options)["ge.exec.overflow"] = "1";
  // enable deterministic
  (*ge_options)[::ge::DETERMINISTIC] = ms_context_ptr->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON" ? "1" : "0";

  SetPassthroughGeOptions(true, ge_options);
}

void GeDeviceContext::SetDumpOptions(std::map<std::string, std::string> *ge_options) const {
  MS_EXCEPTION_IF_NULL(ge_options);
  // set up dump options
  auto &dump_parser = DumpJsonParser::GetInstance();
  dump_parser.Parse();
  if (dump_parser.async_dump_enabled() && !dump_parser.IsAclDump()) {
    (*ge_options)["ge.exec.enableDump"] = std::to_string(static_cast<int>(dump_parser.async_dump_enabled()));
    auto dump_path = FileUtils::CreateNotExistDirs(dump_parser.path());
    if (!dump_path.has_value()) {
      MS_LOG(EXCEPTION) << "Invalid dump path: " << dump_parser.path();
    }
    (*ge_options)["ge.exec.dumpPath"] = dump_path.value();
    // Parse() make sure that input_output is less than 3.
    (*ge_options)["ge.exec.dumpMode"] = kGeDumpMode[dump_parser.input_output()];
    // DumpStep is set to "all" by default
    if (dump_parser.iteration_string() != "all") {
      (*ge_options)["ge.exec.dumpStep"] = dump_parser.iteration_string();
    }
    if (dump_parser.dump_mode() == 1) {
      (*ge_options)["ge.exec.dumpLayer"] = dump_parser.dump_layer();
      MS_LOG(INFO) << "Set dumplayer to: " << (*ge_options)["ge.exec.dumpLayer"];
    }
    if (dump_parser.op_debug_mode() > 0) {
      (*ge_options)["ge.exec.enableDump"] = "0";
      (*ge_options)["ge.exec.enableDumpDebug"] = "1";
      switch (dump_parser.op_debug_mode()) {
        case 1:
          (*ge_options)["ge.exec.dumpDebugMode"] = "aicore_overflow";
          break;
        case 2:
          (*ge_options)["ge.exec.dumpDebugMode"] = "atomic_overflow";
          break;
        case 3:
          (*ge_options)["ge.exec.dumpDebugMode"] = "all";
          break;
        default:
          break;
      }
    }

    MS_LOG(INFO) << "The enable dump state is " << (*ge_options)["ge.exec.enableDump"] << ", save dump path is "
                 << (*ge_options)["ge.exec.dumpPath"] << ", dump mode is " << kGeDumpMode[dump_parser.input_output()]
                 << ", dump step is " << dump_parser.iteration_string() << ".";
  }
}

void GeDeviceContext::SetHcclOptions(const std::shared_ptr<MsContext> &inst_context,
                                     std::map<std::string, std::string> *ge_options) {
  MS_EXCEPTION_IF_NULL(inst_context);
  MS_EXCEPTION_IF_NULL(ge_options);
  auto env_table_file = common::GetEnv("MINDSPORE_HCCL_CONFIG_PATH");
  if (env_table_file.empty()) {
    env_table_file = common::GetEnv("RANK_TABLE_FILE");
  }
  auto simulation_level = common::GetEnv(kSimulationLevel);
  if (!simulation_level.empty()) {
    env_table_file = "";
  }
  auto env_rank_id = common::GetEnv("RANK_ID");
  auto env_device_id = std::to_string(inst_context->get_param<uint32_t>(MS_CTX_DEVICE_ID));
  auto env_cluster_info = common::GetEnv("HELP_CLUSTER");
  auto enable_hccl = inst_context->get_param<bool>(MS_CTX_ENABLE_HCCL);
  auto escluster_config_path = common::GetEnv("ESCLUSTER_CONFIG_PATH");

  MS_LOG(INFO) << "Values for hccl options: env_table_file[" << env_table_file << "], simulation_level["
               << simulation_level << "], env_rank_id[" << env_rank_id << "], env_device_id[" << env_device_id
               << "], enable_hccl[" << enable_hccl << "], UseDynamicCluster[" << common::UseDynamicCluster() << "].";
  if (enable_hccl &&
      (!(env_table_file.empty() || env_rank_id.empty()) || !(env_cluster_info.empty() || env_rank_id.empty()) ||
       hccl::HcclAdapter::GetInstance().UseHcclCM()) &&
      !(common::UseDynamicCluster() && !env_table_file.empty())) {
    MS_LOG(INFO) << "Initialize Ge for distribute parameter";
    if (!env_table_file.empty()) {
      MS_LOG(INFO) << "Use hccl, make sure hccl lib is set in OPTION_EXEC_EXTERN_PLUGIN_PATH.";
      (*ge_options)["ge.exec.rankTableFile"] = env_table_file;
    } else if (hccl::HcclAdapter::GetInstance().UseHcclCM()) {
      hccl::HcclAdapter::AddCMEnvToHcclOption(ge_options);
    }

    (*ge_options)["ge.exec.isUseHcom"] = "1";
    (*ge_options)["ge.exec.deviceId"] = env_device_id;
    (*ge_options)["ge.exec.rankId"] = env_rank_id;
    (*ge_options)["ge.exec.podName"] = env_rank_id;
  } else if (!escluster_config_path.empty()) {
    (*ge_options)["ge.exec.deviceId"] = env_device_id;
    (*ge_options)["ge.exec.rankTableFile"] = env_table_file;
    (*ge_options)["ge.exec.rankId"] = env_rank_id;
  } else {
    // device id is still needed for non-distribute case
    (*ge_options)["ge.exec.deviceId"] = env_device_id;
    MS_LOG(INFO) << "No hccl mode. If use hccl, make sure [RANK_TABLE_FILE,RANK_ID,DEVICE_ID] all be set in ENV.";
  }
}

bool GeDeviceContext::FinalizeGe(const std::shared_ptr<MsContext> &inst_context) {
  MS_EXCEPTION_IF_NULL(inst_context);
  if (inst_context->get_param<uint32_t>(MS_CTX_GE_REF) == 0) {
    return true;
  }
  inst_context->decrease_param<uint32_t>(MS_CTX_GE_REF);
  if (inst_context->get_param<uint32_t>(MS_CTX_GE_REF) == 0) {
    inst_context->set_param<uint32_t>(MS_CTX_GE_REF, 0);
    try {
      transform::ClearGeSessionAndRunner();
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Error occurred when deleting GE graph runner and session fail. Error: " << e.what();
    } catch (...) {
      std::string exName(abi::__cxa_current_exception_type()->name());
      MS_LOG(ERROR) << "Error occurred when deleting GE graph runner and session fail. Exception name: " << exName;
    }
    if (::ge::GEFinalize() != ::ge::GRAPH_SUCCESS) {
      MS_LOG(WARNING) << "Finalize GE failed!";
    }
    inst_context->set_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT, false);
  } else {
    MS_LOG(INFO) << "Ge is used, no need to finalize, tsd reference = "
                 << inst_context->get_param<uint32_t>(MS_CTX_GE_REF) << ".";
  }
  return true;
}

void GeDeviceContext::InitDump() const {
  auto &dump_parser = DumpJsonParser::GetInstance();
  dump_parser.Parse();
  if (!dump_parser.async_dump_enabled()) {
    return;
  }
  if (dump_parser.FileFormatIsNpy()) {
    if (dump_parser.IsCallbackRegistered()) {
      MS_LOG(INFO) << "DumpDataCallback already registered, no need to register again.";
      return;
    }
    (void)acldumpRegCallback(mindspore::ascend::DumpDataCallBack, 0);
    dump_parser.SetCallbackRegistered();
  }
}

void GeDeviceContext::FinalizeDump() const {
  auto &dump_parser = DumpJsonParser::GetInstance();
  dump_parser.Parse();
  if (!dump_parser.async_dump_enabled()) {
    return;
  }
  if (dump_parser.FileFormatIsNpy() && dump_parser.IsTensorDump()) {
    mindspore::ascend::AscendAsyncDumpManager::GetInstance().WaitForWriteFileFinished();
  }
}

DeprecatedInterface *GeDeviceContext::GetDeprecatedInterface() {
  // need lock when multi-threads
  if (deprecated_interface_ == nullptr) {
    deprecated_interface_ = std::make_unique<AscendDeprecatedInterface>(this);
  }
  return deprecated_interface_.get();
}

uint32_t GeDeviceContext::GetDeviceCount() {
  uint32_t device_count = 0;
  auto ret = CALL_ASCEND_API(aclrtGetDeviceCount, &device_count);
  if (ret != ACL_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtGetDeviceCount, ret[" << static_cast<int>(ret) << "]";
  }
  return device_count;
}

std::string GeDeviceContext::GetDeviceName(uint32_t) {
  const char *name = CALL_ASCEND_API(aclrtGetSocName);
  std::string device_name = (name == nullptr) ? "" : name;
  return device_name;
}

AscendDeviceProperties GeDeviceContext::GetDeviceProperties(uint32_t) {
  AscendDeviceProperties device_properties;
  const char *name = CALL_ASCEND_API(aclrtGetSocName);
  device_properties.name = (name == nullptr) ? "" : name;

  size_t free_size{0}, total_size{0};
  auto ret = CALL_ASCEND_API(aclrtGetMemInfo, ACL_HBM_MEM, &free_size, &total_size);
  if (ret != ACL_SUCCESS) {
    MS_LOG(WARNING) << "Failed get memory info for current device. Error number: " << ret;
  }
  device_properties.total_memory = total_size;
  device_properties.free_memory = free_size;
  return device_properties;
}

MS_REGISTER_DEVICE(kAscendDevice, GeDeviceContext);
#ifdef WITH_BACKEND
namespace {
void SetContextSocVersion(MsContext *ctx) {
  const std::map<std::string, std::string> kAscendSocVersions = {
    {"Ascend910A", "ascend910"},    {"Ascend910B", "ascend910"},    {"Ascend910PremiumA", "ascend910"},
    {"Ascend910ProA", "ascend910"}, {"Ascend910ProB", "ascend910"}, {"Ascend910B1", "ascend910b"},
    {"Ascend910B2", "ascend910b"},  {"Ascend910B2C", "ascend910b"}, {"Ascend910B3", "ascend910b"},
    {"Ascend910B4", "ascend910b"},  {"Ascend910C1", "ascend910c"},  {"Ascend910C2", "ascend910c"},
    {"Ascend910C3", "ascend910c"},  {"Ascend910C4", "ascend910c"},  {"Ascend310P", "ascend310p"},
    {"Ascend310P3", "ascend310p"},  {"Ascend310B4", "ascend310b"},  {"Ascend310B1", "ascend310b"}};
  const char *soc_name_c = CALL_ASCEND_API(aclrtGetSocName);
  if (soc_name_c == nullptr) {
    MS_LOG(ERROR) << "Get soc name failed.";
    return;
  }
  std::string version(soc_name_c);
  MS_LOG(INFO) << "The soc version :" << version;
  ctx->set_ascend_soc_name(version);
  auto iter = kAscendSocVersions.find(version);
  if (iter == kAscendSocVersions.end()) {
    ctx->set_ascend_soc_version(version);
  } else {
    ctx->set_ascend_soc_version(iter->second);
  }
}
}  // namespace

MSCONTEXT_REGISTER_INIT_FUNC(kAscendDevice, [](MsContext *ctx) -> void {
  MS_EXCEPTION_IF_NULL(ctx);
  if (ctx->backend_policy() != "ge") {
    (void)ctx->set_backend_policy("ge");
  }
  // change some Environment Variables name
  auto format_mode = common::GetEnv("MS_ENABLE_FORMAT_MODE");
  if (!format_mode.empty()) {
    MS_LOG(WARNING)
      << "The Environment Variable MS_ENABLE_FORMAT_MODE will be discarded, please use MS_FORMAT_MODE instead.";
    common::SetEnv("MS_FORMAT_MODE", format_mode.c_str());
  }

  transform::LoadAscendApiSymbols();
  SetContextSocVersion(ctx);
});
#endif

// Register functions to _c_expression so python hal module could call Ascend device interfaces.
void PybindAscendStatelessFunc(py::module *m) {
  MS_EXCEPTION_IF_NULL(m);
  (void)py::class_<AscendDeviceProperties>(*m, "AscendDeviceProperties")
    .def_readonly("name", &AscendDeviceProperties::name)
    .def_readonly("total_memory", &AscendDeviceProperties::total_memory)
    .def_readonly("free_memory", &AscendDeviceProperties::free_memory)
    .def("__repr__", [](const AscendDeviceProperties &p) {
      std::ostringstream s;
      s << "AscendDeviceProperties(name='" << p.name << "', total_memory=" << p.total_memory / (1024 * 1024)
        << "MB, free_memory=" << p.free_memory / (1024 * 1024) << "MB)";
      return s.str();
    });
  (void)m->def("ascend_get_device_count", &GeDeviceContext::GetDeviceCount, "Get Ascend device count.");
  (void)m->def("ascend_get_device_name", &GeDeviceContext::GetDeviceName,
               "Get Ascend device name of specified device id.");
  (void)m->def("ascend_get_device_properties", &GeDeviceContext::GetDeviceProperties,
               "Get Ascend device properties of specified device id.");
}
REGISTER_DEV_STATELESS_FUNC_CB(kAscendDevice, PybindAscendStatelessFunc);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
