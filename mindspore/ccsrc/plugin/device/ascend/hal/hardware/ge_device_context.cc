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
#include <map>
#include <set>
#include "include/transform/graph_ir/types.h"
#include "include/transform/graph_ir/utils.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/common.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/scoped_long_running.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/hal/device/cpu_memory_manager.h"
#include "include/backend/debug/profiler/profiling.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/hal/hardware/ge_utils.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "runtime/config.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/common/utils/compile_cache_context.h"
#include "mindspore/core/utils/file_utils.h"
#include "toolchain/adx_datadump_server.h"
#include "plugin/device/ascend/hal/device/dump/ascend_dump.h"
#include "plugin/device/ascend/optimizer/ge_backend_optimization.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
constexpr auto kOpDebugConfigFile = "ge_op_debug_config.ini";
constexpr char kGeDumpMode[3][7] = {"all", "input", "output"};

bool IsDynamicShapeFuncGraph(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return false;
  }
  auto nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  return std::any_of(nodes.begin(), nodes.end(), [](const AnfNodePtr &node) {
    if (node == nullptr || common::AnfAlgo::IsCallNode(node)) {
      return false;
    }
    return common::AnfAlgo::IsDynamicShape(node);
  });
}
}  // namespace

bool GeDeviceContext::PartitionGraph(const FuncGraphPtr &func_graph) const {
  if (IsDynamicShapeFuncGraph(func_graph)) {
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
          continue;
        }
        if (!transform::DynamicShapeSupportCheck(node)) {
          all_support = false;
          common::AnfAlgo::SetNodeAttr(kAttrGraphSplitGroup, MakeValue<std::string>(kKernelGroup), node);
          continue;
        }
        if (!transform::SinkGraphCheck(node)) {
          all_support = false;
          common::AnfAlgo::SetNodeAttr(kAttrGraphSplitGroup, MakeValue<std::string>(kKernelGroup), node);
        }
      }
    }
    return all_support;
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  return context_ptr->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK);
}

RunMode GeDeviceContext::GetRunMode(const FuncGraphPtr &func_graph) const {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  // PyNative is only support ACL now on 910B.
  if (context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    auto enable_ge = common::GetEnv("MS_PYNATIVE_GE");
    return enable_ge == "1" ? RunMode::kGraphMode : RunMode::kKernelMode;
  }
  if (common::GetEnv("GRAPH_OP_RUN") == "1") {
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
  // set overflow mode in ascend910b
  const auto &soc_version = ms_context->ascend_soc_version();
  if (soc_version == "ascend910b") {
    bool is_infnan = (common::GetEnv("MS_ASCEND_CHECK_OVERFLOW_MODE") == "INFNAN_MODE");
    if (is_infnan) {
      auto mode = aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_INFNAN;
      auto ret = aclrtSetDeviceSatMode(mode);
      if (ret != ACL_SUCCESS) {
        MS_LOG(EXCEPTION) << "aclrtSetDeviceSatMode failed";
      }
    }
  }
  MS_EXCEPTION_IF_NULL(device_res_manager_);
  device_res_manager_->Initialize();

  // set MS_CTX_ENABLE_GE_HETEROGENOUS true according to  heterogeneous mode
  int32_t is_heterogenous = 0;
  (void)rtGetIsHeterogenous(&is_heterogenous);
  ms_context->set_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS, is_heterogenous == 1);
  InitGe(ms_context);

  if (IsEnableRefMode()) {
    MS_EXCEPTION_IF_NULL(GetKernelExecutor(false));
    GetKernelExecutor(false)->Initialize();
    // DynamicKernelExecutor and KernenlExecutor should be equal for GE
    MS_EXCEPTION_IF_CHECK_FAIL(GetKernelExecutor(true) == GetKernelExecutor(false),
                               "GE dynamic KernelExecutor and KernenlExecutor is not Equal.");
    MS_EXCEPTION_IF_NULL(GetKernelExecutor(true));
    GetKernelExecutor(true)->Initialize();
  }

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
  (void)FinalizeGe(ms_context);
  if (hccl::HcclAdapter::GetInstance().Inited()) {
    (void)hccl::HcclAdapter::GetInstance().FinalizeHccl();
  }
  if (deprecated_interface_ != nullptr) {
    (void)deprecated_interface_->CloseTsd(MsContext::GetInstance(), true);
  }
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
  inst_context->increase_param<uint32_t>(MS_CTX_GE_REF);
  MS_LOG(INFO) << "Init ge successful, ge reference = " << inst_context->get_param<uint32_t>(MS_CTX_GE_REF) << ".";
  return;
}

void UseOpDebugConfig(std::map<std::string, std::string> *ge_options) {
  auto op_debug_config = common::GetEnv("MS_COMPILER_OP_DEBUG_CONFIG");
  if (!op_debug_config.empty()) {
    auto config_path = Common::GetCompilerCachePath();
    DIR *dir = opendir(config_path.c_str());
    if (dir == nullptr) {
      auto ret = mkdir(config_path.c_str(), S_IRWXU);
      if (ret != 0) {
        MS_LOG(INFO) << "kernel dir: " << config_path << "not exist";
        return;
      }
    }
    auto ge_op_debug_config_file = config_path + kOpDebugConfigFile;
    if (ge_op_debug_config_file.size() > PATH_MAX) {
      MS_LOG(WARNING) << "File path length should be smaller than " << PATH_MAX << ", but got "
                      << ge_op_debug_config_file;
      return;
    }
    (*ge_options)["op_debug_config"] = ge_op_debug_config_file;
    std::string ge_op_debug_config = "op_debug_config = " + op_debug_config;
    std::ofstream file_write;
    file_write.open(ge_op_debug_config_file, std::ios::out | std::ios::trunc);
    if (!file_write.is_open()) {
      MS_LOG(WARNING) << "Create ge op debug config file failed. [" << ge_op_debug_config_file << "]";
      return;
    }
    file_write << ge_op_debug_config << std::endl;
    file_write.close();
    MS_LOG(INFO) << "Use MS_COMPILER_OP_DEBUG_CONFIG:" << ge_op_debug_config;
  }
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
  auto topo_sorting_env = common::GetEnv("GE_TOPO_SORTING_MODE");
  MS_LOG(INFO) << "GE topo sorting mode is: " << topo_sorting_env;
  if (topo_sorting_env == "bfs") {
    topo_sorting_mode = "0";
  } else if (topo_sorting_env == "dfs") {
    topo_sorting_mode = "1";
  } else if (topo_sorting_env == "dfs_postorder" ||
             ms_context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    topo_sorting_mode = "2";
  }

  (*ge_options)["ge.topoSortingMode"] = topo_sorting_mode;
  (*ge_options)["ge.exec.memoryOptimizationPolicy"] = "MemoryPriority";
  MS_LOG(INFO) << "Set GE topo mode to memory-priority.";

  auto ge_use_static_memory = common::GetEnv("GE_USE_STATIC_MEMORY");
  if (ge_use_static_memory.empty()) {
    (*ge_options)["ge.exec.staticMemoryPolicy"] = "2";
    MS_LOG(INFO) << "Set staticMemoryPolicy to default mode.";
  }

  if (ms_context_ptr->get_param<std::string>(MS_CTX_ENABLE_JIT_COMPILE) != "") {
    (*ge_options)["ge.jit_compile"] = ms_context_ptr->get_param<std::string>(MS_CTX_ENABLE_JIT_COMPILE);
    MS_LOG(INFO) << "Set jit_compile " << ms_context_ptr->get_param<std::string>(MS_CTX_ENABLE_JIT_COMPILE) << ".";
  } else {
    (*ge_options)["ge.jit_compile"] = "2";
    MS_LOG(INFO) << "The default value of jit_compile is set to 2.";
  }

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

  (*ge_options)["device_id"] = "0";

  (*ge_options)["ge.exec.formatMode"] = "0";
  if (common::GetEnv("MS_ENABLE_FORMAT_MODE") == "1") {
    (*ge_options)["ge.exec.formatMode"] = "1";
  }

  auto profiler_manager = profiler::ProfilerManager::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_manager);
  (*ge_options)["ge.exec.profilingMode"] = std::to_string(static_cast<int>(profiler_manager->GetProfilingEnableFlag()));
  if (profiler_manager->GetProfilingEnableFlag()) {
    (*ge_options)["ge.exec.profilingOptions"] = profiler_manager->GetProfilingOptions();
  }

  (*ge_options)["rank_table_file"] = "";
  (*ge_options)["graphType"] = "1";

  SetDisableReuseMemoryFlag(ge_options);
  SetHcclOptions(ms_context_ptr, ge_options);
  SetDumpOptions(ge_options);

  auto env_job_id = common::GetEnv("JOB_ID");
  if (!env_job_id.empty()) {
    (*ge_options)["ge.exec.jobId"] = env_job_id;
  } else {
    (*ge_options)["ge.exec.jobId"] = "0";
    MS_LOG(INFO) << "JOB_ID is not set in ENV. Now set to default value 0";
  }

  if (CompileCacheEnable()) {
    auto ge_cache_path = Common::GetCompilerCachePath() + kGeCache;
    (void)FileUtils::CreateNotExistDirs(ge_cache_path, true);
    (*ge_options)[kGeGraphCompilerCacheDir] = ge_cache_path;
    MS_LOG(INFO) << "Use GE graph compile cache, GE graph compile cache dir:" << ge_cache_path;
  }

  auto proto_lib_path = common::GetEnv("OPTION_PROTO_LIB_PATH");
  if (!proto_lib_path.empty()) {
    char real_path[PATH_MAX] = {0};
    if (realpath(proto_lib_path.c_str(), real_path)) {
      proto_lib_path = real_path;
      (*ge_options)["ge.opsProtoLibPath"] = proto_lib_path;
    }
  } else {
    MS_LOG(INFO) << "Set proto lib path failed!";
  }

  SetAscendConfig(ms_context_ptr, ge_options);

  auto op_debug_level = common::GetEnv("MS_COMPILER_OP_LEVEL");
  if (!op_debug_level.empty()) {
    (*ge_options)["ge.opDebugLevel"] = op_debug_level;
    MS_LOG(INFO) << "Use MS_COMPILER_OP_LEVEL, op debug level:" << op_debug_level;
  }

  // Disable the global variable acc, only enable it while adding training graph in pipeline
  (*ge_options)["ge.exec.variable_acc"] = "0";

  // ge heterogeneous mode
  if (ms_context_ptr->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
    (*ge_options)["ge.socVersion"] = "Ascend310P3";
  }

  // enable overflow detection
  (*ge_options)["ge.exec.overflow"] = "1";
  // enable deterministic
  (*ge_options)[::ge::DETERMINISTIC] = ms_context_ptr->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON" ? "1" : "0";
  UseOpDebugConfig(ge_options);
}

void GeDeviceContext::SetDumpOptions(std::map<std::string, std::string> *ge_options) const {
  MS_EXCEPTION_IF_NULL(ge_options);
  // set up dump options
  auto &dump_parser = DumpJsonParser::GetInstance();
  dump_parser.Parse();
  if (dump_parser.async_dump_enabled()) {
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

void GeDeviceContext::SetDisableReuseMemoryFlag(std::map<std::string, std::string> *ge_options) const {
  MS_EXCEPTION_IF_NULL(ge_options);
  auto env_disable_reuse_memory = common::GetEnv("DISABLE_REUSE_MEMORY");
  if (!env_disable_reuse_memory.empty()) {
    (*ge_options)["ge.exec.disableReuseMemory"] = env_disable_reuse_memory;
  } else {
    (*ge_options)["ge.exec.disableReuseMemory"] = "0";
    MS_LOG(INFO) << "DISABLE_REUSE_MEMORY is not set in ENV. Now set to default value 0";
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
  auto env_rank_id = common::GetEnv("RANK_ID");
  auto env_device_id = std::to_string(inst_context->get_param<uint32_t>(MS_CTX_DEVICE_ID));
  auto env_cluster_info = common::GetEnv("HELP_CLUSTER");
  auto enable_hccl = inst_context->get_param<bool>(MS_CTX_ENABLE_HCCL);
  if (enable_hccl &&
      (!(env_table_file.empty() || env_rank_id.empty()) || !(env_cluster_info.empty() || env_rank_id.empty()) ||
       hccl::HcclAdapter::GetInstance().UseHcclCM())) {
    MS_LOG(INFO) << "Initialize Ge for distribute parameter";
    if (!env_table_file.empty()) {
      MS_LOG(INFO) << "Use hccl, make sure hccl lib is set in OPTION_EXEC_EXTERN_PLUGIN_PATH.";
      (*ge_options)["ge.exec.rankTableFile"] = env_table_file;
    } else if (hccl::HcclAdapter::GetInstance().UseHcclCM()) {
      hccl::HcclAdapter::AddCMEnvToHcclOption(ge_options);
    }
    auto env_hccl_flag = common::GetEnv("HCCL_FLAG");
    if (!env_hccl_flag.empty()) {
      (*ge_options)["ge.exec.hcclFlag"] = env_hccl_flag;
    }
    (*ge_options)["ge.exec.isUseHcom"] = "1";
    (*ge_options)["ge.exec.deviceId"] = env_device_id;
    (*ge_options)["ge.exec.rankId"] = env_rank_id;
    (*ge_options)["ge.exec.podName"] = env_rank_id;
  } else {
    // device id is still needed for non-distribute case
    (*ge_options)["ge.exec.deviceId"] = env_device_id;
    MS_LOG(INFO) << "No hccl mode. "
                 << "If use hccl, make sure [RANK_TABLE_FILE,RANK_ID,DEVICE_ID] all be set in ENV.";
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
    (void)Adx::AdxRegDumpProcessCallBack(mindspore::ascend::DumpDataCallBack);
  }
  if (AdxDataDumpServerInit() != 0) {
    MS_LOG(EXCEPTION) << "Adx data dump server init failed";
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
  if (AdxDataDumpServerUnInit() != 0) {
    MS_LOG(EXCEPTION) << "Adx data dump server init failed";
  }
}

DeprecatedInterface *GeDeviceContext::GetDeprecatedInterface() {
  // need lock when multi-threads
  if (deprecated_interface_ == nullptr) {
    deprecated_interface_ = std::make_unique<AscendDeprecatedInterface>(this);
  }
  return deprecated_interface_.get();
}

constexpr auto kGeDevice = "GE";
MS_REGISTER_DEVICE(kGeDevice, GeDeviceContext);
#ifdef ASCEND_910B
MS_REGISTER_DEVICE(kAscendDevice, GeDeviceContext);
#endif
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
