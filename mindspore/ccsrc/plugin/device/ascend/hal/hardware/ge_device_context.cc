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
#include <utility>
#include <map>
#include <set>
#include "include/transform/graph_ir/types.h"
#include "include/transform/graph_ir/utils.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/scoped_long_running.h"
#include "abstract/abstract_value.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/hal/device/cpu_memory_manager.h"
#include "profiler/device/profiling.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/hal/hardware/ge_utils.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "runtime/config.h"
#include "runtime/dev.h"
#include "distributed/init.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
constexpr auto kMindsporeDumpConfig = "MINDSPORE_DUMP_CONFIG";
constexpr auto kOpDebugConfigFile = "ge_op_debug_config.ini";
constexpr char kGeDumpMode[3][7] = {"all", "input", "output"};
const std::set<std::string> kAscend910BVersions = {"Ascend910B1", "Ascend910B2", "Ascend910B3", "Ascend910B4"};
}  // namespace

bool GeDeviceContext::PartitionGraph(const FuncGraphPtr &func_graph) const { return true; }

RunMode GeDeviceContext::GetRunMode(const FuncGraphPtr &func_graph) const { return RunMode::kGraphMode; }

void GeDeviceContext::Initialize() {
  if (initialized_) {
    return;
  }

  MS_EXCEPTION_IF_NULL(device_res_manager_);
  device_res_manager_->Initialize();

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // set MS_CTX_ENABLE_GE_HETEROGENOUS true according to  heterogeneous mode
  int32_t is_heterogenous = 0;
  (void)rtGetIsHeterogenous(&is_heterogenous);
  ms_context->set_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS, is_heterogenous == 1);
  InitGe(ms_context);
  initialized_ = true;
}

void GeDeviceContext::Destroy() {
  (void)FinalizeGe(MsContext::GetInstance());
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

  (void)setenv("GE_TRAIN", IsGeTrain() ? "1" : "0", 1);
  std::map<std::string, std::string> ge_options;
  GetGeOptions(inst_context, &ge_options);
  {
    // Release GIL before calling into (potentially long-running) C++ code
    mindspore::ScopedLongRunning long_running;
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
    auto config_path = kernel::tbe::TbeUtils::GetOpDebugPath();
    DIR *dir = opendir(config_path.c_str());
    if (dir == nullptr) {
      auto ret = mkdir(config_path.c_str(), S_IRWXG | S_IRWXU);
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
void GeDeviceContext::SetAscendConfig(const std::shared_ptr<MsContext> &ms_context_ptr,
                                      std::map<std::string, std::string> *ge_options) const {
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  MS_EXCEPTION_IF_NULL(ge_options);
  if (ms_context_ptr->get_param<std::string>(MS_CTX_PRECISION_MODE) != "") {
    (*ge_options)["ge.exec.precision_mode"] = ms_context_ptr->get_param<std::string>(MS_CTX_PRECISION_MODE);
    MS_LOG(INFO) << "Set precision_mode " << ms_context_ptr->get_param<std::string>(MS_CTX_PRECISION_MODE) << ".";
  } else if (IsGeTrain()) {
    auto soc_version = device::ascend::GetSocVersion();
    if (kAscend910BVersions.count(soc_version) != 0) {
      MS_LOG(INFO) << "The default value of precision_mode is set by CANN. soc_version is " << soc_version;
    } else {
      (*ge_options)["ge.exec.precision_mode"] = "allow_fp32_to_fp16";
      MS_LOG(INFO) << "Set precision_mode allow_fp32_to_fp16. soc_version is " << soc_version;
    }
  } else {
    (*ge_options)["ge.exec.precision_mode"] = "force_fp16";
  }

  if (ms_context_ptr->get_param<std::string>(MS_CTX_ENABLE_JIT_COMPILE) != "") {
    (*ge_options)["ge.jit_compile"] = ms_context_ptr->get_param<std::string>(MS_CTX_ENABLE_JIT_COMPILE);
    MS_LOG(INFO) << "Set jit_compile " << ms_context_ptr->get_param<std::string>(MS_CTX_ENABLE_JIT_COMPILE) << ".";
  } else {
    MS_LOG(INFO) << "The default value of jit_compile is set by CANN.";
  }
}

void GeDeviceContext::GetGeOptions(const std::shared_ptr<MsContext> &ms_context_ptr,
                                   std::map<std::string, std::string> *ge_options) {
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  MS_EXCEPTION_IF_NULL(ge_options);

  (*ge_options)["device_id"] = "0";
  auto env_atomic_clean_policy = common::GetEnv("MS_GE_ATOMIC_CLEAN_POLICY");
  if (!env_atomic_clean_policy.empty()) {
    (*ge_options)["ge.exec.atomicCleanPolicy"] = env_atomic_clean_policy;
  }
  // set up dump options
  auto dump_env = common::GetEnv(kMindsporeDumpConfig);
  if (!dump_env.empty()) {
    auto &dump_parser = DumpJsonParser::GetInstance();
    dump_parser.Parse();
    (*ge_options)["ge.exec.enableDump"] = std::to_string(static_cast<int>(dump_parser.async_dump_enabled()));
    (*ge_options)["ge.exec.dumpPath"] = dump_parser.path();
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
    MS_LOG(INFO) << "The enable dump state is " << (*ge_options)["ge.exec.enableDump"] << ", save dump path is "
                 << (*ge_options)["ge.exec.dumpPath"] << ", dump mode is " << kGeDumpMode[dump_parser.input_output()]
                 << ", dump step is " << dump_parser.iteration_string() << ".";
  }
  auto profiler_manager = profiler::ProfilerManager::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_manager);
  (*ge_options)["ge.exec.profilingMode"] = std::to_string(static_cast<int>(profiler_manager->GetProfilingEnableFlag()));
  if (profiler_manager->GetProfilingEnableFlag()) {
    (*ge_options)["ge.exec.profilingOptions"] = profiler_manager->GetProfilingOptions();
  }

  (*ge_options)["rank_table_file"] = "";
  auto env_ddk_version = common::GetEnv("DDK_VERSION");
  if (!env_ddk_version.empty()) {
    (*ge_options)["ge.DDK_version"] = env_ddk_version;
  } else {
    (*ge_options)["ge.DDK_version"] = "1.60.T17.B830";
  }
  (*ge_options)["graphType"] = "1";

  bool training = IsGeTrain();
  if (training) {
    (*ge_options)["ge.graphRunMode"] = "1";
  }

  SetDisableReuseMemoryFlag(ge_options);
  SetHcclOptions(ms_context_ptr, ge_options);

  auto env_job_id = common::GetEnv("JOB_ID");
  if (!env_job_id.empty()) {
    (*ge_options)["ge.exec.jobId"] = env_job_id;
  } else {
    (*ge_options)["ge.exec.jobId"] = "0";
    MS_LOG(WARNING) << "JOB_ID is not set in ENV. Now set to default value 0";
  }

  auto env_fe_flag = common::GetEnv("FE_FLAG");
  if (!env_fe_flag.empty()) {
    (*ge_options)["ge.feFlag"] = env_fe_flag;
    MS_LOG(INFO) << "Use FE, make sure fe lib is set in OPTION_EXEC_EXTERN_PLUGIN_PATH.";
  }

  auto env_aicpu_flag = common::GetEnv("AICPU_FLAG");
  if (!env_aicpu_flag.empty()) {
    (*ge_options)["ge.aicpuFlag"] = env_aicpu_flag;
    MS_LOG(INFO) << "Use AICPU, make sure aicpu lib is set in OPTION_EXEC_EXTERN_PLUGIN_PATH.";
  }

  auto env_op_precision = common::GetEnv("MS_GE_OP_PRECISION");
  if (!env_op_precision.empty()) {
    (*ge_options)["ge.exec.op_precision_mode"] = env_op_precision;
    MS_LOG(INFO) << "Use MS_GE_OP_PRECISION, op precision mode path:" << env_op_precision;
  }

  auto proto_lib_path = common::GetEnv("OPTION_PROTO_LIB_PATH");
  if (!proto_lib_path.empty()) {
    char real_path[PATH_MAX] = {0};
    if (realpath(proto_lib_path.c_str(), real_path)) {
      proto_lib_path = real_path;
      (*ge_options)["ge.opsProtoLibPath"] = proto_lib_path;
    }
  } else {
    MS_LOG(WARNING) << "Set proto lib path failed!";
  }

  SetAscendConfig(ms_context_ptr, ge_options);

  // Disable the global variable acc, only enable it while adding training graph in pipeline
  (*ge_options)["ge.exec.variable_acc"] = "0";

  // ge heterogeneous mode
  if (ms_context_ptr->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
    (*ge_options)["ge.socVersion"] = "Ascend310P3";
  }

  // enable overflow detection
  (*ge_options)["ge.exec.overflow"] = "1";

  UseOpDebugConfig(ge_options);
}

void GeDeviceContext::SetDisableReuseMemoryFlag(std::map<std::string, std::string> *ge_options) const {
  MS_EXCEPTION_IF_NULL(ge_options);
  auto env_disable_reuse_memory = common::GetEnv("DISABLE_REUSE_MEMORY");
  if (!env_disable_reuse_memory.empty()) {
    (*ge_options)["ge.exec.disableReuseMemory"] = env_disable_reuse_memory;
  } else {
    (*ge_options)["ge.exec.disableReuseMemory"] = "0";
    MS_LOG(WARNING) << "DISABLE_REUSE_MEMORY is not set in ENV. Now set to default value 0";
  }
}

void GeDeviceContext::SetHcclOptions(const std::shared_ptr<MsContext> &inst_context,
                                     std::map<std::string, std::string> *ge_options) {
  MS_EXCEPTION_IF_NULL(inst_context);
  MS_EXCEPTION_IF_NULL(ge_options);
  auto env_table_file = common::GetEnv("RANK_TABLE_FILE");
  auto env_rank_id = common::GetEnv("RANK_ID");
  auto env_device_id = std::to_string(inst_context->get_param<uint32_t>(MS_CTX_DEVICE_ID));
  auto env_cluster_info = common::GetEnv("HELP_CLUSTER");
  if (!(env_table_file.empty() || env_rank_id.empty()) || !(env_cluster_info.empty() || env_rank_id.empty())) {
    MS_LOG(INFO) << "Initialize Ge for distribute parameter";
    if (!env_table_file.empty()) {
      MS_LOG(INFO) << "Use hccl, make sure hccl lib is set in OPTION_EXEC_EXTERN_PLUGIN_PATH.";
      (*ge_options)["ge.exec.rankTableFile"] = env_table_file;
    }
    auto env_hccl_flag = common::GetEnv("HCCL_FLAG");
    if (!env_hccl_flag.empty()) {
      (*ge_options)["ge.exec.hcclFlag"] = env_hccl_flag;
    }
    (*ge_options)["ge.exec.isUseHcom"] = "1";
    (*ge_options)["ge.exec.deviceId"] = env_device_id;
    (*ge_options)["ge.exec.rankId"] = env_rank_id;
    (*ge_options)["ge.exec.podName"] = env_rank_id;
    (*ge_options)["ge.graphRunMode"] = "1";
  } else {
    // device id is still needed for non-distribute case
    (*ge_options)["ge.exec.deviceId"] = env_device_id;
    MS_LOG(INFO) << "No hccl mode. "
                 << "If use hccl, make sure [RANK_TABLE_FILE,RANK_ID,DEVICE_ID,DEPLOY_MODE] all be set in ENV.";
  }

  auto env_deploy_mode = common::GetEnv("DEPLOY_MODE");
  if (!env_deploy_mode.empty()) {
    (*ge_options)["ge.exec.deployMode"] = env_deploy_mode;
  } else {
    (*ge_options)["ge.exec.deployMode"] = "0";
    MS_LOG(WARNING) << "DEPLOY_MODE is not set in ENV. Now set to default value 0";
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

DeprecatedInterface *GeDeviceContext::GetDeprecatedInterface() {
  // need lock when multi-threads
  if (deprecated_interface_ == nullptr) {
    deprecated_interface_ = std::make_unique<AscendDeprecatedInterface>(this);
  }
  return deprecated_interface_.get();
}

constexpr auto kGeDevice = "GE";
MS_REGISTER_DEVICE(kGeDevice, GeDeviceContext);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
