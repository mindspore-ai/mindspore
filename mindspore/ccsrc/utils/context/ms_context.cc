/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "utils/context/ms_context.h"
#include <thread>
#include <atomic>
#include <fstream>
#include "./common.h"
#include "utils/convert_utils.h"
#include "utils/tensorprint_utils.h"
#ifndef NO_DLIB
#include "tdt/tsd_client.h"
#include "tdt/tdt_host_interface.h"
#include "tdt/data_common.h"
#endif
#include "transform/df_graph_manager.h"
#include "ir/meta_tensor.h"

namespace mindspore {
using mindspore::transform::DfGraphManager;
using transform::GraphRunner;
using transform::GraphRunnerOptions;

std::atomic<bool> thread_1_must_end(false);

std::shared_ptr<MsContext> MsContext::inst_context_ = nullptr;
std::map<std::string, MsBackendPolicy> MsContext::policy_map_ = {{"ge", kMsBackendGePrior},
                                                                 {"vm", kMsBackendVmOnly},
                                                                 {"ms", kMsBackendMsPrior},
                                                                 {"ge_only", kMsBackendGeOnly},
                                                                 {"vm_prior", kMsBackendVmPrior}};

MsContext::MsContext(const std::string& policy, const std::string& target) {
  save_graphs_flag_ = false;
  save_graphs_path_ = ".";
  save_ms_model_flag_ = false;
  save_ms_model_path_ = "./model.ms";
  enable_dump_ = false;
  save_dump_path_ = ".";
  tsd_ref_ = 0;
  ge_ref_ = 0;
  is_multi_graph_sink_ = false;
  is_pynative_ge_init_ = false;
  enable_reduce_precision_ = true;
  auto env_device = common::GetEnv("DEVICE_ID");
  if (!env_device.empty()) {
    device_id_ = UlongToUint(std::stoul(env_device.c_str()));
  } else {
    device_id_ = 0;
  }
  backend_policy_ = policy_map_[policy];
  device_target_ = target;
  execution_mode_ = kGraphMode;
  enable_task_sink_ = true;
  ir_fusion_flag_ = true;
  enable_hccl_ = false;
  enable_loop_sink_ = false;
  enable_mem_reuse_ = true;
  enable_gpu_summary_ = true;
  precompile_only_ = false;
  auto_mixed_precision_flag_ = true;
  enable_pynative_infer_ = false;
  enable_dynamic_mem_pool_ = false;
  graph_memory_max_size_ = "0";
  variable_memory_max_size_ = "0";
  MS_LOG(INFO) << "Create context with backend policy:" << policy << ", device target:" << target << ".";
}

std::shared_ptr<MsContext> MsContext::GetInstance() {
  if (inst_context_ == nullptr) {
    MS_LOG(DEBUG) << "Create new mindspore context";
    inst_context_.reset(new (std::nothrow) MsContext("ge", kAscendDevice));
  }
  return inst_context_;
}

bool MsContext::set_backend_policy(const std::string& policy) {
  if (policy_map_.find(policy) == policy_map_.end()) {
    MS_LOG(ERROR) << "invalid backend policy name: " << policy;
    return false;
  }
  backend_policy_ = policy_map_[policy];
  MS_LOG(INFO) << "ms set context backend policy:" << policy;
  return true;
}

std::string MsContext::backend_policy() const {
  auto res = std::find_if(
    policy_map_.begin(), policy_map_.end(),
    [&, this](const std::pair<std::string, MsBackendPolicy>& item) { return item.second == backend_policy_; });
  if (res != policy_map_.end()) {
    return res->first;
  }
  return "unknown";
}

void MsContext::set_execution_mode(int execution_mode) {
  if (execution_mode != kGraphMode && execution_mode != kPynativeMode) {
    MS_LOG(EXCEPTION) << "The execution mode is invalid!";
  }
  execution_mode_ = execution_mode;
}

bool MsContext::set_device_target(const std::string& target) {
  if (kTargetSet.find(target) == kTargetSet.end()) {
    MS_LOG(ERROR) << "invalid device target name: " << target;
    return false;
  }
  if (target == kDavinciDevice) {
    device_target_ = kAscendDevice;
  } else {
    device_target_ = target;
  }
  MS_LOG(INFO) << "ms set context device target:" << target;
  return true;
}

bool MsContext::set_device_id(uint32_t device_id) {
  device_id_ = device_id;
  MS_LOG(INFO) << "ms set context device id:" << device_id;
  return true;
}

#ifndef NO_DLIB
// Open tdt dataset
bool MsContext::OpenTsd() {
  if (is_pynative_ge_init_) {
    return true;
  }

  if (tsd_ref_) {
    MS_LOG(DEBUG) << "TDT Dataset client is already opened.";
    tsd_ref_++;
    return true;
  }

  unsigned int device_id;
  unsigned int rank_size = 1;

  device_id = device_id_;

  auto rank_size_env = common::GetEnv("RANK_SIZE");
  if (rank_size_env.empty()) {
    MS_LOG(INFO) << "Should config rank size.";
    rank_size = 1;
  } else {
    int rank_env = std::stoi(rank_size_env);
    if (rank_env <= 0) {
      MS_LOG(EXCEPTION) << "Error rank size " << rank_env << ".";
    }
    rank_size = IntToUint(rank_env);
  }

  MS_LOG(INFO) << "Device id = " << device_id << ", rank size = " << rank_size << ".";

  TDT_StatusT status = tdt::TsdClient::GetInstance()->Open(device_id, rank_size);
  if (status != TDT_OK) {
    MS_LOG(EXCEPTION) << "Device " << device_id << " is occupied, open tsd failed, status = " << status << ".";
    return false;
  }
  tsd_ref_++;
#ifdef ENABLE_TDTQUE
  int32_t initStatus = tdt::TdtHostInit(device_id);
  if (initStatus != TDT_OK_CODE) {
    MS_LOG(EXCEPTION) << "Init tsd failed, status = " << initStatus << ".";
    return false;
  }
  tdt_print_ = std::thread(TensorPrint());
#endif
  MS_LOG(INFO) << "Open and init tsd successful, tsd reference = " << tsd_ref_ << ".";
  return true;
}

bool MsContext::CloseTsd(bool force) {
  if (tsd_ref_ == 0) {
    return true;
  }
  tsd_ref_--;
  if (force || tsd_ref_ == 0) {
    tsd_ref_ = 0;
#ifdef ENABLE_TDTQUE
    int32_t stopStatus = tdt::TdtHostStop(KNpuLog);
    if (stopStatus != TDT_OK_CODE) {
      MS_LOG(EXCEPTION) << "Stop tsd failed, status = " << stopStatus << ".";
      return false;
    }
    py::gil_scoped_release gil_release;
    int32_t destroyStatus = tdt::TdtHostDestroy();
    if (destroyStatus != TDT_OK_CODE) {
      MS_LOG(EXCEPTION) << "Destroy tsd failed, status = " << destroyStatus << ".";
      return false;
    }
    try {
      if (tdt_print_.joinable()) {
        MS_LOG(INFO) << "join tdt host receive process";
        tdt_print_.join();
      }
    } catch (const std::exception& e) {
      MS_LOG(ERROR) << "tdt thread join failed: " << e.what();
    }
#endif
    TDT_StatusT status = tdt::TsdClient::GetInstance()->Close();
    if (status != TDT_OK) {
      MS_LOG(EXCEPTION) << "Close tsd failed, status = " << status << ".";
      return false;
    }
    is_pynative_ge_init_ = false;
    MS_LOG(INFO) << "Destroy and close tsd successful, status = " << status << ".";
  } else {
    MS_LOG(DEBUG) << "TDT Dataset client is used, no need to close, tsd reference = " << tsd_ref_ << ".";
  }

  return true;
}
#else
bool MsContext::OpenTsd() { return true; }

bool MsContext::CloseTsd(bool) { return true; }
#endif

void MsContext::SetHcclOptions(std::map<std::string, std::string>* ge_options) const {
  auto env_table_file = common::GetEnv("RANK_TABLE_FILE");
  auto env_rank_id = common::GetEnv("RANK_ID");
  auto env_device_id = std::to_string(device_id_);
  if (!(env_table_file.empty() || env_rank_id.empty())) {
    MS_LOG(INFO) << "Initialize Ge for distribute parameter";
    MS_LOG(INFO) << "Use hccl, make sure hccl lib is set in OPTION_EXEC_EXTERN_PLUGIN_PATH.";
    auto env_hccl_flag = common::GetEnv("HCCL_FLAG");
    if (!env_hccl_flag.empty()) {
      (*ge_options)["ge.exec.hcclFlag"] = env_hccl_flag;
    }
    (*ge_options)["ge.exec.isUseHcom"] = "1";
    (*ge_options)["ge.exec.deviceId"] = env_device_id;
    (*ge_options)["ge.exec.rankId"] = env_rank_id;
    (*ge_options)["ge.exec.podName"] = env_rank_id;
    (*ge_options)["ge.exec.rankTableFile"] = env_table_file;
    (*ge_options)["ge.graphRunMode"] = "1";
  } else {
    // device id is still needed for non-distribute case
    (*ge_options)["ge.exec.deviceId"] = env_device_id;
    MS_LOG(INFO) << "No hccl mode. "
                    "If use hccl, make sure [RANK_TABLE_FILE,RANK_ID,DEVICE_ID,DEPLOY_MODE] all be set in ENV.";
  }

  auto env_deploy_mode = common::GetEnv("DEPLOY_MODE");
  if (!env_deploy_mode.empty()) {
    (*ge_options)["ge.exec.deployMode"] = env_deploy_mode;
  } else {
    (*ge_options)["ge.exec.deployMode"] = "0";
    MS_LOG(WARNING) << "DEPLOY_MODE is not set in ENV. Now set to default value 0";
  }
}

void MsContext::GetGeOptions(std::map<std::string, std::string>* ge_options) const {
#ifdef ENABLE_GE
  (*ge_options)["device_id"] = "0";
  (*ge_options)["ge.exec.enableDump"] = enable_dump_;
  (*ge_options)["ge.exec.dumpPath"] = save_dump_path_;
  // only not supported in ge
  auto tbe_plugin_path = common::GetEnv("ME_TBE_PLUGIN_PATH");
  if (!tbe_plugin_path.empty()) {
    char real_path[PATH_MAX] = {0};
    if (nullptr == realpath(tbe_plugin_path.c_str(), real_path)) {
      MS_LOG(ERROR) << "Ms tbe plugin Path error, " << tbe_plugin_path;
    } else {
      tbe_plugin_path = real_path;
      (*ge_options)["ge.TBE_plugin_path"] = tbe_plugin_path;
    }
  } else {
    MS_LOG(ERROR) << "Set TBE plugin path failed!";
  }
  (*ge_options)["rank_table_file"] = "";
  auto env_ddk_version = common::GetEnv("DDK_VERSION");
  if (!env_ddk_version.empty()) {
    (*ge_options)["ge.DDK_version"] = env_ddk_version;
  } else {
    (*ge_options)["ge.DDK_version"] = "1.60.T17.B830";
  }
  (*ge_options)["graphType"] = "1";

  if (graph_memory_max_size_ != "0") {
    (*ge_options)["ge.graphMemoryMaxSize"] = graph_memory_max_size_;
  }

  if (variable_memory_max_size_ != "0") {
    (*ge_options)["ge.variableMemoryMaxSize"] = variable_memory_max_size_;
  }

#if ENABLE_TRAIN == 1
  (*ge_options)["ge.graphRunMode"] = "1";
#endif
  SetDisableReuseMemoryFlag(ge_options);
  SetHcclOptions(ge_options);

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

  // all libs are set in same env variable "OPTION_EXEC_EXTERN_PLUGIN_PATH", such as FE, HCCL, AICPU, etc
  auto load_path = common::GetEnv("OPTION_EXEC_EXTERN_PLUGIN_PATH");
  if (!load_path.empty()) {
    char real_path[PATH_MAX] = {0};
    if (realpath(load_path.c_str(), real_path)) {
      load_path = real_path;
      (*ge_options)["ge.soLoadPath"] = load_path;
    }
  } else {
    MS_LOG(ERROR) << "Set lib load path failed!";
  }

  auto proto_lib_path = common::GetEnv("OPTION_PROTO_LIB_PATH");
  if (!proto_lib_path.empty()) {
    char real_path[PATH_MAX] = {0};
    if (realpath(proto_lib_path.c_str(), real_path)) {
      proto_lib_path = real_path;
      (*ge_options)["ge.opsProtoLibPath"] = proto_lib_path;
    }
  } else {
    MS_LOG(ERROR) << "Set proto lib path failed!";
  }
#endif
}

void MsContext::SetDisableReuseMemoryFlag(std::map<std::string, std::string>* ge_options) const {
  auto env_disable_reuse_memory = common::GetEnv("DISABLE_REUSE_MEMORY");
  if (!env_disable_reuse_memory.empty()) {
    (*ge_options)["ge.exec.disableReuseMemory"] = env_disable_reuse_memory;
  } else {
    (*ge_options)["ge.exec.disableReuseMemory"] = "0";
    MS_LOG(WARNING) << "DISABLE_REUSE_MEMORY is not set in ENV. Now set to default value 0";
  }
}

bool MsContext::InitGe() {
#ifdef ENABLE_GE
  if (is_pynative_ge_init_) {
    return true;
  }

  if (ge_ref_) {
    ge_ref_++;
    return true;
  }

  std::map<std::string, std::string> ge_options;
  GetGeOptions(&ge_options);
  {
    // Release GIL before calling into (potentially long-running) C++ code
    py::gil_scoped_release release;
    if (ge::GEInitialize(ge_options) != ge::GRAPH_SUCCESS) {
      MS_LOG(EXCEPTION) << "Initialize GE failed!";
    }
  }
  ge_ref_++;
  MS_LOG(INFO) << "Init ge successful, ge reference = " << ge_ref_ << ".";
#endif
  return true;
}

bool MsContext::FinalizeGe(bool force) {
#ifdef ENABLE_GE
  if (ge_ref_ == 0) {
    return true;
  }
  ge_ref_--;
  if (force || ge_ref_ == 0) {
    ge_ref_ = 0;
    try {
      DfGraphManager::GetInstance().DeleteGraphRunner();
      DfGraphManager::GetInstance().DeleteGeSession();
    } catch (const std::exception& e) {
      MS_LOG(ERROR) << "Error occurred when deleting GE graph runner and session fail. Error: " << e.what();
    } catch (...) {
      std::string exName(abi::__cxa_current_exception_type()->name());
      MS_LOG(ERROR) << "Error occurred when deleting GE graph runner and session fail. Exception name: " << exName;
    }
    if (ge::GEFinalize() != ge::GRAPH_SUCCESS) {
      MS_LOG(WARNING) << "Finalize GE failed!";
    }
    is_pynative_ge_init_ = false;
  } else {
    MS_LOG(INFO) << "Ge is used, no need to finalize, tsd reference = " << ge_ref_ << ".";
  }
#endif
  return true;
}

bool MsContext::PynativeInitGe() {
  if (is_pynative_ge_init_ || ge_ref_ || tsd_ref_) {
    return true;
  }
  (void)OpenTsd();
  (void)InitGe();
  is_pynative_ge_init_ = true;
  return true;
}
}  // namespace mindspore
