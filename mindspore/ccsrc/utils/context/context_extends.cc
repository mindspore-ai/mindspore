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

#include "utils/context/context_extends.h"
#include <map>
#include <string>
#include <memory>
#include <thread>
#include "pybind11/pybind11.h"
#include "utils/ms_utils.h"
#include "utils/convert_utils_base.h"
#ifndef NO_DLIB
#include "acl/acl_tdt.h"
#include "runtime/dev.h"
#include "toolchain/plog.h"
#include "common/util/error_manager/error_manager.h"
#endif
#ifdef ENABLE_GE
#include "transform/graph_ir/df_graph_manager.h"
#endif
#include "profiler/device/profiling.h"

namespace py = pybind11;

namespace mindspore {
namespace context {
#ifdef ENABLE_GE
using mindspore::transform::DfGraphManager;
#endif

constexpr auto kUnknowErrorString = "Unknown error occurred";

#ifndef NO_DLIB
// Open tdt dataset
bool OpenTsd(const std::shared_ptr<MsContext> &ms_context_ptr) {
  if (ms_context_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "nullptr";
  }

  if (ms_context_ptr->get_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT)) {
    return true;
  }

  if (ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF)) {
    MS_LOG(DEBUG) << "ACLTDT Dataset client is already opened.";
    ms_context_ptr->increase_param<uint32_t>(MS_CTX_TSD_REF);
    return true;
  }

  auto role = common::GetEnv("MS_ROLE");
  if (strcmp(role.c_str(), "MS_SCHED") == 0 || strcmp(role.c_str(), "MS_PSERVER") == 0) {
    return true;
  }

  uint32_t rank_size = 1;
  uint32_t device_id = ms_context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);

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

  int log_ret = DlogReportInitialize();
  if (log_ret != 0) {
    MS_LOG(WARNING) << "Init slog failed, ret = " << log_ret;
  }

  MS_LOG(INFO) << "Device id = " << device_id << ", rank size = " << rank_size << ".";
  auto ret = rtSetDevice(static_cast<int32_t>(device_id));
  if (ret != RT_ERROR_NONE) {
    const std::string &error_message = ErrorManager::GetInstance().GetErrorMessage();
    if (!error_message.empty() && error_message.find(kUnknowErrorString) == std::string::npos) {
      MS_LOG(ERROR) << "Ascend error occurred, error message:\n" << error_message;
    }
    MS_LOG(EXCEPTION) << "Device " << device_id << " call rtSetDevice failed, ret[" << static_cast<int>(ret) << "]";
  }
  ms_context_ptr->increase_param<uint32_t>(MS_CTX_TSD_REF);
#ifdef ENABLE_TDTQUE
  auto thread_crt = [](const std::string &path, const acltdtChannelHandle *acl_handle) {
    return std::thread(TensorPrint(path, acl_handle));
  };
  ms_context_ptr->CreateTensorPrintThread(thread_crt);
#endif
  return true;
}

bool CloseTsd(const std::shared_ptr<MsContext> &ms_context_ptr, bool force) {
  if (ms_context_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "ms_context_prt is nullptr";
  }
  if (ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) == 0) {
    return true;
  }
  ms_context_ptr->decrease_param<uint32_t>(MS_CTX_TSD_REF);
  if (force || ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) == 0) {
    ms_context_ptr->set_param<uint32_t>(MS_CTX_TSD_REF, 0);

#ifdef ENABLE_TDTQUE
    py::gil_scoped_release gil_release;
    ms_context_ptr->DestroyTensorPrintThread();
#endif
    uint32_t device_id = ms_context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto ret = rtDeviceReset(static_cast<int32_t>(device_id));
    if (ret != RT_ERROR_NONE) {
      const std::string &error_message = ErrorManager::GetInstance().GetErrorMessage();
      if (!error_message.empty() && error_message.find(kUnknowErrorString) == std::string::npos) {
        MS_LOG(ERROR) << "Ascend error occurred, error message:\n" << error_message;
      }
      MS_LOG(EXCEPTION) << "Device " << device_id << " call rtDeviceReset failed, ret[" << static_cast<int>(ret) << "]";
      return false;
    }
    ms_context_ptr->set_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT, false);
    MS_LOG(INFO) << "Call rtDeviceReset, destroy and close tsd successful, ret[" << static_cast<int>(ret) << "]";
    (void)DlogReportFinalize();
  } else {
    MS_LOG(DEBUG) << "Acltdt Dataset client is used, no need to close, tsd reference = "
                  << ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) << ".";
  }
  return true;
}
#else
bool OpenTsd(const std::shared_ptr<MsContext> &ms_context_ptr) { return true; }
bool CloseTsd(const std::shared_ptr<MsContext> &ms_context_ptr, bool) { return true; }
#endif

void SetDisableReuseMemoryFlag(std::map<std::string, std::string> *ge_options) {
  auto env_disable_reuse_memory = common::GetEnv("DISABLE_REUSE_MEMORY");
  if (!env_disable_reuse_memory.empty()) {
    (*ge_options)["ge.exec.disableReuseMemory"] = env_disable_reuse_memory;
  } else {
    (*ge_options)["ge.exec.disableReuseMemory"] = "0";
    MS_LOG(WARNING) << "DISABLE_REUSE_MEMORY is not set in ENV. Now set to default value 0";
  }
}

void GetGeOptions(const std::shared_ptr<MsContext> &ms_context_ptr, std::map<std::string, std::string> *ge_options) {
  if (ms_context_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
#ifdef ENABLE_GE
  (*ge_options)["device_id"] = "0";
  (*ge_options)["ge.exec.enableDump"] = std::to_string(ms_context_ptr->get_param<bool>(MS_CTX_ENABLE_DUMP));
  (*ge_options)["ge.exec.dumpPath"] = ms_context_ptr->get_param<std::string>(MS_CTX_SAVE_DUMP_PATH);
  (*ge_options)["ge.exec.dumpMode"] = "output";
  MS_LOG(INFO) << "The enable dump state is " << std::to_string(ms_context_ptr->get_param<bool>(MS_CTX_ENABLE_DUMP))
               << " and save dump path is " << ms_context_ptr->get_param<std::string>(MS_CTX_SAVE_DUMP_PATH) << ".";

  auto profiler_manager = profiler::ProfilerManager::GetInstance();
  if (profiler_manager == nullptr) {
    MS_LOG(EXCEPTION) << "Profiler manager is nullptr";
  }
  (*ge_options)["ge.exec.profilingMode"] = std::to_string(profiler_manager->GetProfilingEnableFlag());
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

  if (ms_context_ptr->get_param<std::string>(MS_CTX_GRAPH_MEMORY_MAX_SIZE) != "0") {
    (*ge_options)["ge.graphMemoryMaxSize"] = ms_context_ptr->get_param<std::string>(MS_CTX_GRAPH_MEMORY_MAX_SIZE);
  }

  if (ms_context_ptr->get_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE) != "0") {
    (*ge_options)["ge.variableMemoryMaxSize"] = ms_context_ptr->get_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE);
  }

#if ENABLE_TRAIN == 1
  (*ge_options)["ge.graphRunMode"] = "1";
#endif
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

  (*ge_options)["ge.exec.precision_mode"] = "force_fp16";

  // Disable the global variable acc, only enable it while adding training graph in pipeline
  (*ge_options)["ge.exec.variable_acc"] = "0";
#endif
}

void SetHcclOptions(const std::shared_ptr<MsContext> &ms_context_ptr, std::map<std::string, std::string> *ge_options) {
  if (ms_context_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  auto env_table_file = common::GetEnv("RANK_TABLE_FILE");
  auto env_rank_id = common::GetEnv("RANK_ID");
  auto env_device_id = std::to_string(ms_context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID));
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

bool InitGe(const std::shared_ptr<MsContext> &ms_context_ptr) {
  if (ms_context_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
#ifdef ENABLE_GE
  if (ms_context_ptr->get_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT)) {
    return true;
  }

  if (ms_context_ptr->get_param<uint32_t>(MS_CTX_GE_REF)) {
    ms_context_ptr->increase_param<uint32_t>(MS_CTX_GE_REF);
    return true;
  }

  std::map<std::string, std::string> ge_options;
  GetGeOptions(ms_context_ptr, &ge_options);
  {
    // Release GIL before calling into (potentially long-running) C++ code
    py::gil_scoped_release release;
    if (ge::GEInitialize(ge_options) != ge::GRAPH_SUCCESS) {
      MS_LOG(EXCEPTION) << "Initialize GE failed!";
    }
  }
  ms_context_ptr->increase_param<uint32_t>(MS_CTX_GE_REF);
  MS_LOG(INFO) << "Init ge successful, ge reference = " << ms_context_ptr->get_param<uint32_t>(MS_CTX_GE_REF) << ".";
#endif
  return true;
}

bool PynativeInitGe(const std::shared_ptr<MsContext> &ms_context_ptr) {
  if (ms_context_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  if (ms_context_ptr->get_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT) ||
      ms_context_ptr->get_param<uint32_t>(MS_CTX_GE_REF) || ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF)) {
    return true;
  }

  (void)OpenTsd(ms_context_ptr);
  (void)InitGe(ms_context_ptr);
  ms_context_ptr->set_param(MS_CTX_IS_PYNATIVE_GE_INIT, true);
  return true;
}

bool FinalizeGe(const std::shared_ptr<MsContext> &ms_context_ptr, bool force) {
  if (ms_context_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
#ifdef ENABLE_GE
  if (ms_context_ptr->get_param<uint32_t>(MS_CTX_GE_REF) == 0) {
    return true;
  }
  ms_context_ptr->decrease_param<uint32_t>(MS_CTX_GE_REF);
  if (force || ms_context_ptr->get_param<uint32_t>(MS_CTX_GE_REF) == 0) {
    ms_context_ptr->set_param<uint32_t>(MS_CTX_GE_REF, 0);
    try {
      DfGraphManager::GetInstance().DeleteGraphRunner();
      DfGraphManager::GetInstance().DeleteGeSession();
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Error occurred when deleting GE graph runner and session fail. Error: " << e.what();
    } catch (...) {
      std::string exName(abi::__cxa_current_exception_type()->name());
      MS_LOG(ERROR) << "Error occurred when deleting GE graph runner and session fail. Exception name: " << exName;
    }
    if (ge::GEFinalize() != ge::GRAPH_SUCCESS) {
      MS_LOG(WARNING) << "Finalize GE failed!";
    }
    ms_context_ptr->set_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT, false);
  } else {
    MS_LOG(INFO) << "Ge is used, no need to finalize, tsd reference = "
                 << ms_context_ptr->get_param<uint32_t>(MS_CTX_GE_REF) << ".";
  }
#endif
  return true;
}

bool IsTsdOpened(const std::shared_ptr<MsContext> &ms_context_ptr) {
  if (ms_context_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  return ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) > 0;
}

bool IsGeInited(const std::shared_ptr<MsContext> &ms_context_ptr) {
  if (ms_context_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  return ms_context_ptr->get_param<uint32_t>(MS_CTX_GE_REF) > 0;
}

// Register for device type.
struct DeviceTypeSetRegister {
  DeviceTypeSetRegister() {
    MsContext::device_type_seter([](std::shared_ptr<MsContext> &device_type_seter) {
#ifdef ENABLE_GE
      device_type_seter.reset(new (std::nothrow) MsContext("ge", kAscendDevice));
#elif defined(ENABLE_D)
      device_type_seter.reset(new (std::nothrow) MsContext("ms", kAscendDevice));
#elif defined(ENABLE_GPU)
      device_type_seter.reset(new (std::nothrow) MsContext("ms", kGPUDevice));
#else
      device_type_seter.reset(new (std::nothrow) MsContext("vm", kCPUDevice));
#endif
    });
  }
  DeviceTypeSetRegister(const DeviceTypeSetRegister &) = delete;
  DeviceTypeSetRegister &operator=(const DeviceTypeSetRegister &) = delete;
  ~DeviceTypeSetRegister() = default;
} device_type_set_regsiter;
}  // namespace context
}  // namespace mindspore
