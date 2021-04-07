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

#include "runtime/device/ascend/profiling/profiling_manager.h"
#include <stdlib.h>
#include <vector>
#include "securec/include/securec.h"
#include "./prof_mgr_core.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "utils/convert_utils.h"
#include "runtime/base.h"
#include "toolchain/prof_acl_api.h"
#include "runtime/device/ascend/profiling/profiling_callback_register.h"

namespace {
constexpr uint32_t kProfilingDeviceNum = 1;
constexpr auto kRtSetDeviceRegName = "profiling";
constexpr Status PROF_SUCCESS = 0;
constexpr Status PROF_FAILED = 0xFFFFFFFF;
}  // namespace

namespace mindspore {
namespace device {
namespace ascend {
ProfilingManager &ProfilingManager::GetInstance() {
  static ProfilingManager inst;
  return inst;
}

ProfilingManager::ProfilingManager() : device_id_(0), prof_cb_({0}), hccl_enabled_bef_profiling_enabled_(false) {}

uint64_t ProfilingManager::GetJobId() const {
  const char *job_id = std::getenv("JOB_ID");
  return ((job_id != nullptr) ? std::strtoul(job_id, nullptr, 10) : 0);
}

bool ProfilingManager::ReportProfilingData(const map<uint32_t, string> &op_taskId_map) const {
  if (!IsProfiling()) {
    MS_LOG(INFO) << "No need profiling. please export PROFILING_MODE and in train mode.";
    return false;
  }
  if (op_taskId_map.empty()) {
    MS_LOG(WARNING) << "op_taskId_map is empty.";
    return false;
  }

  MS_LOG(INFO) << "DistributeTask: op tasId map size = " << op_taskId_map.size();

  ReporterData reporter_data = {};
  for (const auto &iter : op_taskId_map) {
    auto data = iter.second + ' ' + std::to_string(iter.first) + ';';
    reporter_data.deviceId = UintToInt(device_id_);
    reporter_data.data = (unsigned char *)(const_cast<char *>(data.c_str()));
    reporter_data.dataLen = data.size();
    auto ret = memcpy_s(reporter_data.tag, MSPROF_ENGINE_MAX_TAG_LEN + 1, "framework", sizeof("framework"));
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return false;
    }
    int32_t cb_ret = CallMsprofReport(NOT_NULL(&reporter_data));
    if (cb_ret != 0) {
      MS_LOG(ERROR) << "reporter data fail, errorno(" << cb_ret << ")";
      return false;
    }
  }
  return true;
}

uint64_t GetProfilingModule() {
  return PROF_MODEL_EXECUTE_MASK | PROF_RUNTIME_API_MASK | PROF_RUNTIME_TRACE_MASK | PROF_SCHEDULE_TIMELINE_MASK |
         PROF_SCHEDULE_TRACE_MASK | PROF_TASK_TIME_MASK | PROF_SUBTASK_TIME_MASK | PROF_AICPU_TRACE_MASK |
         PROF_AICORE_METRICS_MASK | PROF_AIVECTORCORE_METRICS_MASK | PROF_MODEL_LOAD_MASK;
}

Status ProfilingManager::PluginInit() const {
  if (prof_cb_.msprofReporterCallback == nullptr) {
    MS_LOG(ERROR) << "MsprofReporterCallback callback is nullptr.";
    return PROF_FAILED;
  }
  return prof_cb_.msprofReporterCallback(static_cast<uint32_t>(MsprofReporterModuleId::MSPROF_MODULE_FRAMEWORK),
                                         static_cast<uint32_t>(MsprofReporterCallbackType::MSPROF_REPORTER_INIT),
                                         nullptr, 0);
}

void ProfilingManager::PluginUnInit() const {
  if (prof_cb_.msprofReporterCallback == nullptr) {
    MS_LOG(ERROR) << "MsprofReporterCallback callback is nullptr.";
    return;
  }
  int32_t cb_ret = prof_cb_.msprofReporterCallback(
    static_cast<uint32_t>(MsprofReporterModuleId::MSPROF_MODULE_FRAMEWORK),
    static_cast<uint32_t>(MsprofReporterCallbackType::MSPROF_REPORTER_UNINIT), nullptr, 0);
  if (cb_ret != 0) {
    MS_LOG(WARNING) << "profiling plugin uninit failed, ret:%d" << cb_ret;
  }
}

Status ProfilingManager::GetProfConf(const NotNull<MsprofGeOptions *> prof) {
  string job_id = std::to_string(GetJobId());
  if (memcpy_s(prof->jobId, sizeof(prof->jobId), job_id.c_str(), strlen(job_id.c_str())) != EOK) {
    MS_LOG(ERROR) << "Copy job_id failed.";
    return PROF_FAILED;
  }

  auto context = MsContext::GetInstance();
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return PROF_FAILED;
  }

  const string prof_options_str = context->get_param<std::string>(MS_CTX_PROFILING_OPTIONS);

  if (memcpy_s(prof->options, MSPROF_OPTIONS_DEF_LEN_MAX, prof_options_str.c_str(), prof_options_str.size()) != EOK) {
    MS_LOG(ERROR) << "Copy profiling_options failed";
    return PROF_FAILED;
  }
  return PROF_SUCCESS;
}

bool ProfilingManager::StartupProfiling(uint32_t device_id) {
  auto is_profiling = IsProfiling();
  if (!is_profiling) {
    MS_LOG(INFO) << "No need profiling. please export PROFILING_MODE and in train mode.";
    return true;
  }

  if (hccl_enabled_bef_profiling_enabled_) {
    MS_LOG(ERROR)
      << "Please check the Profiler object initialized before mindspore.context.set_auto_parallel_context() "
         "and mindspore.communication.management.init(). Profiler should be initialized before these code.";
    return false;
  }

  device_id_ = device_id;

  struct MsprofGeOptions prof_conf = {0};
  if (GetProfConf(NOT_NULL(&prof_conf)) != PROF_SUCCESS) {
    MS_LOG(ERROR) << "Get prof conf failed.";
    return false;
  }

  if (!ProfStartUp(NOT_NULL(&prof_conf))) {
    MS_LOG(ERROR) << "ProfMgrStartUp failed.";
    return false;
  }
  return true;
}

uint32_t GetCurrentDeviceId() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
}

bool ProfilingManager::ProfStartUp(const NotNull<MsprofGeOptions *> prof_conf) const {
  MS_LOG(INFO) << "Prof start up. ";

  if (prof_cb_.msprofCtrlCallback == nullptr) {
    MS_LOG(ERROR) << "MsprofCtrlCallback callback is nullptr.";
    return false;
  }

  // call profiling start up api
  int32_t cb_ret =
    prof_cb_.msprofCtrlCallback(static_cast<uint32_t>(MsprofCtrlCallbackType::MSPROF_CTRL_INIT_GE_OPTIONS),
                                static_cast<void *>(prof_conf.get()), sizeof(MsprofGeOptions));
  if (cb_ret != PROF_SUCCESS) {
    MS_LOG(ERROR) << "Call msprofCtrlCallback failed, ret: " << cb_ret;
    return false;
  }

  MS_LOG(INFO) << "Start up profiling success.";
  return true;
}

bool ProfilingManager::StopProfiling() {
  MS_LOG(INFO) << "StopProfiling";
  if (!IsProfiling()) {
    MS_LOG(INFO) << "No need profiling. please export PROFILING_MODE and in train mode.";
    return true;
  }

  // plugin unregister
  PluginUnInit();
  // stop runtime profiler
  auto module = GetProfilingModule();
  uint32_t device_ids[kProfilingDeviceNum] = {GetCurrentDeviceId()};

  auto rt_ret = rtProfilerStop(module, kProfilingDeviceNum, device_ids);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtProfilerStop failed";
    return false;
  }

  // stop profiling
  if (prof_cb_.msprofCtrlCallback == nullptr) {
    MS_LOG(ERROR) << "MsprofCtrlCallback callback is nullptr.";
    return false;
  }

  int32_t cb_ret =
    prof_cb_.msprofCtrlCallback(static_cast<uint32_t>(MsprofCtrlCallbackType::MSPROF_CTRL_FINALIZE), nullptr, 0);
  if (cb_ret != 0) {
    MS_LOG(WARNING) << "Call msprofCtrlCallback failed, ret: " << cb_ret;
    return false;
  }
  return true;
}

Status ProfilingManager::CallMsprofReport(const NotNull<ReporterData *> reporter_data) const {
  if (prof_cb_.msprofReporterCallback == nullptr) {
    MS_LOG(ERROR) << "MsprofReporterCallback callback is nullptr.";
    return PROF_FAILED;
  }
  return prof_cb_.msprofReporterCallback(static_cast<uint32_t>(MsprofReporterModuleId::MSPROF_MODULE_FRAMEWORK),
                                         static_cast<uint32_t>(MsprofReporterCallbackType::MSPROF_REPORTER_REPORT),
                                         static_cast<void *>(reporter_data.get()), sizeof(ReporterData));
}

Status RegProfCtrlCallback(MsprofCtrlCallback func) {
  if (func == nullptr) {
    MS_LOG(ERROR) << "Msprof ctrl callback is nullptr.";
    return PROF_FAILED;
  }
  if (ProfilingManager::GetInstance().GetMsprofCallback().msprofCtrlCallback != nullptr) {
    MS_LOG(WARNING) << "Msprof ctrl callback is exist, just ignore it.";
  } else {
    MS_LOG(INFO) << "GE register Msprof ctrl callback.";
    ProfilingManager::GetInstance().SetMsprofCtrlCallback(func);
  }
  return PROF_SUCCESS;
}

Status RegProfSetDeviceCallback(MsprofSetDeviceCallback func) {
  if (func == nullptr) {
    MS_LOG(ERROR) << "MsprofSetDeviceCallback callback is nullptr.";
    return PROF_FAILED;
  }
  ProfilingManager::GetInstance().SetMsprofSetDeviceCallback(func);
  // Pass MsprofSetDeviceCallback to runtime
  MS_LOG(INFO) << "GE pass setdevice callback to runtime.";
  Status rt_ret = rtRegDeviceStateCallback(kRtSetDeviceRegName, static_cast<rtDeviceStateCallback>(func));
  if (rt_ret != PROF_SUCCESS) {
    MS_LOG(WARNING) << "Pass MsprofSetDeviceCallback to runtime failed.";
    return rt_ret;
  }
  return PROF_SUCCESS;
}

Status RegProfReporterCallback(MsprofReporterCallback func) {
  if (func == nullptr) {
    MS_LOG(ERROR) << "MsprofReporterCallback callback is nullptr.";
    return PROF_FAILED;
  }
  if (ProfilingManager::GetInstance().GetMsprofCallback().msprofReporterCallback != nullptr) {
    MS_LOG(WARNING) << "Msprof reporter callback is exist, just ignore it.";
  } else {
    MS_LOG(INFO) << "GE register Msprof reporter callback.";
    ProfilingManager::GetInstance().SetMsprofReporterCallback(func);
    // Pass MsprofReporterCallback to runtime
    Status rt_ret = rtSetMsprofReporterCallback(func);
    if (rt_ret != PROF_SUCCESS) {
      MS_LOG(WARNING) << "Pass MsprofReporterCallback to runtime failed, ret: " << rt_ret;
      return rt_ret;
    }
    // Pass MsprofReporterCallback to hccl
  }
  return PROF_SUCCESS;
}

Status ProfCommandHandle(ProfCommandHandleType type, void *data, uint32_t len) {
  MS_LOG(INFO) << "ProfCommandHandle start, type:" << type;
  if (type == kProfCommandhandleInit) {
    auto cb_ret = ProfilingManager::GetInstance().PluginInit();
    if (cb_ret != PROF_SUCCESS) {
      MS_LOG(ERROR) << "Profiling plugin int failed.";
      return PROF_FAILED;
    }

    // call runtime profiler API
    auto module = GetProfilingModule();
    auto device_id = GetCurrentDeviceId();
    auto ret = rtProfilerStart(module, kProfilingDeviceNum, &device_id);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Call rtProfilerStart failed, ret:" << ret;
      return PROF_FAILED;
    }
  }
  return PROF_SUCCESS;
}

bool DoRegiste() {
  MS_LOG(INFO) << "VM profiling register start";
  return VMCallbackRegister::GetInstance().Register(RegProfCtrlCallback, RegProfSetDeviceCallback,
                                                    RegProfReporterCallback, ProfCommandHandle);
}
static bool doRegiste = DoRegiste();
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
