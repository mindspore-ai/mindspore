/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/profiling/profiling_manager.h"
#include <cstdlib>
#include "common/util/error_manager/error_manager.h"
#include "securec/include/securec.h"
#include "./prof_mgr_core.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "include/common/utils/convert_utils.h"
#include "runtime/base.h"
#include <nlohmann/json.hpp>
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_utils.h"
#include "plugin/device/ascend/hal/profiler/ascend_profiling.h"

using mindspore::device::ascend::ProfilingUtils;

namespace {
constexpr Status PROF_SUCCESS = 0;
constexpr Status PROF_FAILED = 0xFFFFFFFF;
}  // namespace

namespace mindspore {
namespace device {
namespace ascend {
ProfilingManager &ProfilingManager::GetInstance() {
  static ProfilingManager inst{};
  return inst;
}

ProfilingManager::ProfilingManager()
    : device_id_(0), prof_cb_({0}), cur_state_(kProfilingInvalid), profiling_path_("") {}

uint64_t ProfilingManager::GetJobId() const { return 0; }

uint64_t GetProfilingModule() {
  return PROF_MODEL_EXECUTE_MASK | PROF_RUNTIME_API_MASK | PROF_RUNTIME_TRACE_MASK | PROF_SCHEDULE_TIMELINE_MASK |
         PROF_SCHEDULE_TRACE_MASK | PROF_TASK_TIME_MASK | PROF_SUBTASK_TIME_MASK | PROF_AICPU_TRACE_MASK |
         PROF_AICORE_METRICS_MASK | PROF_AIVECTORCORE_METRICS_MASK | PROF_MODEL_LOAD_MASK;
}

Status ProfilingManager::PluginInit() const {
  int32_t ret = MsprofReportData(static_cast<uint32_t>(MsprofReporterModuleId::MSPROF_MODULE_FRAMEWORK),
                                 static_cast<uint32_t>(MsprofReporterCallbackType::MSPROF_REPORTER_INIT), nullptr, 0);
  if (ret != UintToInt(PROF_SUCCESS)) {
    MS_LOG(ERROR) << "MsprofReporter init failed, ret: " << ret;
    return PROF_FAILED;
  }
  return PROF_SUCCESS;
}

void ProfilingManager::PluginUnInit() const {
  int32_t cb_ret =
    MsprofReportData(static_cast<uint32_t>(MsprofReporterModuleId::MSPROF_MODULE_FRAMEWORK),
                     static_cast<uint32_t>(MsprofReporterCallbackType::MSPROF_REPORTER_UNINIT), nullptr, 0);
  if (cb_ret != 0) {
    MS_LOG(WARNING) << "profiling plugin uninit failed, ret:%d" << cb_ret;
  }
}

Status ProfilingManager::GetProfConf(const NotNull<MsprofGeOptions *> prof) const {
  string job_id = std::to_string(GetJobId());
  if (memcpy_s(prof->jobId, sizeof(prof->jobId), job_id.c_str(), strlen(job_id.c_str())) != EOK) {
    MS_LOG(ERROR) << "Copy job_id failed.";
    return PROF_FAILED;
  }

  auto profiler_manager = profiler::ProfilerManager::GetInstance();
  if (profiler_manager == nullptr) {
    MS_LOG(ERROR) << "Profiler manager instance is nullptr.";
    return PROF_FAILED;
  }
  const string prof_options_str = profiler_manager->GetProfilingOptions();

  const nlohmann::json options_all = nlohmann::json::parse(prof_options_str);
  nlohmann::json options_for_cann;
  options_for_cann["output"] = options_all["output"];
  options_for_cann["fp_point"] = options_all["fp_point"];
  options_for_cann["bp_point"] = options_all["bp_point"];
  options_for_cann["training_trace"] = options_all["training_trace"];
  options_for_cann["task_trace"] = options_all["task_trace"];
  options_for_cann["aic_metrics"] = options_all["aic_metrics"];
  options_for_cann["aicpu"] = options_all["aicpu"];

  const string options_for_cann_str = options_for_cann.dump();
  if (memcpy_s(prof->options, MSPROF_OPTIONS_DEF_LEN_MAX, options_for_cann_str.c_str(), options_for_cann_str.size()) !=
      EOK) {
    MS_LOG(ERROR) << "Copy profiling_options failed";
    return PROF_FAILED;
  }
  return PROF_SUCCESS;
}

bool ProfilingManager::InitProfiling(const std::string &profiling_path, uint32_t device_id) {
  profiling_path_ = profiling_path;
  device_id_ = device_id;

  bool ret = ProfRegisterCtrlCallback();
  if (ret == false) {
    return ret;
  }

  return true;
}

bool ProfilingManager::ProfRegisterCtrlCallback() const {
  rtError_t rt_ret = MsprofRegisterCallback(GE, CtrlCallbackHandle);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtProfRegisterCtrlCallback failed.";
    return false;
  }

  return true;
}

rtError_t CtrlCallbackHandle(uint32_t rt_type, void *data, uint32_t /* len */) {
  if (rt_type == RT_PROF_CTRL_REPORTER) {
    ProfilingManager::GetInstance().SetMsprofReporterCallback(reinterpret_cast<MsprofReporterCallback>(data));
    MS_LOG(INFO) << "Set MsprofReporterCallback success.";
  } else if (rt_type == RT_PROF_CTRL_SWITCH) {
    Status ret = ProfCtrlSwitchHandle(data);
    if (ret != PROF_SUCCESS) {
      MS_LOG(ERROR) << "Start runtime profiler failed.";
    }
  }

  return RT_ERROR_NONE;
}

Status ProfilingManager::CallMsprofReport(const NotNull<ReporterData *> reporter_data) const {
  int32_t ret = MsprofReportData(static_cast<int32_t>(MsprofReporterModuleId::MSPROF_MODULE_FRAMEWORK),
                                 static_cast<int32_t>(MsprofReporterCallbackType::MSPROF_REPORTER_REPORT),
                                 static_cast<void *>(reporter_data.get()), sizeof(ReporterData));

  if (ret != UintToInt(PROF_SUCCESS)) {
    return PROF_FAILED;
  }
  return PROF_SUCCESS;
}

Status ProfilingManager::ProfHandleInit() {
  MS_LOG(INFO) << "Begin to init profiling. Current profiling state is " << cur_state_;
  cur_state_ = kProfilingInit;
  auto cb_ret = ProfilingManager::GetInstance().PluginInit();
  if (cb_ret != PROF_SUCCESS) {
    MS_LOG(ERROR) << "Failed to init profiling.";
    return PROF_FAILED;
  }

  return PROF_SUCCESS;
}

Status ProfilingManager::ProfHandleStart() {
  MS_LOG(INFO) << "Begin to start profiling. Current profiling state is " << cur_state_;
  cur_state_ = kProfilingStart;

  // Report graph data if there is any cache data.
  ProfilingUtils::ReportAllGraphProfilingData();

  return PROF_SUCCESS;
}

Status ProfilingManager::ProfHandleStop() {
  MS_LOG(INFO) << "Begin to stop profiling. Current profiling state is " << cur_state_;
  cur_state_ = kProfilingStop;
  return PROF_SUCCESS;
}

Status ProfilingManager::ProfHandleFinalize() {
  MS_LOG(INFO) << "Begin to finalize profiling. Current profiling state is " << cur_state_;
  cur_state_ = kProfilingFinalize;
  ProfilingManager::GetInstance().PluginUnInit();

  return PROF_SUCCESS;
}

Status ProfilingManager::ProfCommandHandle(ProfCommandHandleType type) {
  // Only need process "Init"/“Start”/“Stop”/“Finalize”
  if (type == kProfCommandhandleInit) {
    return ProfHandleInit();
  } else if (type == kProfCommandhandleStart) {
    return ProfHandleStart();
  } else if (type == kProfCommandhandleStop) {
    return ProfHandleStop();
  } else if (type == kProfCommandhandleFinalize) {
    return ProfHandleFinalize();
  }

  MS_LOG(ERROR) << "Receive invalid profiling type " << type << ". Current profiling state is << " << cur_state_;
  return PROF_FAILED;
}

Status ProfCtrlSwitchHandle(void *data) {
  if (data == nullptr) {
    MS_LOG(ERROR) << "Ctrl switch handl data is nullptr.";
    return PROF_FAILED;
  }

  rtProfCommandHandle_t *prof_config_param = reinterpret_cast<rtProfCommandHandle_t *>(data);
  auto type = static_cast<ProfCommandHandleType>(prof_config_param->type);
  return ProfCommandHandle(type);
}

Status ProfCommandHandle(ProfCommandHandleType type) { return ProfilingManager::GetInstance().ProfCommandHandle(type); }

void ProfilingManager::QueryHashId(const int32_t &device_id, const std::string &src_str, uint64_t *hash_id) {
  // when some profiling data size exceeds the specified size, query its hashId instead.
  MsprofHashData hash_data{};
  hash_data.deviceId = device_id;
  hash_data.dataLen = src_str.size();
  hash_data.data = reinterpret_cast<unsigned char *>(const_cast<char *>(src_str.c_str()));

  const int32_t ret = MsprofReportData(static_cast<int32_t>(MsprofReporterModuleId::MSPROF_MODULE_FRAMEWORK),
                                       static_cast<int32_t>(MsprofReporterCallbackType::MSPROF_REPORTER_HASH),
                                       &hash_data, sizeof(MsprofHashData));
  if (ret != UintToInt(PROF_SUCCESS)) {
    MS_LOG(EXCEPTION) << "[Profiling] Query hash id of long string failed, src string is " << src_str.c_str()
                      << ", ret is " << ret << "." << GetErrorMessage(true);
  }

  *hash_id = hash_data.hashId;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
