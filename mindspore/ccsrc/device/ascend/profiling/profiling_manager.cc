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

#include "device/ascend/profiling/profiling_manager.h"

#include <stdlib.h>
#include <vector>

#include <nlohmann/json.hpp>
#include "securec/include/securec.h"
#include "./prof_mgr_core.h"
#include "device/ascend/profiling/plugin_impl.h"
#include "device/ascend/profiling/profiling_engine_impl.h"
#include "utils/log_adapter.h"
#include "utils/context/ms_context.h"
#include "common/utils.h"
#include "utils/convert_utils.h"

using std::vector;
using Json = nlohmann::json;

namespace mindspore {
namespace device {
namespace ascend {
ProfilingManager &ProfilingManager::GetInstance() {
  static ProfilingManager inst;
  return inst;
}

ProfilingManager::ProfilingManager() : device_id_(0), prof_handle_(nullptr) {
  engine_0_ = std::make_shared<ProfilingEngineImpl>();
}

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
  auto reporter = PluginImpl::GetPluginReporter();
  if (reporter == nullptr) {
    MS_LOG(ERROR) << "No profiling data report!";
    return false;
  }
  MS_LOG(INFO) << "DistributeTask: op tasId map size = " << op_taskId_map.size();

  Msprof::Engine::ReporterData reporter_data = {};
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
    ret = reporter->Report(&reporter_data);
    if (ret != 0) {
      MS_LOG(ERROR) << "reporter data fail, errorno(" << ret << ")";
      return false;
    }
  }
  return true;
}

static std::vector<std::string> Split(const std::string &str, const char delim) {
  std::vector<std::string> elems;

  if (str.empty()) {
    elems.emplace_back("");
    return elems;
  }

  std::stringstream ss(str);
  std::string item;

  while (getline(ss, item, delim)) {
    elems.push_back(item);
  }
  auto str_size = str.size();
  if (str_size > 0 && str[str_size - 1] == delim) {
    elems.emplace_back("");
  }

  return elems;
}

bool ProfilingManager::StartupProfiling(uint32_t device_id) {
  auto is_profiling = IsProfiling();
  if (!is_profiling) {
    MS_LOG(INFO) << "No need profiling. please export PROFILING_MODE and in train mode.";
    return true;
  }
  device_id_ = device_id;
  // exp: export PROFILING_MODE=true
  // export PROFILING_OPTIONS=training_trace
  const char *prof_options_str = std::getenv("PROFILING_OPTIONS");
  // register Framework to profiling
  int result = Msprof::Engine::RegisterEngine("Framework", engine_0_.get());
  if (result != 0) {
    MS_LOG(ERROR) << "Register profiling Engine failed.";
    return false;
  }

  if (prof_options_str != nullptr) {
    const string prof_options_str_tmp = prof_options_str;
    vector<string> opts = Split(prof_options_str_tmp, ':');
    if (!opts.empty()) {
      // current one docker only use one device`
      Json p_device;

      // JOBID
      auto job_id = GetJobId();
      p_device["jobID"] = std::to_string(job_id);

      // device_id
      p_device["deviceID"] = std::to_string(device_id);

      // features:'training_trace', 'task_trace'  etc
      Json features;
      for (vector<string>::size_type i = 0; i < opts.size(); i++) {
        Json f;
        f["name"] = opts[i];
        features[i] = f;
      }
      p_device["features"] = features;

      // only one device, but sProfMgrStartUp API require for device list
      Json devices;
      devices[0] = p_device;

      Json startCfg;
      startCfg["startCfg"] = devices;

      // convert json to string
      std::stringstream ss;
      ss << startCfg;
      std::string cfg = ss.str();

      MS_LOG(INFO) << "profiling config " << cfg;

      // call profiling startup API
      ProfMgrCfg prof_cfg = {cfg};
      prof_handle_ = ProfMgrStartUp(&prof_cfg);
      if (prof_handle_ == nullptr) {
        MS_LOG(ERROR) << "Startup profiling failed.";
        return false;
      }
    }
  }

  return true;
}

bool ProfilingManager::StopProfiling() const {
  MS_LOG(INFO) << "StopProfiling";
  if (!IsProfiling()) {
    MS_LOG(INFO) << "No need profiling. please export PROFILING_MODE and in train mode.";
    return true;
  }
  Msprof::Engine::Reporter *reporter = PluginImpl::GetPluginReporter();
  if (reporter != nullptr) {
    MS_LOG(INFO) << "report data end, ret = " << reporter->Flush();
  }

  if (prof_handle_ != nullptr) {
    int result = ProfMgrStop(prof_handle_);
    if (result != 0) {
      MS_LOG(ERROR) << "ProfMgr stop return fail:" << result << ".";
      return false;
    }
  }

  return true;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
