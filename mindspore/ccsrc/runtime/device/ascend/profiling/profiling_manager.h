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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_MANAGER_H_

#include <map>
#include <cstring>
#include <string>
#include <memory>
#include <nlohmann/json.hpp>
#include "utils/contract.h"
#include "utils/ms_context.h"
#include "toolchain/prof_callback.h"
#include "toolchain/prof_acl_api.h"
#include "toolchain/slog.h"
#include "runtime/base.h"
#include "profiler/device/profiling.h"

using std::map;
using std::string;

namespace mindspore {
namespace device {
namespace ascend {
struct MsprofCallback {
  MsprofCtrlCallback msprofCtrlCallback;
  MsprofSetDeviceCallback msprofSetDeviceCallback;
  MsprofReporterCallback msprofReporterCallback;
};

enum ProfCommandHandleType {
  kProfCommandhandleInit = 0,
  kProfCommandhandleStart,
  kProfCommandhandleStop,
  kProfCommandhandleFinalize,
  kProfCommandhandleModelSubscribe,
  kProfCommandhandleModelUnsubscribe
};

class ProfilingManager {
 public:
  static ProfilingManager &GetInstance();
  uint64_t GetJobId() const;
  bool ProfRegisterCtrlCallback() const;
  bool StartupProfiling(uint32_t device_id);
  bool StopProfiling() const;

  inline bool IsProfiling() const {
    auto profiler_manager = profiler::ProfilerManager::GetInstance();
    MS_EXCEPTION_IF_NULL(profiler_manager);
    return profiler_manager->GetProfilingEnableFlag();
  }
  Status PluginInit() const;
  void PluginUnInit() const;
  Status CallMsprofReport(NotNull<ReporterData *> reporter_data) const;
  const struct MsprofCallback &GetMsprofCallback() { return prof_cb_; }
  void SetMsprofCtrlCallback(MsprofCtrlCallback func) { prof_cb_.msprofCtrlCallback = func; }
  void SetMsprofReporterCallback(MsprofReporterCallback func) { prof_cb_.msprofReporterCallback = func; }
  void SetMsprofSetDeviceCallback(MsprofSetDeviceCallback func) { prof_cb_.msprofSetDeviceCallback = func; }
  Status GetProfConf(NotNull<MsprofGeOptions *> prof);
  void SetHcclEnabledBefProfilingEnabled() { hccl_enabled_bef_profiling_enabled_ = true; }

 protected:
  ProfilingManager();
  ~ProfilingManager() {}

 private:
  bool ProfStartUp(NotNull<MsprofGeOptions *> prof_conf) const;
  uint32_t device_id_;
  MsprofCallback prof_cb_;
  bool hccl_enabled_bef_profiling_enabled_;
  bool has_started_;
};

Status ProfCommandHandle(ProfCommandHandleType type);
rtError_t CtrlCallbackHandle(uint32_t rt_type, void *data, uint32_t /*len*/);
Status ProfCtrlSwitchHandle(void *data);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PROFILING_PROFILING_MANAGER_H_
