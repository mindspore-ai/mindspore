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
#ifndef MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_PROFILING_MANAGER_H_
#define MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_PROFILING_MANAGER_H_

#include <map>
#include <cstring>
#include <string>
#include <memory>
#include <nlohmann/json.hpp>
#include "utils/contract.h"
#include "utils/context/ms_context.h"

using std::map;
using std::string;
namespace mindspore {
namespace device {
namespace ascend {
class ProfilingEngineImpl;
class ProfilingManager {
 public:
  static ProfilingManager &GetInstance();
  uint64_t GetJobId() const;
  bool ReportProfilingData(const map<uint32_t, string> &op_taskId_map) const;
  bool StartupProfiling(uint32_t device_id);
  bool StopProfiling();

  inline bool IsProfiling() const {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    return context->enable_profiling();
  }

 protected:
  ProfilingManager();
  ~ProfilingManager() { prof_handle_ = nullptr; }

 private:
  bool ProfStartUp(NotNull<nlohmann::json *> json);
  std::shared_ptr<ProfilingEngineImpl> engine_0_;
  uint32_t device_id_;
  void *prof_handle_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_PROFILING_MANAGER_H_
