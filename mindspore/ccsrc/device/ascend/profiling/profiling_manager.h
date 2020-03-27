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
using std::map;
using std::string;

namespace mindspore {
namespace device {
namespace ascend {
// PROFILING_CUSTOM_LOGID_START 3
const uint64_t kProfilingFpStartLogId = 1;
const uint64_t kProfilingBpEndLogId = 2;
const uint64_t kProfilingAllReduce1Start = 3;
const uint64_t kProfilingAllReduce1End = 4;
const uint64_t kProfilingAllReduce2Start = 5;
const uint64_t kProfilingAllReduce2End = 6;
const uint64_t kProfilingIterEndLogId = 255;

class ProfilingEngineImpl;
class ProfilingManager {
 public:
  static ProfilingManager &GetInstance();
  uint64_t GetJobId() const;
  bool ReportProfilingData(const map<uint32_t, string> &op_taskId_map) const;
  bool StartupProfiling(uint32_t device_id);
  bool StopProfiling() const;

  inline bool IsProfiling() const {
    const char *is_profiling = std::getenv("PROFILING_MODE");
    return (is_profiling != nullptr && (strcmp("true", is_profiling) == 0));
  }

 protected:
  ProfilingManager();
  ~ProfilingManager() { prof_handle_ = nullptr; }

 private:
  std::shared_ptr<ProfilingEngineImpl> engine_0_;
  uint32_t device_id_;
  void *prof_handle_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_ASCEND_PROFILING_PROFILING_MANAGER_H_
