/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_PROFILER_PROFILER_H_
#define MINDSPORE_CCSRC_RUNTIME_PROFILER_PROFILER_H_

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <thread>
#include <mutex>
#include "nlohmann/json.hpp"
#include "utils/os.h"
#include "utils/ms_utils.h"
#include "utils/hash_map.h"
#include "utils/log_adapter.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace runtime {
static const char kDefaultOpName[] = "Default";
static const size_t kPercent = 100;

enum class ProfilerModule { kPython, kRuntime, kPynative, kKernel, kOther };

enum class ProfilerEvent {
  kDefault,
  kKernelInfer,
  kKernelResize,
  kKernelLaunch,
  kKernelUpdate,
  kGraphLaunch,
  kInputProcess,
  kOutputProcess,
  kWaitTaskFinish,
  kPreLaunch,
  kPostLaunch,
  kSendOutput,
  kMemoryAlloc,
  kMemoryFree,
  kCopyData,
  kStreamSync,

  // Inner event is not counted in the total time.
  kKernelInferInner,
  kKernelInferDataSync,
};

#define PROFILER_START(start_time)                                          \
  do {                                                                      \
    if (runtime::ProfilerAnalyzer::GetInstance().profiler_enable()) {       \
      start_time = runtime::ProfilerAnalyzer::GetInstance().GetTimeStamp(); \
    }                                                                       \
  } while (0);

#define PROFILER_END(start_time, module, event, op_name, is_inner_event)                                           \
  do {                                                                                                             \
    if (runtime::ProfilerAnalyzer::GetInstance().profiler_enable()) {                                              \
      auto end_time = runtime::ProfilerAnalyzer::GetInstance().GetTimeStamp();                                     \
      auto brief_name = runtime::ProfilerAnalyzer::GetInstance().GetBriefName(op_name);                            \
      runtime::ProfilerAnalyzer::GetInstance().RecordData(                                                         \
        std::make_shared<runtime::ProfilerData>(module, event, brief_name, is_inner_event, start_time, end_time)); \
    }                                                                                                              \
  } while (0);

// Record the profiler data by the constructor and destructor of this class.
class BACKEND_EXPORT ProfilerRecorder {
 public:
  ProfilerRecorder(ProfilerModule module, ProfilerEvent event, const std::string &op_name, bool is_inner_event = false);
  ~ProfilerRecorder();

 private:
  ProfilerModule module_;
  ProfilerEvent event_;
  std::string op_name_;
  uint64_t start_time_;
  bool is_inner_event_;
};

struct ProfilerData {
  ProfilerData(ProfilerModule module, ProfilerEvent event, const std::string &op_name, bool is_inner_event,
               uint64_t start_time, uint64_t end_time)
      : module_(module),
        event_(event),
        op_name_(op_name),
        is_inner_event_(is_inner_event),
        start_time_(start_time),
        end_time_(end_time) {
    dur_time_ = end_time - start_time;
    tid_ = std::this_thread::get_id();
    pid_ = getpid();
  }
  ProfilerModule module_;
  ProfilerEvent event_;
  std::string op_name_;
  bool is_inner_event_;
  uint64_t start_time_;
  uint64_t end_time_;
  uint64_t dur_time_;
  std::thread::id tid_;
  int32_t pid_;
};
using ProfilerDataPtr = std::shared_ptr<ProfilerData>;

struct ProfilerStatisticsInfo {
  explicit ProfilerStatisticsInfo(const std::string &name, bool is_inner_info = false)
      : name_(name), is_inner_info_(is_inner_info), count_(0), total_time_(0), average_time_(0), percent_(0) {}
  std::string name_;
  bool is_inner_info_;
  size_t count_;
  uint64_t total_time_;
  double average_time_;
  double percent_;

  void AccumulateTime(uint64_t time) {
    total_time_ += time;
    ++count_;
  }
  void Average() {
    MS_EXCEPTION_IF_ZERO("count", count_);
    average_time_ = static_cast<double>(total_time_) / count_;
  }
  void CalculatePercent(uint64_t step_total_time) {
    MS_EXCEPTION_IF_ZERO("step_total_time", step_total_time);
    percent_ = (static_cast<double>(total_time_) / step_total_time) * kPercent;
  }
};
using ProfilerStatisticsInfoPtr = std::shared_ptr<ProfilerStatisticsInfo>;

struct ProfilerEventInfo {
  ProfilerStatisticsInfoPtr event_statistics_info_;
  mindspore::HashMap<std::string, ProfilerStatisticsInfoPtr> op_infos_;
};
using ProfilerEventInfoPtr = std::shared_ptr<ProfilerEventInfo>;

struct ProfilerModuleInfo {
  ProfilerStatisticsInfoPtr module_statistics_info_;
  std::map<ProfilerEvent, ProfilerEventInfoPtr> event_infos_;
};
using ProfilerModuleInfoPtr = std::shared_ptr<ProfilerModuleInfo>;

class BACKEND_EXPORT ProfilerAnalyzer {
 public:
  static ProfilerAnalyzer &GetInstance() noexcept;

  // Increase step at the step begin.
  void StartStep();
  // Analyze and output profiler data at the step end.
  void EndStep();

  void Clear();

  // The used by ProfilerRecorder to record data.
  bool profiler_enable() const { return profiler_enable_; }
  void RecordData(const ProfilerDataPtr &data);
  uint64_t GetTimeStamp();
  std::string GetBriefName(const std::string &scope_name);

 private:
  ProfilerAnalyzer() = default;
  ~ProfilerAnalyzer() = default;
  DISABLE_COPY_AND_ASSIGN(ProfilerAnalyzer);

  void Initialize();

  // Process data.
  void SaveJsonData(const ProfilerDataPtr &data);
  void AnalyzeSummaryData(const ProfilerDataPtr &data);
  void AnalyzeEventSummaryData(std::map<ProfilerEvent, ProfilerEventInfoPtr> *const event_infos,
                               const ProfilerDataPtr &data);
  void AnalyzeOpSummaryData(mindspore::HashMap<std::string, ProfilerStatisticsInfoPtr> *const op_infos,
                            const ProfilerDataPtr &data);

  // Dump data.
  void DumpJsonData();
  void DumpDetailData();
  void DumpSummaryData();
  void DumpModuleSummaryData(std::stringstream &string_stream);
  void DumpEventSummaryData(const std::map<ProfilerEvent, ProfilerEventInfoPtr> &event_infos,
                            std::stringstream &string_stream);
  void DumpOpSummaryData(const mindspore::HashMap<std::string, ProfilerStatisticsInfoPtr> &op_infos,
                         std::stringstream &string_stream);

  // The relevant members of step.
  size_t step_{0};
  uint64_t step_total_time_{0};
  std::vector<ProfilerDataPtr> data_;
  std::mutex data_mutex_;
  nlohmann::json json_infos_;
  // The data analyzed level is module-->event-->op.
  std::map<ProfilerModule, ProfilerModuleInfoPtr> module_infos_;

  // Save file name.
  std::string json_file_name_;
  std::string summary_info_file_name_;
  std::string detail_info_file_name_;

  // The relevant members of init.
  int show_top_num_{0};
  bool profiler_enable_{false};
  bool init_{false};
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_PROFILER_PROFILER_H_
