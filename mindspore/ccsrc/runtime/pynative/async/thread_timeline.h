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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_THREAD_TIMELINE_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_THREAD_TIMELINE_H_

#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <mutex>
#include "utils/profile.h"
#include "utils/ms_utils.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace pynative {
enum kTimelineType {
  kTypePyTask = 0,
  kTypeFpTask,
  kTypeBpTask,
  kTypeBackendTask,
  kTypeWaitValueTask,
  kTypeWaitAbsTask,
  kTypeGraphTask,
  kTypeDoCastTask,
  kTypeInferTask,
  kTypeGetOutputTask,
  kTypeOpGradTask,
};

struct BACKEND_EXPORT TimelineObj {
  TimelineObj(std::string name, std::thread::id tid, kTimelineType type, uint64_t ts_start, uint64_t ts_end);
  std::string name_;
  std::thread::id tid_;
  kTimelineType type_;
  uint64_t ts_start_;
  uint64_t ts_end_;
};

// Add a timeline tool for PyNative multi-stage pipeline.
// The tool is thread-safe and lock-free when recording the time data.
// Each thread has a thread_local data and need to merge the thread_local data
// before the thread exits.
class BACKEND_EXPORT ThreadTimeline {
 public:
  static ThreadTimeline &GetInstance();

  void Init();

  // Gets the current timestamp in microseconds.
  // The unit of timestamps in the Timeline JSON file is required to be microseconds.
  static uint64_t GetTime();

  void Record(const TimelineObj &obj) const noexcept;

  // Parse the Timeline JSON.
  void Parse();

  // Combine thread_local data before thread exit.
  void Combine();

 private:
  ThreadTimeline() = default;
  ~ThreadTimeline() = default;
  DISABLE_COPY_AND_ASSIGN(ThreadTimeline);

  void DumpJson(const std::string &json) const;

  // The data after Combine thread_local data.
  std::unordered_map<std::thread::id, std::vector<TimelineObj>> timeline_data_map_;
  // Initialize the starting timestamp.
  uint64_t start_{0};
  std::mutex combine_mutex_;
};

// To add the Guard class using the C++ RAII mechanism.
// Only need to add a line to the function that needs to be counted time-consuming.
// The Guard class will record the start timestamp when constructing,
// and the end timestamp will be recorded when Guard destroying.
class BACKEND_EXPORT TimelineGuard {
 public:
  TimelineGuard(std::string op_name, kTimelineType timeline_type);
  ~TimelineGuard();

  static void enable() { enable_ = true; }

 private:
  std::string op_name_;
  kTimelineType timeline_type_;
  uint64_t ts_start_;
  inline static bool enable_{false};
};
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_THREAD_TIMELINE_H_
