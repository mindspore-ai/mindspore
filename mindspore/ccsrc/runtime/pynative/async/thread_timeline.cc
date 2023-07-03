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

#include "runtime/pynative/async/thread_timeline.h"

#include <map>
#include <utility>
#include "nlohmann/json.hpp"
#include "include/common/debug/common.h"

namespace mindspore {
namespace pynative {
namespace {
enum class kTimelinePh { kTimelineBegin = 0, kTimelineEnd };
constexpr auto kStrName = "name";
constexpr auto kStrPh = "ph";
constexpr auto kStrPid = "pid";
constexpr auto kStrTid = "tid";
constexpr auto kStrTs = "ts";
std::string GetPhString(kTimelinePh ph) {
  static const std::map<kTimelinePh, std::string> kPhMap = {{kTimelinePh::kTimelineBegin, "B"},
                                                            {kTimelinePh::kTimelineEnd, "E"}};
  return kPhMap.find(ph)->second;
}

std::string GetTypeString(kTimelineType type) {
  static const std::map<kTimelineType, std::string> kTidMap = {{kTimelineType::kTypePyTask, "py_"},
                                                               {kTimelineType::kTypeFpTask, "fp_"},
                                                               {kTimelineType::kTypeBpTask, "bp_"},
                                                               {kTimelineType::kTypeBackendTask, "backend_"},
                                                               {kTimelineType::kTypeWaitValueTask, "WaitValue"},
                                                               {kTimelineType::kTypeWaitAbsTask, "WaitAbs"},
                                                               {kTimelineType::kTypeGraphTask, "Graph_"},
                                                               {kTimelineType::kTypeDoCastTask, "DoCast_"},
                                                               {kTimelineType::kTypeInferTask, "Infer_"},
                                                               {kTimelineType::kTypeGetOutputTask, "GetOut_"},
                                                               {kTimelineType::kTypeOpGradTask, "OpGrad_"}};
  return kTidMap.find(type)->second;
}

// Each thread has independent data. So it's lock-free.
thread_local std::vector<TimelineObj> t_timeline_data_;
}  // namespace

uint64_t ThreadTimeline::GetTime() {
  auto now = std::chrono::steady_clock::now();
  int64_t us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
  return static_cast<uint64_t>(us);
}

TimelineObj::TimelineObj(std::string name, std::thread::id tid, kTimelineType type, uint64_t ts_start, uint64_t ts_end)
    : name_(std::move(name)), tid_(tid), type_(type), ts_start_(ts_start), ts_end_(ts_end) {}

ThreadTimeline &ThreadTimeline::GetInstance() {
  static ThreadTimeline instance;
  return instance;
}

void ThreadTimeline::Init() {
  TimelineGuard::enable();
  start_ = ThreadTimeline::GetTime();
}

void ThreadTimeline::Record(const TimelineObj &obj) const noexcept { (void)t_timeline_data_.emplace_back(obj); }

void ThreadTimeline::Combine() {
  thread_local static bool combined = false;
  if (combined) {
    return;
  }
  std::unique_lock<std::mutex> lock(combine_mutex_);
  for (auto &item : t_timeline_data_) {
    (void)timeline_data_map_[item.tid_].emplace_back(item);
  }
  combined = true;
}

std::string TidToString(const std::thread::id &tid) {
  std::stringstream ss;
  ss << tid;
  return ss.str();
}

void ThreadTimeline::Parse() {
  if (timeline_data_map_.empty()) {
    return;
  }
  nlohmann::json out;
  for (auto &timeline_data : timeline_data_map_) {
    for (auto &item : timeline_data.second) {
      nlohmann::json record_start;
      record_start[kStrName] = GetTypeString(item.type_) + item.name_;
      record_start[kStrPh] = GetPhString(kTimelinePh::kTimelineBegin);
      record_start[kStrPid] = std::to_string(getpid());
      record_start[kStrTid] = TidToString(item.tid_);
      record_start[kStrTs] = item.ts_start_ - start_;

      nlohmann::json record_end;
      record_end[kStrName] = GetTypeString(item.type_) + item.name_;
      record_end[kStrPh] = GetPhString(kTimelinePh::kTimelineEnd);
      record_end[kStrPid] = std::to_string(getpid());
      record_end[kStrTid] = TidToString(item.tid_);
      record_end[kStrTs] = item.ts_end_ - start_;

      (void)out.emplace_back(record_start);
      (void)out.emplace_back(record_end);
    }
  }
  DumpJson(out.dump());
  timeline_data_map_.clear();
}

void ThreadTimeline::DumpJson(const std::string &json) const {
  auto ts = ThreadTimeline::GetTime();
  std::string filename = "async_timeline_" + std::to_string(ts) + ".json";
  (void)Common::SaveStringToFile(filename, json);
}

TimelineGuard::TimelineGuard(std::string op_name, kTimelineType timeline_type) {
  if (enable_) {
    op_name_ = std::move(op_name);
    timeline_type_ = timeline_type;
    ts_start_ = ThreadTimeline::GetTime();
  }
}

TimelineGuard::~TimelineGuard() {
  if (enable_) {
    auto &profile = ThreadTimeline::GetInstance();
    auto ts_end = ThreadTimeline::GetTime();
    profile.Record(TimelineObj(op_name_, std::this_thread::get_id(), timeline_type_, ts_start_, ts_end));
  }
}
}  // namespace pynative
}  // namespace mindspore
