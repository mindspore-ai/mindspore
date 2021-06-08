/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include "profiler/device/ascend/ascend_profiling.h"
#include <cstdarg>
#include <iomanip>
#include <utility>
#include "utils/log_adapter.h"
#include "./securec.h"
namespace mindspore {
namespace profiler {
namespace ascend {
const int kMaxEvents = 10000;
const int kEventDescMax = 256;
const int kMaxEventTypes = 8;
const int kIndent = 8;

AscendProfiler::AscendProfiler() : counter_(0) { Reset(); }

void AscendProfiler::Dump(std::ostream &output_stream) {
  MS_LOG(INFO) << "start dump async profiling info";
  if (events_.empty()) {
    return;
  }

  auto first_evt = events_[0];
  auto start = first_evt.timestamp;
  std::vector<decltype(start)> prev_timestamps;
  prev_timestamps.resize(kMaxEventTypes, start);

  for (uint32_t i = 0; i < counter_; ++i) {
    auto &evt = events_[i];
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(evt.timestamp - start).count();
    auto &prev_ts = prev_timestamps[evt.event_type];
    auto cost = std::chrono::duration_cast<std::chrono::microseconds>(evt.timestamp - prev_ts).count();
    prev_ts = evt.timestamp;
    output_stream << std::setw(kIndent) << elapsed << "\t\t" << cost << "\t\t" << evt.desc << std::endl;
  }

  events_.clear();
  MS_LOG(INFO) << "end";
}

void AscendProfiler::Reset() {
  counter_ = 0;
  events_.clear();
  events_.resize(kMaxEvents);
}
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
