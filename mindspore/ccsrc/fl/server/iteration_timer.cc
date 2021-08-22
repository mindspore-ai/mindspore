/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "fl/server/iteration_timer.h"

namespace mindspore {
namespace fl {
namespace server {
void IterationTimer::Start(const std::chrono::milliseconds &duration) {
  if (running_.load()) {
    MS_LOG(WARNING) << "The timer already started.";
    return;
  }
  running_ = true;
  end_time_ = CURRENT_TIME_MILLI + duration;
  monitor_thread_ = std::thread([&]() {
    while (running_.load()) {
      if (CURRENT_TIME_MILLI > end_time_) {
        timeout_callback_(false, "");
        running_ = false;
      }
      // The time tick is 1 millisecond.
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
}

void IterationTimer::Stop() {
  running_ = false;
  if (monitor_thread_.joinable()) {
    monitor_thread_.join();
  }
}

void IterationTimer::SetTimeOutCallBack(const TimeOutCb &timeout_cb) {
  timeout_callback_ = timeout_cb;
  return;
}

bool IterationTimer::IsTimeOut(const std::chrono::milliseconds &timestamp) const {
  return timestamp > end_time_ ? true : false;
}

bool IterationTimer::IsRunning() const { return running_; }
}  // namespace server
}  // namespace fl
}  // namespace mindspore
