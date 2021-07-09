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

#ifndef MINDSPORE_CCSRC_FL_SERVER_ITERATION_TIMER_H_
#define MINDSPORE_CCSRC_FL_SERVER_ITERATION_TIMER_H_

#include <chrono>
#include <atomic>
#include <thread>
#include <functional>
#include "fl/server/common.h"

namespace mindspore {
namespace fl {
namespace server {
// IterationTimer controls the time window for the purpose of eliminating trailing time of each iteration.
class IterationTimer {
 public:
  IterationTimer() : running_(false), end_time_(0) {}
  ~IterationTimer() = default;

  // Start timing. The timer will stop after parameter 'duration' milliseconds.
  void Start(const std::chrono::milliseconds &duration);

  // Caller could use this method to manually stop timing, otherwise the timer will keep timing until it expires.
  void Stop();

  // Set the callback which will be called when the timer expires.
  void SetTimeOutCallBack(const TimeOutCb &timeout_cb);

  // Judge whether current timestamp is out of time window's range since the Start function is called.
  bool IsTimeOut(const std::chrono::milliseconds &timestamp) const;

  // Judge whether the timer is keeping timing.
  bool IsRunning() const;

 private:
  // The running state for the timer.
  std::atomic<bool> running_;

  // The timestamp in millesecond at which the timer should stop timing.
  std::chrono::milliseconds end_time_;

  // The thread that keeps timing and call timeout_callback_ when the timer expires.
  std::thread monitor_thread_;
  TimeOutCb timeout_callback_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_ITERATION_TIMER_H_
