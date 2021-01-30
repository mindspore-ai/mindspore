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

#ifndef __LITEBUS_TIMER_HPP__
#define __LITEBUS_TIMER_HPP__

#include "actor/aid.h"
#include "timer/timewatch.h"

namespace mindspore {
class Timer {
 public:
  Timer();
  ~Timer();
  bool operator==(const Timer &that) const;
  // run this timer's thunk.
  void operator()() const;
  TimeWatch GetTimeWatch() const;
  AID GetTimerAID() const;
  uint64_t GetTimerID() const;

 private:
  friend class TimerTools;

  Timer(uint64_t timerId, const TimeWatch &timeWatch, const AID &timeAid, const std::function<void()> &handler);

  uint64_t id;
  TimeWatch t;
  AID aid;
  std::function<void()> thunk;
};

}  // namespace mindspore

#endif
