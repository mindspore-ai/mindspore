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

#ifndef __LITEBUS_TIMETOOLS_HPP__
#define __LITEBUS_TIMETOOLS_HPP__

#include <atomic>
#include <list>
#include <map>
#include <set>

#include "timer/duration.h"
#include "timer/timer.h"

namespace mindspore {
class TimerTools {
 public:
  static bool Initialize();
  static void Finalize();
  static Timer AddTimer(const Duration &duration, const AID &aid, const std::function<void()> &thunk);
  static bool Cancel(const Timer &timer);
  static std::atomic_bool g_initStatus;
};
}  // namespace mindspore

#endif
