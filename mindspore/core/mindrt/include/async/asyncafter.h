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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_ASYNCAFTER_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_ASYNCAFTER_H

#include "async/async.h"

#include "timer/timertools.h"

constexpr mindspore::Duration MILLISECONDS = 1;
constexpr mindspore::Duration SECONDS = 1000;

namespace mindspore {
template <typename T>
Timer AsyncAfter(const Duration &duration, const AID &aid, void (T::*method)()) {
  return TimerTools::AddTimer(duration, aid, [=]() { Async(aid, method); });
}

template <typename T, typename Arg0, typename Arg1>
Timer AsyncAfter(const Duration &duration, const AID &aid, void (T::*method)(Arg0), Arg1 &&arg) {
  return TimerTools::AddTimer(duration, aid, [=]() { Async(aid, method, arg); });
}

template <typename T, typename... Args0, typename... Args1>
Timer AsyncAfter(const Duration &duration, const AID &aid, void (T::*method)(Args0...), Args1 &&... args) {
  std::function<void(Args0...)> f([=](Args0... args0) { Async(aid, method, args0...); });

  auto handler = std::bind(f, args...);

  return TimerTools::AddTimer(duration, aid, [=]() { Async(aid, std::move(handler)); });
}

};  // namespace mindspore

#endif
