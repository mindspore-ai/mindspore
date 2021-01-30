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

#include "timer/timer.h"

namespace mindspore {

Timer::Timer() : id(0), t(TimeWatch()), aid(AID()), thunk(&abort) {}

Timer::~Timer() {}

bool Timer::operator==(const Timer &that) const { return id == that.id; }

void Timer::operator()() const { thunk(); }

TimeWatch Timer::GetTimeWatch() const { return t; }

AID Timer::GetTimerAID() const { return aid; }

uint64_t Timer::GetTimerID() const { return id; }

Timer::Timer(uint64_t timerId, const TimeWatch &timeWatch, const AID &timeAid, const std::function<void()> &handler)
    : id(timerId), t(timeWatch), aid(timeAid), thunk(handler) {}

}  // namespace mindspore
