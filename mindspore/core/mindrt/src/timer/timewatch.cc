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

#include "timer/timewatch.h"

namespace mindspore {

TimeWatch::TimeWatch() : duration(Now()) {}

TimeWatch::TimeWatch(const Duration &d) : duration(d) {}

TimeWatch::TimeWatch(const TimeWatch &that) : duration(that.duration) {}

TimeWatch::~TimeWatch() {}

Duration TimeWatch::Now() {
  struct timespec ts = {0, 0};
  (void)clock_gettime(CLOCK_MONOTONIC, &ts);
  Duration duration = ts.tv_sec * SECTOMILLI + (ts.tv_nsec / MICRTONANO) / MILLITOMICR;
  return duration;
}

TimeWatch TimeWatch::In(const Duration &duration) { return TimeWatch(Now() + duration); }

TimeWatch &TimeWatch::operator=(const TimeWatch &that) {
  if (&that != this) {
    duration = that.duration;
  }

  return *this;
}

TimeWatch &TimeWatch::operator=(const Duration &d) {
  duration = Now() + d;
  return *this;
}

bool TimeWatch::operator==(const TimeWatch &that) const { return duration == that.duration; }

bool TimeWatch::operator<(const TimeWatch &that) const { return duration < that.duration; }

bool TimeWatch::operator<=(const TimeWatch &that) const { return duration <= that.duration; }

Duration TimeWatch::Time() const { return duration; }

Duration TimeWatch::Remaining() const {
  Duration nowTime = Now();
  return duration > nowTime ? (duration - nowTime) : 0;
}

bool TimeWatch::Expired() const { return duration <= Now(); }

}  // namespace mindspore
