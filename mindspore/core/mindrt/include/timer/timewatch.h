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

#ifndef __LITEBUS_TIMEWATCH_HPP__
#define __LITEBUS_TIMEWATCH_HPP__

#include "timer/duration.h"
namespace mindspore {
constexpr Duration MICRTONANO = 1000;
constexpr Duration MILLITOMICR = 1000;
constexpr Duration SECTOMILLI = 1000;

class TimeWatch {
 public:
  TimeWatch();

  TimeWatch(const Duration &duration);

  TimeWatch(const TimeWatch &that);
  ~TimeWatch();

  // Constructs a Time instance that is the 'duration' from now.
  static TimeWatch In(const Duration &duration);

  static Duration Now();

  TimeWatch &operator=(const TimeWatch &that);

  TimeWatch &operator=(const Duration &duration);

  bool operator==(const TimeWatch &that) const;

  bool operator<(const TimeWatch &that) const;

  bool operator<=(const TimeWatch &that) const;

  // Returns the value of the timewatch as a Duration object.
  Duration Time() const;

  // Returns the amount of time remaining.
  Duration Remaining() const;

  // return true if the time expired.
  bool Expired() const;

 private:
  Duration duration;
};

}  // namespace mindspore

#endif
