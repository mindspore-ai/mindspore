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

#include <sys/time.h>
#include <string>
#include "utils/log_adapter.h"

namespace mindspore {
// export GetTimeString for all sub modules
std::string GetTimeString() {
  const size_t buf_len = 80;
  char buf[buf_len] = {'\0'};
#if defined(_WIN32) || defined(_WIN64)
  time_t time_seconds = time(0);
  struct tm now_time;
  localtime_s(&now_time, &time_seconds);
  sprintf_s(buf, buf_len, "%d-%d-%d %d:%d:%d", now_time.tm_year + 1900, now_time.tm_mon + 1, now_time.tm_mday,
            now_time.tm_hour, now_time.tm_min, now_time.tm_sec);
#else
  struct timeval cur_time;
  (void)gettimeofday(&cur_time, nullptr);

  struct tm now;
  (void)localtime_r(&cur_time.tv_sec, &now);
  (void)strftime(buf, buf_len, "%Y-%m-%d-%H:%M:%S", &now);  // format date and time
  // set micro-second
  buf[27] = '\0';
  int idx = 26;
  const int ten = 10;
  auto num = cur_time.tv_usec;
  constexpr int interval_number = 3;
  for (int i = 5; i >= 0; i--) {
    buf[idx--] = static_cast<char>(num % ten + '0');
    num /= ten;
    if (i % interval_number == 0) {
      buf[idx--] = '.';
    }
  }
#endif
  return std::string(buf);
}
}  // namespace mindspore
