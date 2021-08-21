/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef SOFT_DP_TOOLS_H
#define SOFT_DP_TOOLS_H

#include <cstdint>
#include <string>
#include <utility>

template <typename T1, typename T2>
T1 AlignUp(T1 num, T2 align) {
  if (num % align) {
    num = (num / align + 1) * align;
  }

  return num;
}

template <typename T1, typename T2>
T1 AlignDown(T1 num, T2 align) {
  if (num % align) {
    num = num / align * align;
  }

  return num;
}

template <typename T>
bool IsInTheScope(T num, T left_point, T right_point) {
  return num >= left_point && num <= right_point;
}

template <typename T>
T TruncatedFunc(T num, T min, T max) {
  if (num < min) {
    return min;
  }
  if (num > max) {
    return max;
  }

  return num;
}

std::pair<bool, std::string> GetRealpath(const std::string &path);

bool IsDirectory(const std::string &path);

#endif  // SOFT_DP_TOOLS_H
