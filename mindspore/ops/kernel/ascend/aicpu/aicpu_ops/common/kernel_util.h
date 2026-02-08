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
#ifndef AICPU_OPS_AICPU_COMMON_KERNEL_UTIL_H_
#define AICPU_OPS_AICPU_COMMON_KERNEL_UTIL_H_
#include <climits>
#include <limits>
#include "common/kernel_log.h"
#ifndef AICPU_VISIBILITY_API
#define AICPU_VISIBILITY_API __attribute__((visibility("default")))
inline size_t IntToSize(int u) {
  if (u < 0) {
    AICPU_LOGE("The int value [%d] is less than 0.", u);
    return SIZE_MAX;
  }
  return static_cast<size_t>(u);
}

inline int SizeToInt(size_t u) {
  if (u > static_cast<size_t>((std::numeric_limits<int>::max)())) {
    AICPU_LOGE("The size_t value [%lu] exceeds the maximum value of int.", u);
    return INT_MAX;
  }
  return static_cast<int>(u);
}

inline size_t LongToSize(int64_t u) {
  if (u < 0) {
    AICPU_LOGE("The int64_t value [%ld] is less than 0.", u);
    return SIZE_MAX;
  }
  return static_cast<size_t>(u);
}

inline int32_t LongToInt(int64_t u) {
  if (u > static_cast<int64_t>((std::numeric_limits<int32_t>::max)())) {
    AICPU_LOGE("The int64_t value [%ld] exceeds the maximum value of int32_t.", u);
    return INT_MAX;
  } else if (u < static_cast<int64_t>((std::numeric_limits<int32_t>::min)())) {
    AICPU_LOGE("The int64_t value [%ld] is less than the minimum value of int32_t.", u);
    return INT_MIN;
  }
  return static_cast<int32_t>(u);
}
#endif
#endif  // AICPU_OPS_AICPU_COMMON_KERNEL_UTIL_H_
