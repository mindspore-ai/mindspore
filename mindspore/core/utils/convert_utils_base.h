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

#ifndef MINDSPORE_CORE_UTILS_CONVERT_UTILS_BASE_H_
#define MINDSPORE_CORE_UTILS_CONVERT_UTILS_BASE_H_

#include <limits>
#include <memory>

#include "utils/log_adapter.h"

namespace mindspore {
inline int SizeToInt(size_t u) {
  if (u > static_cast<size_t>((std::numeric_limits<int>::max)())) {
    MS_LOG(EXCEPTION) << "The size_t value(" << u << ") exceeds the maximum value of int.";
  }
  return static_cast<int>(u);
}

inline uint32_t SizeToUint(size_t u) {
  if (u > static_cast<size_t>((std::numeric_limits<uint32_t>::max)())) {
    MS_LOG(EXCEPTION) << "The size_t value(" << u << ") exceeds the maximum value of uint32_t.";
  }
  return static_cast<uint32_t>(u);
}

inline int64_t SizeToLong(size_t u) {
  if (u > static_cast<size_t>((std::numeric_limits<int64_t>::max)())) {
    MS_LOG(EXCEPTION) << "The size_t value(" << u << ") exceeds the maximum value of int64_t.";
  }
  return static_cast<int64_t>(u);
}

inline uint64_t SizeToUlong(size_t u) { return static_cast<uint64_t>(u); }

inline size_t IntToSize(int u) {
  if (u < 0) {
    MS_LOG(WARNING) << "The int value(" << u << ") is less than 0.";
    return SIZE_MAX;
  }
  return static_cast<size_t>(u);
}

inline size_t LongToSize(int64_t u) {
  if (u < 0) {
    MS_LOG(EXCEPTION) << "The int64_t value(" << u << ") is less than 0.";
  }
  return static_cast<size_t>(u);
}

inline size_t FloatToSize(float u) {
  if (u < 0) {
    MS_LOG(EXCEPTION) << "The float value(" << u << ") is less than 0.";
  }

  if (u > static_cast<float>((std::numeric_limits<size_t>::max)())) {
    MS_LOG(EXCEPTION) << "The float value(" << u << ") exceeds the maximum value of size_t.";
  }
  return static_cast<size_t>(u);
}
inline float IntToFloat(int32_t v) { return static_cast<float>(v); }

inline int FloatToInt(float u) {
  if (u > static_cast<float>((std::numeric_limits<int>::max)())) {
    MS_LOG(EXCEPTION) << "The float value(" << u << ") exceeds the maximum value of int.";
  }
  return static_cast<int>(u);
}

inline float SizeToFloat(size_t v) { return static_cast<float>(v); }

inline double LongToDouble(int64_t v) { return static_cast<double>(v); }

inline double FloatToDouble(float v) { return static_cast<double>(v); }

inline uint32_t IntToUint(int32_t u) {
  if (u < 0) {
    MS_LOG(EXCEPTION) << "The int32_t value(" << u << ") is less than 0.";
  }
  return static_cast<uint32_t>(u);
}

inline int32_t UintToInt(uint32_t u) {
  if (u > static_cast<uint32_t>((std::numeric_limits<int32_t>::max)())) {
    MS_LOG(EXCEPTION) << "The uint32_t value(" << u << ") exceeds the maximum value of int32_t.";
  }
  return static_cast<int32_t>(u);
}

inline uint64_t LongToUlong(int64_t u) {
  if (u < 0) {
    MS_LOG(EXCEPTION) << "The int64_t value(" << u << ") is less than 0.";
  }
  return static_cast<uint64_t>(u);
}

inline int32_t LongToInt(int64_t u) {
  if (u > static_cast<int64_t>((std::numeric_limits<int32_t>::max)())) {
    MS_LOG(EXCEPTION) << "The size_t value(" << u << ") exceeds the maximum value of int.";
  }
  return static_cast<int32_t>(u);
}

inline int64_t UlongToLong(uint64_t u) {
  if (u > static_cast<uint64_t>((std::numeric_limits<int64_t>::max)())) {
    MS_LOG(EXCEPTION) << "The uint64_t value(" << u << ") exceeds the maximum value of int64_t.";
  }
  return static_cast<int64_t>(u);
}

inline unsigned int UlongToUint(uint64_t u) {
  if (u > static_cast<uint64_t>((std::numeric_limits<unsigned int>::max)())) {
    MS_LOG(EXCEPTION) << "The size_t value(" << u << ") exceeds the maximum value of unsigned int.";
  }
  return static_cast<unsigned int>(u);
}

inline int IntMulWithOverflowCheck(int a, int b) {
  int out = a * b;
  if (a != 0) {
    bool overflow = ((out / a) != b);
    if (overflow) {
      MS_LOG(EXCEPTION) << "Mul: a(" << a << ") * b(" << b << ") result is overflow";
    }
  }
  return out;
}

inline int64_t LongMulWithOverflowCheck(int64_t a, int64_t b) {
  int64_t out = a * b;
  if (a != 0) {
    bool overflow = ((out / a) != b);
    if (overflow) {
      MS_LOG(EXCEPTION) << "Mul: a(" << a << ") * b(" << b << ") result is overflow";
    }
  }
  return out;
}

inline size_t SizetMulWithOverflowCheck(size_t a, size_t b) {
  size_t out = a * b;
  if (a != 0) {
    if ((out / a) != b) {
      MS_LOG(EXCEPTION) << "Mul: a(" << a << ") * b(" << b << ") result is overflow";
    }
  }
  return out;
}

inline uint8_t *AddressOffset(void *address, size_t offset) {
  MS_EXCEPTION_IF_NULL(address);
  return static_cast<uint8_t *>(address) + offset;
}
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_CONVERT_UTILS_BASE_H_
