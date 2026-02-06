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
#include <vector>
#include <algorithm>
#include <string>

#include "utils/log_adapter.h"

namespace mindspore {
const size_t kGBToByte = 1024 << 20;
const size_t kMBToByte = 1024 << 10;

inline int SizeToInt(size_t u) {
  if (u > static_cast<size_t>((std::numeric_limits<int>::max)())) {
    MS_LOG(INTERNAL_EXCEPTION) << "The size_t value(" << u << ") exceeds the maximum value of int.";
  }
  return static_cast<int>(u);
}

inline uint32_t SizeToUint(size_t u) {
  if (u > static_cast<size_t>((std::numeric_limits<uint32_t>::max)())) {
    MS_LOG(INTERNAL_EXCEPTION) << "The size_t value(" << u << ") exceeds the maximum value of uint32_t.";
  }
  return static_cast<uint32_t>(u);
}

inline int64_t SizeToLong(size_t u) {
  if (u > static_cast<size_t>((std::numeric_limits<int64_t>::max)())) {
    MS_LOG(INTERNAL_EXCEPTION) << "The size_t value(" << u << ") exceeds the maximum value of int64_t.";
  }
  return static_cast<int64_t>(u);
}

inline uint64_t SizeToUlong(size_t u) { return static_cast<uint64_t>(u); }

inline size_t IntToSize(int u) {
  if (u < 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "The int value(" << u << ") is less than 0.";
  }
  return static_cast<size_t>(u);
}

inline size_t LongToSize(int64_t u) {
  if (u < 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "The int64_t value(" << u << ") is less than 0.";
  }
  if (static_cast<uint64_t>(u) > std::numeric_limits<size_t>::max()) {
    MS_LOG(INTERNAL_EXCEPTION) << "The int value(" << u << ") exceeds the maximum value of size_t.";
  }
  return static_cast<size_t>(u);
}

inline size_t LongToSizeClipNeg(int64_t u) { return u < 0 ? 0 : LongToSize(u); }

inline std::vector<size_t> LongVecToSizeVec(const std::vector<int64_t> &vec) {
  std::vector<size_t> result;
  result.reserve(vec.size());
  (void)std::transform(vec.begin(), vec.end(), std::back_inserter(result), LongToSize);
  return result;
}

inline uint32_t LongToUint(int64_t u) {
  if (u < 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "The int64_t value(" << u << ") is less than 0.";
  }
  if (u > static_cast<int64_t>((std::numeric_limits<uint32_t>::max)())) {
    MS_LOG(INTERNAL_EXCEPTION) << "The int64_t value(" << u << ") exceeds the maximum value of uint32_t.";
  }
  return static_cast<uint32_t>(u);
}

inline size_t FloatToSize(float u) {
  if (u < 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "The float value(" << u << ") is less than 0.";
  }

  if (u > static_cast<float>((std::numeric_limits<size_t>::max)())) {
    MS_LOG(INTERNAL_EXCEPTION) << "The float value(" << u << ") exceeds the maximum value of size_t.";
  }
  return static_cast<size_t>(u);
}
inline float IntToFloat(int32_t v) { return static_cast<float>(v); }

inline int FloatToInt(float u) {
  if (u > static_cast<float>((std::numeric_limits<int>::max)())) {
    MS_LOG(INTERNAL_EXCEPTION) << "The float value(" << u << ") exceeds the maximum value of int.";
  }
  return static_cast<int>(u);
}

inline int FloatToLong(float u) {
  if (u > static_cast<float>((std::numeric_limits<int64_t>::max)())) {
    MS_LOG(INTERNAL_EXCEPTION) << "The float value(" << u << ") exceeds the maximum value of int64_t.";
  }
  return static_cast<int64_t>(u);
}

inline int64_t DoubleToLong(double u) {
  if (u > static_cast<double>((std::numeric_limits<int64_t>::max)())) {
    MS_LOG(INTERNAL_EXCEPTION) << "The double value(" << u << ") exceeds the maximum value of int64_t.";
  }
  return static_cast<int64_t>(u);
}

inline float SizeToFloat(size_t v) { return static_cast<float>(v); }

inline double LongToDouble(int64_t v) { return static_cast<double>(v); }

inline float LongToFloat(int64_t v) { return static_cast<float>(v); }

inline double FloatToDouble(float v) { return static_cast<double>(v); }

inline uint32_t IntToUint(int32_t u) {
  if (u < 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "The int32_t value(" << u << ") is less than 0.";
  }
  return static_cast<uint32_t>(u);
}

inline int32_t UintToInt(uint32_t u) {
  if (u > static_cast<uint32_t>((std::numeric_limits<int32_t>::max)())) {
    MS_LOG(INTERNAL_EXCEPTION) << "The uint32_t value(" << u << ") exceeds the maximum value of int32_t.";
  }
  return static_cast<int32_t>(u);
}

inline uint64_t LongToUlong(int64_t u) {
  if (u < 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "The int64_t value(" << u << ") is less than 0.";
  }
  return static_cast<uint64_t>(u);
}

inline int32_t LongToInt(int64_t u) {
  if (u > static_cast<int64_t>((std::numeric_limits<int32_t>::max)())) {
    MS_LOG(INTERNAL_EXCEPTION) << "The size_t value(" << u << ") exceeds the maximum value of int.";
  }
  return static_cast<int32_t>(u);
}

inline int64_t IntToLong(int32_t v) { return static_cast<int64_t>(v); }

inline int64_t UlongToLong(uint64_t u) {
  if (u > static_cast<uint64_t>((std::numeric_limits<int64_t>::max)())) {
    MS_LOG(INTERNAL_EXCEPTION) << "The uint64_t value(" << u << ") exceeds the maximum value of int64_t.";
  }
  return static_cast<int64_t>(u);
}

inline unsigned int UlongToUint(uint64_t u) {
  if (u > static_cast<uint64_t>((std::numeric_limits<unsigned int>::max)())) {
    MS_LOG(INTERNAL_EXCEPTION) << "The size_t value(" << u << ") exceeds the maximum value of unsigned int.";
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

inline uint32_t Uint32tMulWithOverflowCheck(uint32_t a, uint32_t b) {
  uint32_t out = a * b;
  if (a != 0) {
    if ((out / a) != b) {
      MS_LOG(EXCEPTION) << "Mul: a(" << a << ") * b(" << b << ") result is overflow";
    }
  }
  return out;
}

inline size_t SizetAddWithOverflowCheck(size_t x, size_t y) {
  size_t sum = x + y;
  if (sum < x || sum < y) {
    MS_LOG(EXCEPTION) << "Add: a(" << x << ") + b(" << y << ") result is overflow";
  }
  return sum;
}

inline uint32_t Uint32tAddWithOverflowCheck(uint32_t x, uint32_t y) {
  uint32_t sum = x + y;
  if (sum < x || sum < y) {
    MS_LOG(EXCEPTION) << "Add: a(" << x << ") + b(" << y << ") result is overflow";
  }
  return sum;
}

inline uint8_t *AddressOffset(void *address, size_t offset) {
  MS_EXCEPTION_IF_NULL(address);
  return static_cast<uint8_t *>(address) + offset;
}

inline size_t CalAddressOffset(void *dst_address, void *ori_address) {
  MS_EXCEPTION_IF_NULL(dst_address);
  MS_EXCEPTION_IF_NULL(ori_address);
  return static_cast<uint8_t *>(dst_address) - static_cast<uint8_t *>(ori_address);
}

inline std::vector<int64_t> Convert2Int(const std::vector<size_t> &v) {
  std::vector<int64_t> result;
  (void)std::transform(v.begin(), v.end(), std::back_inserter(result), SizeToInt);
  return result;
}

inline std::vector<int64_t> Convert2Long(const std::vector<size_t> &v) {
  std::vector<int64_t> result;
  (void)std::transform(v.begin(), v.end(), std::back_inserter(result), SizeToLong);
  return result;
}

inline std::vector<size_t> Convert2SizeT(const std::vector<int64_t> &v) {
  std::vector<size_t> result;
  (void)std::transform(v.begin(), v.end(), std::back_inserter(result), LongToSize);
  return result;
}

inline std::vector<size_t> Convert2SizeTClipNeg(const std::vector<int64_t> &v) {
  std::vector<size_t> result;
  auto ConvertFunc = [](int64_t v) -> size_t { return v < 0 ? 0 : static_cast<int64_t>(v); };
  (void)std::transform(v.begin(), v.end(), std::back_inserter(result), ConvertFunc);
  return result;
}

inline bool ShapeVectorIsSame(const std::vector<int64_t> &shape, const std::vector<int64_t> &check_shape) {
  if (shape.size() != check_shape.size()) {
    return false;
  } else {
    for (size_t idx = 0; idx < shape.size(); ++idx) {
      if (shape[idx] != check_shape[idx]) {
        return false;
      }
    }
  }
  return true;
}

inline std::string ShapeVectorToStr(const std::vector<int64_t> &shp) {
  std::ostringstream buffer;
  bool f_begin = true;
  buffer << "(";
  for (auto &x : shp) {
    if (!f_begin) {
      buffer << ", ";
    } else {
      f_begin = false;
    }
    buffer << x;
  }
  buffer << ")";
  return buffer.str();
}
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_CONVERT_UTILS_BASE_H_
