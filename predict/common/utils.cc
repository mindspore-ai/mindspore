/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "common/utils.h"

namespace mindspore {
namespace predict {
uint64_t GetTimeUs() {
  struct timespec ts = {0, 0};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return 0;
  }
  // USECS_IN_SEC *NSECS_IN_USEC;
  auto retval = static_cast<uint64_t>((ts.tv_sec * USEC) + (ts.tv_nsec / MSEC));
  return retval;
}

static const unsigned int FP32_BIT_SIZE = 32;
static const unsigned int FP32_EXPONENT_BIAS = 127;
static const unsigned int FP32_SIGNIFICAND = 23;

static const unsigned int FP32_EXPONENT_MAX = 255;

static const unsigned int FP16_BIT_SIZE = 16;
static const unsigned int FP16_EXPONENT_BIAS = 15;
static const unsigned int FP16_SIGNIFICAND = 10;

static const int FP16_EXPONENT_MAX = 30;
static const int FP16_EXPONENT_MIN = -10;

float ShortToFloat32(int16_t srcValue) {
  uint16_t expHalf16 = srcValue & 0x7C00;
  int exp1 = static_cast<int>(expHalf16);
  uint16_t mantissa16 = srcValue & 0x03FF;
  int mantissa1 = static_cast<int>(mantissa16);
  int sign = static_cast<int>(srcValue & 0x8000);
  sign = sign << FP16_BIT_SIZE;

  // nan or inf
  if (expHalf16 == 0x7C00) {
    // nan
    if (mantissa16 > 0) {
      int res = (0x7FC00000 | sign);
      int *iRes = &res;
      MS_ASSERT(iRes != nullptr);
      auto fres = static_cast<float>(*iRes);
      return fres;
    }
    // inf
    int res = (0x7F800000 | sign);
    int *iRes = &res;
    MS_ASSERT(iRes != nullptr);
    auto fres = static_cast<float>(*iRes);
    return fres;
  }
  if (expHalf16 != 0) {
    exp1 += ((FP32_EXPONENT_BIAS - FP16_EXPONENT_BIAS) << FP16_SIGNIFICAND);  // exponents converted to float32 bias
    int res = (exp1 | mantissa1);
    res = res << (FP32_SIGNIFICAND - FP16_SIGNIFICAND);
    res = (res | sign);
    int *iRes = &res;

    auto fres = static_cast<float>(*iRes);
    return fres;
  }

  int xmm1 = exp1 > (1 << FP16_SIGNIFICAND) ? exp1 : (1 << FP16_SIGNIFICAND);
  xmm1 = (xmm1 << (FP32_SIGNIFICAND - FP16_SIGNIFICAND));
  xmm1 += ((FP32_EXPONENT_BIAS - FP16_EXPONENT_BIAS - FP16_SIGNIFICAND)
           << FP32_SIGNIFICAND);  // add the bias difference to xmm1
  xmm1 = xmm1 | sign;             // Combine with the sign mask

  auto res = static_cast<float>(mantissa1);  // Convert mantissa to float
  res *= static_cast<float>(xmm1);

  return res;
}

int16_t Float32ToShort(float srcValue) {
  auto srcValueBit = static_cast<unsigned int>(srcValue);
  int sign = srcValueBit >> (FP32_BIT_SIZE - 1);
  int mantissa = srcValueBit & 0x007FFFFF;
  // exponent
  int exp = ((srcValueBit & 0x7F800000) >> FP32_SIGNIFICAND) + FP16_EXPONENT_BIAS - FP32_EXPONENT_BIAS;
  int16_t res;
  if (exp > 0 && exp < FP16_EXPONENT_MAX) {
    // use rte rounding mode, round the significand, combine sign, exponent and significand into a short.
    res = (sign << (FP16_BIT_SIZE - 1)) | (exp << FP16_SIGNIFICAND) |
          ((mantissa + 0x00001000) >> (FP32_SIGNIFICAND - FP16_SIGNIFICAND));
  } else if (srcValueBit == 0) {
    res = 0;
  } else {
    if (exp <= 0) {
      if (exp < FP16_EXPONENT_MIN) {
        // value is less than min half float point
        res = 0;
      } else {
        // normalized single, magnitude is less than min normal half float point.
        mantissa = (mantissa | 0x00800000) >> (1 - exp);
        // round to nearest
        if ((mantissa & 0x00001000) > 0) {
          mantissa = mantissa + 0x00002000;
        }
        // combine sign & mantissa (exp is zero to get denormalized number)
        res = (sign << FP16_EXPONENT_BIAS) | (mantissa >> (FP32_SIGNIFICAND - FP16_SIGNIFICAND));
      }
    } else if (exp == (FP32_EXPONENT_MAX - FP32_EXPONENT_BIAS + FP16_EXPONENT_BIAS)) {
      if (mantissa == 0) {
        // input float is infinity, return infinity half
        res = (sign << FP16_EXPONENT_BIAS) | 0x7C00;
      } else {
        // input float is NaN, return half NaN
        res = (sign << FP16_EXPONENT_BIAS) | 0x7C00 | (mantissa >> (FP32_SIGNIFICAND - FP16_SIGNIFICAND));
      }
    } else {
      // exp > 0, normalized single, round to nearest
      if ((mantissa & 0x00001000) > 0) {
        mantissa = mantissa + 0x00002000;
        if ((mantissa & 0x00800000) > 0) {
          mantissa = 0;
          exp = exp + 1;
        }
      }
      if (exp > FP16_EXPONENT_MAX) {
        // exponent overflow - return infinity half
        res = (sign << FP16_EXPONENT_BIAS) | 0x7C00;
      } else {
        // combine sign, exp and mantissa into normalized half
        res = (sign << FP16_EXPONENT_BIAS) | (exp << FP16_SIGNIFICAND) |
              (mantissa >> (FP32_SIGNIFICAND - FP16_SIGNIFICAND));
      }
    }
  }
  return res;
}
std::string Remove(const std::string &from, const std::string &subStr, Mode mode) {
  std::string result = from;
  if (mode == PREFIX) {
    if (from.substr(0, subStr.length()) == subStr) {
      result = from.substr(subStr.size());
    }
  } else if (mode == SUFFIX) {
    if (from.rfind(subStr) == from.size() - subStr.size()) {
      result = from.substr(0, from.size() - subStr.size());
    }
  } else {
    size_t index;
    while ((index = result.find(subStr)) != std::string::npos) {
      result = result.erase(index, subStr.size());
    }
  }

  return result;
}

std::vector<std::string> StrSplit(const std::string &str, const std::string &pattern) {
  std::string::size_type pos;
  std::vector<std::string> result;
  std::string tmpStr(str + pattern);
  std::string::size_type size = tmpStr.size();

  for (std::string::size_type i = 0; i < size; i++) {
    pos = tmpStr.find(pattern, i);
    if (pos < size) {
      std::string s = tmpStr.substr(i, pos - i);
      result.push_back(s);
      i = pos + pattern.size() - 1;
    }
  }
  return result;
}

std::vector<std::string> Tokenize(const std::string &src, const std::string &delimiters,
                                  const Option<size_t> &maxTokenNum) {
  if (maxTokenNum.IsSome() && maxTokenNum.Get() == 0) {
    return {};
  }

  std::vector<std::string> tokens;
  size_t offset = 0;

  while (true) {
    size_t nonDelimiter = src.find_first_not_of(delimiters, offset);
    if (nonDelimiter == std::string::npos) {
      break;
    }
    size_t delimiter = src.find_first_of(delimiters, nonDelimiter);
    if (delimiter == std::string::npos || (maxTokenNum.IsSome() && tokens.size() == maxTokenNum.Get() - 1)) {
      tokens.push_back(src.substr(nonDelimiter));
      break;
    }

    tokens.push_back(src.substr(nonDelimiter, delimiter - nonDelimiter));
    offset = delimiter;
  }
  return tokens;
}

void ShortToFloat32(const int16_t *srcdata, float *dstdata, size_t elementSize) {
  MS_ASSERT(srcdata != nullptr);
  MS_ASSERT(dstdata != nullptr);
  for (size_t i = 0; i < elementSize; i++) {
    dstdata[i] = ShortToFloat32(srcdata[i]);
  }
}

void Float32ToShort(const float *srcdata, int16_t *dstdata, size_t elementSize) {
  MS_ASSERT(srcdata != nullptr);
  MS_ASSERT(dstdata != nullptr);
  for (size_t i = 0; i < elementSize; i++) {
    dstdata[i] = Float32ToShort(srcdata[i]);
  }
}
}  // namespace predict
}  // namespace mindspore
