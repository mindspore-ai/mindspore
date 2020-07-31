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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_QUANTIZATION_FIXED_POINT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_QUANTIZATION_FIXED_POINT_H_

#include <limits.h>
#include "include/infer_log.h"
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif

// returns the high-32 bits of a * b with rounding
// assume that a and b is divided by 2^31, who fall into [-1, 1]
// so the mantissa of a * b is (a / 2^31) * (b / 2^31) * 2^31= (a * b) / 2^31
// actually we compute 2 * a * b / 2^32
// and take 32 bits of mantissa for rounding
inline int SaturatingRoundingDoublingHighMul(int a, int b) {
  if (a == INT_MIN && b == INT_MIN) {
    return INT_MAX;
  }
  int64_t ab = ((int64_t)a) * ((int64_t)b);
  int64_t rounding = ab >= 0 ? (1ll << 30) : (1ll - (1ll << 30));
  // do not apply right shift to potential negetive values
  int ab_mantissa = (int) ((ab + rounding) / (1ll << 31));
  return ab_mantissa;
}

// division by a 2^exponent with rounding
// or arithmetic right shift with rouding
inline int RoundingDivideByPOT(int x, int exponent) {
  MS_ASSERT(exponent >= 0);
  MS_ASSERT(exponent <= 31);
  const int mask = (1ll << exponent) - 1;
  const int remainder = x & mask;
  const int threshold = (mask >> 1) + (x < 0 ? 1 : 0);
  return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

inline int MultiplyByQuantizedMultiplier(int32_t value, int32_t multiplier, int32_t left_shift, int32_t right_shift) {
  return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(value * (1 << left_shift), multiplier), -right_shift);
}

#ifdef ENABLE_NEON
inline int32x4_t RoundingDivideByPOTInt32x4(int32x4_t x, int exponent) {
  const int32x4_t shift_vec = vdupq_n_s32(-exponent);
  const int32x4_t fixup = vshrq_n_s32(vandq_s32(x, shift_vec), 31);
  const int32x4_t fixed_up_x = vqaddq_s32(x, fixup);
  return vrshlq_s32(fixed_up_x, shift_vec);
}

inline int32x4_t SaturatingRoundingDoublingHighMulInt32x4(int32x4_t a, int32x4_t b) {
  return vqrdmulhq_s32(a, b);
}
#endif

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_QUANTIZATION_FIXED_POINT_H_

