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

#include "nnacl/nnacl_common.h"

typedef union float32_bits {
  unsigned int u;
  float f;
} float32_bits;

float ShortToFloat32(uint16_t src_value) {
  const float32_bits magic = {113 << 23};
  const unsigned int shifted_exp = 0x7c00 << 13;
  float32_bits o;

  o.u = (src_value & 0x7fff) << 13;
  unsigned int exp = shifted_exp & o.u;
  o.u += (127 - 15) << 23;

  if (exp == shifted_exp) {
    o.u += (128 - 16) << 23;
  } else if (exp == 0) {
    o.u += 1 << 23;
    o.f -= magic.f;
  }

  o.u |= (src_value & 0x8000) << 16;
  return o.f;
}

uint16_t Float32ToShort(float src_value) {
  float *psrcValue = NULL;
  psrcValue = &src_value;
  unsigned int srcValueBit = (unsigned int)(*psrcValue);
  unsigned int sign = srcValueBit >> (FP32_BIT_SIZE - 1);
  unsigned int mantissa = srcValueBit & 0x007FFFFF;
  // exponent
  int exp = ((srcValueBit & 0x7F800000) >> FP32_SIGNIFICAND) + FP16_EXPONENT_BIAS - FP32_EXPONENT_BIAS;
  uint16_t res;
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
