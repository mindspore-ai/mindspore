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

#include "nnacl/fp32/common_func_fp32.h"

void PostConvFuncComm(const float *src_ptr_, float *out_ptr, const float *bias_ptr, size_t output_channel,
                      size_t plane_size, size_t plane_stride, size_t oc_stride, ActType relu_type, int size) {
  int oc_div = 0, oc_mod = 0;
  for (int oc = 0; oc < output_channel; oc++) {
    if (size != 0) {
      oc_div = oc / size;
      oc_mod = oc % size;
    } else {
      return;
    }
    for (int hw = 0; hw < plane_size; hw++) {
      int src_index = oc_div * size * plane_stride + hw * size + oc_mod;
      int dst_index = hw * oc_stride + oc;
      float value = src_ptr_[src_index];
      if (bias_ptr != NULL) {
        value = value + bias_ptr[oc];
      }
      value = (relu_type == ActType_Relu || relu_type == ActType_Relu6) ? (MSMAX(0.f, value)) : (value);
      value = (relu_type == ActType_Relu6) ? (MSMIN(6.f, value)) : (value);
      out_ptr[dst_index] = value;
    }
  }
  return;
}

void PostConvFuncFp32C8(const float *c8_out_ptr, float *out_ptr, const float *bias_ptr, size_t output_channel,
                        size_t plane_size, size_t stride, size_t relu_type) {
#if !defined(ENABLE_ARM) && !defined(ENABLE_SSE)
  PostConvFuncComm(c8_out_ptr, out_ptr, bias_ptr, output_channel, plane_size, plane_size, stride, relu_type, C8NUM);
#else
  size_t oc8mod = output_channel % C8NUM;
  size_t oc8div = output_channel - oc8mod;
  size_t stride_size = stride * sizeof(float);
  PostFuncBiasReluC8(out_ptr, c8_out_ptr, bias_ptr, oc8div, oc8mod, plane_size, stride_size, relu_type);
#endif
  return;
}

void PostConvFuncFp32C4(const float *c4_out_ptr, float *out_ptr, const float *bias_ptr, size_t output_channel,
                        size_t plane_size, size_t plane_stride, size_t relu_type) {
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  size_t oc4mod = output_channel % C4NUM;
  size_t oc4div = output_channel - oc4mod;
  size_t stride_size = (plane_stride - plane_size) * C4NUM * sizeof(float);
  PostFuncBiasReluC4(out_ptr, c4_out_ptr, bias_ptr, oc4div, oc4mod, plane_size, stride_size, relu_type);
#else
  PostConvFuncComm(c4_out_ptr, out_ptr, bias_ptr, output_channel, plane_size, plane_stride, output_channel, relu_type,
                   C4NUM);
#endif
  return;
}

#if !defined(ENABLE_ARM) && !defined(ENABLE_SSE)
void WinogradTransLeft(const float *S, const float *B, float *M, size_t w, size_t h, size_t k, size_t length) {
  const int unitStep = 4 * length;
  for (int y = 0; y < h; ++y) {
    float *dstY = M + y * w * unitStep;
    for (int x = 0; x < w; ++x) {
      float *dstX = dstY + x * unitStep;
      const float *srcX = S + x * unitStep;
      memset(dstX, 0, unitStep * sizeof(float));
      for (int i = 0; i < k; ++i) {
        float b = B[i * h + y];
        const float *srcY = srcX + i * w * unitStep;
        if (0.0f == b) {
          continue;
        }
        for (int j = 0; j < unitStep; ++j) {
          dstX[j] += srcY[j] * b;
        }
      }
    }
  }
}

// M = S * B , M = w*h * l, S = k*h * l, B = w*k
void WinogradTransRight(const float *S, const float *B, float *M, size_t w, size_t h, size_t k, size_t length) {
  const int unitStep = 4 * length;
  for (int y = 0; y < h; ++y) {
    float *dstY = M + y * w * unitStep;
    const float *srcY = S + y * k * unitStep;

    for (int x = 0; x < w; ++x) {
      float *dstX = dstY + x * unitStep;
      memset(dstX, 0, unitStep * sizeof(float));
      for (int i = 0; i < k; ++i) {
        const float *srcX = srcY + i * unitStep;
        float b = B[i * h + x];
        if (0.0f == b) {
          continue;
        }
        for (int j = 0; j < unitStep; ++j) {
          dstX[j] += srcX[j] * b;
        }
      }
    }
  }
}
#endif

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

static const unsigned int FP32_BIT_SIZE = 32;
static const unsigned int FP32_EXPONENT_BIAS = 127;
static const unsigned int FP32_SIGNIFICAND = 23;

static const unsigned int FP32_EXPONENT_MAX = 255;

static const unsigned int FP16_BIT_SIZE = 16;
static const unsigned int FP16_EXPONENT_BIAS = 15;
static const unsigned int FP16_SIGNIFICAND = 10;

static const int FP16_EXPONENT_MAX = 30;
static const int FP16_EXPONENT_MIN = -10;

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
