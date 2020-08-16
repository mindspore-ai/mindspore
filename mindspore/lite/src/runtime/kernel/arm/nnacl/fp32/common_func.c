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

#include "nnacl/fp32/common_func.h"

#ifndef __aarch64__
void MatrixAdd(const float *a_ptr, const float *b_ptr, float *dst, size_t a_stride, size_t b_stride, size_t c_stride,
               size_t row, size_t col) {
  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++) {
      int a_index = c * a_stride + r * C4NUM;
      int b_index = c * b_stride + r * C4NUM;
      int c_index = c * c_stride + r * C4NUM;
      for (int i = 0; i < C4NUM; i++) {
        dst[c_index + i] = a_ptr[a_index + i] + b_ptr[b_index + i];
      }
    }
  }
  return;
}

void MatrixSub(const float *a_ptr, const float *b_ptr, float *dst, size_t a_stride, size_t b_stride, size_t c_stride,
               size_t row, size_t col) {
  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++) {
      int a_index = c * a_stride + r * C4NUM;
      int b_index = c * b_stride + r * C4NUM;
      int c_index = c * c_stride + r * C4NUM;
      for (int i = 0; i < C4NUM; i++) {
        dst[c_index + i] = a_ptr[a_index + i] - b_ptr[b_index + i];
      }
    }
  }
  return;
}
#endif

void MatrixMultiAdd(float *c11, float *c12, float *c21, float *c22, float *x_ptr, size_t row, size_t col,
                    size_t c_stride, size_t x_stride) {
  /* U2 = P1 + P6 */
  MatrixAdd(x_ptr, c12, c12, x_stride, c_stride, c_stride, row, col);
  /* U3 = U2 + P7 */
  MatrixAdd(c12, c21, c21, c_stride, c_stride, c_stride, row, col);
  /* U4 = U2 + P5 */
  MatrixAdd(c12, c22, c12, c_stride, c_stride, c_stride, row, col);
  /* U7 = U3 + P5 */
  MatrixAdd(c21, c22, c22, c_stride, c_stride, c_stride, row, col);
  /* U5 = U4 + P3 */
  MatrixAdd(c12, c11, c12, c_stride, c_stride, c_stride, row, col);
  return;
}

void PostConvFuncComm(const float *src_ptr_, float *out_ptr, const float *bias_ptr, size_t output_channel,
                      size_t plane_size, size_t stride, bool is_relu, bool is_relu6, int size) {
  if (size == 0) {
    return;
  }
  for (int oc = 0; oc < output_channel; oc++) {
    int oc_div = oc / size, oc_mod = oc % size;
    for (int hw = 0; hw < plane_size; hw++) {
      int src_index = oc_div * size * plane_size + hw * size + oc_mod;
      int dst_index = hw * stride + oc;
      float value = src_ptr_[src_index];
      if (bias_ptr != NULL) {
        value = value + bias_ptr[oc];
      }
      value = (is_relu || is_relu6) ? (MSMAX(0.f, value)) : (value);
      value = (is_relu6) ? (MSMIN(6.f, value)) : (value);
      out_ptr[dst_index] = value;
    }
  }
  return;
}

void PostConvFuncFp32C4(const float *c4_out_ptr, float *out_ptr, const float *bias_ptr, size_t output_channel,
                        size_t plane_size, size_t stride, bool is_relu, bool is_relu6) {
#ifndef ENABLE_ARM64
  PostConvFuncComm(c4_out_ptr, out_ptr, bias_ptr, output_channel, plane_size, stride, is_relu, is_relu6, C4NUM);
#else
  if (bias_ptr != NULL) {
    if (is_relu) {
      C4BiasAddRelu(out_ptr, c4_out_ptr, bias_ptr, output_channel, plane_size, stride * sizeof(float));
    } else if (is_relu6) {
      C4BiasAddRelu6(out_ptr, c4_out_ptr, bias_ptr, output_channel, plane_size, stride * sizeof(float));
    } else {
      C4BiasAdd(out_ptr, c4_out_ptr, bias_ptr, output_channel, plane_size, stride * sizeof(float));
    }
  } else {
    if (is_relu) {
      C4Relu(out_ptr, c4_out_ptr, output_channel, plane_size, stride * sizeof(float));
    } else if (is_relu6) {
      C4Relu6(out_ptr, c4_out_ptr, output_channel, plane_size, stride * sizeof(float));
    } else {
      // do nothing
    }
  }
#endif
  return;
}

void PostConvFuncFp32C8(const float *c8_out_ptr, float *out_ptr, const float *bias_ptr, size_t output_channel,
                        size_t plane_size, size_t stride, bool is_relu, bool is_relu6) {
#ifndef ENABLE_ARM64
  PostConvFuncComm(c8_out_ptr, out_ptr, bias_ptr, output_channel, plane_size, stride, is_relu, is_relu6, C8NUM);
#else
  size_t oc8mod = output_channel % C8NUM;
  size_t oc8div = output_channel - oc8mod;
  size_t stride_size = stride * sizeof(float);
  size_t relu_type = is_relu ? 1 : 0;
  relu_type = is_relu6 ? 2 : relu_type;
  PostFuncBiasReluC8(out_ptr, c8_out_ptr, bias_ptr, oc8div, oc8mod, plane_size, stride_size, relu_type);
#endif
  return;
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
  int exp1 = (int)(expHalf16);
  uint16_t mantissa16 = srcValue & 0x03FF;
  int mantissa1 = (int)(mantissa16);
  int sign = (int)(srcValue & 0x8000);
  sign = sign << FP16_BIT_SIZE;

  // nan or inf
  if (expHalf16 == 0x7C00) {
    // nan
    if (mantissa16 > 0) {
      int res = (0x7FC00000 | sign);
      int *iRes = &res;
      auto fres = (float)(*iRes);
      return fres;
    }
    // inf
    int res = (0x7F800000 | sign);
    int *iRes = &res;
    auto fres = (float)(*iRes);
    return fres;
  }
  if (expHalf16 != 0) {
    exp1 += ((FP32_EXPONENT_BIAS - FP16_EXPONENT_BIAS) << FP16_SIGNIFICAND);  // exponents converted to float32 bias
    int res = (exp1 | mantissa1);
    res = res << (FP32_SIGNIFICAND - FP16_SIGNIFICAND);
    res = (res | sign);
    int *iRes = &res;
    auto fres = (float)(*iRes);
    return fres;
  }

  int xmm1 = exp1 > (1 << FP16_SIGNIFICAND) ? exp1 : (1 << FP16_SIGNIFICAND);
  xmm1 = (xmm1 << (FP32_SIGNIFICAND - FP16_SIGNIFICAND));
  xmm1 += ((FP32_EXPONENT_BIAS - FP16_EXPONENT_BIAS - FP16_SIGNIFICAND)
    << FP32_SIGNIFICAND);  // add the bias difference to xmm1
  xmm1 = xmm1 | sign;             // Combine with the sign mask

  auto res = (float)(mantissa1);  // Convert mantissa to float
  int *ixmm1 = NULL;
  ixmm1 = &xmm1;
  res *= (float)(*ixmm1);

  return res;
}

// __gnu_f2h_ieee
int16_t Float32ToShort(float srcValue) {
  float *psrcValue = NULL;
  psrcValue = &srcValue;
  auto srcValueBit = (unsigned int)(*psrcValue);
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
