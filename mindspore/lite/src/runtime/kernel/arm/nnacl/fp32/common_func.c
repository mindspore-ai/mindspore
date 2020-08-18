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

union float32_bits {
  unsigned int u;
  float f;
};
typedef union float32_bits float32_bits;

float ShortToFloat32(uint16_t srcValue) {
  const float32_bits magic = {113 << 23};
  const unsigned int shifted_exp = 0x7c00 << 13;  // exponent mask after shift
  float32_bits o;

  o.u = (srcValue & 0x7fff) << 13;       // exponent/mantissa bits
  unsigned int exp = shifted_exp & o.u;  // just the exponent
  o.u += (127 - 15) << 23;               // exponent adjust

  // handle exponent special cases
  if (exp == shifted_exp) {   // Inf/NaN?
    o.u += (128 - 16) << 23;  // extra exp adjust
  } else if (exp == 0) {      // Zero/Denormal?
    o.u += 1 << 23;           // extra exp adjust
    o.f -= magic.f;           // renormalize
  }

  o.u |= (srcValue & 0x8000) << 16;  // sign bit
  return o.f;
}

uint16_t Float32ToShort(float srcValue) {
  float32_bits f;
  f.f = srcValue;

  const float32_bits f32infty = {255 << 23};
  const float32_bits f16max = {(127 + 16) << 23};
  const float32_bits denorm_magic = {((127 - 15) + (23 - 10) + 1) << 23};
  unsigned int sign_mask = 0x80000000u;
  uint16_t o;

  unsigned int sign = f.u & sign_mask;
  f.u ^= sign;

  // NOTE all the integer compares in this function can be safely
  // compiled into signed compares since all operands are below
  // 0x80000000. Important if you want fast straight SSE2 code
  // (since there's no unsigned PCMPGTD).

  if (f.u >= f16max.u) {                       // result is Inf or NaN (all exponent bits set)
    o = (f.u > f32infty.u) ? 0x7e00 : 0x7c00;  // NaN->qNaN and Inf->Inf
  } else {                                     // (De)normalized number or zero
    if (f.u < (113 << 23)) {                   // resulting FP16 is subnormal or zero
      // use a magic value to align our 10 mantissa bits at the bottom of
      // the float. as long as FP addition is round-to-nearest-even this
      // just works.
      f.f += denorm_magic.f;

      // and one integer subtract of the bias later, we have our final float!
      o = (uint16_t)(f.u - denorm_magic.u);
    } else {
      unsigned int mant_odd = (f.u >> 13) & 1;  // resulting mantissa is odd

      // update exponent, rounding bias part 1
      f.u += ((unsigned int)(15 - 127) << 23) + 0xfff;
      // rounding bias part 2
      f.u += mant_odd;
      // take the bits!
      o = (uint16_t)(f.u >> 13);
    }
  }

  o |= (uint16_t)(sign >> 16);
  return o;
}
