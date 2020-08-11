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
