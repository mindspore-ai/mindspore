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

#include "nnacl/common_func.h"

int offset(const int *shape, const int dim0, const int dim1, const int dim2, const int dim3) {
  return ((dim0 * shape[1] + dim1) * shape[2] + dim2) * shape[3] + dim3;
}

int offsetComm(const int *shape, const int dim0, const int dim1, const int dim2) {
  return ((dim0 * shape[1] + dim1) * shape[2] + dim2) * shape[3];
}

int offset4d(const int *shape, const int *dims) { return offset(shape, dims[0], dims[1], dims[2], dims[3]); }

int8_t MinInt8(int8_t a, int8_t b) { return b ^ ((a ^ b) & -(a < b)); }

int8_t MaxInt8(int8_t a, int8_t b) { return a ^ ((a ^ b) & -(a < b)); }

void ReluFp32(float *data, float *dst, int ele_num) {
  int four_block = UP_DIV(ele_num, C4NUM);
  for (int i = 0; i < four_block - 1; i++) {
    int index = i * C4NUM;
#ifdef ENABLE_NEON
    float32x4_t relu_data = vld1q_f32(data + index);
    float32x4_t zero_data = vdupq_n_f32(0);
    relu_data = vmaxq_f32(relu_data, zero_data);
    vst1q_f32(dst + index, relu_data);
#else
    data[index] = data[index] < 0 ? 0 : data[index];
    data[index + 1] = data[index + 1] < 0 ? 0 : data[index + 1];
    data[index + 2] = data[index + 2] < 0 ? 0 : data[index + 2];
    data[index + 3] = data[index + 3] < 0 ? 0 : data[index + 3];
#endif
  }
  for (int j = (four_block - 1) * C4NUM; j < ele_num; ++j) {
    data[j] = data[j] < 0 ? 0 : data[j];
  }
}

void Relu6Fp32(float *data, float *dst, int ele_num) {
  int four_block = UP_DIV(ele_num, C4NUM);
  for (int i = 0; i < four_block - 1; i++) {
    int index = i * C4NUM;
#ifdef ENABLE_NEON
    float32x4_t relu6_data = vld1q_f32(data + index);
    float32x4_t zero_data = vdupq_n_f32(0);
    float32x4_t six_data = vdupq_n_f32(6);
    relu6_data = vmaxq_f32(relu6_data, zero_data);
    relu6_data = vminq_f32(relu6_data, six_data);
    vst1q_f32(dst + index, relu6_data);
#else
    data[index] = data[index] < 0 ? 0 : data[index];
    data[index] = data[index] > 6 ? 6 : data[index];
    data[index + 1] = data[index + 1] < 0 ? 0 : data[index + 1];
    data[index + 1] = data[index + 1] > 6 ? 6 : data[index + 1];
    data[index + 2] = data[index + 2] < 0 ? 0 : data[index + 2];
    data[index + 2] = data[index + 2] > 6 ? 6 : data[index + 2];
    data[index + 3] = data[index + 3] < 0 ? 0 : data[index + 3];
    data[index + 3] = data[index + 3] > 6 ? 6 : data[index + 3];
#endif
  }
  for (int j = (four_block - 1) * C4NUM; j < ele_num; ++j) {
    data[j] = data[j] < 0 ? 0 : data[j];
    data[j] = data[j] > 6 ? 6 : data[j];
  }
}

#ifdef ENABLE_AVX
#ifdef WIN32
void ReluFp32C8(float *data, float *dst, int ele_num) {
  int four_block = UP_DIV(ele_num, C8NUM);
  for (int i = 0; i < four_block - 1; i++) {
    int index = i * C8NUM;
    data[index] = data[index] < 0 ? 0 : data[index];
    data[index + 1] = data[index + 1] < 0 ? 0 : data[index + 1];
    data[index + 2] = data[index + 2] < 0 ? 0 : data[index + 2];
    data[index + 3] = data[index + 3] < 0 ? 0 : data[index + 3];
    data[index + 4] = data[index + 4] < 0 ? 0 : data[index + 4];
    data[index + 5] = data[index + 5] < 0 ? 0 : data[index + 5];
    data[index + 6] = data[index + 6] < 0 ? 0 : data[index + 6];
    data[index + 7] = data[index + 7] < 0 ? 0 : data[index + 7];
  }
  for (int j = (four_block - 1) * C8NUM; j < ele_num; ++j) {
    data[j] = data[j] < 0 ? 0 : data[j];
  }
}

void Relu6Fp32C8(float *data, float *dst, int ele_num) {
  int four_block = UP_DIV(ele_num, C8NUM);
  for (int i = 0; i < four_block - 1; i++) {
    int index = i * C8NUM;
    data[index] = data[index] < 0 ? 0 : data[index];
    data[index] = data[index] > 6 ? 6 : data[index];
    data[index + 1] = data[index + 1] < 0 ? 0 : data[index + 1];
    data[index + 1] = data[index + 1] > 6 ? 6 : data[index + 1];
    data[index + 2] = data[index + 2] < 0 ? 0 : data[index + 2];
    data[index + 2] = data[index + 2] > 6 ? 6 : data[index + 2];
    data[index + 3] = data[index + 3] < 0 ? 0 : data[index + 3];
    data[index + 3] = data[index + 3] > 6 ? 6 : data[index + 3];
    data[index + 4] = data[index + 4] < 0 ? 0 : data[index + 4];
    data[index + 4] = data[index + 4] > 6 ? 6 : data[index + 4];
    data[index + 5] = data[index + 5] < 0 ? 0 : data[index + 5];
    data[index + 5] = data[index + 5] > 6 ? 6 : data[index + 5];
    data[index + 6] = data[index + 6] < 0 ? 0 : data[index + 6];
    data[index + 6] = data[index + 6] > 6 ? 6 : data[index + 6];
    data[index + 7] = data[index + 7] < 0 ? 0 : data[index + 7];
    data[index + 7] = data[index + 7] > 6 ? 6 : data[index + 7];
  }
  for (int j = (four_block - 1) * C8NUM; j < ele_num; ++j) {
    data[j] = data[j] < 0 ? 0 : data[j];
    data[j] = data[j] > 6 ? 6 : data[j];
  }
}
#endif
#endif
