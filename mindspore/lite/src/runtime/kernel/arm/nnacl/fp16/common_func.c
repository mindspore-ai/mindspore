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
#include "nnacl/fp16/common_func.h"

void ReluFp16(float16_t *data, float16_t *dst, int ele_num) {
  int eight_block = UP_DIV(ele_num, C8NUM);
  for (int i = 0; i < eight_block - 1; i++) {
    int index = i * C8NUM;
#ifdef ENABLE_NEON
    float16x8_t relu_data = vld1q_f16(data + index);
    float16x8_t zero_data = vdupq_n_f16(0);
    relu_data = vmaxq_f16(relu_data, zero_data);
    vst1q_f16(dst + index, relu_data);
#else
    data[index] = data[index] < 0 ? 0 : data[index];
    data[index + 1] = data[index + 1] < 0 ? 0 : data[index + 1];
    data[index + 2] = data[index + 2] < 0 ? 0 : data[index + 2];
    data[index + 3] = data[index + 3] < 0 ? 0 : data[index + 3];
#endif
  }
  for (int j = (eight_block - 1) * C8NUM; j < ele_num; ++j) {
    data[j] = data[j] < 0 ? 0 : data[j];
  }
}

void Relu6Fp16(float16_t *data, float16_t *dst, int ele_num) {
  int eight_block = UP_DIV(ele_num, C8NUM);
  for (int i = 0; i < eight_block - 1; i++) {
    int index = i * C8NUM;
#ifdef ENABLE_NEON
    float16x8_t relu6_data = vld1q_f16(data + index);
    float16x8_t zero_data = vdupq_n_f16(0);
    float16x8_t six_data = vdupq_n_f16(6);
    relu6_data = vmaxq_f16(relu6_data, zero_data);
    relu6_data = vminq_f16(relu6_data, six_data);
    vst1q_f16(dst + index, relu6_data);
#else
    for (int j = 0; j < C8NUM; ++j) {
      data[index + j] = data[index + j] < 0 ? 0 : data[index + j];
      data[index + j] = data[index + j] > 6 ? 6 : data[index + j];
    }
#endif
  }
  for (int j = (eight_block - 1) * C8NUM; j < ele_num; ++j) {
    data[j] = data[j] < 0 ? 0 : data[j];
    data[j] = data[j] > 6 ? 6 : data[j];
  }
}
