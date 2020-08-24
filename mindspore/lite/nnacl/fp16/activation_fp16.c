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

#include "nnacl/fp16/activation_fp16.h"
#include "nnacl/errorcode.h"

int ReluFp16(const float16_t *src, float16_t *dst, int ele_num) {
  int eight_block = UP_DIV(ele_num, C8NUM);
  int i;
  for (i = 0; i < eight_block - 1; i++) {
    int index = i * C8NUM;
#ifdef ENABLE_NEON
    float16x8_t relu_src = vld1q_f16(src + index);
    float16x8_t zero_src = vdupq_n_f16(0);
    relu_src = vmaxq_f16(relu_src, zero_src);
    vst1q_f16(dst + index, relu_src);
#else
    int j;
    for (j = 0; j < C8NUM; j++) {
      dst[index + j] = src[index + j] < 0 ? 0 : src[index + j];
    }
#endif
  }
  for (int j = (eight_block - 1) * C8NUM; j < ele_num; ++j) {
    dst[j] = src[j] < 0 ? 0 : src[j];
  }
  return NNACL_OK;
}

int Relu6Fp16(const float16_t *data, float16_t *dst, int ele_num) {
  int eight_block = UP_DIV(ele_num, C8NUM);
  int i;
  for (i = 0; i < eight_block - 1; i++) {
    int index = i * C8NUM;
#ifdef ENABLE_NEON
    float16x8_t relu6_data = vld1q_f16(data + index);
    float16x8_t zero_data = vdupq_n_f16(0);
    float16x8_t six_data = vdupq_n_f16(6);
    relu6_data = vmaxq_f16(relu6_data, zero_data);
    relu6_data = vminq_f16(relu6_data, six_data);
    vst1q_f16(dst + index, relu6_data);
#else
    int j;
    for (j = 0; j < C8NUM; ++j) {
      dst[index + j] = data[index + j] < 0 ? 0 : data[index + j];
      dst[index + j] = dst[index + j] > 6 ? 6 : dst[index + j];
    }
#endif
  }
  for (int j = (eight_block - 1) * C8NUM; j < ele_num; ++j) {
    dst[j] = data[j] < 0 ? 0 : data[j];
    dst[j] = dst[j] > 6 ? 6 : dst[j];
  }
  return NNACL_OK;
}

int LReluFp16(const float16_t *src, float16_t *dst, int ele_num, float16_t alpha) {
  for (int i = 0; i < ele_num; ++i) {
    dst[i] = src[i] > (float16_t)0.0f ? src[i] : (src[i] * alpha);
  }
  return NNACL_OK;
}

int SigmoidFp16(const float16_t *src, float16_t *dst, int ele_num) {
  for (int i = 0; i < ele_num; ++i) {
    dst[i] = (float16_t)1.0f / (float16_t)(1.0f + exp(-src[i]));
  }
  return NNACL_OK;
}

int TanhFp16(const float16_t *src, float16_t *dst, int ele_num) {
  for (int i = 0; i < ele_num; ++i) {
    dst[i] = (float16_t)1.0f - (float16_t)2.0f / (float16_t)(exp(2 * src[i]) + 1);
  }
  return NNACL_OK;
}

int HSwishFp16(const float16_t *src, float16_t *dst, int ele_num) {
  for (int i = 0; i < ele_num; ++i) {
    float16_t in = src[i];
    float16_t relu6 = MSMIN(MSMAX(in + 3, 0), 6);
    dst[i] = in * relu6 / (float16_t)6.0f;
  }
  return NNACL_OK;
}
