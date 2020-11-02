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
  int offset = 0;
#ifdef ENABLE_NEON
  float16x8_t zero = vdupq_n_f16(0);
  for (; offset <= ele_num - C8NUM; offset += C8NUM) {
    float16x8_t src_value = vld1q_f16(src + offset);
    float16x8_t rst_value = vmaxq_f16(src_value, zero);
    vst1q_f16(dst + offset, rst_value);
  }
#endif
  for (; offset < ele_num; offset++) {
    dst[offset] = src[offset] < 0 ? 0 : src[offset];
  }
  return NNACL_OK;
}

int Relu6Fp16(const float16_t *data, float16_t *dst, int ele_num) {
  int offset = 0;
#ifdef ENABLE_NEON
  float16x8_t zero_data = vdupq_n_f16(0);
  float16x8_t six_data = vdupq_n_f16(6);
  for (; offset <= ele_num - C8NUM; offset += C8NUM) {
    float16x8_t relu6_data = vld1q_f16(data + offset);
    relu6_data = vmaxq_f16(relu6_data, zero_data);
    relu6_data = vminq_f16(relu6_data, six_data);
    vst1q_f16(dst + offset, relu6_data);
  }
#endif
  for (; offset < ele_num; offset++) {
    dst[offset] = data[offset] < 0 ? 0 : data[offset];
    dst[offset] = dst[offset] > 6 ? 6 : dst[offset];
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

float16_t TanhOptFp16(float16_t src) {
  if (src > 5.0) {
    return 1.0f;
  } else if (src < -5.0) {
    return -1.0f;
  } else {
    float square = src * src;
    float a = (((square + 378.0f) * square + 17325.0f) * square + 135135.0f) * src;
    float b = ((28.0f * square + 3150.0f) * square + 62370.0f) * square + 135135.0f;
    return a / b;
  }
}

int TanhFp16(const float16_t *src, float16_t *dst, int ele_num) {
  for (int i = 0; i < ele_num; ++i) {
    dst[i] = TanhOptFp16(src[i]);
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

int SwishFp16(const float16_t *src, float16_t *dst, int ele_num) {
  int ret = SigmoidFp16(src, dst, ele_num);
  if (ret != NNACL_OK) {
    return NNACL_ERR;
  }
  int index = 0;
  for (; index < ele_num; index++) {
    dst[index] = src[index] * dst[index];
  }
  return NNACL_OK;
}
