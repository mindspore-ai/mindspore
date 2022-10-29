/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
 *
 */
#include "nnacl/int8/dynamic_gather_int8.h"
#include "nnacl/op_base.h"

void DynamicGather(const int8_t *input, int outer_size, int inner_size, int limit, const int32_t *indices,
                   int indices_element_size, float *output, const float *scale_in, const int32_t *zp_in) {
  for (int m = 0; m < outer_size; ++m) {
    const int8_t *int8_in_m = input + inner_size * m * limit;
    float *int8_out_m = output + inner_size * m * indices_element_size;
    for (int i = 0; i < indices_element_size; ++i) {
      int index = indices[i];
      index = index < 0 ? index + limit : index;
      const float scale = scale_in[index];
      const int zp = zp_in[index];
      float *out = int8_out_m + i * inner_size;
      const int8_t *src = int8_in_m + index * inner_size;
#ifndef ENABLE_ARM64
      for (int j = 0; j < inner_size; ++j) {
        out[j] = (src[j] - zp) * scale;
      }
#else
      int count_16 = DOWN_ROUND(inner_size, C16NUM);
      DynamicGatherArm64(src, out, count_16, zp, scale);
      for (int j = count_16; j < inner_size; ++j) {
        out[j] = (src[j] - zp) * scale;
      }
#endif
    }
  }
  return;
}

#ifdef ENABLE_FP16
void DynamicGatherForFp16(const int8_t *input, int outer_size, int inner_size, int limit, const int32_t *indices,
                          int indices_element_size, float16_t *output, const float *scale_in, const int32_t *zp_in) {
  for (int m = 0; m < outer_size; ++m) {
    const int8_t *int8_in_m = input + inner_size * m * limit;
    float16_t *int8_out_m = output + inner_size * m * indices_element_size;
    for (int i = 0; i < indices_element_size; ++i) {
      int index = indices[i];
      index = index < 0 ? index + limit : index;
      const float scale = scale_in[index];
      const int zp = zp_in[index];
      float16_t *out = int8_out_m + i * inner_size;
      const int8_t *src = int8_in_m + index * inner_size;
#ifndef ENABLE_ARM64
      for (int j = 0; j < inner_size; ++j) {
        out[j] = (float16_t)(src[j] - zp) * scale;
      }
#else
      int count_16 = DOWN_ROUND(inner_size, C16NUM);
      DynamicGatherArm64ForFp16(src, out, count_16, zp, scale);
      for (int j = count_16; j < inner_size; ++j) {
        out[j] = (float16_t)((src[j] - zp) * scale);
      }
#endif
    }
  }
  return;
}
#endif
