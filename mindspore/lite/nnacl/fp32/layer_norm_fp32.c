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
#include "nnacl/fp32/layer_norm_fp32.h"
#include <math.h>
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"

int LayerNorm(const int outer_size, const int inner_size, const float *src_data, const float *gamma_data,
              const float *beta_data, const bool affine, const float epsilon, float *dst_data, const int tid,
              const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  if (affine && (gamma_data == NULL || beta_data == NULL)) {
    return NNACL_NULL_PTR;
  }
  for (int j = tid; j < outer_size; j += thread_num) {
    const float *src = src_data + j * inner_size;
    float *dst = dst_data + j * inner_size;
    float mean = 0.0f;
    float square_mean = 0.0f;
    for (int i = 0; i < inner_size; i++) {
      mean += src[i];
      square_mean += src[i] * src[i];
    }
    mean /= (float)inner_size;
    square_mean /= (float)inner_size;
    const float deno = 1 / sqrtf(square_mean - mean * mean + epsilon);
    for (int i = 0; i < inner_size; ++i) {
      dst[i] = (src[i] - mean) * deno;
      if (affine) {
        dst[i] = dst[i] * gamma_data[i] + beta_data[i];
      }
    }
  }
  return NNACL_OK;
}
