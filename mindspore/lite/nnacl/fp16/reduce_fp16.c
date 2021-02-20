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

#include <float.h>
#include "nnacl/fp16/reduce_fp16.h"
#include "nnacl/errorcode.h"

int ReduceMeanFp16(const int outer_size, const int inner_size, const int axis_size, const float16_t *src_data,
                   float16_t *dst_data, const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const float16_t *outer_src = src_data + j * axis_size * inner_size;
    float16_t *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const float16_t *inner_src = outer_src + k;
      float16_t *inner_dst = outer_dst + k;
      float tmp = 0.0;
      for (i = 0; i < axis_size; i++) {
        tmp += inner_src[i * inner_size];
      }
      *inner_dst = (float16_t)(tmp / axis_size);
    }
  }
  return NNACL_OK;
}

int ReduceMaxFp16(int outer_size, int inner_size, int axis_size, const float16_t *src_data, float16_t *dst_data,
                  int tid, int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const float16_t *outer_src = src_data + j * axis_size * inner_size;
    float16_t *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const float16_t *inner_src = outer_src + k;
      float16_t *inner_dst = outer_dst + k;
      float tmp = -FLT_MAX;
      for (i = 0; i < axis_size; i++) {
        tmp = tmp > inner_src[i * inner_size] ? tmp : inner_src[i * inner_size];
      }
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}
