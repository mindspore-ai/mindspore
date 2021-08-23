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

int ReduceMeanFp16(int outer_size, int inner_size, int axis_size, const float16_t *src_data, float16_t *dst_data,
                   int tid, int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  if (axis_size == 0) {
    return NNACL_ERR;
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

int ReduceSumFp16(int outer_size, int inner_size, int axis_size, const float16_t *src_data, float16_t *dst_data,
                  int tid, int thread_num) {
  int stride = UP_DIV(outer_size, thread_num);
  int start = stride * tid;
  int end = MSMIN(outer_size, start + stride);
  int num = end - start;
#ifdef ENABLE_NEON
  int block_c8 = inner_size - inner_size % C8NUM;
#endif

  int src_stride = axis_size * inner_size;
  src_data += start * src_stride;
  dst_data += start * inner_size;

  for (int i = 0; i < num; i++, src_data += src_stride, dst_data += inner_size) {
    int j = 0;
#ifdef ENABLE_NEON
    for (; j < block_c8; j += C8NUM) {
      const float16_t *inner_src = src_data + j;
      float16_t *inner_dst = dst_data + j;
      float16x8_t tmp = {0, 0, 0, 0, 0, 0, 0, 0};
      for (int k = 0; k < axis_size; k++) {
        tmp = vaddq_f16(tmp, vld1q_f16(inner_src + k * inner_size));
      }
      vst1q_f16(inner_dst, tmp);
    }
#endif
    for (; j < inner_size; j++) {
      const float16_t *inner_src = src_data + j;
      float16_t *inner_dst = dst_data + j;
      float tmp = 0.0f;
      for (int k = 0; k < axis_size; k++) {
        tmp += inner_src[k * inner_size];
      }
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}
