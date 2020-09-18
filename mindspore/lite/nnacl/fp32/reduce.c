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

#include "nnacl/fp32/reduce.h"
#include <float.h>
#include "nnacl/errorcode.h"
#include "nnacl/common_func.h"

int ReduceMean(const int outer_size, const int inner_size, const int axis_size, const float *src_data, float *dst_data,
               const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const float *outer_src = src_data + j * axis_size * inner_size;
    float *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const float *inner_src = outer_src + k;
      float *inner_dst = outer_dst + k;
      float tmp = 0.0f;
      for (i = 0; i < axis_size; i++) {
        tmp += inner_src[i * inner_size];
      }
      *inner_dst = tmp / (float)axis_size;
    }
  }
  return NNACL_OK;
}
int ReduceSum(const int outer_size, const int inner_size, const int axis_size, const float *src_data, float *dst_data,
              const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j;
#ifdef ENABLE_NEON
  int block_mod = inner_size % C4NUM;
  int block_c4 = inner_size - block_mod;
#endif
  for (j = tid; j < outer_size; j += thread_num) {
    const float *outer_src = src_data + j * axis_size * inner_size;
    float *outer_dst = dst_data + j * inner_size;
    int k = 0;
#ifdef ENABLE_NEON
    for (; k < block_c4; k += C4NUM) {
      const float *inner_src = outer_src + k;
      float *inner_dst = outer_dst + k;
      float32x4_t tmp = {0, 0, 0, 0};
      for (i = 0; i < axis_size; i++) {
        tmp = vaddq_f32(tmp, vld1q_f32(inner_src + i * inner_size));
      }
      vst1q_f32(inner_dst, tmp);
    }
#endif
    for (; k < inner_size; k++) {
      const float *inner_src = outer_src + k;
      float *inner_dst = outer_dst + k;
      float tmp = 0.0f;
      for (i = 0; i < axis_size; i++) {
        tmp += inner_src[i * inner_size];
      }
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}
int ReduceMax(const int outer_size, const int inner_size, const int axis_size, const float *src_data, float *dst_data,
              const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const float *outer_src = src_data + j * axis_size * inner_size;
    float *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const float *inner_src = outer_src + k;
      float *inner_dst = outer_dst + k;
      float tmp = -FLT_MAX;
      for (i = 0; i < axis_size; i++) {
        tmp = tmp > inner_src[i * inner_size] ? tmp : inner_src[i * inner_size];
      }
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}
int ReduceMin(const int outer_size, const int inner_size, const int axis_size, const float *src_data, float *dst_data,
              const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const float *outer_src = src_data + j * axis_size * inner_size;
    float *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const float *inner_src = outer_src + k;
      float *inner_dst = outer_dst + k;
      float tmp = FLT_MAX;
      for (i = 0; i < axis_size; i++) {
        tmp = tmp < inner_src[i * inner_size] ? tmp : inner_src[i * inner_size];
      }
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}
int ReduceProd(const int outer_size, const int inner_size, const int axis_size, const float *src_data, float *dst_data,
               const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const float *outer_src = src_data + j * axis_size * inner_size;
    float *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const float *inner_src = outer_src + k;
      float *inner_dst = outer_dst + k;
      float tmp = 1.0f;
      for (i = 0; i < axis_size; i++) {
        tmp *= inner_src[i * inner_size];
      }
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}

int IntReduceProd(const int outer_size, const int inner_size, const int axis_size, const int *src_data, int *dst_data,
                  const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const int *outer_src = src_data + j * axis_size * inner_size;
    int *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const int *inner_src = outer_src + k;
      int *inner_dst = outer_dst + k;
      int tmp = 1;
      for (i = 0; i < axis_size; i++) {
        if (isMulOverflow(tmp, inner_src[i * inner_size])) {
          return NNACL_ERRCODE_MUL_OVERFLOW;
        }
        tmp *= inner_src[i * inner_size];
      }
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}
int ReduceSumSquare(const int outer_size, const int inner_size, const int axis_size, const float *src_data,
                    float *dst_data, const int tid, const int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const float *outer_src = src_data + j * axis_size * inner_size;
    float *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const float *inner_src = outer_src + k;
      float *inner_dst = outer_dst + k;
      float tmp = 0.0f;
      for (i = 0; i < axis_size; i++) {
        tmp += inner_src[i * inner_size] * inner_src[i * inner_size];
      }
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}
