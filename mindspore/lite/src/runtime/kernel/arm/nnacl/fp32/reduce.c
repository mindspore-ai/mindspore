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
#include "nnacl/fp32/reduce.h"
#include "nnacl/errorcode.h"

int ReduceMean(const int outer_size, const int inner_size, const int axis_size, const float *src_data,
               const int *src_shape, float *dst_data, const int tid, const int thread_num) {
  if (src_data == NULL || src_shape == NULL || dst_data == NULL) {
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
int ReduceSum(const int outer_size, const int inner_size, const int axis_size, const float *src_data,
              const int *src_shape, float *dst_data, const int tid, const int thread_num) {
  if (src_data == NULL || src_shape == NULL || dst_data == NULL) {
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
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}
int ReduceMax(const int outer_size, const int inner_size, const int axis_size, const float *src_data,
              const int *src_shape, float *dst_data, const int tid, const int thread_num) {
  if (src_data == NULL || src_shape == NULL || dst_data == NULL) {
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
int ReduceMin(const int outer_size, const int inner_size, const int axis_size, const float *src_data,
              const int *src_shape, float *dst_data, const int tid, const int thread_num) {
  if (src_data == NULL || src_shape == NULL || dst_data == NULL) {
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
int ReduceProd(const int outer_size, const int inner_size, const int axis_size, const float *src_data,
               const int *src_shape, float *dst_data, const int tid, const int thread_num) {
  if (src_data == NULL || src_shape == NULL || dst_data == NULL) {
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
int ReduceSumSquare(const int outer_size, const int inner_size, const int axis_size, const float *src_data,
                    const int *src_shape, float *dst_data, const int tid, const int thread_num) {
  if (src_data == NULL || src_shape == NULL || dst_data == NULL) {
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
