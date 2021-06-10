/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32_grad/softmax.h"
#include <math.h>
#include <float.h>
#include "nnacl/fp32/exp_fp32.h"

void ExpFp32Offset(const float *src, float *dst, float sub_bias, int num) {
  int i = 0;
#ifdef ENABLE_ARM64
  int count = (num / C4NUM) * C4NUM;
  for (; i < count; i += C4NUM) {
    MS_FLOAT32X4 input = vld1q_f32(src + i);
    MS_FLOAT32X4 bias = vdupq_n_f32(sub_bias);
    MS_FLOAT32X4 i1 = vsubq_f32(input, bias);
    simd_exp(i1, dst + i);
  }
#endif
  for (; i < num; ++i) {
    single_exp(src[i] - sub_bias, dst + i);
  }
}

// output = exp(input) / reduce_sum(exp(input), axis)
static void SoftMaxP1Simple(const float *input_ptr, float *output_ptr, float *sum_data, int start, int count,
                            int length) {
  for (int i = start; i < start + count; i++) {
    int inner_offset = i * length;
    float max_data = input_ptr[inner_offset];
    for (int j = 0; j < length; j++) {
      int axis_offset = inner_offset + j;
      max_data = max_data > input_ptr[axis_offset] ? max_data : input_ptr[axis_offset];
    }
    ExpFp32Offset(input_ptr + inner_offset, output_ptr + inner_offset, max_data, length);
    float _sum_data = 0;
    for (int j = 0; j < length; j++) {
      int axis_offset = inner_offset + j;
      _sum_data += output_ptr[axis_offset];
    }
    sum_data[i] = _sum_data;
  }
}

void SoftMaxP1(const float *input_ptr, float *output_ptr, float *sum_data, int start, int count, int length,
               int inner_size) {
  if (inner_size == 1) {
    SoftMaxP1Simple(input_ptr, output_ptr, sum_data, start, count, length);
    return;
  }
  for (int i = start; i < start + count; i++) {
    int outter_offset = i * length * inner_size;
    int sum_outter_offset = i * inner_size;
    for (int k = 0; k < inner_size; k++) {
      int inner_offset = outter_offset + k;
      float max_data = input_ptr[inner_offset];
      for (int j = 0; j < length; j++) {
        int axis_offset = inner_offset + j * inner_size;
        max_data = max_data > input_ptr[axis_offset] ? max_data : input_ptr[axis_offset];
      }
      for (int j = 0; j < length; j++) {
        int axis_offset = inner_offset + j * inner_size;
        output_ptr[axis_offset] = exp(input_ptr[axis_offset] - max_data);
      }
      float _sum_data = 0;
      for (int j = 0; j < length; j++) {
        int axis_offset = inner_offset + j * inner_size;
        _sum_data += output_ptr[axis_offset];
      }
      sum_data[k + sum_outter_offset] = _sum_data;
    }
  }
}

void SoftMaxP2(const float *input_ptr, float *output_ptr, const float *sum_data, int start, int count, int length,
               int inner_size) {
  for (int i = start; i < start + count; i++) {
    int outter_offset = i * length * inner_size;
    int sum_outter_offset = i * inner_size;
    for (int j = 0; j < length; j++) {
      int axis_offset = outter_offset + j * inner_size;
      for (int k = 0; k < inner_size; k++) {
        int inner_offset = axis_offset + k;
        output_ptr[inner_offset] = output_ptr[inner_offset] / sum_data[k + sum_outter_offset];
      }
    }
  }
}
