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
#include "nnacl/fp32/softmax_fp32.h"
#include <math.h>
#include <float.h>
#include "nnacl/fp32/exp_fp32.h"

void SoftmaxNorm(const float *src, float *dst, int batch, int channel) {
  int cur_batch_offset = 0;
  for (int i = 0; i < batch; i++, cur_batch_offset += channel) {
    int j = 0;
#ifdef ENABLE_NEON
    float32x4_t max4 = vdupq_n_f32(-FLT_MAX);
    int count = (channel / C4NUM) * C4NUM;
    for (; j < count; j += C4NUM) {
      float32x4_t input4 = vld1q_f32(src + cur_batch_offset + j);
      max4 = vmaxq_f32(max4, input4);
    }
#ifdef ENABLE_ARM64
    float max = vmaxvq_f32(max4);
#else
    float max = max4[0];
    for (int m = 1; m < 4; ++m) {
      max = MSMAX(max, max4[m]);
    }
#endif
#else
    float max = -FLT_MAX;
#endif
    for (; j < channel; j++) {
      float input = src[cur_batch_offset + j];
      if (input > max) {
        max = input;
      }
    }
    int k = 0;
#ifdef ENABLE_NEON
    int count2 = (channel / C4NUM) * C4NUM;
    for (; k < count2; k += C4NUM) {
      float32x4_t input4 = vld1q_f32(src + cur_batch_offset + k);
      float32x4_t output4 = vsubq_f32(input4, vdupq_n_f32(max));
      vst1q_f32(dst + cur_batch_offset + k, output4);
    }
#endif
    for (; k < channel; k++) {
      int offset = cur_batch_offset + k;
      dst[offset] = src[offset] - max;
    }
  }
}

void SumAndDiv(const float *src, float *dst, int batch, int channel) {
  int cur_batch_offset = 0;
  for (int i = 0; i < batch; i++, cur_batch_offset += channel) {
    float sum = 0;
    int j = 0;
#ifdef ENABLE_NEON
    float32x4_t sum4 = vdupq_n_f32(0);
    int count = (channel / C4NUM) * C4NUM;
    for (; j < count; j += C4NUM) {
      sum4 = vaddq_f32(sum4, vld1q_f32(src + cur_batch_offset + j));
    }
#ifdef ENABLE_ARM64
    sum = vaddvq_f32(sum4);
#else
    sum = sum4[0] + sum4[1] + sum4[2] + sum4[3];
#endif
#endif
    for (; j < channel; j++) {
      sum += src[cur_batch_offset + j];
    }
    int k = 0;
#ifdef ENABLE_NEON
    const float div = 1.0f / sum;
    for (; k < count; k += C4NUM) {
      vst1q_f32(dst + cur_batch_offset + k, vmulq_n_f32(vld1q_f32(src + cur_batch_offset + k), div));
    }
#endif
    for (; k < channel; k++) {
      dst[cur_batch_offset + k] = src[cur_batch_offset + k] / sum;
    }
  }
}

void SoftmaxLastAxis(const float *src, float *dst, int batch, int channel) {
  SoftmaxNorm(src, dst, batch, channel);
  ExpFp32(dst, dst, batch * channel);
  SumAndDiv(dst, dst, batch, channel);
}

// output = exp(input) / reduce_sum(exp(input), axis)
void Softmax(const float *input_ptr, float *output_ptr, float *sum_data, const SoftmaxParameter *parameter) {
  int axis = parameter->axis_;
  int n_dim = parameter->n_dim_;
  const int *input_shape = parameter->input_shape_;
  int inner_size = 1;
  int outter_size = 1;

  for (int i = 0; i < axis; i++) {
    outter_size *= input_shape[i];
  }
  for (int i = axis + 1; i < n_dim; i++) {
    inner_size *= input_shape[i];
  }
  for (int i = 0; i < outter_size; i++) {
    int outter_offset = i * input_shape[axis] * inner_size;
    int sum_outter_offset = i * inner_size;
    for (int k = 0; k < inner_size; k++) {
      int inner_offset = outter_offset + k;
      float max_data = input_ptr[inner_offset];
      for (int j = 0; j < input_shape[axis]; j++) {
        int axis_offset = inner_offset + j * inner_size;
        max_data = max_data > input_ptr[axis_offset] ? max_data : input_ptr[axis_offset];
      }
      for (int j = 0; j < input_shape[axis]; j++) {
        int axis_offset = inner_offset + j * inner_size;
        output_ptr[axis_offset] = exp(input_ptr[axis_offset] - max_data);
        sum_data[k + sum_outter_offset] += output_ptr[axis_offset];
      }
    }
  }
  for (int i = 0; i < outter_size; i++) {
    int outter_offset = i * input_shape[axis] * inner_size;
    int sum_outter_offset = i * inner_size;
    for (int j = 0; j < input_shape[axis]; j++) {
      int axis_offset = outter_offset + j * inner_size;
      for (int k = 0; k < inner_size; k++) {
        int inner_offset = axis_offset + k;
        output_ptr[inner_offset] = output_ptr[inner_offset] / sum_data[k + sum_outter_offset];
      }
    }
  }
}
