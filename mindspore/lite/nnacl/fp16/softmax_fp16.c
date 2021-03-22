/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "nnacl/fp16/softmax_fp16.h"
#include <math.h>
#include "nnacl/fp16/exp_fp16.h"

void SoftmaxNormFp16(const float16_t *src, float16_t *dst, int batch, int channel) {
  int cur_batch_offset = 0;
  for (int i = 0; i < batch; i++, cur_batch_offset += channel) {
    int j = 0;
#ifdef ENABLE_ARM64
    float16x8_t max_8 = vdupq_n_f16(-FLT16_MAX);
    int count = (channel / C8NUM) * C8NUM;
    for (; j < count; j += C8NUM) {
      float16x8_t input_8 = vld1q_f16(src + cur_batch_offset + j);
      max_8 = vmaxq_f16(max_8, input_8);
    }
    float16_t max = vmaxvq_f16(max_8);
#else
    float16_t max = -FLT_MAX;
#endif
    for (; j < channel; j++) {
      float16_t input = src[cur_batch_offset + j];
      if (input > max) {
        max = input;
      }
    }
    int k = 0;
#ifdef ENABLE_NEON
    int count2 = (channel / C8NUM) * C8NUM;
    for (; k < count2; k += C8NUM) {
      float16x8_t input_8 = vld1q_f16(src + cur_batch_offset + k);
      float16x8_t output_8 = vsubq_f16(input_8, vdupq_n_f16(max));
      vst1q_f16(dst + cur_batch_offset + k, output_8);
    }
#endif
    for (; k < channel; k++) {
      int offset = cur_batch_offset + k;
      dst[offset] = src[offset] - max;
    }
  }
}

void SumAndDivFp16(const float16_t *src, float16_t *dst, int batch, int channel) {
  int cur_batch_offset = 0;
  for (int i = 0; i < batch; i++, cur_batch_offset += channel) {
    float16_t sum = 0;
    int j = 0;
#ifdef ENABLE_NEON
    float16x8_t sum8 = vdupq_n_f16(0);
    int count = (channel / C8NUM) * C8NUM;
    for (; j < count; j += C8NUM) {
      sum8 = vaddq_f16(sum8, vld1q_f16(src + cur_batch_offset + j));
    }
    sum = sum8[0] + sum8[1] + sum8[2] + sum8[3] + sum8[4] + sum8[5] + sum8[6] + sum8[7];
#endif
    for (; j < channel; j++) {
      sum += src[cur_batch_offset + j];
    }
    int k = 0;
#ifdef ENABLE_NEON
    const float16_t div = 1.0f / sum;
    for (; k < count; k += C8NUM) {
      vst1q_f16(dst + cur_batch_offset + k, vmulq_n_f16(vld1q_f16(src + cur_batch_offset + k), div));
    }
#endif
    for (; k < channel; k++) {
      dst[cur_batch_offset + k] = src[cur_batch_offset + k] / sum;
    }
  }
}

void SoftmaxLastAxisFp16(const float16_t *src, float16_t *dst, int batch, int channel) {
  SoftmaxNormFp16(src, dst, batch, channel);
  ExpFp16(dst, dst, batch * channel);
  SumAndDivFp16(dst, dst, batch, channel);
}

// output = exp(input) / reduce_sum(exp(input), axis)
void SoftmaxFp16(const float16_t *input_ptr, float16_t *output_ptr, float16_t *sum_data, SoftmaxParameter *parameter) {
  int axis = parameter->axis_;
  int n_dim = parameter->n_dim_;
  int *input_shape = parameter->input_shape_;
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
      float16_t max_data = input_ptr[inner_offset];
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
