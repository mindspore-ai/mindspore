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

#include "nnacl/fp16/log_softmax_fp16.h"
#include <math.h>
#include <float.h>
#include "nnacl/fp16/softmax_fp16.h"
#include "nnacl/fp16/exp_fp16.h"

void LogSoftmaxLastAxisFp16(const float16_t *src, float16_t *dst, float16_t *exp_data, int batch, int channel) {
  SoftmaxNormFp16(src, dst, batch, channel);
  ExpFp16(dst, exp_data, batch * channel);
  int cur_batch_offset = 0;
  for (int i = 0; i < batch; i++, cur_batch_offset += channel) {
    float16_t sum = 0;
    int j = 0;
#ifdef ENABLE_NEON
    float16x8_t sum8 = vdupq_n_f16(0);
    int count = (channel / C8NUM) * C8NUM;
    for (; j < count; j += C8NUM) {
      sum8 = vaddq_f16(sum8, vld1q_f16(exp_data + cur_batch_offset + j));
    }
    sum = sum8[0] + sum8[1] + sum8[2] + sum8[3] + sum8[4] + sum8[5] + sum8[6] + sum8[7];
#endif
    for (; j < channel; j++) {
      sum += exp_data[cur_batch_offset + j];
    }
    for (int k = 0; k < channel; k++) {
      dst[cur_batch_offset + k] = dst[cur_batch_offset + k] - log(sum);
    }
  }
}

// output = (input - reduce_max(input, axis)) - log(reduce_sum(exp(input - reduce_max(input, axis)), axis))
void LogSoftmaxFp16(const float16_t *input_ptr, float16_t *output_ptr, float16_t *sum_data,
                    const SoftmaxParameter *parameter) {
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
      float16_t max_data = input_ptr[inner_offset];
      sum_data[k + sum_outter_offset] = 0;
      for (int j = 0; j < input_shape[axis]; j++) {
        int axis_offset = inner_offset + j * inner_size;
        max_data = max_data > input_ptr[axis_offset] ? max_data : input_ptr[axis_offset];
      }
      for (int j = 0; j < input_shape[axis]; j++) {
        int axis_offset = inner_offset + j * inner_size;
        output_ptr[axis_offset] = input_ptr[axis_offset] - max_data;
        sum_data[k + sum_outter_offset] += exp(output_ptr[axis_offset]);
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
        output_ptr[inner_offset] = output_ptr[inner_offset] - log(sum_data[k + sum_outter_offset]);
      }
    }
  }
}
