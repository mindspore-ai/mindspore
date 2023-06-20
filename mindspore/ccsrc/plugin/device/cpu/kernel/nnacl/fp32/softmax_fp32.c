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
#include "nnacl/errorcode.h"
#include "nnacl/softmax_fp32_simd.h"

void SoftmaxNorm(const float *src, float *dst, int batch, int channel) {
  int cur_batch_offset = 0;
  for (int i = 0; i < batch; i++, cur_batch_offset += channel) {
    int index = 0;
    float max = -FLT_MAX;

    SIMD_RUN_NO_SCALAR(SoftmaxNormGetMax, index, src, cur_batch_offset, &max, channel);
    for (; index < channel; index++) {
      float input = src[cur_batch_offset + index];
      if (input > max) {
        max = input;
      }
    }

    index = 0;
    SIMD_RUN_NO_SCALAR(SoftmaxNormCalcNorm, index, src, dst, cur_batch_offset, max, channel);
    for (; index < channel; index++) {
      int offset = cur_batch_offset + index;
      dst[offset] = src[offset] - max;
    }
  }
}

int SoftmaxLastAxis(const float *src, float *dst, int batch, int channel) {
  int cur_batch_offset = 0;
  for (int i = 0; i < batch; i++, cur_batch_offset += channel) {
    int index = 0;

    // get channel's max value
    float max = -FLT_MAX;
    SIMD_RUN_NO_SCALAR(SoftmaxNormGetMax, index, src, cur_batch_offset, &max, channel);
    for (; index < channel; index++) {
      float input = src[cur_batch_offset + index];
      if (input > max) {
        max = input;
      }
    }

    // get channel's exp sum value
    float exp_sum = 0.0f;
    index = 0;
    SIMD_RUN_NO_SCALAR(SoftmaxLastAxisGetExpSum, index, src, dst, cur_batch_offset, max, &exp_sum, channel);
    for (; index < channel; index++) {
      int offset = cur_batch_offset + index;
      float exp_out = simd_exp32_f32(src[offset] - max);
      exp_sum += exp_out;
      dst[offset] = exp_out;
    }

    // get result
    NNACL_CHECK_TRUE_RET(exp_sum != 0, NNACL_ERR);
    exp_sum = 1.0f / exp_sum;
    index = 0;
    SIMD_RUN_NO_SCALAR(SoftmaxLastAxisGetResult, index, dst, dst, cur_batch_offset, exp_sum, channel);
    for (; index < channel; index++) {
      dst[cur_batch_offset + index] = dst[cur_batch_offset + index] * exp_sum;
    }
  }
  return NNACL_OK;
}

// output = exp(input) / reduce_sum(exp(input), axis)
void Softmax(const float *input_ptr, float *output_ptr, float *sum_data, int axis, int n_dim,
             const int32_t *input_shape) {
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
      sum_data[k + sum_outter_offset] = 0;
      for (int j = 0; j < input_shape[axis]; j++) {
        int axis_offset = inner_offset + j * inner_size;
        max_data = max_data > input_ptr[axis_offset] ? max_data : input_ptr[axis_offset];
      }
      for (int j = 0; j < input_shape[axis]; j++) {
        int axis_offset = inner_offset + j * inner_size;
        output_ptr[axis_offset] = expf(input_ptr[axis_offset] - max_data);
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
