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

#include "nnacl/fp32/local_response_norm_fp32.h"
#include <math.h>
#include "nnacl/errorcode.h"

int LocalResponseNorm(const float *input_ptr, int out_size, int channel, float *output_ptr,
                      const LocalResponseNormParameter *param) {
  NNACL_CHECK_NULL_RETURN_ERR(input_ptr);
  NNACL_CHECK_NULL_RETURN_ERR(output_ptr);
  NNACL_CHECK_NULL_RETURN_ERR(param);
  int64_t depth_radius = param->depth_radius_;
  float bias = param->bias_;
  float alpha = param->alpha_;
  float beta = param->beta_;

  for (int i = 0; i < out_size; i++) {
    const float *in_data = input_ptr + i * channel;
    float *out_data = output_ptr + i * channel;
    // border_left
    for (int j = 0; j < MSMIN(depth_radius, channel); j++) {
      int left = MSMAX(0, j - depth_radius);
      int right = MSMIN(channel - 1, j + depth_radius);
      float sum = 0.0f;
      for (int k = left; k <= right; k++) {
        const float in_val = in_data[k];
        sum += in_val * in_val;
      }
      out_data[j] = in_data[j] * (float)(powf(sum * alpha + bias, -beta));
    }
    // center
    if (2 * depth_radius + 1 < channel) {
      float tmp_sum = 0.0f;
      for (int j = 0; j < depth_radius * 2 + 1; ++j) {
        tmp_sum += in_data[j] * in_data[j];
      }
      out_data[depth_radius] = in_data[depth_radius] * (powf(tmp_sum * alpha + bias, -beta));
      for (int j = depth_radius + 1; j < channel - depth_radius; ++j) {
        tmp_sum -= in_data[j - depth_radius - 1] * in_data[j - depth_radius - 1];
        tmp_sum += in_data[j + depth_radius] * in_data[j + depth_radius];
        out_data[j] = in_data[j] * (float)(powf(tmp_sum * alpha + bias, -beta));
      }
    }
    // border_right
    for (int j = MSMAX(0, channel - depth_radius); j < channel; j++) {
      int left = MSMAX(0, j - depth_radius);
      int right = MSMIN(channel - 1, j + depth_radius);
      float sum = 0.0f;
      for (int k = left; k <= right; k++) {
        const float in_val = in_data[k];
        sum += in_val * in_val;
      }
      out_data[j] = in_data[j] * (float)(powf(sum * alpha + bias, -beta));
    }
  }
  return 0;
}
