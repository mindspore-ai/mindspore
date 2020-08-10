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

#include "nnacl/fp32/local_response_norm.h"
#include <math.h>

int LocalResponseNorm(float *input_ptr, int out_size, int channel, float *output_ptr,
                      LocalResponseNormParameter *param) {
  int i, j, k;
  int left, right;

  float depth_radius = param->depth_radius_;
  float bias = param->bias_;
  float alpha = param->alpha_;
  float beta = param->beta_;

  for (i = 0; i < out_size; i++) {
    float *in_data = input_ptr + i * channel;
    float *out_data = output_ptr + i * channel;

    for (j = 0; j < channel; j++) {
      left = MSMAX(0, j - depth_radius);
      right = MSMIN(channel - 1, j + depth_radius);

      float sum = 0.0;
      for (k = left; k <= right; k++) {
        const float in_val = in_data[k];
        sum += in_val * in_val;
      }
      out_data[j] = in_data[j] * (float)(pow((double)(sum * alpha + bias), -beta));
    }
  }
  return 0;
}
