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

#include "src/runtime/kernel/arm/nnacl/fp32/batchnorm.h"

void BatchNorm(const float *input_ptr, const float *mean_ptr, const float *variance_ptr, int units, int channel,
               float epsilon, float *output_ptr) {
  for (int u = 0; u < units; u++) {
    for (int c = 0; c < channel; c++) {
      auto variance_sqrt = sqrt(variance_ptr[c] + epsilon);
      output_ptr[u * channel + c] = (input_ptr[u * channel + c] - mean_ptr[c]) / variance_sqrt;
    }
  }
}
