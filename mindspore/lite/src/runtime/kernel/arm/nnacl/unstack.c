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

#include "nnacl/unstack.h"
#include <string.h>

void Unistack(float *input, float **output, UnstackParameter *para) {
  for (int j = 0; j < para->num_; j++) {
    float *out_addr = output[j];
    int out_offset = 0;
    for (int i = 0; i < para->pre_dims_; i++) {
      int in_offset = i * para->axis_dim_ * para->after_dims_ + j * para->after_dims_;
      (void)memcpy(out_addr + out_offset, input + in_offset, para->after_dims_ * sizeof(float));
      out_offset += para->after_dims_;
    }
  }
}
