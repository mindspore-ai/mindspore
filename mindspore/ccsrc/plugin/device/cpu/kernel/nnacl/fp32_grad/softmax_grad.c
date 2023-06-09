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

#include "nnacl/fp32_grad/softmax_grad.h"
#include <string.h>

void SoftmaxGrad(const float *input_ptr, const float *yt_ptr, float *output_ptr, float *sum_data, float *sum_mul,
                 const int *input_shape, int n_dim, int ele_size, int32_t axis) {
  int dim = 1;
  int inner_size = 1, outter_size = 1;
  for (int i = 0; i < axis; i++) {
    outter_size *= input_shape[i];
  }
  for (int i = axis + 1; i < n_dim; i++) {
    inner_size *= input_shape[i];
  }
  NNACL_CHECK_ZERO_RETURN(outter_size);
  for (int i = 0; i < inner_size * input_shape[axis]; i++) sum_mul[i] = 1.0;
  for (int i = 0; i < n_dim; i++) dim *= input_shape[i];
  dim /= outter_size;
  memcpy(output_ptr, yt_ptr, (size_t)(ele_size) * sizeof(float));

  const int M = input_shape[axis];
  const int N = inner_size;
  for (int i = 0; i < outter_size; i++) {
    int outter_offset = i * dim;
    memset(sum_data, 0, (size_t)(inner_size) * sizeof(float));
    for (int k = 0; k < inner_size; k++) {
      int inner_offset = outter_offset + k;
      for (int j = 0; j < input_shape[axis]; j++) {
        int offset = inner_offset + j * inner_size;
        sum_data[k] += output_ptr[offset] * input_ptr[offset];
      }
    }
    for (int k = 0; k < M; ++k) {
      float a = -sum_mul[k];
      for (int j = 0; j < N; ++j) {
        *(output_ptr + outter_offset + k * N + j) += a * sum_data[j];
      }
    }
  }

  for (int i = 0; i < ele_size; i++) {
    output_ptr[i] *= input_ptr[i];
  }
}
