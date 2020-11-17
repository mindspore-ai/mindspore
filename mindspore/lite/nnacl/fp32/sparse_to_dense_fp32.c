/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
// * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "nnacl/fp32/sparse_to_dense_fp32.h"

void SparseToDense(int **sparse_indices, const int *output_shape, const float *sparse_values, float default_value,
                   float *output, bool isScalar, int index_start, int index_end, int out_width) {
  for (int i = index_start; i < index_end; i++) {
    for (int j = 0; j < out_width; j++) {
      output[i * out_width + j] = default_value;
    }
  }

  int d1 = output_shape[1] * output_shape[2] * output_shape[3];
  int d2 = output_shape[2] * output_shape[3];
  int d3 = output_shape[3];

  int index;
  if (isScalar == true) {
    for (int i = index_start; i < index_end; i++) {
      index = d1 * sparse_indices[i][0] + d2 * sparse_indices[i][1] + d3 * sparse_indices[i][2] + sparse_indices[i][3];
      output[index] = sparse_values[0];
    }
  } else {
    for (int i = index_start; i < index_end; i++) {
      index = d1 * sparse_indices[i][0] + d2 * sparse_indices[i][1] + d3 * sparse_indices[i][2] + sparse_indices[i][3];
      output[index] = sparse_values[i];
    }
  }
}
