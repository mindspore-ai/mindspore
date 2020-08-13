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
#include "nnacl/sparse_to_dense.h"

void SparseToDense(int *input, int *output_shape_, float *snum, float *dnum, int sp_num, float *output,
                   SparseToDenseParameter *s2d_param_, int task_id) {
  int m;
  for (int i = task_id; i < output_shape_[0]; i += s2d_param_->op_parameter_.thread_num_) {
    for (int j = 0; j < output_shape_[1]; j++) {
      m = i * output_shape_[1] + j;
      output[m] = dnum[0];
    }
  }

  for (int j = 0; j < sp_num; j++) {
    int temp = j * 2;
    int temp1 = j * 2 + 1;
    int tempout1 = input[temp] * output_shape_[1] + input[temp1];
    output[tempout1] = snum[j];
  }
}
