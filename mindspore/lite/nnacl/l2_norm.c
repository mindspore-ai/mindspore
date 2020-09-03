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

#include "nnacl/l2_norm.h"
#include <math.h>

int L2NormFp32(const float *input_ptr, float *output_ptr,
               L2NormParameter *param) {
  int *axis = param->axis_;
  size_t axis_num = param->axis_num_;
  float epsilon = param->epsilon_;
  int shape_num = param->shape_num_;

  // default case, axis is set default
  if (shape_num == axis_num) {
    bool default_case_flag = true;
    for (int i = 0; i < axis_num; i++) {
      if (axis[i] != i) {
        default_case_flag = false;
      }
    }
    if (default_case_flag) {
      int data_num = param->data_num_;
      float sum = 0;
      for (int i = 0; i < data_num; i++) {
        sum = sum + input_ptr[i] * input_ptr[i];
      }
      float res = sqrt(sum > epsilon ? sum : epsilon);
      for (int i = 0; i < data_num; i++) {
        output_ptr[i] = input_ptr[i] / res;
      }
      return 0;
    }
  } else {
    return -1;
  }
  return 0;
}
