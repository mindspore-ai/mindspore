/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/where_fp32.h"
#include "nnacl/common_func.h"

void WhereWithTripleInputs(const float *x, const float *y, float *output, const WhereArgs *param, int task_id,
                           int thread_num) {
  const bool *condition = param->condition_;
  int stride = UP_DIV(param->max_num_, thread_num);
  int begin = task_id * stride;
  int end = MSMIN(begin + stride, param->max_num_);

  for (int i = begin; i < end; ++i) {
    bool cond = condition[param->condition_num_ > 1 ? i : 0];
    if (cond) {
      output[i] = x[param->x_num_ > 1 ? i : 0];
    } else {
      output[i] = y[param->y_num_ > 1 ? i : 0];
    }
  }
}
