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

#include "nnacl/fp16/batchnorm_fp16.h"
#include <math.h>

void BatchNormFp16(const float16_t *input, const void *mean, const void *variance, const BatchNormParameter *param,
                   int task_id, float16_t *output) {
  NNACL_CHECK_ZERO_RETURN(param->op_parameter_.thread_num_);
  int units_per_thread = UP_DIV(param->unit_, param->op_parameter_.thread_num_);
  int completed_units = task_id * units_per_thread;
  int cur_unit = MSMIN(units_per_thread, param->unit_ - completed_units);
  int cur_offset = completed_units * param->channel_;

  for (int i = 0; i < cur_unit; i++) {
    for (int c = 0; c < param->channel_; c++) {
      float16_t variance_sqrt = sqrtf(((const float16_t *)variance)[c] + param->epsilon_);
      if (variance_sqrt != 0) {
        output[cur_offset + c] = (input[cur_offset + c] - ((const float16_t *)mean)[c]) / variance_sqrt;
      }
    }
    cur_offset += param->channel_;
  }
}

void FusedBatchNormFp16(const void *input, const void *scale, const void *offset, const void *mean,
                        const void *variance, const BatchNormParameter *param, int task_id, void *output) {
  NNACL_CHECK_ZERO_RETURN(param->op_parameter_.thread_num_);
  int units_per_thread = UP_DIV(param->unit_, param->op_parameter_.thread_num_);
  int completed_units = task_id * units_per_thread;
  int cur_unit = MSMIN(units_per_thread, param->unit_ - completed_units);
  int cur_offset = completed_units * param->channel_;

  for (int i = 0; i < cur_unit; i++) {
    for (int c = 0; c < param->channel_; c++) {
      float16_t variance_sqrt = sqrtf(((const float16_t *)variance)[c] + param->epsilon_);
      if (variance_sqrt != 0) {
        float16_t norm_val =
          (((const float16_t *)input)[cur_offset + c] - ((const float16_t *)mean)[c]) / variance_sqrt;
        ((float16_t *)output)[cur_offset + c] =
          norm_val * ((const float16_t *)scale)[c] + ((const float16_t *)offset)[c];
      }
    }
    cur_offset += param->channel_;
  }
}

void FusedBatchNormFp16MeanVar(const float16_t *input, float16_t *run_mean, float16_t *run_var,
                               const BatchNormParameter *param, float16_t *save_mean, float16_t *save_var) {
  const float N = (float)param->unit_;
  const float VN = N;
  const float VNUB = (N > 1.0f) ? (N - 1.0f) : 1.0f;
  const float momentum = (1.0f - param->momentum_);

  for (int i = 0; i < param->unit_; i++) {
    for (int c = 0; c < param->channel_; c++) {
      int idx = i * param->channel_ + c;
      run_mean[c] += input[idx];
    }
  }
  for (int c = 0; c < param->channel_; c++) {
    run_mean[c] /= (float16_t)N;
  }
  for (int i = 0; i < param->unit_; i++) {
    for (int c = 0; c < param->channel_; c++) {
      int idx = i * param->channel_ + c;
      run_var[c] += (float16_t)((float)(input[idx] - run_mean[c]) * (float)(input[idx] - run_mean[c]));
    }
  }
  for (int c = 0; c < param->channel_; c++) {
    float unbiased_var = ((float)run_var[c] / VNUB);
    run_var[c] = (float16_t)((float)run_var[c] / VN);
    save_mean[c] = (float16_t)(momentum * (float)save_mean[c] + (1.0f - momentum) * (float)run_mean[c]);
    save_var[c] = (float16_t)(momentum * (float)save_var[c] + (1.0f - momentum) * unbiased_var);
  }
}
