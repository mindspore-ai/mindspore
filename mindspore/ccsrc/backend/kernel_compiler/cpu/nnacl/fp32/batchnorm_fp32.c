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

#include "nnacl/fp32/batchnorm_fp32.h"
#include <math.h>
#include "nnacl/batchnorm_parameter.h"
#include "nnacl/op_base.h"

void BatchNormFp32(const float *input, const float *mean, const float *variance, const BatchNormParameter *param,
                   int task_id, float *output) {
  if (param->op_parameter_.thread_num_ == 0) {
    return;
  }
  int units_per_thread = UP_DIV(param->unit_, param->op_parameter_.thread_num_);
  int completed_units = task_id * units_per_thread;
  int cur_unit = MSMIN(units_per_thread, param->unit_ - completed_units);
  int cur_offset = completed_units * param->channel_;

  for (int i = 0; i < cur_unit; i++) {
    const float *unit_input = input + cur_offset;
    float *unit_output = output + cur_offset;
    int c = 0;
#ifdef ENABLE_ARM
    for (; c <= param->channel_ - C4NUM; c += C4NUM) {
      MS_FLOAT32X4 input_4 = MS_LDQ_F32(unit_input + c);
      MS_FLOAT32X4 mean_4 = MS_LDQ_F32(mean + c);
      MS_FLOAT32X4 variance_4 = MS_LDQ_F32(variance + c);
      MS_FLOAT32X4 variance_sqrt = MS_SQRTFX4_F32(MS_ADDQ_F32(variance_4, MS_MOVQ_F32(param->epsilon_)));
      MS_FLOAT32X4 output_4 = MS_DIVQ_F32(MS_SUBQ_F32(input_4, mean_4), variance_sqrt);
      MS_STQ_F32(unit_output + c, output_4);
    }
#endif
    for (; c < param->channel_; c++) {
      float variance_sqrt = sqrtf(variance[c] + param->epsilon_);
      unit_output[c] = (unit_input[c] - mean[c]) / variance_sqrt;
    }
    cur_offset += param->channel_;
  }
}

void FusedBatchNormFp32(const float *input, const float *scale, const float *offset, const float *mean,
                        const float *variance, const BatchNormParameter *param, int task_id, float *output) {
  if (param->op_parameter_.thread_num_ == 0) {
    return;
  }
  int units_per_thread = UP_DIV(param->unit_, param->op_parameter_.thread_num_);
  int completed_units = task_id * units_per_thread;
  int cur_unit = MSMIN(units_per_thread, param->unit_ - completed_units);
  int cur_offset = completed_units * param->channel_;

  for (int i = 0; i < cur_unit; i++) {
    const float *unit_input = input + cur_offset;
    float *unit_output = output + cur_offset;
    int c = 0;
#ifdef ENABLE_ARM
    for (; c <= param->channel_ - C4NUM; c += C4NUM) {
      MS_FLOAT32X4 input_4 = MS_LDQ_F32(unit_input + c);
      MS_FLOAT32X4 scale_4 = MS_LDQ_F32(scale + c);
      MS_FLOAT32X4 offset_4 = MS_LDQ_F32(offset + c);
      MS_FLOAT32X4 mean_4 = MS_LDQ_F32(mean + c);
      MS_FLOAT32X4 variance_4 = MS_LDQ_F32(variance + c);
      MS_FLOAT32X4 variance_sqrt = MS_SQRTFX4_F32(MS_ADDQ_F32(variance_4, MS_MOVQ_F32(param->epsilon_)));
      MS_FLOAT32X4 norm_val = MS_DIVQ_F32(MS_SUBQ_F32(input_4, mean_4), variance_sqrt);
      MS_FLOAT32X4 output_4 = MS_ADDQ_F32(MS_MULQ_F32(norm_val, scale_4), offset_4);
      MS_STQ_F32(unit_output + c, output_4);
    }
#endif
    for (; c < param->channel_; c++) {
      float variance_sqrt = sqrtf(variance[c] + param->epsilon_);
      float norm_val = (unit_input[c] - mean[c]) / variance_sqrt;
      unit_output[c] = norm_val * scale[c] + offset[c];
    }
    cur_offset += param->channel_;
  }
}

void FusedBatchNormFp32MeanVar(const float *input, float *run_mean, float *run_var, const BatchNormParameter *param,
                               float *save_mean, float *save_var) {
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
    run_mean[c] /= N;
  }
  for (int i = 0; i < param->unit_; i++) {
    for (int c = 0; c < param->channel_; c++) {
      int idx = i * param->channel_ + c;
      run_var[c] += (input[idx] - run_mean[c]) * (input[idx] - run_mean[c]);
    }
  }
  for (int c = 0; c < param->channel_; c++) {
    float unbiased_var = (run_var[c] / VNUB);
    run_var[c] = (run_var[c] / VN);
    save_mean[c] = momentum * save_mean[c] + (1.0f - momentum) * run_mean[c];
    save_var[c] = momentum * save_var[c] + (1.0f - momentum) * unbiased_var;
  }
}
