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
#include "nnacl/intrinsics/ms_simd_instructions_fp16.h"

void BatchNormFp16(const float16_t *input, const float16_t *mean, const float16_t *variance,
                   const BatchNormStruct *param, int task_id, int thread_num, float16_t *output) {
  int units_per_thread = UP_DIV(param->unit_, thread_num);
  int completed_units = task_id * units_per_thread;
  int cur_unit = MSMIN(units_per_thread, param->unit_ - completed_units);
  int cur_offset = completed_units * param->channel_;

  for (int i = 0; i < cur_unit; i++) {
    const float16_t *unit_input = input + cur_offset;
    float16_t *unit_output = output + cur_offset;
    int c = 0;
#ifdef ENABLE_ARM
    for (; c <= param->channel_ - C8NUM; c += C8NUM) {
      MS_FLOAT16X8 input_8 = MS_LDQ_F16(unit_input + c);
      MS_FLOAT16X8 mean_8 = MS_LDQ_F16(mean + c);
      MS_FLOAT16X8 variance_8 = MS_LDQ_F16(variance + c);
      MS_FLOAT16X8 variance_sqrt = MS_SQRTFX8_F16(MS_ADDQ_F16(variance_8, MS_MOVQ_F16(param->epsilon_)));
      MS_FLOAT16X8 output_8 = MS_DIVQ_F16(MS_SUBQ_F16(input_8, mean_8), variance_sqrt);
      MS_STQ_F16(unit_output + c, output_8);
    }
#endif
    for (; c < param->channel_; c++) {
      float16_t variance_sqrt = sqrtf(variance[c] + param->epsilon_);
      unit_output[c] = (unit_input[c] - mean[c]) / variance_sqrt;
    }
    cur_offset += param->channel_;
  }
}

void FusedBatchNormFp16(const float16_t *input, const float16_t *scale, const float16_t *offset, const float16_t *mean,
                        const float16_t *variance, const BatchNormStruct *param, int task_id, int thread_num,
                        float16_t *output) {
  int units_per_thread = UP_DIV(param->unit_, thread_num);
  int completed_units = task_id * units_per_thread;
  int cur_unit = MSMIN(units_per_thread, param->unit_ - completed_units);
  int cur_offset = completed_units * param->channel_;

  for (int i = 0; i < cur_unit; i++) {
    const float16_t *unit_input = input + cur_offset;
    float16_t *unit_output = output + cur_offset;
    int c = 0;
#ifdef ENABLE_ARM
    for (; c <= param->channel_ - C8NUM; c += C8NUM) {
      MS_FLOAT16X8 input_8 = MS_LDQ_F16(unit_input + c);
      MS_FLOAT16X8 scale_8 = MS_LDQ_F16(scale + c);
      MS_FLOAT16X8 offset_8 = MS_LDQ_F16(offset + c);
      MS_FLOAT16X8 mean_8 = MS_LDQ_F16(mean + c);
      MS_FLOAT16X8 variance_8 = MS_LDQ_F16(variance + c);
      MS_FLOAT16X8 variance_sqrt = MS_SQRTFX8_F16(MS_ADDQ_F16(variance_8, MS_MOVQ_F16(param->epsilon_)));
      MS_FLOAT16X8 norm_val = MS_DIVQ_F16(MS_SUBQ_F16(input_8, mean_8), variance_sqrt);
      MS_FLOAT16X8 output_8 = MS_ADDQ_F16(MS_MULQ_F16(norm_val, scale_8), offset_8);
      MS_STQ_F16(unit_output + c, output_8);
    }
#endif
    for (; c < param->channel_; c++) {
      float16_t variance_sqrt = sqrtf(variance[c] + param->epsilon_);
      float16_t norm_val = (unit_input[c] - mean[c]) / variance_sqrt;
      unit_output[c] = norm_val * scale[c] + offset[c];
    }
    cur_offset += param->channel_;
  }
}

void FusedBatchNormFp16MeanVar(const float16_t *input, float16_t *run_mean, float16_t *run_var,
                               const BatchNormStruct *param, float16_t *save_mean, float16_t *save_var) {
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
