/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdBatchNormFp32CoreCalc(block_size, block_num, unit_input, mean, variance, param, unit_output, c)        \
  for (int block_max_size = param->channel_ - block_num + 1; c < block_max_size; c += block_num) {                 \
    MS_FLOAT_32xN(block_num) input = MS_LD_F32(block_size, unit_input + c);                                        \
    MS_FLOAT_32xN(block_num) mean_ = MS_LD_F32(block_size, mean + c);                                              \
    MS_FLOAT_32xN(block_num) variance_ = MS_LD_F32(block_size, variance + c);                                      \
    MS_FLOAT_32xN(block_num) variance_sqrt =                                                                       \
      MS_SQRT_F32(block_size, MS_ADD_F32(block_size, variance_, MS_MOVN_F32(block_size, param->epsilon_)));        \
    MS_FLOAT_32xN(block_num) output = MS_DIV_F32(block_size, MS_SUB_F32(block_size, input, mean_), variance_sqrt); \
    MS_ST_F32(block_size, unit_output + c, output);                                                                \
  }

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

    MS_SIMD_RUN_NO_SCALAR(SimdBatchNormFp32CoreCalc, unit_input, mean, variance, param, unit_output, c);

    for (; c < param->channel_; c++) {
      float variance_sqrt = sqrtf(variance[c] + param->epsilon_);
      unit_output[c] = (unit_input[c] - mean[c]) / variance_sqrt;
    }
    cur_offset += param->channel_;
  }
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdFusedBatchNormFp32CoreCalc(block_size, block_num, unit_input, scale, mean, offset, variance, param,      \
                                       unit_output, c)                                                               \
  for (int block_max_size = param->channel_ - block_num + 1; c < block_max_size; c += block_num) {                   \
    MS_FLOAT_32xN(block_num) input = MS_LD_F32(block_size, unit_input + c);                                          \
    MS_FLOAT_32xN(block_num) scale_ = MS_LD_F32(block_size, scale + c);                                              \
    MS_FLOAT_32xN(block_num) offset_ = MS_LD_F32(block_size, offset + c);                                            \
    MS_FLOAT_32xN(block_num) mean_ = MS_LD_F32(block_size, mean + c);                                                \
    MS_FLOAT_32xN(block_num) variance_ = MS_LD_F32(block_size, variance + c);                                        \
    MS_FLOAT_32xN(block_num) variance_sqrt =                                                                         \
      MS_SQRT_F32(block_size, MS_ADD_F32(block_size, variance_, MS_MOVN_F32(block_size, param->epsilon_)));          \
    MS_FLOAT_32xN(block_num) norm_val = MS_DIV_F32(block_size, MS_SUB_F32(block_size, input, mean_), variance_sqrt); \
    MS_FLOAT_32xN(block_num) output = MS_ADD_F32(block_size, MS_MUL_F32(block_size, norm_val, scale_), offset_);     \
    MS_ST_F32(block_size, unit_output + c, output);                                                                  \
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

    MS_SIMD_RUN_NO_SCALAR(SimdFusedBatchNormFp32CoreCalc, unit_input, scale, mean, offset, variance, param, unit_output,
                          c);

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
