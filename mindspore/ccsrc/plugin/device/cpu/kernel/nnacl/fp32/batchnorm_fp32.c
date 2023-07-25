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
#include "nnacl/op_base.h"
#include "nnacl/batchnorm_fp32_simd.h"
#include "nnacl/kernel/fused_batch_norm.h"
#include "nnacl/tensor_c_utils.h"

int FusedBatchNormEval(KernelBase *self) {
  FusedBatchNormStruct *fused_batch_norm = (FusedBatchNormStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(fused_batch_norm);

  if (fused_batch_norm->trained_) {
    TensorC *scale_tensor = fused_batch_norm->bn_.base_.in_[SECOND_INPUT];
    TensorC *offset_tensor = fused_batch_norm->bn_.base_.in_[THIRD_INPUT];
    TensorC *mean_tensor = fused_batch_norm->bn_.base_.in_[FOURTH_INPUT];
    TensorC *var_tensor = fused_batch_norm->bn_.base_.in_[FIFTH_INPUT];
    (void)memcpy(fused_batch_norm->scale_, scale_tensor->data_, GetSize(scale_tensor));
    (void)memcpy(fused_batch_norm->offset_, offset_tensor->data_, GetSize(offset_tensor));
    (void)memcpy(fused_batch_norm->bn_.mean_, mean_tensor->data_, GetSize(mean_tensor));
    (void)memcpy(fused_batch_norm->bn_.variance_, var_tensor->data_, GetSize(var_tensor));
  }
  return NNACL_OK;
}

void BatchNormSetupVirtualBatch(KernelBase *self, int virtual_batch_multiplier, int momentum) {
  BatchNormStruct *bn = (BatchNormStruct *)self;
  NNACL_CHECK_NULL_RETURN_VOID(bn);
  if (virtual_batch_multiplier > 0) {
    float new_momentum = (momentum < 0.0f) ? (bn->momentum_ / virtual_batch_multiplier) : momentum;
    bn->momentum_ = new_momentum;
  }
  return;
}

void BatchNormFp32(const float *input, const float *mean, const float *variance, const BatchNormStruct *param,
                   int task_id, int thread_num, float *output) {
  int units_per_thread = UP_DIV(param->unit_, thread_num);
  int completed_units = task_id * units_per_thread;
  int cur_unit = MSMIN(units_per_thread, param->unit_ - completed_units);
  int channel = param->channel_;
  int cur_offset = completed_units * channel;
  float epsilon = param->epsilon_;

  for (int i = 0; i < cur_unit; i++) {
    const float *unit_input = input + cur_offset;
    float *unit_output = output + cur_offset;
    int c = 0;

    SIMD_RUN_NO_SCALAR(BatchNormFp32, c, unit_input, mean, variance, channel, epsilon, unit_output);

    for (; c < channel; c++) {
      float variance_sqrt = sqrtf(variance[c] + epsilon);
      unit_output[c] = (unit_input[c] - mean[c]) / variance_sqrt;
    }
    cur_offset += channel;
  }
}

void FusedBatchNormFp32(const float *input, const float *scale, const float *offset, const float *mean,
                        const float *variance, const BatchNormStruct *param, int task_id, int thread_num,
                        float *output) {
  int units_per_thread = UP_DIV(param->unit_, thread_num);
  int completed_units = task_id * units_per_thread;
  int cur_unit = MSMIN(units_per_thread, param->unit_ - completed_units);
  int channel = param->channel_;
  float epsilon = param->epsilon_;
  int cur_offset = completed_units * channel;

  for (int i = 0; i < cur_unit; i++) {
    const float *unit_input = input + cur_offset;
    float *unit_output = output + cur_offset;
    int c = 0;

    SIMD_RUN_NO_SCALAR(FusedBatchNormFp32, c, unit_input, scale, offset, mean, variance, channel, epsilon, unit_output);

    for (; c < channel; c++) {
      float variance_sqrt = sqrtf(variance[c] + epsilon);
      float norm_val = (unit_input[c] - mean[c]) / variance_sqrt;
      unit_output[c] = norm_val * scale[c] + offset[c];
    }
    cur_offset += channel;
  }
}

void FusedBatchNormFp32MeanVar(const float *input, float *run_mean, float *run_var, const BatchNormStruct *param,
                               float *save_mean, float *save_var, bool isBatchNorm2d) {
  const float N = (float)param->unit_;
  const float VN = N;
  const float VNUB = (isBatchNorm2d == false) ? N : ((N > 1.0f) ? (N - 1.0f) : 1.0f);
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
