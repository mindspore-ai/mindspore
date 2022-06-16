/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/group_norm_fp32.h"
#include <math.h>
#include "nnacl/group_norm_parameter.h"
#include "nnacl/op_base.h"
#include "nnacl/errorcode.h"
#include "nnacl/group_norm_fp32_simd.h"

static void GroupNormFp32MeanVar(const float *input, float *run_mean, float *run_var, int completed_group,
                                 int cur_groups, const GroupNormParameter *param);

int GroupNormFp32(const float *input, const float *scale, const float *offset, float *mean, float *variance,
                  const GroupNormParameter *param, int task_id, float *output) {
  if (param->op_parameter_.thread_num_ == 0) {
    return NNACL_ERR;
  }
  const int frame_elem_num = param->unit_ * param->channel_;
  const int groups_per_thread = UP_DIV(param->num_groups_, param->op_parameter_.thread_num_);
  const int completed_group = task_id * groups_per_thread;
  const int cur_group = MSMIN(groups_per_thread, param->num_groups_ - completed_group);
  const int num_of_ch_per_group = param->channel_ / param->num_groups_;
  int cur_offset = completed_group * num_of_ch_per_group * param->unit_;

  for (int b = 0; b < param->batch_; b++) {
    const float *b_in = input + b * frame_elem_num;
    float *b_out = output + b * frame_elem_num;
    int b_offset = cur_offset;
    GroupNormFp32MeanVar(b_in, mean, variance, completed_group, cur_group, param);
    for (int g = 0; g < cur_group; g++) {
      int grp_idx = g + completed_group;
      int c_offset = grp_idx * num_of_ch_per_group;
      float m = mean[grp_idx];
      float v = variance[grp_idx];
      float variance_sqrt = sqrtf(v + param->epsilon_);
      if (variance_sqrt == 0) {
        return NNACL_ERR;
      }
      for (int c = 0; c < num_of_ch_per_group; c++) {
        const float *unit_input = b_in + b_offset;
        float *unit_output = b_out + b_offset;
        float s = scale[c_offset + c];
        float o = offset[c_offset + c];
        int u = 0;
        SIMD_RUN_NO_SCALAR(GroupNormFp32, u, unit_input, s, o, m, variance_sqrt, param->unit_, unit_output);
        for (; u < param->unit_; u++) {
          float norm_val = (unit_input[u] - m) / variance_sqrt;
          unit_output[u] = norm_val * s + o;
        }
        b_offset += param->unit_;
      }
    }
  }
  return NNACL_OK;
}

#define SimdReduceSum(block_size, block_num, in, i, sum)                                          \
  do {                                                                                            \
    for (int block_max_size = param->unit_ - block_num + 1; i < block_max_size; i += block_num) { \
      MS_FLOAT_32xN(block_num) input = MS_LD_F32(block_size, in + i);                             \
      sum += MS_GET_SUM_F32(block_size, input);                                                   \
    }                                                                                             \
  } while (0)

#define SimdReduceVar(block_size, block_num, in, m, i, sum)                                         \
  do {                                                                                              \
    MS_FLOAT_32xN(block_num) mean = MS_MOVN_F32(block_size, m);                                     \
    MS_FLOAT_32xN(block_num) tmp = MS_MOVN_F32(block_size, 0);                                      \
    for (int block_max_size = param->unit_ - block_num + 1; i < block_max_size; i += block_num) {   \
      MS_FLOAT_32xN(block_num) input = MS_SUB_F32(block_size, MS_LD_F32(block_size, in + i), mean); \
      tmp = MS_ADD_F32(block_size, tmp, MS_MUL_F32(block_size, input, input));                      \
    }                                                                                               \
    sum += MS_GET_SUM_F32(block_size, tmp);                                                         \
  } while (0)

static void GroupNormFp32MeanVar(const float *input, float *run_mean, float *run_var, int completed_group,
                                 int cur_groups, const GroupNormParameter *param) {
  const int num_of_ch_per_group = param->channel_ / param->num_groups_;
  const float N = (float)(param->unit_ * num_of_ch_per_group);

  // calc mean
  for (int g = 0; g < cur_groups; g++) {
    int g_idx = g + completed_group;
    float sum = 0;
    for (int c = 0; c < num_of_ch_per_group; c++) {
      const float *in = input + (num_of_ch_per_group * g_idx + c) * param->unit_;
      int i = 0;
      SIMD_RUN_NO_SCALAR(GroupNormReduceSum, i, in, &sum, param->unit_);
      for (; i < param->unit_; i++) {
        sum += in[i];
      }
    }
    run_mean[g_idx] = sum / N;
  }

  // calc variance
  for (int g = 0; g < cur_groups; g++) {
    int g_idx = g + completed_group;
    float var = 0;
    run_var[g_idx] = 0;
    for (int c = 0; c < num_of_ch_per_group; c++) {
      const float *in = input + (num_of_ch_per_group * g_idx + c) * param->unit_;
      int i = 0;
      SIMD_RUN_NO_SCALAR(GroupNormReduceVar, i, in, run_mean[g_idx], &var, param->unit_);
      for (; i < param->unit_; i++) {
        var += (in[i] - run_mean[g_idx]) * (in[i] - run_mean[g_idx]);
      }
    }
    run_var[g_idx] = var / N;
  }
}
