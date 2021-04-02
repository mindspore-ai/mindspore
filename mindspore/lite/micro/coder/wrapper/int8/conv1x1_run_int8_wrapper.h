/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_INT8_CONV1X1_RUN_H_
#define MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_INT8_CONV1X1_RUN_H_

#include <stdint.h>
#include <stdbool.h>
#include "nnacl/conv_parameter.h"
#include "nnacl/matmul_parameter.h"

typedef struct {
  int32_t *input_sum_;     /* per-oc */
  int32_t *filter_zp_ptr_; /* per-oc up round  */
  int32_t *left_shift_;    /* per-oc up round  */
  int32_t *right_shift_;   /* per-oc up round  */
  int32_t *multiplier_;    /* per-oc up round  */
  int8_t *packed_weight_;
  int32_t *bias_data_;
  int8_t *packed_input_;
  int8_t *input_ptr_;
  int8_t *output_ptr_;
  size_t thread_count_hw;
  size_t thread_stride_hw_;
  size_t thread_count_oc;
  size_t thread_stride_oc_;
  ConvParameter *conv_param_;
  MatMulParameter *matmul_param_;
  MATMUL_OPT_DP_FUNC matmul_func_;
  bool pre_trans_input_;
  bool support_optimize_;
  bool filter_peroc_;
  bool parallel_by_oc_;
} Conv1x1Args;

void Conv1x1PreRun(Conv1x1Args *args, int thread_num);
void Pre1x1Trans(Conv1x1Args *args, int8_t *src_input, int8_t *src_output);
int OcOptPre(void *cdata, int task_id);
int RunArm64OptOc(void *cdata, int task_id);
int RunArmOc(void *cdata, int task_id);
int RunArm64OptHw(void *cdata, int task_id);
int RunArmHw(void *cdata, int task_id);

#endif  // MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_INT8_CONV1X1_RUN_H_
