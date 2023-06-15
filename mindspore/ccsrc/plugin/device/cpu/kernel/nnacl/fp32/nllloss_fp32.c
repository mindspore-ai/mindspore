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

#include "nnacl/fp32/nllloss_fp32.h"
#include "nnacl/op_base.h"
#include "nnacl/errorcode.h"

int NLLLoss(const float *logits, const int32_t *labels, const float *weight, float *loss, float *total_weight,
            const NLLLossStruct *nllloss, const ReductionType reduction_type) {
  NNACL_CHECK_NULL_RETURN_ERR(logits);
  NNACL_CHECK_NULL_RETURN_ERR(labels);
  NNACL_CHECK_NULL_RETURN_ERR(weight);
  NNACL_CHECK_NULL_RETURN_ERR(loss);
  NNACL_CHECK_NULL_RETURN_ERR(total_weight);

  float total_loss = 0.0;
  float tmp_total_weight = 0.0;
  for (int i = 0; i < nllloss->batch_; i++) {
    int index = i * nllloss->class_num_ + labels[i];
    float n_weight = weight[labels[i]];
    float n_loss = -logits[index] * n_weight;
    tmp_total_weight += n_weight;
    total_loss += n_loss;
    if (reduction_type == Reduction_None) {
      loss[i] = n_loss;
    }
  }

  *total_weight = tmp_total_weight;
  if (reduction_type == Reduction_Sum) {
    *loss = total_loss;
  } else if (reduction_type == Reduction_Mean) {
    *loss = total_loss / tmp_total_weight;
  }
  return NNACL_OK;
}
