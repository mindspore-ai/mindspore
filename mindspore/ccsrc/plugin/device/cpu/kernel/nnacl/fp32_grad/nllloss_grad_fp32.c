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

#include "nnacl/fp32_grad/nllloss_grad_fp32.h"

#include "nnacl/op_base.h"
#include "nnacl/errorcode.h"

int NLLLossGrad(const float *logits, const float *loss_grad, const int *labels, const float *weight,
                const float *total_weight, float *logits_grad, int batch, int class_num, ReductionType reduction_type) {
  if (logits == NULL || loss_grad == NULL || labels == NULL || weight == NULL || total_weight == NULL ||
      logits_grad == NULL) {
    return NNACL_NULL_PTR;
  }

  memset(logits_grad, 0, batch * class_num * sizeof(float));
  for (int i = 0; i < batch; i++) {
    int index = i * class_num + labels[i];
    float n_weight = weight[labels[i]];
    if (reduction_type == Reduction_Sum) {
      logits_grad[index] = -loss_grad[0] * n_weight;
    } else if (reduction_type == Reduction_Mean) {
      logits_grad[index] = -loss_grad[0] * n_weight / *total_weight;
    } else {
      logits_grad[index] = -loss_grad[i] * n_weight;
    }
  }
  return NNACL_OK;
}
