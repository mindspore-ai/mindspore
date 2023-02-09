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

#include "nnacl/fp32/bce_with_logits_loss_fp32.h"
#include "nnacl/bce_with_logits_loss_fp32_simd.h"

void BCEWithLogitLoss(const float *logits, const float *label, const float *weight, const float *pos_weight, int length,
                      bool reduction, float *output, float *reduction_sum) {
  int i = 0;
  float simd_reduction_output = 0.0f;
  SIMD_RUN_NO_SCALAR(BCEWithLogitLoss, i, logits, label, weight, pos_weight, length, reduction, output,
                     &simd_reduction_output);
  for (; i < length; ++i) {
    float logits_value = logits[i];
    float label_value = label[i];
    float weight_value = weight[i];
    float post_weight_value = pos_weight[i];
    float max_value = -logits_value;
    max_value = max_value > 0.f ? max_value : 0.f;
    float log_weight = (post_weight_value - 1.0f) * label_value + 1.0f;
    float log_exp_value = logf(expf(-max_value) + expf(-logits_value - max_value));
    float loss = (1.0f - label_value) * logits_value + log_weight * (log_exp_value + max_value);
    if (reduction) {
      simd_reduction_output += loss * weight_value;
    } else {
      output[i] = loss * weight_value;
    }
  }
  if (reduction) {
    *reduction_sum = simd_reduction_output;
  }
}
