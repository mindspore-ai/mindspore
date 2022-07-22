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

#include "nnacl/fp32_grad/softmax_cross_entropy_with_logits.h"
#include <math.h>

void ForwardPostExecute(const float *labels, const float *logits, float *grads, float *output2,
                        size_t number_of_classes, int batch_size) {
  float eps = 1e-6;
  if (grads != NULL) {
    for (size_t i = 0; i < (size_t)(batch_size); ++i) {
      float loss = 0.f;
      for (size_t j = 0; j < number_of_classes; ++j) {
        float logit = -logf(logits[i * number_of_classes + j] <= 0.0 ? eps : logits[i * number_of_classes + j]);
        grads[i * number_of_classes + j] = (logits[i * number_of_classes + j] - labels[i * number_of_classes + j]);
        loss += labels[i * number_of_classes + j] * logit;
      }
      output2[i] = loss;
    }
  } else {
    for (size_t i = 0; i < (size_t)(batch_size); ++i) {
      float loss = 0.f;
      for (size_t j = 0; j < number_of_classes; ++j) {
        float logit = -logf(logits[i * number_of_classes + j] <= 0.0 ? eps : logits[i * number_of_classes + j]);
        loss += labels[i * number_of_classes + j] * logit;
      }
      output2[i] = loss;
    }
  }
}
