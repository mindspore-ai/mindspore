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
#include "nnacl/fp32_grad/apply_proximal_adagrad_fp32.h"
#include "nnacl/intrinsics/ms_simd_instructions.h"
#include "nnacl/apply_proximal_adagrad_fp32_simd.h"

int Sign(float x) {
  if (x > 0) {
    return 1;
  }
  if (x < 0) {
    return -1;
  }
  return 0;
}

void ApplyProximalAdagradOpt(float *var, float *accum, float lr, float l1, float l2, float *grad,
                             int64_t input_elements) {
  int64_t i = 0;

  SIMD_RUN_NO_SCALAR(ApplyProximalAdagradOpt, i, var, accum, lr, l1, l2, grad, input_elements);

  for (; i < input_elements; ++i) {
    accum[i] += grad[i] * grad[i];
    float learning_rate = lr / sqrt(accum[i]);
    float prox_v = var[i];
    prox_v -= grad[i] * learning_rate;

    if (l1 > 0) {
      var[i] = Sign(prox_v) * fmax(fabs(prox_v) - learning_rate * l1, 0.0) / (1 + l2 * learning_rate);
    } else {
      var[i] = prox_v / (1 + l2 * learning_rate);
    }
  }
}
