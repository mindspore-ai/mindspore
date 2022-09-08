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
#include "nnacl/fp32_grad/apply_proximal_gradient_descent_fp32.h"
#include "nnacl/intrinsics/ms_simd_instructions.h"
#include "nnacl/apply_proximal_gradient_descent_fp32_simd.h"

void ApplyProximalGradientDescentOpt(float *var, float alpha, float l1, float l2, float *delta,
                                     int64_t input_elements) {
  int64_t i = 0;
  SIMD_RUN_NO_SCALAR(ApplyProximalGradientDescentOpt, i, var, alpha, l1, l2, delta, input_elements);
  for (; i < input_elements; ++i) {
    float prox_v = var[i];
    prox_v -= delta[i] * alpha;

    if (l1 > 0) {
      var[i] = SignFp32(prox_v) * fmax(fabs(prox_v) - alpha * l1, 0.0) / (1 + l2 * alpha);
    } else {
      var[i] = prox_v / (1 + l2 * alpha);
    }
  }
}

float SignFp32(const float x) {
  if (x > 0.0) {
    return 1.0;
  }
  if (x < 0.0) {
    return -1.0;
  }
  return 0.0;
}
