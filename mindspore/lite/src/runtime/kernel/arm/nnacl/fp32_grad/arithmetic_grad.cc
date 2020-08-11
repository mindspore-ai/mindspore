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

#include "nnacl/fp32_grad/arithmetic_grad.h"

void ElementDivNegSquare(const float *nom, const float *denom, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = -nom[i] / (denom[i] * denom[i]);
  }
}

void ElementMulAndDivNegSquare(const float *a, const float *b, const float *denom, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = -a[i] * b[i] / (denom[i] * denom[i]);
  }
}
