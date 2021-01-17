/*
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

#include "nnacl/fp32_grad/binary_cross_entropy_grad.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

int BinaryCrossEntropyGrad(const int input_size, const int reduction, const float *input_x, const float *input_y,
                           const float *weight, const float *dloss, float *dx) {
  const float epsilon = 1e-12f;
  if (reduction == 0) {
    for (int i = 0; i < input_size; i++) {
      float denominator = MAX(input_x[i] * (1 - input_x[i]), epsilon);
      float value = weight[i] * (input_x[i] - input_y[i]) / denominator;
      dx[i] = value * dloss[i];
    }
  } else {
    float dloss1 = dloss[0];
    if (reduction == 1) {
      dloss1 = dloss[0] / input_size;
    }
    for (int i = 0; i < input_size; i++) {
      float denominator = MAX(input_x[i] * (1 - input_x[i]), epsilon);
      float value = weight[i] * (input_x[i] - input_y[i]) / denominator;
      dx[i] = value * dloss1;
    }
  }
  return 0;
}
