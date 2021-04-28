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

#include <math.h>
#include "nnacl/fp32_grad/binary_cross_entropy.h"

static void BinaryCrossEntropyLossKernel(const int input_size, const int reduction, const float *input_x,
                                         const float *input_y, const float *weight, float *loss, float *tmp_loss) {
  const float epsilon = 1e-12;
  if (reduction == 0) {
    for (int i = 0; i < input_size; i++) {
      float value =
        -weight[i] * (input_y[i] * logf(input_x[i] + epsilon) + (1 - input_y[i]) * logf(1 - input_x[i] + epsilon));
      loss[i] = value;
    }
  } else {
    for (int i = 0; i < input_size; i++) {
      float value =
        -weight[i] * (input_y[i] * logf(input_x[i] + epsilon) + (1 - input_y[i]) * logf(1 - input_x[i] + epsilon));
      tmp_loss[i] = value;
    }
  }
}

void BinaryCrossEntropy(const int input_size, const int reduction, const float *input_x, const float *input_y,
                        const float *weight, float *loss, float *tmp_loss) {
  loss[0] = 0.0f;
  BinaryCrossEntropyLossKernel(input_size, reduction, input_x, input_y, weight, loss, tmp_loss);
  if (reduction != 0) {
    if (input_size % 2 == 1) {
      tmp_loss[0] += tmp_loss[input_size - 1];
    }
    for (int stride = input_size / 2; stride > 0; stride = stride / 2) {
      for (int i = 0; i < stride; i++) {
        tmp_loss[i] += tmp_loss[i + stride];
      }

      if (stride > 2 && stride % 2 == 1) {
        tmp_loss[0] += tmp_loss[stride - 1];
      }
    }
    loss[0] += tmp_loss[0];
    if (reduction == 1) {
      loss[0] /= input_size;
    }
  }
}
