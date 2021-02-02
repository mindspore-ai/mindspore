/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32_grad/strided_slice_grad.h"
#include "nnacl/errorcode.h"

static size_t CalcIndex(const int *shape, size_t size, int i, size_t pos) {
  size_t res = 1;
  for (size_t j = 0; j < size; j++) {
    res *= shape[(i + 1) + j];
  }
  return (pos / res % shape[i]);
}

int DoStridedSliceGrad(const float *inputs, float *output, const int *dx_shape, StridedSliceParameter *param) {
  if (inputs == NULL || output == NULL || param == NULL) {
    return NNACL_NULL_PTR;
  }
  if (param->num_axes_ > DIMENSION_7D) {
    return NNACL_PARAM_INVALID;
  }

  size_t size = 1;
  int *s = param->strides_;
  int *b = param->begins_;
  for (int i = 0; i < DIMENSION_7D; i++) {
    size *= param->in_shape_[i];
  }

  for (size_t pos = 0; pos < size; pos++) {
    size_t i = CalcIndex(param->in_shape_, 6, 0, pos);
    size_t j = CalcIndex(param->in_shape_, 5, 1, pos);
    size_t k = CalcIndex(param->in_shape_, 4, 2, pos);
    size_t l = CalcIndex(param->in_shape_, 3, 3, pos);
    size_t m = CalcIndex(param->in_shape_, 2, 4, pos);
    size_t n = CalcIndex(param->in_shape_, 1, 5, pos);
    size_t o = CalcIndex(param->in_shape_, 0, 6, pos);

    size_t input_idx =
      (i * s[0] + b[0]) * dx_shape[1] * dx_shape[2] * dx_shape[3] * dx_shape[4] * dx_shape[5] * dx_shape[6] +
      (j * s[1] + b[1]) * dx_shape[2] * dx_shape[3] * dx_shape[4] * dx_shape[5] * dx_shape[6] +
      (k * s[2] + b[2]) * dx_shape[3] * dx_shape[4] * dx_shape[5] * dx_shape[6] +
      (l * s[3] + b[3]) * dx_shape[4] * dx_shape[5] * dx_shape[6] + (m * s[4] + b[4]) * dx_shape[5] * dx_shape[6] +
      (n * s[5] + b[5]) * dx_shape[6] + (o * s[6] + b[6]);
    output[input_idx] = inputs[pos];
  }
  return NNACL_OK;
}
