/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
    res *= shape[((size_t)(i) + 1) + j];
  }
  NNACL_CHECK_ZERO_RETURN_ERR(res);
  NNACL_CHECK_ZERO_RETURN_ERR(shape[i]);
  return (pos / res % shape[i]);
}

int DoStridedSliceGrad(const float *inputs, float *output, const int *dx_shape, const StridedSliceParameter *param) {
  if (inputs == NULL || output == NULL || param == NULL) {
    return NNACL_NULL_PTR;
  }
  if (param->num_axes_ > DIMENSION_8D) {
    return NNACL_PARAM_INVALID;
  }

  size_t size = 1;
  const int *s = param->strides_;
  const int *b = param->begins_;
  for (int i = 0; i < DIMENSION_8D; i++) {
    size *= (size_t)(param->in_shape_[i]);
  }

  for (size_t pos = 0; pos < size; pos++) {
    size_t i = CalcIndex(param->in_shape_, C7NUM, C0NUM, pos);
    size_t j = CalcIndex(param->in_shape_, C6NUM, C1NUM, pos);
    size_t k = CalcIndex(param->in_shape_, C5NUM, C2NUM, pos);
    size_t l = CalcIndex(param->in_shape_, C4NUM, C3NUM, pos);
    size_t m = CalcIndex(param->in_shape_, C3NUM, C4NUM, pos);
    size_t n = CalcIndex(param->in_shape_, C2NUM, C5NUM, pos);
    size_t o = CalcIndex(param->in_shape_, C1NUM, C6NUM, pos);
    size_t p = CalcIndex(param->in_shape_, C0NUM, C7NUM, pos);
    size_t input_idx =
      (i * s[C0NUM] + b[C0NUM]) * dx_shape[C1NUM] * dx_shape[C2NUM] * dx_shape[C3NUM] * dx_shape[C4NUM] *
        dx_shape[C5NUM] * dx_shape[C6NUM] * dx_shape[C7NUM] +
      (j * s[C1NUM] + b[C1NUM]) * dx_shape[C2NUM] * dx_shape[C3NUM] * dx_shape[C4NUM] * dx_shape[C5NUM] *
        dx_shape[C6NUM] * dx_shape[C7NUM] +
      (k * s[C2NUM] + b[C2NUM]) * dx_shape[C3NUM] * dx_shape[C4NUM] * dx_shape[C5NUM] * dx_shape[C6NUM] *
        dx_shape[C7NUM] +
      (l * s[C3NUM] + b[C3NUM]) * dx_shape[C4NUM] * dx_shape[C5NUM] * dx_shape[C6NUM] * dx_shape[C7NUM] +
      (m * s[C4NUM] + b[C4NUM]) * dx_shape[C5NUM] * dx_shape[C6NUM] * dx_shape[C7NUM] +
      (n * s[C5NUM] + b[C5NUM]) * dx_shape[C6NUM] * dx_shape[C7NUM] + (o * s[C6NUM] + b[C6NUM]) * dx_shape[C7NUM] +
      (p * s[C7NUM] + b[C7NUM]);
    output[input_idx] = inputs[pos];
  }
  return NNACL_OK;
}
