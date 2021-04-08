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

#include "nnacl/fp32/l2_norm_fp32.h"
#include <math.h>
#include "nnacl/errorcode.h"

int CalcThreadSquareSum(const float *input_ptr, float *sum, int begin, int end) {
  *sum = 0.0f;
  int i;
  for (i = begin; i < end; ++i) {
    *sum += input_ptr[i] * input_ptr[i];
  }
  return NNACL_OK;
}

int ThreadDivSqrtSum(const float *input_ptr, float *output_ptr, const L2NormParameter *param, const float sqrt_sum,
                     const int begin, const int end) {
  bool is_relu = param->act_type_ == ActType_Relu;
  bool is_relu6 = param->act_type_ == ActType_Relu6;
  int i;
  if (sqrt_sum == 0) {
    return NNACL_ERRCODE_DIVISOR_ZERO;
  }
  for (i = begin; i < end; i++) {
    float tmp = input_ptr[i] / sqrt_sum;
    if (is_relu) {
      output_ptr[i] = MSMAX(0, tmp);
    } else if (is_relu6) {
      output_ptr[i] = MSMIN(6, MSMAX(0, tmp));
    } else {
      output_ptr[i] = tmp;
    }
  }
  return NNACL_OK;
}

int ThreadTrailingAxis(const float *input_ptr, float *output_ptr, const L2NormParameter *param, const int begin,
                       const int end) {
  bool is_relu = param->act_type_ == ActType_Relu;
  bool is_relu6 = param->act_type_ == ActType_Relu6;

  const int c = param->shape_[param->shape_num_ - 1];
  int i = 0;
  for (i = begin; i < end; ++i) {
    float square_sum = 0.0f;
    int j = 0;
    for (j = 0; j < c; ++j) {
      const float val = input_ptr[i * c + j];
      square_sum += val * val;
    }
    float sqrt_sum = sqrt(square_sum > param->epsilon_ ? square_sum : param->epsilon_);
    for (j = 0; j < c; ++j) {
      float tmp = input_ptr[i * c + j] / sqrt_sum;
      if (is_relu) {
        output_ptr[i * c + j] = MSMAX(0, tmp);
      } else if (is_relu6) {
        output_ptr[i * c + j] = MSMIN(6, MSMAX(0, tmp));
      } else {
        output_ptr[i * c + j] = tmp;
      }
    }
  }
  return NNACL_OK;
}
