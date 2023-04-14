/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "nnacl/base/arithmetic_base.h"
#include "nnacl/kernel/arithmetic.h"

void CalcMultiplesAndStrides(ArithmeticParameter *param) {
  for (size_t i = 0; i < param->ndim_; i++) {
    if (param->in_shape0_[i] != 0) {
      param->multiples0_[i] = param->out_shape_[i] / param->in_shape0_[i];
    }
    if (param->in_shape1_[i] != 0) {
      param->multiples1_[i] = param->out_shape_[i] / param->in_shape1_[i];
    }
  }
  // cal strides
  ComputeStrides(param->in_shape0_, param->in_strides0_, param->ndim_);
  ComputeStrides(param->in_shape1_, param->in_strides1_, param->ndim_);
  ComputeStrides(param->out_shape_, param->out_strides_, param->ndim_);
}

void CalcStructMultiplesAndStrides(ArithmeticStruct *arithmetic) {
  for (size_t i = 0; i < arithmetic->ndim_; i++) {
    if (arithmetic->in_shape0_[i] != 0) {
      arithmetic->multiples0_[i] = arithmetic->out_shape_[i] / arithmetic->in_shape0_[i];
    }
    if (arithmetic->in_shape1_[i] != 0) {
      arithmetic->multiples1_[i] = arithmetic->out_shape_[i] / arithmetic->in_shape1_[i];
    }
  }
  // cal strides
  ComputeStrides(arithmetic->in_shape0_, arithmetic->in_strides0_, arithmetic->ndim_);
  ComputeStrides(arithmetic->in_shape1_, arithmetic->in_strides1_, arithmetic->ndim_);
  ComputeStrides(arithmetic->out_shape_, arithmetic->out_strides_, arithmetic->ndim_);
}
