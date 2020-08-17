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

#include "nnacl/fp32/constant_of_shape.h"

int ConstantOfShape(float *output, int tid, ConstantOfShapeParameter *param) {
  int size = param->unit_;
  float data = param->value_;
  int ind_st = MSMIN(tid * size, param->element_sz_);
  int ind_end = MSMIN(param->element_sz_, (tid + 1) * size);
  for (int i = ind_st; i < ind_end; ++i) {
    output[i] = data;
  }
  return NNACL_OK;
}
