/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_NNACL_SQUARED_DIFFERENCE_H_
#define MINDSPORE_LITE_NNACL_SQUARED_DIFFERENCE_H_

#include "nnacl/fp32/squared_difference.h"
#include "nnacl/fp32/sub_fp32.h"
#include "nnacl/fp32/mul_fp32.h"

int ElementSquaredDifference(const float *in0, const float *in1, float *out, int size) {
  ElementSub(in0, in1, out, size);
  return ElementMul(out, out, out, size);
}

#endif  // MINDSPORE_LITE_NNACL_SQUARED_DIFFERENCE_H_
