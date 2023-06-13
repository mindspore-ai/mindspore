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
#ifndef NNACL_FP16_RANGE_FP16_H_
#define NNACL_FP16_RANGE_FP16_H_

#include "nnacl/op_base.h"

void RangeFp16(float16_t *output_ptr, float16_t start, float16_t delta, int nums) {
  for (int i = 0; i < nums; ++i, start += delta) {
    output_ptr[i] = start;
  }
}

#endif  //  NNACL_FP16_RANGE_FP16_H_
