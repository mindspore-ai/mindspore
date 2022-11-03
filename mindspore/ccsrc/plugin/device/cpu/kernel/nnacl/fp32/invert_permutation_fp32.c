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

#include "nnacl/fp32/invert_permutation_fp32.h"
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"

int InvertPermutation(const int *input, int *output, size_t num) {
  NNACL_CHECK_NULL_RETURN_ERR(input);
  NNACL_CHECK_NULL_RETURN_ERR(output);
  for (size_t i = 0; i < num; i++) {
    size_t index = (size_t)input[i];
    if (index >= num) {
      return NNACL_ERR;
    }
    output[index] = i;
  }
  return NNACL_OK;
}
