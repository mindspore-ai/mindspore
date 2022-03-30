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

#include "nnacl/base/fill_base.h"

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdFillFp32CoreCalc(block_size, block_num, output, size, data, index)                  \
  for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
    MS_ST_F32(block_size, output + index, MS_MOVN_F32(block_size, data));                       \
  }

int FillFp32(float *output, int size, float data) {
  if (output == NULL) {
    return NNACL_NULL_PTR;
  }

  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdFillFp32CoreCalc, output, size, data, index);

  for (; index < size; ++index) {
    output[index] = data;
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdFillInt32CoreCalc(block_size, block_num, output, size, data, index)                 \
  for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
    MS_ST_EPI32(block_size, output + index, MS_MOVN_EPI32(block_size, data));                   \
  }

int FillInt32(int *output, int size, int data) {
  if (output == NULL) {
    return NNACL_NULL_PTR;
  }

  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdFillInt32CoreCalc, output, size, data, index);

  for (; index < size; ++index) {
    output[index] = data;
  }
  return NNACL_OK;
}
