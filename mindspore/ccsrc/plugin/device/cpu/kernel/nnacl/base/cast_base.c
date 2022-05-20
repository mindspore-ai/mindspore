/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/base/cast_base.h"
#include "nnacl/intrinsics/ms_simd_instructions.h"

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdInt32ToFloat32CoreCalc(block_size, block_num, input, output, number, index)           \
  for (int block_max_size = number - block_num + 1; index < block_max_size; index += block_num) { \
    MS_INT_32xN(block_num) value = MS_LD_EPI32(block_size, input + index);                        \
    MS_ST_F32(block_size, output + index, MS_INT32_TO_FLOAT32(block_size, value));                \
  }

void Int32ToFloat32(const int32_t *input, float *output, int number) {
  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdInt32ToFloat32CoreCalc, input, output, number, index);

  for (; index < number; ++index) {
    output[index] = (float)input[index];
  }
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdFloat32ToInt32CoreCalc(block_size, block_num, input, output, number, index)           \
  for (int block_max_size = number - block_num + 1; index < block_max_size; index += block_num) { \
    MS_FLOAT_32xN(block_num) value = MS_LD_F32(block_size, input + index);                        \
    MS_ST_EPI32(block_size, output + index, MS_FLOAT32_TO_INT32(block_size, value));              \
  }

void Float32ToInt32(const float *input, int32_t *output, int number) {
  int index = 0;

  MS_SIMD_RUN_X86_NO_SCALAR(SimdFloat32ToInt32CoreCalc, input, output, number, index);

  for (; index < number; ++index) {
    output[index] = (int32_t)input[index];
  }
}
