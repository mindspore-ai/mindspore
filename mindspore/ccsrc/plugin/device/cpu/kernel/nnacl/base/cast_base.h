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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_BASE_CAST_BASE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_BASE_CAST_BASE_H_

#include "nnacl/op_base.h"
#include "nnacl/nnacl_common.h"

#ifdef __cplusplus
extern "C" {
#endif

inline void BoolToFloat32(const bool *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float)input[i];
  }
}

inline void Uint8ToFloat32(const uint8_t *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float)input[i];
  }
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdInt32ToFloat32CoreCalc(block_size, block_num, input, output, number, index)           \
  for (int block_max_size = number - block_num + 1; index < block_max_size; index += block_num) { \
    MS_INT_32xN(block_num) value = MS_LD_EPI32(block_size, input + index);                        \
    MS_ST_F32(block_size, output + index, MS_INT32_TO_FLOAT32(block_size, value));                \
  }

inline void Int32ToFloat32(const int32_t *input, float *output, int number) {
  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdInt32ToFloat32CoreCalc, input, output, number, index);

  for (; index < number; ++index) {
    output[index] = (float)input[index];
  }
}

inline void Int64ToFloat32(const int64_t *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float)input[i];
  }
}

#ifdef ENABLE_FP16
inline void Int64ToFp16(const int64_t *input, float16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float16_t)input[i];
  }
}

inline void Int32ToFp16(const int32_t *input, float16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float16_t)input[i];
  }
}

inline void BoolToFp16(const bool *input, float16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float16_t)input[i];
  }
}

inline void Uint8ToFp16(const uint8_t *input, float16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float16_t)input[i];
  }
}

inline void Float32ToFp16(const float *input, float16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float16_t)(input[i]);
  }
}

inline void Fp16ToFloat32(const float16_t *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float)(input[i]);
  }
}
#else
inline void Fp16ToFloat32(const uint16_t *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = ShortToFloat32(input[i]);
  }
}

inline void Float32ToFp16(const float *input, uint16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = Float32ToShort(input[i]);
  }
}
#endif

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdFloat32ToInt32CoreCalc(block_size, block_num, input, output, number, index)           \
  for (int block_max_size = number - block_num + 1; index < block_max_size; index += block_num) { \
    MS_FLOAT_32xN(block_num) value = MS_LD_F32(block_size, input + index);                        \
    MS_ST_EPI32(block_size, output + index, MS_FLOAT32_TO_INT32(block_size, value));              \
  }

inline void Float32ToInt32(const float *input, int32_t *output, int number) {
  int index = 0;

  MS_SIMD_RUN_X86_NO_SCALAR(SimdFloat32ToInt32CoreCalc, input, output, number, index);

  for (; index < number; ++index) {
    output[index] = (int32_t)input[index];
  }
}

inline void Float32ToInt64(const float *input, int64_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (int64_t)input[i];
  }
}

inline void Int32ToInt64(const int32_t *input, int64_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (int64_t)input[i];
  }
}

inline void Float32ToInt16(const float *input, int16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (int16_t)input[i];
  }
}

inline void BoolToInt32(const bool *input, int32_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (int32_t)input[i];
  }
}

inline void Float32ToBool(const float *input, bool *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (bool)input[i];
  }
}

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_BASE_CAST_BASE_H_
