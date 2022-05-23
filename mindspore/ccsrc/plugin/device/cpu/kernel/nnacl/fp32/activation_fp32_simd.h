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
#ifndef MINDSPORE_NNACL_FP32_ACTIVATION_SIMD_H_
#define MINDSPORE_NNACL_FP32_ACTIVATION_SIMD_H_

#include "nnacl/intrinsics/ms_simd_instructions.h"

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define RegSimdFp32ReluSIMDFunc(_SIMD, block_num)                                                   \
  static inline int Fp32Relu##_SIMD(int index, const float *src, int length, float *dst) {          \
    SIMD_F32 zero = SIMD_MOV_F32(0.0f);                                                             \
    for (int block_max_size = length - block_num + 1; index < block_max_size; index += block_num) { \
      SIMD_ST_F32(dst + index, SIMD_MAX_F32(SIMD_LD_F32(src + index), zero));                       \
    }                                                                                               \
    return index;                                                                                   \
  }

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define RegInt32ReluSIMDFunc(_SIMD, block_num)                                                      \
  static inline int Int32Relu##_SIMD(int index, const int32_t *src, int length, int32_t *dst) {     \
    SIMD_EPI32 zero = SIMD_MOV_EPI32(0.0f);                                                         \
    for (int block_max_size = length - block_num + 1; index < block_max_size; index += block_num) { \
      SIMD_ST_EPI32(dst + index, SIMD_MAX_EPI32(SIMD_LD_EPI32(src + index), zero));                 \
    }                                                                                               \
    return index;                                                                                   \
  }

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define RegFp32Relu6SIMDFunc(_SIMD, block_num)                                                      \
  static inline int Fp32Relu6##_SIMD(int index, const float *src, int length, float *dst) {         \
    SIMD_F32 zero = SIMD_MOV_F32(0.0f);                                                             \
    SIMD_F32 six = SIMD_MOV_F32(6.0f);                                                              \
    for (int block_max_size = length - block_num + 1; index < block_max_size; index += block_num) { \
      SIMD_ST_F32(dst + index, SIMD_CLAMP_F32(SIMD_LD_F32(src + index), zero, six));                \
    }                                                                                               \
    return index;                                                                                   \
  }

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define RegLReluSIMDFunc(_SIMD, block_num)                                                           \
  static inline int LRelu##_SIMD(int index, const float *src, int length, float *dst, float alpha) { \
    SIMD_F32 alpha_data = SIMD_MOV_F32(alpha);                                                       \
    for (int block_max_size = length - block_num + 1; index < block_max_size; index += block_num) {  \
      SIMD_F32 src_tmp = SIMD_LD_F32(src + index);                                                   \
      SIMD_MASK mask = SIMD_CMPGT_F32(SIMD_MOV_F32(0.0f), src_tmp);                                  \
      SIMD_ST_F32(dst + index, SIMD_BLEND_F32(src_tmp, SIMD_MUL_F32(src_tmp, alpha_data), mask));    \
    }                                                                                                \
    return index;                                                                                    \
  }

#define RegSigmoidSIMDFunc(_SIMD, block_num)                                                                     \
  static inline int Sigmoid##_SIMD(int index, const float *src, int length, float *dst) {                        \
    for (int block_max_size = length - block_num + 1; index < block_max_size; index += block_num) {              \
      SIMD_EXP_ST_F32(SIMD_SUB_F32(SIMD_MOV_F32(0.0f), (SIMD_LD_F32(src + index))), dst + index);                \
      SIMD_ST_F32(dst + index,                                                                                   \
                  SIMD_DIV_F32(SIMD_MOV_F32(1.0f), SIMD_ADD_F32(SIMD_MOV_F32(1.0f), SIMD_LD_F32(dst + index)))); \
    }                                                                                                            \
    return index;                                                                                                \
  }

#define RegTanhSIMDFunc(_SIMD, block_num)                                                           \
  static inline int Tanh##_SIMD(int index, const float *src, int length, float *dst) {              \
    for (int block_max_size = length - block_num + 1; index < block_max_size; index += block_num) { \
      SIMD_F32 input = SIMD_LD_F32(src + index);                                                    \
      SIMD_ST_F32(dst + index, SIMD_TANH_F32(input));                                               \
    }                                                                                               \
    return index;                                                                                   \
  }

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define RegSwishSIMDFunc(_SIMD, block_num)                                                          \
  static inline int Swish##_SIMD(int index, const float *src, int length, float *dst) {             \
    for (int block_max_size = length - block_num + 1; index < block_max_size; index += block_num) { \
      SIMD_F32 src_value = SIMD_LD_F32(src + index);                                                \
      SIMD_F32 sigmoid_value = SIMD_LD_F32(dst + index);                                            \
      SIMD_F32 result = SIMD_MUL_F32(src_value, sigmoid_value);                                     \
      SIMD_ST_F32(dst + index, result);                                                             \
    }                                                                                               \
    return index;                                                                                   \
  }

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define RegHSwishSIMDFunc(_SIMD, block_num)                                                         \
  static inline int HSwish##_SIMD(int index, const float *src, int length, float *dst) {            \
    for (int block_max_size = length - block_num + 1; index < block_max_size; index += block_num) { \
      SIMD_F32 src_value = SIMD_LD_F32(src + index);                                                \
      SIMD_F32 relu6 = SIMD_CLAMP_N_F32(SIMD_ADD_N_F32(src_value, 3), 0, 6);                        \
      SIMD_ST_F32(dst + index, SIMD_DIV_N_F32(SIMD_MUL_F32(src_value, relu6), 6));                  \
    }                                                                                               \
    return index;                                                                                   \
  }

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define RegHSigmoidSIMDFunc(_SIMD, block_num)                                                       \
  static inline int HSigmoid##_SIMD(int index, const float *src, int length, float *dst) {          \
    for (int block_max_size = length - block_num + 1; index < block_max_size; index += block_num) { \
      SIMD_F32 src_value = SIMD_LD_F32(src + index);                                                \
      SIMD_F32 relu6 = SIMD_CLAMP_N_F32(SIMD_ADD_N_F32(src_value, 3), 0, 6);                        \
      SIMD_ST_F32(dst + index, SIMD_DIV_N_F32(relu6, 6));                                           \
    }                                                                                               \
    return index;                                                                                   \
  }

#define RegHardTanhNoLimitMinSIMDFunc(_SIMD, block_num)                                                           \
  static inline int HardTanhNoLimitMin##_SIMD(int index, const float *src, int length, float *dst, float min_val, \
                                              float max_val) {                                                    \
    for (int block_max_size = length - block_num + 1; index < block_max_size; index += block_num) {               \
      SIMD_ST_F32(dst + index, SIMD_MIN_N_F32(SIMD_LD_F32(src + index), max_val));                                \
    }                                                                                                             \
    return index;                                                                                                 \
  }

#define RegHardTanhNoLimitMaxSIMDFunc(_SIMD, block_num)                                                           \
  static inline int HardTanhNoLimitMax##_SIMD(int index, const float *src, int length, float *dst, float min_val, \
                                              float max_val) {                                                    \
    for (int block_max_size = length - block_num + 1; index < block_max_size; index += block_num) {               \
      SIMD_ST_F32(dst + index, SIMD_MAX_N_F32(SIMD_LD_F32(src + index), min_val));                                \
    }                                                                                                             \
    return index;                                                                                                 \
  }

#define RegHardTanhLimitMinMaxSIMDFunc(_SIMD, block_num)                                                           \
  static inline int HardTanhLimitMinMax##_SIMD(int index, const float *src, int length, float *dst, float min_val, \
                                               float max_val) {                                                    \
    for (int block_max_size = length - block_num + 1; index < block_max_size; index += block_num) {                \
      SIMD_ST_F32(dst + index, SIMD_CLAMP_N_F32(SIMD_LD_F32(src + index), min_val, max_val));                      \
    }                                                                                                              \
    return index;                                                                                                  \
  }

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define RegGeluSIMDFunc(_SIMD, block_num)                                                                          \
  static inline int Gelu##_SIMD(int index, const float *src, int length, float *dst) {                             \
    for (int block_max_size = length - block_num + 1; index < block_max_size; index += block_num) {                \
      SIMD_F32 in = SIMD_LD_F32(src + index);                                                                      \
      SIMD_F32 tmp1 = SIMD_MUL_F32(SIMD_MUL_N_F32(in, 0.035677408136f), in);                                       \
      SIMD_F32 tmp2 = SIMD_MUL_F32(SIMD_ADD_N_F32(tmp1, 0.79788456080287f), in);                                   \
      SIMD_ST_F32(dst + index, SIMD_MUL_F32(SIMD_MUL_N_F32(in, 0.5f), SIMD_ADD_N_F32(SIMD_TANH_F32(tmp2), 1.0f))); \
    }                                                                                                              \
    return index;                                                                                                  \
  }

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define RegEluSIMDFunc(_SIMD, block_num)                                                            \
  static inline int Elu##_SIMD(int index, const float *src, int length, float *dst, float alpha) {  \
    for (int block_max_size = length - block_num + 1; index < block_max_size; index += block_num) { \
      SIMD_F32 src_tmp = SIMD_LD_F32(src + index);                                                  \
      SIMD_F32 exp_tmp = SIMD_SUB_N_F32(SIMD_EXP_F32(src_tmp), 1.0f);                               \
      SIMD_MASK mask = SIMD_CMPLE_F32(src_tmp, SIMD_MOV_F32(0.0f));                                 \
      SIMD_ST_F32(dst + index, SIMD_BLEND_F32(src_tmp, SIMD_MUL_N_F32(exp_tmp, alpha), mask));      \
    }                                                                                               \
    return index;                                                                                   \
  }

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define RegCeluSIMDFunc(_SIMD, block_num)                                                           \
  static inline int Celu##_SIMD(int index, const float *src, int length, float *dst, float alpha) { \
    for (int block_max_size = length - block_num + 1; index < block_max_size; index += block_num) { \
      SIMD_F32 src_tmp = SIMD_LD_F32(src + index);                                                  \
      SIMD_F32 exp_tmp = SIMD_SUB_N_F32(SIMD_EXP_F32(SIMD_DIV_N_F32(src_tmp, alpha)), 1.0f);        \
      SIMD_MASK mask = SIMD_CMPLE_F32(src_tmp, SIMD_MOV_F32(0.0f));                                 \
      SIMD_ST_F32(dst + index, SIMD_BLEND_F32(src_tmp, SIMD_MUL_N_F32(exp_tmp, alpha), mask));      \
    }                                                                                               \
    return index;                                                                                   \
  }

#ifdef ENABLE_AVX512
#pragma GCC push_options
#pragma GCC target("avx512f")

#define MS_SIMD_INSTRUCTION MS_SIMD_AVX512_INSTRUCTION
RegSimdFp32ReluSIMDFunc(AVX512, 16);
RegInt32ReluSIMDFunc(AVX512, 16);
RegFp32Relu6SIMDFunc(AVX512, 16);
RegLReluSIMDFunc(AVX512, 16);
RegSigmoidSIMDFunc(AVX512, 16);
RegTanhSIMDFunc(AVX512, 16);
RegSwishSIMDFunc(AVX512, 16);
RegHSwishSIMDFunc(AVX512, 16);
RegHSigmoidSIMDFunc(AVX512, 16);
RegHardTanhNoLimitMinSIMDFunc(AVX512, 16);
RegHardTanhNoLimitMaxSIMDFunc(AVX512, 16);
RegHardTanhLimitMinMaxSIMDFunc(AVX512, 16);
RegGeluSIMDFunc(AVX512, 16);
RegEluSIMDFunc(AVX512, 16);
RegCeluSIMDFunc(AVX512, 16);
#undef MS_SIMD_INSTRUCTION
#pragma GCC pop_options
#endif

#ifdef ENABLE_AVX
#pragma GCC push_options
#pragma GCC target("avx", "avx2")

#define MS_SIMD_INSTRUCTION MS_SIMD_AVX_INSTRUCTION
RegSimdFp32ReluSIMDFunc(AVX, 8);
RegInt32ReluSIMDFunc(AVX, 8);
RegFp32Relu6SIMDFunc(AVX, 8);
RegLReluSIMDFunc(AVX, 8);
RegSigmoidSIMDFunc(AVX, 8);
RegTanhSIMDFunc(AVX, 8);
RegSwishSIMDFunc(AVX, 8);
RegHSwishSIMDFunc(AVX, 8);
RegHSigmoidSIMDFunc(AVX, 8);
RegHardTanhNoLimitMinSIMDFunc(AVX, 8);
RegHardTanhNoLimitMaxSIMDFunc(AVX, 8);
RegHardTanhLimitMinMaxSIMDFunc(AVX, 8);
RegGeluSIMDFunc(AVX, 8);
RegEluSIMDFunc(AVX, 8);
RegCeluSIMDFunc(AVX, 8);
#undef MS_SIMD_INSTRUCTION
#pragma GCC pop_options
#endif

#ifdef ENABLE_SSE
#pragma GCC push_options
#pragma GCC target("sse4.1")

#define MS_SIMD_INSTRUCTION MS_SIMD_SSE_INSTRUCTION
RegSimdFp32ReluSIMDFunc(SSE, 4);
RegInt32ReluSIMDFunc(SSE, 4);
RegFp32Relu6SIMDFunc(SSE, 4);
RegLReluSIMDFunc(SSE, 4);
RegSigmoidSIMDFunc(SSE, 4);
RegTanhSIMDFunc(SSE, 4);
RegSwishSIMDFunc(SSE, 4);
RegHSwishSIMDFunc(SSE, 4);
RegHSigmoidSIMDFunc(SSE, 4);
RegHardTanhNoLimitMinSIMDFunc(SSE, 4);
RegHardTanhNoLimitMaxSIMDFunc(SSE, 4);
RegHardTanhLimitMinMaxSIMDFunc(SSE, 4);
RegGeluSIMDFunc(SSE, 4);
RegEluSIMDFunc(SSE, 4);
RegCeluSIMDFunc(SSE, 4);
#undef MS_SIMD_INSTRUCTION
#pragma GCC pop_options
#endif

#ifdef ENABLE_ARM
#define MS_SIMD_INSTRUCTION MS_SIMD_NEON_INSTRUCTION
RegSimdFp32ReluSIMDFunc(NEON, 4);
RegInt32ReluSIMDFunc(NEON, 4);
RegFp32Relu6SIMDFunc(NEON, 4);
RegLReluSIMDFunc(NEON, 4);
RegSigmoidSIMDFunc(NEON, 4);
RegTanhSIMDFunc(NEON, 4);
RegSwishSIMDFunc(NEON, 4);
RegHSwishSIMDFunc(NEON, 4);
RegHSigmoidSIMDFunc(NEON, 4);
RegHardTanhNoLimitMinSIMDFunc(NEON, 4);
RegHardTanhNoLimitMaxSIMDFunc(NEON, 4);
RegHardTanhLimitMinMaxSIMDFunc(NEON, 4);
RegGeluSIMDFunc(NEON, 4);
RegEluSIMDFunc(NEON, 4);
RegCeluSIMDFunc(NEON, 4);
#undef MS_SIMD_INSTRUCTION
#endif
#endif
