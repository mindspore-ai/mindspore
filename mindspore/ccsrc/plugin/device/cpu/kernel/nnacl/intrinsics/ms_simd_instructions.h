/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#define NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#include <math.h>
#include "nnacl/intrinsics/ms_simd_cpu_info.h"

#ifdef ENABLE_AVX512
#include "nnacl/intrinsics/ms_simd_avx512_instructions.h"
#endif

#ifdef ENABLE_AVX
#include "nnacl/intrinsics/ms_simd_avx_instructions.h"
#endif

#ifdef ENABLE_SSE
#include "nnacl/intrinsics/ms_simd_sse_instructions.h"
#endif

#ifdef ENABLE_ARM
#include "nnacl/intrinsics/ms_simd_neon_instructions.h"
#endif

#define MS_SIMD_AVX512_INSTRUCTION(instruction, suffix) instruction##512##suffix
#define MS_SIMD_AVX_INSTRUCTION(instruction, suffix) instruction##256##suffix
#define MS_SIMD_SSE_INSTRUCTION(instruction, suffix) instruction##128##suffix
#define MS_SIMD_NEON_INSTRUCTION(instruction, suffix) instruction##128##suffix

#define MS_SIMD_INSTRUCTION_F32(instruction) MS_SIMD_INSTRUCTION(instruction, _F32)
#define MS_SIMD_INSTRUCTION_EPI32(instruction) MS_SIMD_INSTRUCTION(instruction, _EPI32)
#define MS_SIMD_INSTRUCTION_MASK(instruction) MS_SIMD_INSTRUCTION(instruction, _MASK)

// define (float/int) data
#define SIMD_F32 MS_SIMD_INSTRUCTION_F32(MS_FLOAT)
#define SIMD_EPI32 MS_SIMD_INSTRUCTION_EPI32(MS_INT)
#define SIMD_MASK MS_SIMD_INSTRUCTION(MS_MASK, _TYPE)

// read scaler data
#define SIMD_F32_GETI MS_SIMD_INSTRUCTION(MS, _F32_GETI)

// move (float/int) data
#define SIMD_MOV_F32 MS_SIMD_INSTRUCTION_F32(MS_MOV)
#define SIMD_MOV_EPI32 MS_SIMD_INSTRUCTION_EPI32(MS_MOV)
#define SIMD_SET0_F32 MS_SIMD_INSTRUCTION(MS_MOV, _VAL0_F32)

// load (float/int) data
#define SIMD_LD_F32 MS_SIMD_INSTRUCTION_F32(MS_LD)
#define SIMD_LD_EPI32 MS_SIMD_INSTRUCTION_EPI32(MS_LD)
#define SIMD_LD_HALF_EPI32 MS_SIMD_INSTRUCTION(MS_LD, _HALF_EPI32)

// load 4 (float/int) data
#define SIMD_LDX4_F32 MS_SIMD_INSTRUCTION(MS_LOAD, X4_F32)
#define SIMD_LDX4_EPI32 MS_SIMD_INSTRUCTION(MS_LOAD, X4_EPI32)

// stored (float/int) data
#define SIMD_ST_F32 MS_SIMD_INSTRUCTION_F32(MS_ST)
#define SIMD_ST_EPI32 MS_SIMD_INSTRUCTION_EPI32(MS_ST)
#define SIMD_ST_HALF_EPI32 MS_SIMD_INSTRUCTION(MS_ST, _HALF_EPI32)

// sign
#define SIMD_SIGN_F32 MS_SIMD_INSTRUCTION_F32(SIMD_SIGN)
#define SIMD_SIGNABS_F32 MS_SIMD_INSTRUCTION_F32(SIMD_SIGNABS)

// add (float/int) op
#define SIMD_ADD_F32 MS_SIMD_INSTRUCTION_F32(MS_ADD)
#define SIMD_ADD_EPI32 MS_SIMD_INSTRUCTION_EPI32(MS_ADD)
#define SIMD_ADD_N_F32(val1, val2) MS_EXPAND(SIMD_ADD_F32(val1, SIMD_MOV_F32(val2)))
#define SIMD_ADD_N_EPI32(val1, val2) MS_EXPAND(SIMD_ADD_EPI32(val1, SIMD_MOV_EPI32(val2)))

// sub (float/int) op
#define SIMD_SUB_F32 MS_SIMD_INSTRUCTION_F32(MS_SUB)
#define SIMD_SUB_EPI32 MS_SIMD_INSTRUCTION_EPI32(MS_SUB)
#define SIMD_SUB_N_F32(val1, val2) MS_EXPAND(SIMD_SUB_F32(val1, SIMD_MOV_F32(val2)))
#define SIMD_SUB_N_EPI32(val1, val2) MS_EXPAND(SIMD_SUB_EPI32(val1, SIMD_MOV_EPI32(val2)))

// div (float/int) op
#define SIMD_DIV_F32 MS_SIMD_INSTRUCTION_F32(MS_DIV)
#define SIMD_DIV_EPI32 MS_SIMD_INSTRUCTION_EPI32(MS_DIV)
#define SIMD_DIV_N_F32(val1, val2) MS_EXPAND(SIMD_DIV_F32(val1, SIMD_MOV_F32(val2)))
#define SIMD_DIV_N_EPI32(val1, val2) MS_EXPAND(SIMD_DIV_EPI32(val1, SIMD_MOV_EPI32(val2)))

// sqrt (float) op
#define SIMD_SQRT_F32 MS_SIMD_INSTRUCTION_F32(MS_SQRT)

// rsqrt (float) op
#define SIMD_RSQRT_F32 MS_SIMD_INSTRUCTION_F32(MS_RSQRT)

// log (float) op
#define SIMD_LOG_F32 MS_SIMD_INSTRUCTION(MS, _LOG_F32)

// cos (float) op
#define SIMD_COS_F32 MS_SIMD_INSTRUCTION_F32(MS_COS)

// sin (float) op
#define SIMD_SIN_F32 MS_SIMD_INSTRUCTION_F32(MS_SIN)

// erf (float) op
#define SIMD_ERF_F32 MS_SIMD_INSTRUCTION(MS, _ERF_F32)

// abs (float) op
#define SIMD_ABS_F32 MS_SIMD_INSTRUCTION_F32(MS_ABS)
#define SIMD_ABS_EPI32 MS_SIMD_INSTRUCTION_EPI32(MS_ABS)

// round (float) op
#define SIMD_ROUND_F32 MS_SIMD_INSTRUCTION_F32(MS_ROUND)

// ceil (float) op
#define SIMD_CEIL_F32 MS_SIMD_INSTRUCTION_F32(MS_CEIL)

// floor (float) op
#define SIMD_FLOOR_F32 MS_SIMD_INSTRUCTION_F32(MS_FLOOR)

// tanh (float) op
#define SIMD_TANH_F32 MS_SIMD_INSTRUCTION_F32(MS_TANH)

// min (float/int) op
#define SIMD_MIN_F32 MS_SIMD_INSTRUCTION_F32(MS_MIN)
#define SIMD_MIN_EPI32 MS_SIMD_INSTRUCTION_EPI32(MS_MIN)
#define SIMD_MIN_N_F32(val1, val2) MS_EXPAND(SIMD_MIN_F32(val1, SIMD_MOV_F32(val2)))
#define SIMD_MIN_N_EPI32(val1, val2) MS_EXPAND(SIMD_MIN_EPI32(val1, SIMD_MOV_EPI32(val2)))

// max (float/int) op
#define SIMD_MAX_F32 MS_SIMD_INSTRUCTION_F32(MS_MAX)
#define SIMD_MAX_EPI32 MS_SIMD_INSTRUCTION_EPI32(MS_MAX)
#define SIMD_MAX_N_F32(val1, val2) MS_EXPAND(SIMD_MAX_F32(val1, SIMD_MOV_F32(val2)))
#define SIMD_MAX_N_EPI32(val1, val2) MS_EXPAND(SIMD_MAX_EPI32(val1, SIMD_MOV_EPI32(val2)))

// get max (float/int) op
#define SIMD_GET_MAX_F32 MS_SIMD_INSTRUCTION_F32(MS_GET_MAX)
#define SIMD_GET_MAX_EPI32 MS_SIMD_INSTRUCTION_EPI32(MS_GET_MAX)

// get max (float/int) op
#define SIMD_GET_SUM_F32 MS_SIMD_INSTRUCTION_F32(MS_GET_SUM)
#define SIMD_REDUCE_ADD_F32 MS_SIMD_INSTRUCTION(MS_REDUCE_ADD, _F32)

// clamp (float/int) op
#define SIMD_CLAMP_F32(val, min_val, max_val) SIMD_MIN_F32(SIMD_MAX_F32(val, min_val), max_val)
#define SIMD_CLAMP_EPI32(val, min_val, max_val) SIMD_MIN_EPI32(SIMD_MAX_EPI32(val, min_val), max_val)
#define SIMD_CLAMP_N_F32(val, min_val, max_val) \
  SIMD_MIN_F32(SIMD_MAX_F32(val, SIMD_MOV_F32(min_val)), SIMD_MOV_F32(max_val))
#define SIMD_CLAMP_N_EPI32(val, min_val, max_val) \
  SIMD_MIN_EPI32(SIMD_MAX_EPI32(val, SIMD_MOV_EPI32(min_val)), SIMD_MOV_EPI32(max_val))

// mul (float/int) op
#define SIMD_MUL_F32 MS_SIMD_INSTRUCTION_F32(MS_MUL)
#define SIMD_MUL_EPI32 MS_SIMD_INSTRUCTION_EPI32(MS_MUL)
#define SIMD_MUL_N_F32(val1, val2) MS_EXPAND(SIMD_MUL_F32(val1, SIMD_MOV_F32(val2)))
#define SIMD_MUL_N_EPI32(val1, val2) MS_EXPAND(SIMD_MUL_EPI32(val1, SIMD_MOV_EPI32(val2)))

// pow (float) op
#define SIMD_POW_F32 MS_SIMD_INSTRUCTION_F32(MS_POW)

// fma (float/int) op
#define SIMD_FMADD_F32 MS_SIMD_INSTRUCTION_F32(MS_FMADD)

// fms (float/int) op
#define SIMD_FMSUB_F32 MS_SIMD_INSTRUCTION_F32(MS_FMSUB)

// fsm (float) op
#define MS_FSMUL_F32 MS_SIMD_INSTRUCTION_F32(MS_FSMUL)

// square (float/int) op
#define SIMD_MUL_SQUARE_F32(val1) SIMD_MUL_F32(val1, val1)
#define SIMD_MUL_SQUARE_EPI32(val1) SIMD_MUL_EPI32(val1, val1)

// exp (float) op
#define SIMD_EXP_ST_F32 MS_SIMD_INSTRUCTION(simd_exp, )
#define SIMD_EXP_F32 MS_SIMD_INSTRUCTION(simd_exp, _f32)
// exp (float) high precision but a little slow op.
#define SIMD_HEXP_F32 MS_SIMD_INSTRUCTION(simd_hexp, _f32)

// cmp (float/int) op
#define SIMD_CMPLT_F32 MS_SIMD_INSTRUCTION_F32(MS_CMPLT)
#define SIMD_CMPLE_F32 MS_SIMD_INSTRUCTION_F32(MS_CMPLE)
#define SIMD_CMPGT_F32 MS_SIMD_INSTRUCTION_F32(MS_CMPGT)
#define SIMD_BLEND_F32 MS_SIMD_INSTRUCTION_F32(MS_BLEND)

// cast data
#define MS_CAST_F32_S32 MS_SIMD_INSTRUCTION(MS_CAST, _F32_S32)

// logical op
#define SIMD_AND_MASK MS_SIMD_INSTRUCTION_MASK(MS_AND)
#define SIMD_OR_F32 MS_SIMD_INSTRUCTION_F32(MS_OR)
#define SIMD_AND_MASK_F32 MS_SIMD_INSTRUCTION(MS_AND, _MASK_F32)
#define SIMD_AND_F32 MS_SIMD_INSTRUCTION_F32(MS_AND)

#define SIMD_GETSIGN_F32(src)                                                 \
  SIMD_OR_F32(SIMD_AND_F32(src, MS_CAST_F32_S32(SIMD_MOV_EPI32(0x80000000))), \
              MS_CAST_F32_S32(SIMD_MOV_EPI32(0x3F800000)))

// int32/float mutual conversion
#define SIMD_EPI32_TO_F32 MS_SIMD_INSTRUCTION(MS, _INT32_TO_FLOAT32)
#define SIMD_F32_TO_EPI32 MS_SIMD_INSTRUCTION(MS, _FLOAT32_TO_INT32)
#define SIMD_F16_TO_F32 MS_SIMD_INSTRUCTION(MS, _FLOAT16_TO_FLOAT32)
#define SIMD_F32_TO_F16 MS_SIMD_INSTRUCTION(MS, _FLOAT32_TO_FLOAT16)

// enable avx512
#if defined(ENABLE_AVX512)
#define SIMD_RUN_AVX512(function, index, ...)     \
  do {                                            \
    AVX512_HARDWARE_SELF_AWARENESS_BEGIN          \
    index = function##AVX512(index, __VA_ARGS__); \
    AVX512_HARDWARE_SELF_AWARENESS_END            \
  } while (0)
#else
#define SIMD_RUN_AVX512(function, index, ...)
#endif

// enable avx256
#if defined(ENABLE_AVX)
#define SIMD_RUN_AVX(function, index, ...) index = function##AVX(index, __VA_ARGS__)
#else
#define SIMD_RUN_AVX(function, index, ...)
#endif

// enable sse
#if defined(ENABLE_SSE)
#define SIMD_RUN_SSE(function, index, ...) index = function##SSE(index, __VA_ARGS__)
#else
#define SIMD_RUN_SSE(function, index, ...)
#endif

// enable neon
#if defined(ENABLE_NEON)
#define SIMD_RUN_NEON(function, index, ...) index = function##NEON(index, __VA_ARGS__)
#else
#define SIMD_RUN_NEON(function, index, ...)
#endif

#define SIMD_RUN_NO_SCALAR(function, index, ...)   \
  do {                                             \
    SIMD_RUN_AVX512(function, index, __VA_ARGS__); \
    SIMD_RUN_AVX(function, index, __VA_ARGS__);    \
    SIMD_RUN_SSE(function, index, __VA_ARGS__);    \
    SIMD_RUN_NEON(function, index, __VA_ARGS__);   \
  } while (0)

#define SIMD_RUN_X86_NO_SCALAR(function, index, ...) \
  do {                                               \
    SIMD_RUN_AVX512(function, index, __VA_ARGS__);   \
    SIMD_RUN_AVX(function, index, __VA_ARGS__);      \
    SIMD_RUN_SSE(function, index, __VA_ARGS__);      \
  } while (0)

#define SIMD512_BLOCK16 32  // SIMD : 512 = 16 x 32
#define SIMD256_BLOCK16 16  // SIMD : 256 = 16 x 16
#define SIMD128_BLOCK16 8   // SIMD : 128 = 16 x 8

#define SIMD512_BLOCK32 16  // SIMD : 512 = 32 x 16
#define SIMD256_BLOCK32 8   // SIMD : 256 = 32 x 8
#define SIMD128_BLOCK32 4   // SIMD : 128 = 32 x 4

#define SIMD512_BLOCK64 8  // SIMD : 512 = 64 x 8
#define SIMD256_BLOCK64 4  // SIMD : 256 = 64 x 4
#define SIMD128_BLOCK64 2  // SIMD : 128 = 64 x 2

#define MS_EXPAND(...) __VA_ARGS__

// Scaler
#define MS_FLOAT32X1 float
#define MS_INT32X1 int
#define MS_MOV32_F32(value) (value)
#define MS_MOV32_EPI32(value) (value)
#define MS_LD32_F32(address) (*(address))
#define MS_LD32_EPI32(address) (*(address))
#define MS_ST32_F32(address, value) (*(address) = (value))
#define MS_ST32_EPI32(address, value) (*(address) = (value))
#define MS_ADD32_F32(value1, value2) ((value1) + (value2))
#define MS_ADD32_EPI32(value1, value2) ((value1) + (value2))
#define MS_SUB32_F32(value1, value2) ((value1) - (value2))
#define MS_SUB32_EPI32(value1, value2) ((value1) - (value2))
#define MS_MUL32_F32(value1, value2) ((value1) * (value2))
#define MS_MUL32_EPI32(value1, value2) ((value1) * (value2))
#define MS_DIV32_F32(value1, value2) ((value1) / (value2))
#define MS_DIV32_EPI32(value1, value2) ((value1) / (value2))
#define MS_MIN32_F32(value1, value2) (fmin((value1), (value2)))
#define MS_MIN32_EPI32(value1, value2) ((value1) < (value2) ? (value1) : (value2))
#define MS_MAX32_F32(value1, value2) (fmax((value1), (value2)))
#define MS_MAX32_EPI32(value1, value2) ((value1) > (value2) ? (value1) : (value2))
#define MS_SQRT32_F32(value) (sqrt(value))

static inline float simd_exp32_f32(float data) {
  typedef union {
    float f;
    int i;
  } fi;
  static float param[] = {0.693147f, 1.0f / 120, 1.0f / 24, 1.0f / 6, 1.0f / 2, 1.0f};  // Approximate calculation param
#ifdef _WIN32
  if (data < -88.0f) {
    return 0.0f;
  } else if (data > 88.0f) {
    return 1.6516363e+38;  // e^88 = 1.6516363e+38
  }
#else
  data =
    MS_MAX32_F32(-87.3365478515625f, MS_MIN32_F32(88.72283935546875f, data));  // clamp(logf(FLT_MIN), logf(FLT_MAX))
#endif
  int integer = floor(data * 1.44269504088896341f + 0.5f);
  float decimal = data - integer * param[0];
  fi int_exp;
  const int shift = 23;
  const int bias = 126;
  const float factor = 2;
  // 2^n * exp(r) should be counted 2 * 2^(n - 1) * exp(r),
  // because n may be 128, and it is not representable by fp32.
  int_exp.i = (integer + bias) << shift;  // integer num 2^(n - 1) approximate calculation : ((x - 1) + 127) << 23
  // Approximate calculation
  const float decimal_exp =
    1.0f + decimal * (1.0f + decimal * (0.5f + decimal * (param[3] + decimal * (param[2] + decimal * param[1]))));
  return factor * int_exp.f * decimal_exp;
}

// exp(x) = exp(n * ln(2) + r) = 2^n * exp(r) = 2 * 2^(n - 1) * exp(r)
static inline void simd_exp32(float src, float *dst) {
  typedef union {
    float f;
    int i;
  } fi;
  static float param[] = {0.693147f, 1.0f / 120, 1.0f / 24, 1.0f / 6, 1.0f / 2, 1.0f};  // log(2.0f)
  src = MS_MAX32_F32(-87.3365478515625f, MS_MIN32_F32(88.72283935546875f, src));  // clamp(logf(FLT_MIN), logf(FLT_MAX))
  int integer = floor(src * 1.44269504088896341f + 0.5f);
  float decimal = src - integer * param[0];
  fi int_exp;
  const int shift = 23;
  const int bias = 126;
  const float factor = 2;
  // 2^n * exp(r) should be counted 2 * 2^(n - 1) * exp(r),
  // because n may be 128, and it is not representable by fp32.
  int_exp.i = (integer + bias) << shift;  // integer num 2^(n - 1) approximate calculation : ((x - 1) + 127) << 23
  const float decimal_exp =
    1.0f + decimal * (1.0f + decimal * (0.5f + decimal * (param[3] + decimal * (param[2] + decimal * param[1]))));
  *dst = factor * int_exp.f * decimal_exp;
}

// define (float/int) data
#define MS_FLOAT_32xN(byte_num) MS_FLOAT32##X##byte_num
#define MS_INT_32xN(byte_num) MS_INT32##X##byte_num

// move (float/int) data
#define MS_MOVN_F32(byte_num, ...) MS_EXPAND(MS_MOV##byte_num##_F32(__VA_ARGS__))
#define MS_MOVN_EPI32(byte_num, ...) MS_EXPAND(MS_MOV##byte_num##_EPI32(__VA_ARGS__))

// load (float/int) data
#define MS_LD_F32(bit_num, ...) MS_EXPAND(MS_LD##bit_num##_F32(__VA_ARGS__))
#define MS_LD_EPI32(bit_num, ...) MS_EXPAND(MS_LD##bit_num##_EPI32(__VA_ARGS__))

// load 4 (float/int) data
#define MS_LDX4_F32(bit_num, ...) MS_EXPAND(MS_LOAD##bit_num##X4_F32(__VA_ARGS__))
#define MS_LDX4_EPI32(bit_num, ...) MS_EXPAND(MS_LOAD##bit_num##X4_EPI32(__VA_ARGS__))

// stored (float/int) data
#define MS_ST_F32(bit_num, ...) MS_EXPAND(MS_ST##bit_num##_F32(__VA_ARGS__))
#define MS_ST_EPI32(bit_num, ...) MS_EXPAND(MS_ST##bit_num##_EPI32(__VA_ARGS__))

// add (float/int) op
#define MS_ADD_F32(bit_num, ...) MS_EXPAND(MS_ADD##bit_num##_F32(__VA_ARGS__))
#define MS_ADD_EPI32(bit_num, ...) MS_EXPAND(MS_ADD##bit_num##_EPI32(__VA_ARGS__))
#define MS_ADD_N_F32(bit_num, val1, val2) MS_EXPAND(MS_ADD##bit_num##_F32(val1, MS_MOV##bit_num##_F32(val2)))
#define MS_ADD_N_EPI32(bit_num, val1, val2) MS_EXPAND(MS_ADD##bit_num##_EPI32(val1, MS_MOV##bit_num##_F32(val2)))

// sub (float/int) op
#define MS_SUB_F32(bit_num, ...) MS_EXPAND(MS_SUB##bit_num##_F32(__VA_ARGS__))
#define MS_SUB_EPI32(bit_num, ...) MS_EXPAND(MS_SUB##bit_num##_EPI32(__VA_ARGS__))
#define MS_SUB_N_F32(bit_num, val1, val2) MS_EXPAND(MS_SUB##bit_num##_F32(val1, MS_MOV##bit_num##_F32(val2)))
#define MS_SUB_N_EPI32(bit_num, val1, val2) MS_EXPAND(MS_SUB##bit_num##_EPI32(val1, MS_MOV##bit_num##_F32(val2)))

// div (float/int) op
#define MS_DIV_F32(bit_num, ...) MS_EXPAND(MS_DIV##bit_num##_F32(__VA_ARGS__))
#define MS_DIV_EPI32(bit_num, ...) MS_EXPAND(MS_DIV##bit_num##_EPI32(__VA_ARGS__))
#define MS_DIV_N_F32(bit_num, val1, val2) MS_EXPAND(MS_DIV##bit_num##_F32(val1, MS_MOV##bit_num##_F32(val2)))
#define MS_DIV_N_EPI32(bit_num, val1, val2) MS_EXPAND(MS_DIV##bit_num##_EPI32(val1, MS_MOV##bit_num##_EPI32(val2)))

// sqrt (float) op
#define MS_SQRT_F32(bit_num, ...) MS_EXPAND(MS_SQRT##bit_num##_F32(__VA_ARGS__))

// rsqrt (float) op
#define MS_RSQRT_F32(bit_num, ...) MS_EXPAND(MS_RSQRT##bit_num##_F32(__VA_ARGS__))

// log (float) op
#define MS_LOG_F32(bit_num, ...) MS_EXPAND(MS_LOG##bit_num##_F32(__VA_ARGS__))

// cos (float) op
#define MS_COS_F32(bit_num, ...) MS_EXPAND(MS_COS##bit_num##_F32(__VA_ARGS__))

// sin (float) op
#define MS_SIN_F32(bit_num, ...) MS_EXPAND(MS_SIN##bit_num##_F32(__VA_ARGS__))

// erf (float) op
#define MS_ERF_F32(bit_num, ...) MS_EXPAND(MS_ERF##bit_num##_F32(__VA_ARGS__))

// log (float) op
#define MS_ABS_F32(bit_num, ...) MS_EXPAND(MS_ABS##bit_num##_F32(__VA_ARGS__))

// round (float) op
#define MS_ROUND_F32(bit_num, ...) MS_EXPAND(MS_ROUND##bit_num##_F32(__VA_ARGS__))

// ceil (float) op
#define MS_CEIL_F32(bit_num, ...) MS_EXPAND(MS_CEIL##bit_num##_F32(__VA_ARGS__))

// floor (float) op
#define MS_FLOOR_F32(bit_num, ...) MS_EXPAND(MS_FLOOR##bit_num##_F32(__VA_ARGS__))

// min (float/int) op
#define MS_MIN_F32(bit_num, ...) MS_EXPAND(MS_MIN##bit_num##_F32(__VA_ARGS__))
#define MS_MIN_EPI32(bit_num, ...) MS_EXPAND(MS_MIN##bit_num##_EPI32(__VA_ARGS__))
#define MS_MIN_N_F32(bit_num, val, n) MS_MIN_F32(bit_num, val, MS_MOVN_F32(bit_num, n))
#define MS_MIN_N_EPI32(bit_num, val, n) MS_MIN_EPI32(bit_num, val, MS_MOVN_EPI32(bit_num, n))

// max (float/int) op
#define MS_MAX_F32(bit_num, ...) MS_EXPAND(MS_MAX##bit_num##_F32(__VA_ARGS__))
#define MS_MAX_EPI32(bit_num, ...) MS_EXPAND(MS_MAX##bit_num##_EPI32(__VA_ARGS__))

// get max (float/int) op
#define MS_GET_MAX_F32(bit_num, ...) MS_EXPAND(MS_GET_MAX##bit_num##_F32(__VA_ARGS__))
#define MS_GET_MAX_EPI32(bit_num, ...) MS_EXPAND(MS_GET_MAX##bit_num##_EPI32(__VA_ARGS__))

// get max (float/int) op
#define MS_GET_SUM_F32(bit_num, ...) MS_EXPAND(MS_GET_SUM##bit_num##_F32(__VA_ARGS__))

// max n (float/int) op
#define MS_MAX_N_F32(bit_num, val, n) MS_MAX_F32(bit_num, val, MS_MOVN_F32(bit_num, n))
#define MS_MAX_N_EPI32(bit_num, val, n) MS_MAX_EPI32(bit_num, val, MS_MOVN_EPI32(bit_num, n))
#define MS_CLAMP_F32(bit_num, val, min_val, max_val) MS_MIN_F32(bit_num, MS_MAX_F32(bit_num, val, min_val), max_val)
#define MS_CLAMP_EPI32(bit_num, val, min_val, max_val) \
  MS_MIN_EPI32(bit_num, MS_MAX_EPI32(bit_num, val, min_val), max_val)

// clamp n (float/int) op
#define MS_CLAMP_N_F32(bit_num, val, min_val, max_val) \
  MS_MIN_F32(bit_num, MS_MAX_F32(bit_num, val, MS_MOV##bit_num##_F32(min_val)), MS_MOV##bit_num##_F32(max_val))
#define MS_CLAMP_N_EPI32(bit_num, val, min_val, max_val) \
  MS_MIN_EPI32(bit_num, MS_MAX_EPI32(bit_num, val, MS_MOV##bit_num##_EPI32(min_val)), MS_MOV##bit_num##_EPI32(max_val))

// mul (float/int) op
#define MS_MUL_F32(bit_num, ...) MS_EXPAND(MS_MUL##bit_num##_F32(__VA_ARGS__))
#define MS_MUL_EPI32(bit_num, ...) MS_EXPAND(MS_MUL##bit_num##_EPI32(__VA_ARGS__))
#define MS_MUL_N_F32(bit_num, val1, val2) MS_EXPAND(MS_MUL##bit_num##_F32(val1, MS_MOV##bit_num##_F32(val2)))
#define MS_MUL_N_EPI32(bit_num, val1, val2) MS_EXPAND(MS_MUL##bit_num##_EPI32(val1, MS_MOV##bit_num##_EPI32(val2)))

// fma (float/int) op
#define MS_FMADD_F32(bit_num, ...) MS_EXPAND(MS_FMADD##bit_num##_F32(__VA_ARGS__))
#define MS_FMADD_N_F32(bit_num, val1, val2) MS_EXPAND(MS_FMADD##bit_num##_F32(val1, MS_MOV##bit_num##_F32(val2)))

// fms (float/int) op
#define MS_FMSUB_F32(bit_num, ...) MS_EXPAND(MS_FMSUB##bit_num##_F32(__VA_ARGS__))
#define MS_FMSUB_N_F32(bit_num, val1, val2) MS_EXPAND(MS_FMSUB##bit_num##_F32(val1, MS_MOV##bit_num##_F32(val2)))

// square (float/int) op
#define MS_MUL_SQUARE_F32(bit_num, val) MS_EXPAND((MS_MUL##bit_num##_F32(val, val)))
#define MS_MUL_SQUARE_EPI32(bit_num, val) MS_EXPAND((MS_MUL##bit_num##_EPI32(val, val)))

// exp (float) op
#define MS_EXP_ST_F32(bit_num, ...) MS_EXPAND((simd_exp##bit_num(__VA_ARGS__)))
#define MS_EXP_F32(bit_num, ...) MS_EXPAND((simd_exp##bit_num##_f32(__VA_ARGS__)))

#define MS_CMPLT_F32(bit_num, ...) MS_EXPAND((MS_CMPLT##bit_num##_F32(__VA_ARGS__)))
#define MS_CMPLE_F32(bit_num, ...) MS_EXPAND((MS_CMPLE##bit_num##_F32(__VA_ARGS__)))
#define MS_CMPGT_F32(bit_num, ...) MS_EXPAND((MS_CMPGT##bit_num##_F32(__VA_ARGS__)))
#define MS_BLEND_F32(bit_num, ...) MS_EXPAND((MS_BLEND##bit_num##_F32(__VA_ARGS__)))

#define MS_INT16_TO_FLOAT16(bit_num, ...) MS_EXPAND((MS##bit_num##_INT16_TO_FLOAT16(__VA_ARGS__)))
#define MS_FLOAT16_TO_INT16(bit_num, ...) MS_EXPAND((MS##bit_num##_FLOAT16_TO_INT16(__VA_ARGS__)))

#define MS_INT32_TO_FLOAT16(bit_num, ...) MS_EXPAND((MS##bit_num##_INT32_TO_FLOAT16(__VA_ARGS__)))
#define MS_FLOAT16_TO_INT32(bit_num, ...) MS_EXPAND((MS##bit_num##_FLOAT16_TO_INT32(__VA_ARGS__)))

#define MS_INT32_TO_FLOAT32(bit_num, ...) MS_EXPAND((MS##bit_num##_INT32_TO_FLOAT32(__VA_ARGS__)))
#define MS_FLOAT32_TO_INT32(bit_num, ...) MS_EXPAND((MS##bit_num##_FLOAT32_TO_INT32(__VA_ARGS__)))

#define MS_INT64_TO_FLOAT32(bit_num, ...) MS_EXPAND((MS##bit_num##_INT64_TO_FLOAT32(__VA_ARGS__)))
#define MS_FLOAT32_TO_INT64(bit_num, ...) MS_EXPAND((MS##bit_num##_FLOAT32_TO_INT64(__VA_ARGS__)))

#define MS_INT64_TO_FLOAT16(bit_num, ...) MS_EXPAND((MS##bit_num##_INT64_TO_FLOAT16(__VA_ARGS__)))
#define MS_FLOAT16_TO_INT64(bit_num, ...) MS_EXPAND((MS##bit_num##_FLOAT16_TO_INT64(__VA_ARGS__)))

#define MS_INT32_TO_FLOAT64(bit_num, ...) MS_EXPAND((MS##bit_num##_INT32_TO_FLOAT64(__VA_ARGS__)))
#define MS_FLOAT64_TO_INT32(bit_num, ...) MS_EXPAND((MS##bit_num##_FLOAT64_TO_INT32(__VA_ARGS__)))

#define MS_INT64_TO_FLOAT64(bit_num, ...) MS_EXPAND((MS##bit_num##_INT64_TO_FLOAT64(__VA_ARGS__)))
#define MS_FLOAT64_TO_INT64(bit_num, ...) MS_EXPAND((MS##bit_num##_FLOAT64_TO_INT64(__VA_ARGS__)))

// enable avx512
#if defined(ENABLE_AVX512)
#define MS_SIMD_RUN_AVX512(function, ...) MS_EXPAND(function(512, 16, __VA_ARGS__))
#else
#define MS_SIMD_RUN_AVX512(function, ...)
#endif

// enable avx256
#if defined(ENABLE_AVX)
#define MS_SIMD_RUN_AVX(function, ...) MS_EXPAND(function(256, 8, __VA_ARGS__))
#else
#define MS_SIMD_RUN_AVX(function, ...)
#endif

// enable sse
#if defined(ENABLE_SSE)
#define MS_SIMD_RUN_SSE(function, ...) MS_EXPAND(function(128, 4, __VA_ARGS__))
#else
#define MS_SIMD_RUN_SSE(function, ...)
#endif

// enable neon
#if defined(ENABLE_NEON)
#define MS_SIMD_RUN_NEON(function, ...) MS_EXPAND(function(128, 4, __VA_ARGS__))
#else
#define MS_SIMD_RUN_NEON(function, ...)
#endif

// enable neon/sse
#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
#define MS_SIMD_RUN_SSEORNEON128(function, ...) MS_EXPAND(function(128, 4, __VA_ARGS__))
#else
#define MS_SIMD_RUN_SSEORNEON128(function, ...)
#endif

// scalar (c style data)
#define MS_SIMD_RUN_SCALAR(function, ...) MS_EXPAND(function(32, 1, __VA_ARGS__))

#define MS_SIMD_RUN(function, ...)                   \
  do {                                               \
    MS_SIMD_RUN_AVX512(function, __VA_ARGS__);       \
    MS_SIMD_RUN_AVX(function, __VA_ARGS__);          \
    MS_SIMD_RUN_SSEORNEON128(function, __VA_ARGS__); \
    MS_SIMD_RUN_SCALAR(function, __VA_ARGS__);       \
  } while (0)

#define MS_SIMD_RUN_NO_SCALAR(function, ...)         \
  do {                                               \
    MS_SIMD_RUN_AVX512(function, __VA_ARGS__);       \
    MS_SIMD_RUN_AVX(function, __VA_ARGS__);          \
    MS_SIMD_RUN_SSEORNEON128(function, __VA_ARGS__); \
  } while (0)

#define MS_SIMD_RUN_X86(function, ...)         \
  do {                                         \
    MS_SIMD_RUN_AVX512(function, __VA_ARGS__); \
    MS_SIMD_RUN_AVX(function, __VA_ARGS__);    \
    MS_SIMD_RUN_SSE(function, __VA_ARGS__);    \
    MS_SIMD_RUN_SCALAR(function, __VA_ARGS__); \
  } while (0)

#define MS_SIMD_RUN_X86_NO_SCALAR(function, ...) \
  do {                                           \
    MS_SIMD_RUN_AVX512(function, __VA_ARGS__);   \
    MS_SIMD_RUN_AVX(function, __VA_ARGS__);      \
    MS_SIMD_RUN_SSE(function, __VA_ARGS__);      \
  } while (0)

#endif  // NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
