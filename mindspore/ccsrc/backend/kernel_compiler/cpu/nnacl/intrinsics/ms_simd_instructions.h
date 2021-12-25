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

#ifndef MINDSPORE_NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#define MINDSPORE_NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#include <math.h>

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
#define MS_MUL32_F32(value1, value2) ((value1) * (value2))
#define MS_MUL32_EPI32(value1, value2) ((value1) * (value2))
#define MS_DIV32_F32(value1, value2) ((value1) / (value2))
#define MS_DIV32_EPI32(value1, value2) ((value1) / (value2))
#define MS_MIN32_F32(value1, value2) (fmin((value1), (value2)))
#define MS_MIN32_EPI32(value1, value2) ((value1) < (value2) ? (value1) : (value2))
#define MS_MAX32_F32(value1, value2) (fmax((value1), (value2)))
#define MS_MAX32_EPI32(value1, value2) ((value1) > (value2) ? (value1) : (value2))

// define (float/int) data
#define MS_FLOAT_32xN(byte_num) MS_FLOAT32##X##byte_num
#define MS_INT_32xN(byte_num) MS_INT32##X##byte_num

// move (float/int) data
#define MS_MOVN_F32(byte_num, ...) MS_MOV##byte_num##_F32(__VA_ARGS__)
#define MS_MOVN_EPI32(byte_num, ...) MS_MOV##byte_num##_EPI32(__VA_ARGS__)

// load (float/int) data
#define MS_LD_F32(bit_num, ...) MS_LD##bit_num##_F32(__VA_ARGS__)
#define MS_LD_EPI32(bit_num, ...) MS_LD##bit_num##_EPI32(__VA_ARGS__)

// stored (float/int) data
#define MS_ST_F32(bit_num, ...) MS_ST##bit_num##_F32(__VA_ARGS__)
#define MS_ST_EPI32(bit_num, ...) MS_ST##bit_num##_EPI32(__VA_ARGS__)

// add (float/int) op
#define MS_ADD_F32(bit_num, ...) MS_ADD##bit_num##_F32(__VA_ARGS__)
#define MS_ADD_EPI32(bit_num, ...) MS_ADD##bit_num##_EPI32(__VA_ARGS__)

// div (float/int) op
#define MS_DIV_F32(bit_num, ...) MS_DIV##bit_num##_F32(__VA_ARGS__)
#define MS_DIV_EPI32(bit_num, ...) MS_DIV##bit_num##_EPI32(__VA_ARGS__)

// div (float/int) op
#define MS_DIV_N_F32(bit_num, val1, val2) MS_DIV##bit_num##_F32(val1, MS_MOV##bit_num##_F32(val2))
#define MS_DIV_N_EPI32(bit_num, val1, val2) MS_DIV##bit_num##_EPI32(val1, MS_MOV##bit_num##_EPI32(val2))

// min (float/int) op
#define MS_MIN_F32(bit_num, ...) MS_MIN##bit_num##_F32(__VA_ARGS__)
#define MS_MIN_EPI32(bit_num, ...) MS_MIN##bit_num##_EPI32(__VA_ARGS__)

// max (float/int) op
#define MS_MAX_F32(bit_num, ...) MS_MAX##bit_num##_F32(__VA_ARGS__)
#define MS_MAX_EPI32(bit_num, ...) MS_MAX##bit_num##_EPI32(__VA_ARGS__)

// mul (float/int) op
#define MS_MUL_F32(bit_num, ...) MS_MUL##bit_num##_F32(__VA_ARGS__)
#define MS_MUL_EPI32(bit_num, ...) MS_MUL##bit_num##_EPI32(__VA_ARGS__)

// square (float/int) op
#define MS_MUL_SQUARE_F32(bit_num, val) (MS_MUL##bit_num##_F32(val, val))
#define MS_MUL_SQUARE_EPI32(bit_num, val) (MS_MUL##bit_num##_EPI32(val, val))

// enable avx512
#if defined(ENABLE_AVX512)
#define MS_SIMD_RUN_AVX512(function, ...) function(512, 16, __VA_ARGS__)
#else
#define MS_SIMD_RUN_AVX512(function, ...)
#endif

// enable avx256
#if defined(ENABLE_AVX)
#define MS_SIMD_RUN_AVX(function, ...) function(256, 8, __VA_ARGS__)
#else
#define MS_SIMD_RUN_AVX(function, ...)
#endif

// enable neon/sse
#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
#define MS_SIMD_RUN_SSEORNEON128(function, ...) function(128, 4, __VA_ARGS__)
#else
#define MS_SIMD_RUN_SSEORNEON128(function, ...)
#endif

// scalar (c style data)
#define MS_SIMD_RUN_SCALAR(function, ...) function(32, 1, __VA_ARGS__)

#define MS_SIMD_RUN(function, ...)                 \
  MS_SIMD_RUN_AVX512(function, __VA_ARGS__);       \
  MS_SIMD_RUN_AVX(function, __VA_ARGS__);          \
  MS_SIMD_RUN_SSEORNEON128(function, __VA_ARGS__); \
  MS_SIMD_RUN_SCALAR(function, __VA_ARGS__);

#define MS_SIMD_RUN_NO_SCALAR(function, ...) \
  MS_SIMD_RUN_AVX512(function, __VA_ARGS__); \
  MS_SIMD_RUN_AVX(function, __VA_ARGS__);    \
  MS_SIMD_RUN_SSEORNEON128(function, __VA_ARGS__);

#endif  // MINDSPORE_NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
