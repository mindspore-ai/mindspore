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

#ifndef MINDSPORE_LITE_NNACL_OP_BASE_H_
#define MINDSPORE_LITE_NNACL_OP_BASE_H_

#ifdef ENABLE_ARM
#include <arm_neon.h>
#endif

#ifdef ENABLE_SSE
#include <x86intrin.h>
#endif

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define C2NUM 2
#define C4NUM 4
#define C6NUM 6
#define C8NUM 8
#define C12NUM 12
#define C16NUM 16
#define TILE_NUM 8

#define MSMIN(x, y) ((x) < (y) ? (x) : (y))
#define MSMAX(x, y) ((x) > (y) ? (x) : (y))

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define UP_ROUND(x, y) (((x) + (y) - (1)) / (y) * (y))
#define UP_ROUND_DIV(x, y) (x % y == 0 ? (x / y) : (x / y) + 1)
#define DOWN_DIV(x, y) (((x) - (y) + (1)) / (y))

#define MSVALID(left, x, right) (MSMIN((MSMAX(left, x)), right))

#define COMM_SHAPE_SIZE 4
#define MAX_SHAPE_SIZE 8

#define DIMENSION_4D 4
#define DIMENSION_6D 6
#define kInputIndex 0
#define kWeightIndex 1
#define kBiasIndex 2
#define kOutputIndex 0
#define kNHWC_N 0
#define kNHWC_H 1
#define kNHWC_W 2
#define kNHWC_C 3
#define kInputSize1 2
#define kInputSize2 3
#define MAX_LEN 256

typedef enum LiteDataType {
  kDataTypeFloat,
  kDataTypeFloat16,
  kDataTypeInt,
  kDataTypeInt8,
  KDataTypeBool,
} LiteDataType;

typedef enum DataOrder {
  RowMajor,
  ColMajor,
} DataOrder;

typedef struct OpParameter {
  char name_[100];
  int type_;
  int thread_num_;
} OpParameter;

typedef struct QuantArg {
  float scale_;
  int32_t zp_;
} QuantArg;

typedef struct QuantMulArg {
  int32_t multiplier_;
  int left_shift_;
  int right_shift_;
} QuantMulArg;

typedef enum ActType { ActType_No, ActType_Relu, ActType_Sigmod, ActType_Relu6, ActType_Prelu } ActType;
typedef enum PadMode { Pad_No, Pad_Same, Pad_Valid } PadMode;
typedef enum RoundingMode { Rounding_No, Rounding_Away_from_zero, Rounding_Up } RoundingMode;
typedef enum CalFixedMultiplierMode {
  Method_No,
  Method_SinglePrecision,
  Method_DoublePrecision
} CalFixedMultiplierMode;

#ifdef ENABLE_ARM
#define MS_FLOAT32X4 float32x4_t
#define MS_LDQ_F32 vld1q_f32
#define MS_ADDQ_F32 vaddq_f32
#define MS_MOVQ_F32 vmovq_n_f32
#define MS_DUPQ_F32 vdupq_n_f32  // It is recommended to replace with MS_MOVQ_F32.
#define MS_SUBQ_F32 vsubq_f32
#define MS_MLAQ_F32(src1, src2, src3) vmlaq_f32(src1, src2, src3)
#define MS_STQ_F32 vst1q_f32
#define MS_MAXQ_F32 vmaxq_f32
#define MS_MINQ_F32 vminq_f32
#define MS_MULQ_F32(src1, src2) vmulq_n_f32(src1, src2)
#elif defined(ENABLE_SSE)
#define MS_FLOAT32X4 __m128
#define MS_LDQ_F32 _mm_loadu_ps
#define MS_ADDQ_F32 _mm_add_ps
#define MS_MOVQ_F32 _mm_set_ps1
#define MS_DUPQ_F32 _mm_load_ps1  // It is recommended to replace with MS_MOVQ_F32.
#define MS_MLAQ_F32(src1, src2, src3) _mm_add_ps(src1, _mm_mul_ps(src2, src3))
#define MS_STQ_F32 _mm_storeu_ps
#define MS_SUBQ_F32 _mm_sub_ps
#define MS_MAXQ_F32 _mm_max_ps
#define MS_MINQ_F32 _mm_min_ps
#define MS_MULQ_F32(src1, src2) _mm_mul_ps(src1, _mm_set_ps1(src2))
#endif

#endif  // MINDSPORE_LITE_NNACL_OP_BASE_H_
