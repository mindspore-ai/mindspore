/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_OP_BASE_H_
#define MINDSPORE_NNACL_OP_BASE_H_

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#if defined(ENABLE_AVX) || defined(ENABLE_SSE) || defined(ENABLE_ARM)
#include "nnacl/intrinsics/ms_simd_instructions.h"
#endif

#define C1NUM 1
#define C2NUM 2
#define C3NUM 3
#define C4NUM 4
#define C5NUM 5
#define C6NUM 6
#define C8NUM 8
#define C12NUM 12
#define C16NUM 16
#define C20NUM 20
#define C24NUM 24
#define C32NUM 32
#define C40NUM 40
#define C64NUM 64
#define TILE_NUM 8

#define MSMIN(x, y) ((x) < (y) ? (x) : (y))
#define MSMAX(x, y) ((x) > (y) ? (x) : (y))
#define MSCEIL(x) (int)((x) + (((x) - (int)(x)) > 0 ? 1 : 0))

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define UP_ROUND(x, y) (((x) + (y) - (1)) / (y) * (y))
#define UP_ROUND_DIV(x, y) (x % y == 0 ? (x / y) : (x / y) + 1)
#define DOWN_DIV(x, y) ((x) / (y))
#define DOWN_ROUND(x, y) ((x) / (y) * (y))

#define MSVALID(left, x, right) (MSMIN((MSMAX(left, x)), right))
#define SIZE_MUL_OVERFLOW(x, y) (((x) == 0) ? false : (SIZE_MAX / (x)) < (y))
#define INT_MUL_OVERFLOW(x, y)                                                             \
  ((x == 0) ? false                                                                        \
            : ((x) > 0 ? ((y >= 0) ? (INT_MAX / (x)) < (y) : (INT_MAX / (x)) < (-1 * (y))) \
                       : ((y >= 0) ? (INT_MAX / (x)) > (-1 * (y)) : (INT_MAX / (x)) > (y))))

#define INT_MUL_OVERFLOW_THRESHOLD(x, y, threshold)                                                \
  ((x == 0) ? false                                                                                \
            : ((x) > 0 ? ((y >= 0) ? ((threshold) / (x)) < (y) : ((threshold) / (x)) < (-1 * (y))) \
                       : ((y >= 0) ? ((threshold) / (x)) > (-1 * (y)) : ((threshold) / (x)) > (y))))

#define INT_ADD_OVERFLOW(x, y) (INT_MAX - (x)) < (y)

#define INT_ADD_OVERFLOW_THRESHOLD(x, y, threshold) ((threshold) - (x)) < (y)

#define COMM_SHAPE_SIZE 4
#define MAX_SHAPE_SIZE 8

#define FIRST_INPUT 0
#define SECOND_INPUT 1
#define THIRD_INPUT 2
#define FOURTH_INPUT 3
#define FIFTH_INPUT 4

#define DIMENSION_1D 1
#define DIMENSION_2D 2
#define DIMENSION_3D 3
#define DIMENSION_4D 4
#define DIMENSION_5D 5
#define DIMENSION_6D 6
#define DIMENSION_7D 7
#define DIMENSION_8D 8
#define DIMENSION_10D 10
#define DIMENSION_11D 11
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
#define MAX_AXIS_SIZE 6
#define MAX_LEN 256
#define FLT16_MAX 65504
#define NNACL_NC4HW4 13
#define kDefaulLiteMaxSpinCount 300000
#define kDefaulLiteMinSpinCount 1
#define kDefaulLiteIosSpinCount 1
#define INPUT_MAX_NUM 10

#if ENABLE_HIGH_PERFORMANCE
#define MS_CHECK_TRUE_RET(value, errcode)
#define MS_CHECK_TRUE_RET_VOID(value)
#define MS_CHECK_FALSE(value, errcode)
#define MS_CHECK_TRUE_MSG(value, errcode, msg)
#define MS_CHECK_FALSE_MSG(value, errcode, msg)
#define MS_CHECK_LT(value1, value2, errcode)
#define MS_CHECK_GT(value1, value2, errcode)
#define MS_CHECK_LE(value1, value2, errcode)
#define MS_CHECK_GE(value1, value2, errcode)
#define MS_CHECK_PTR_IF_NULL(ptr)

#define MS_CHECK_INT_MUL_NOT_OVERFLOW(value1, value2, errcode)
#define MS_CHECK_INT_ADD_NOT_OVERFLOW(value1, value2, errcode)

#define NNACL_CHECK_ZERO_RETURN_ERR(val)
#define NNACL_CHECK_ZERO_RETURN(val)
#define NNACL_CHECK_NULL_RETURN_ERR(ptr)
#define NNACL_CHECK_NULL_RETURN_VOID(ptr)
#else
// Check whether value is true, if not return 'errcode'
#define MS_CHECK_TRUE_RET(value, errcode) \
  do {                                    \
    if (!(value)) {                       \
      return errcode;                     \
    }                                     \
  } while (0)

#define MS_CHECK_TRUE_RET_VOID(value) \
  do {                                \
    if (!(value)) {                   \
      return;                         \
    }                                 \
  } while (0)

// Check whether value is false, if not return 'errcode'
#define MS_CHECK_FALSE(value, errcode) \
  do {                                 \
    if ((value)) {                     \
      return errcode;                  \
    }                                  \
  } while (0)

// Check whether value is true, if not return 'errcode'
// and print error string msg
#define MS_CHECK_TRUE_MSG(value, errcode, msg) \
  do {                                         \
    if (!(value)) {                            \
      MS_LOG(ERROR) << #msg;                   \
      return errcode;                          \
    }                                          \
  } while (0)

#define MS_CHECK_FALSE_MSG(value, errcode, msg) \
  do {                                          \
    if ((value)) {                              \
      MS_LOG(ERROR) << #msg;                    \
      return errcode;                           \
    }                                           \
  } while (0)

#define MS_CHECK_LT(value1, value2, errcode)                                         \
  do {                                                                               \
    if ((value1) >= (value2)) {                                                      \
      MS_LOG(ERROR) << "check ge fail, value1: " << value1 << " value2: " << value2; \
      return errcode;                                                                \
    }                                                                                \
  } while (0)

#define MS_CHECK_GT(value1, value2, errcode)                                         \
  do {                                                                               \
    if ((value1) <= (value2)) {                                                      \
      MS_LOG(ERROR) << "check gt fail, value1: " << value1 << " value2: " << value2; \
      return errcode;                                                                \
    }                                                                                \
  } while (0)

#define MS_CHECK_LE(value1, value2, errcode)                                         \
  do {                                                                               \
    if ((value1) > (value2)) {                                                       \
      MS_LOG(ERROR) << "check le fail, value1: " << value1 << " value2: " << value2; \
      return errcode;                                                                \
    }                                                                                \
  } while (0)

#define MS_CHECK_GE(value1, value2, errcode)                                         \
  do {                                                                               \
    if ((value1) < (value2)) {                                                       \
      MS_LOG(ERROR) << "check ge fail, value1: " << value1 << " value2: " << value2; \
      return errcode;                                                                \
    }                                                                                \
  } while (0)

#define MS_CHECK_PTR_IF_NULL(ptr)                                \
  do {                                                           \
    if ((ptr) == nullptr) {                                      \
      MS_LOG(ERROR) << ": The pointer[" << #ptr << "] is null."; \
      return;                                                    \
    }                                                            \
  } while (0)

#define MS_CHECK_INT_MUL_NOT_OVERFLOW(value1, value2, errcode) \
  MS_CHECK_TRUE_RET(!(INT_MUL_OVERFLOW(value1, value2)), errcode)
#define MS_CHECK_INT_ADD_NOT_OVERFLOW(value1, value2, errcode) \
  MS_CHECK_TRUE_RET(!(INT_ADD_OVERFLOW(value1, value2)), errcode)

#define NNACL_CHECK_ZERO_RETURN_ERR(val) \
  do {                                   \
    if ((val) == 0) {                    \
      return NNACL_ERR;                  \
    }                                    \
  } while (0)

#define NNACL_CHECK_ZERO_RETURN(val) \
  do {                               \
    if ((val) == 0) {                \
      return;                        \
    }                                \
  } while (0)

#define NNACL_CHECK_NULL_RETURN_ERR(ptr) \
  do {                                   \
    if ((ptr) == NULL) {                 \
      return NNACL_NULL_PTR;             \
    }                                    \
  } while (0)

#define NNACL_CHECK_NULL_RETURN_VOID(ptr) \
  do {                                    \
    if ((ptr) == NULL) {                  \
      return;                             \
    }                                     \
  } while (0)

#endif

typedef enum LiteDataType {
  kDataTypeFloat,
  kDataTypeFloat16,
  kDataTypeInt,
  kDataTypeInt8,
  kDataTypeBool,
  kDataTypeFloat64
} LiteDataType;

typedef enum DataOrder {
  RowMajor,
  ColMajor,
} DataOrder;

typedef struct OpParameter {
  char name_[100];
  int type_;
  int thread_num_;
  int quant_type_;
  bool is_train_session_;
  bool is_zero_shape_;
  void (*destroy_func_)(struct OpParameter *param);
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
typedef enum PadMode { Pad_pad, Pad_same, Pad_valid } PadMode;
typedef enum RoundingMode { Rounding_No, Rounding_Away_from_zero, Rounding_Up } RoundingMode;
typedef enum CalFixedMultiplierMode {
  Method_No,
  Method_SinglePrecision,
  Method_DoublePrecision
} CalFixedMultiplierMode;

#endif  // MINDSPORE_NNACL_OP_BASE_H_
