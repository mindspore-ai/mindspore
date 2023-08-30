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

#ifndef MINDSPORE_CORE_MINDAPI_BASE_TYPES_H_
#define MINDSPORE_CORE_MINDAPI_BASE_TYPES_H_

#include <cstdint>

namespace mindspore {
enum CoordinateTransformMode : int64_t {
  ASYMMETRIC = 0,
  ALIGN_CORNERS = 1,
  HALF_PIXEL = 2,
  CROP_AND_RESIZE = 3,
};

enum class ResizeMethod : int64_t {
  UNKNOWN = -1,
  LINEAR = 0,
  NEAREST = 1,
  CUBIC = 2,
  AREA = 3,
};

enum class NearestMode : int64_t {
  NORMAL = 0,
  ROUND_HALF_DOWN = 1,
  ROUND_HALF_UP = 2,
  FLOOR = 3,
  CEIL = 4,
};

enum RoundMode : int64_t {
  FLOOR = 0,
  CEIL = 1,
};

enum ActivationType : int64_t {
  NO_ACTIVATION = 0,
  RELU = 1,
  SIGMOID = 2,
  RELU6 = 3,
  ELU = 4,
  LEAKY_RELU = 5,
  ABS = 6,
  RELU1 = 7,
  SOFTSIGN = 8,
  SOFTPLUS = 9,
  TANH = 10,
  SELU = 11,
  HSWISH = 12,
  HSIGMOID = 13,
  THRESHOLDRELU = 14,
  LINEAR = 15,
  HARD_TANH = 16,
  SIGN = 17,
  SWISH = 18,
  GELU = 19,
  GLU = 20,
  UNKNOWN = 21,
};

enum ReduceMode : int64_t {
  Reduce_Mean = 0,
  Reduce_Max = 1,
  Reduce_Min = 2,
  Reduce_Prod = 3,
  Reduce_Sum = 4,
  Reduce_Sum_Square = 5,
  Reduce_ASum = 6,
  Reduce_All = 7,
  Reduce_L2 = 8,
  Reduce_L1 = 9,
  Reduce_Log_Sum = 10,
  Reduce_Log_Sum_Exp = 11
};

enum EltwiseMode : int64_t {
  PROD = 0,
  SUM = 1,
  MAXIMUM = 2,
  ELTWISEMODE_UNKNOW = 3,
};

enum Reduction : int64_t {
  REDUCTION_SUM = 0,
  MEAN = 1,
  NONE = 2,
};

enum PadMode : int64_t {
  PAD = 0,
  SAME = 1,
  VALID = 2,
};

enum class LshProjectionType : int64_t {
  UNKNOWN = 0,
  SPARSE = 1,
  DENSE = 2,
};

enum PaddingMode : int64_t {
  CONSTANT = 0,
  REFLECT = 1,
  SYMMETRIC = 2,
  MODE_RESERVED = 3,
};

enum PoolMode : int64_t {
  MAX_POOLING = 0,
  MEAN_POOLING = 1,
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_MINDAPI_BASE_TYPES_H_
