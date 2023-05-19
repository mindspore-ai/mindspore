/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BINARY_FUNC_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BINARY_FUNC_CUH_

#include <limits.h>

enum class BinaryOpType {
  kGreater = 0,
  kLess = 1,
  kMaximum = 2,
  kMinimum = 3,
  kPower = 4,
  kRealDiv = 5,
  kMul = 6,
  kSub = 7,
  kAdd = 8,
  kFloorDiv = 9,
  kAbsGrad = 10,
  kDiv = 11,
  kDivNoNan = 12,
  kEqual = 13,
  kSquaredDifference = 14,
  kMod = 15,
  kFloorMod = 16,
  kAtan2 = 17,
  kGreaterEqual = 18,
  kLessEqual = 19,
  kNotEqual = 20,
  kLogicalAnd = 21,
  kLogicalOr = 22,
  kTruncateDiv = 23,
  kTruncateMod = 24,
  kComplex = 25,
  kXdivy = 26,
  kBitwiseAnd = 27,
  kBitwiseOr = 28,
  kBitwiseXor = 29,
  kMulNoNan = 30,
  kXlogy = 31,
  kInvalid = INT_MAX,
};
#endif
