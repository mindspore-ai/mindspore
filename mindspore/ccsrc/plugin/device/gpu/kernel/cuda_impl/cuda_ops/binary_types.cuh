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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BINARY_TYPES_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BINARY_TYPES_CUH_
#include <limits.h>
enum class BinaryOpType {
  // compare
  kGreater = 0,
  kLess = 1,
  kEqual = 2,
  kGreaterEqual = 3,
  kLessEqual = 4,
  kNotEqual = 5,
  kLogicalAnd = 6,
  kLogicalOr = 7,
  // math
  kMaximum = 8,
  kMinimum = 9,
  kAdd = 10,
  kSub = 11,
  kMul = 12,
  kDiv = 13,
  kPow = 14,
  kRealDiv = 15,
  kBitwiseAnd = 16,
  kBitwiseOr = 17,
  kBitwiseXor = 18,
  kMod = 19,
  kFloorMod = 20,
  kSquaredDifference = 21,
  kAtan2 = 22,
  kTruncateDiv = 23,
  kTruncateMod = 24,
  kAbsGrad = 25,
  kFloorDiv = 26,
  kDivNoNan = 27,
  kMulNoNan = 28,
  kXlogy = 29,
  kXdivy = 30,
  // complex
  kComplex = 31,
  // Ext
  kAddExt = 32,
  kSubExt = 33,
  kInvalid = INT_MAX,
};

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BINARY_TYPES_CUH_
