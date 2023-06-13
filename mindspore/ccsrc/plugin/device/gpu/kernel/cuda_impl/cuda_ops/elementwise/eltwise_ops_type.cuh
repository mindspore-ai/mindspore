/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_OPS_TYPE_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_OPS_TYPE_CUH_
#include <limits.h>
enum class ElwiseOpType {
  // unary ops
  kSin = 0,
  kCos = 1,
  kTan = 2,
  kSinh = 3,
  kCosh = 4,
  kTanh = 5,
  kAsin = 6,
  kAcos = 7,
  kAtan = 8,
  kAsinh = 9,
  kAcosh = 10,
  kAtanh = 11,
  kErfinv = 12,
  kErf = 13,
  kErfc = 14,
  kAbs = 15,
  kSqrt = 16,
  kInvert = 17,
  kRsqrt = 18,
  kSign = 19,
  kSquare = 20,
  kExp = 21,
  kSigmoid = 22,
  kLogicalNot = 23,
  kReLU = 24,
  kLog = 25,
  kNeg = 26,
  kReciprocal = 27,
  kExpm1 = 28,
  kMish = 29,
  kSiLU = 30,
  kSoftsign = 31,
  kTrunc = 32,
  kFloor = 33,
  kCeil = 34,
  kRound = 35,
  kOnesLike = 36,
  kRint = 37,
  kConj = 38,
  kLog1p = 39,
  kComplexAbs = 40,
  kReal = 41,
  kImag = 42,
  // binary ops
  kAsinGrad = 200,
  kACosGrad = 201,
  kAtanGrad = 202,
  kAsinhGrad = 203,
  kAcoshGrad = 204,
  kTanhGrad = 205,
  kSqrtGrad = 206,
  kRsqrtGrad = 207,
  kReciprocalGrad = 208,
  kZeta = 209,
  kSigmoidGrad = 210,
  kSiLUGrad = 211,
  kInvalid = INT_MAX,
};
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_OPS_TYPE_CUH_
