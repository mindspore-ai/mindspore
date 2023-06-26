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

#ifndef MINDSPORE_CORE_BASE_ARITHMETIC_OPS_H_
#define MINDSPORE_CORE_BASE_ARITHMETIC_OPS_H_

#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace prim {
// Arithmetic
constexpr auto kScalarToTensor = "ScalarToTensor";
constexpr auto kScalarAdd = "ScalarAdd";
constexpr auto kScalarSub = "ScalarSub";
constexpr auto kScalarMul = "ScalarMul";
constexpr auto kScalarDiv = "ScalarDiv";
constexpr auto kScalarFloordiv = "ScalarFloordiv";
constexpr auto kScalarMod = "ScalarMod";
constexpr auto kScalarPow = "ScalarPow";
constexpr auto kScalarLog = "ScalarLog";
constexpr auto kScalarTrunc = "ScalarTrunc";
constexpr auto kScalarFloor = "ScalarFloor";
constexpr auto kScalarUadd = "ScalarUadd";
constexpr auto kScalarUsub = "ScalarUsub";
constexpr auto kScalarBitwiseAnd = "bit_and";
constexpr auto kScalarBitwiseOr = "bit_or";
constexpr auto kScalarCast = "ScalarCast";
constexpr auto kAcoshGrad = "AcoshGrad";
constexpr auto kTrunc = "Trunc";
constexpr auto kEuclideanNorm = "EuclideanNorm";
constexpr auto kGer = "Ger";
constexpr auto kZeta = "Zeta";
constexpr auto kLinearSumAssignment = "LinearSumAssignment";
constexpr auto kTensorToScalar = "TensorToScalar";

// Arithmetic
GVAR_DEF(PrimitivePtr, kPrimScalarAdd, std::make_shared<Primitive>(kScalarAdd));
GVAR_DEF(PrimitivePtr, kPrimScalarToTensor, std::make_shared<Primitive>(kScalarToTensor));
GVAR_DEF(PrimitivePtr, kPrimScalarSub, std::make_shared<Primitive>(kScalarSub));
GVAR_DEF(PrimitivePtr, kPrimScalarMul, std::make_shared<Primitive>(kScalarMul));
GVAR_DEF(PrimitivePtr, kPrimScalarDiv, std::make_shared<Primitive>(kScalarDiv));
GVAR_DEF(PrimitivePtr, kPrimScalarFloorDiv, std::make_shared<Primitive>(kScalarFloordiv));
GVAR_DEF(PrimitivePtr, kPrimScalarMod, std::make_shared<Primitive>(kScalarMod));
GVAR_DEF(PrimitivePtr, kPrimScalarPow, std::make_shared<Primitive>(kScalarPow));
GVAR_DEF(PrimitivePtr, kPrimScalarTrunc, std::make_shared<Primitive>(kScalarTrunc));
GVAR_DEF(PrimitivePtr, kPrimScalarFloor, std::make_shared<Primitive>(kScalarFloor));
GVAR_DEF(PrimitivePtr, kPrimScalarUadd, std::make_shared<Primitive>(kScalarUadd));
GVAR_DEF(PrimitivePtr, kPrimScalarUsub, std::make_shared<Primitive>(kScalarUsub));
GVAR_DEF(PrimitivePtr, kPrimScalarCast, std::make_shared<Primitive>(kScalarCast));
GVAR_DEF(PrimitivePtr, kPrimScalarExp, std::make_shared<Primitive>("scalar_exp"));
GVAR_DEF(PrimitivePtr, kPrimScalarLog, std::make_shared<Primitive>(kScalarLog));
GVAR_DEF(PrimitivePtr, kPrimScalarSin, std::make_shared<Primitive>("scalar_sin"));
GVAR_DEF(PrimitivePtr, kPrimScalarCos, std::make_shared<Primitive>("scalar_cos"));
GVAR_DEF(PrimitivePtr, kPrimScalarTan, std::make_shared<Primitive>("scalar_tan"));
GVAR_DEF(PrimitivePtr, kPrimLinearSumAssignment, std::make_shared<Primitive>(kLinearSumAssignment));
GVAR_DEF(PrimitivePtr, kPrimScalarBitwiseAnd, std::make_shared<Primitive>(kScalarBitwiseAnd));
GVAR_DEF(PrimitivePtr, kPrimScalarBitwiseOr, std::make_shared<Primitive>(kScalarBitwiseOr));
GVAR_DEF(PrimitivePtr, kPrimTensorToScalar, std::make_shared<Primitive>(kTensorToScalar));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_ARITHMETIC_OPS_H_
