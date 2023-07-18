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
#include "ops/arithmetic_op_name.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace prim {
// Arithmetic
GVAR_DEF(PrimitivePtr, kPrimScalarAdd, std::make_shared<Primitive>(kScalarAddOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarToTensor, std::make_shared<Primitive>(kScalarToTensorOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarSub, std::make_shared<Primitive>(kScalarSubOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarMul, std::make_shared<Primitive>(kScalarMulOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarDiv, std::make_shared<Primitive>(kScalarDivOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarFloorDiv, std::make_shared<Primitive>(kScalarFloordivOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarMod, std::make_shared<Primitive>(kScalarModOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarPow, std::make_shared<Primitive>(kScalarPowOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarTrunc, std::make_shared<Primitive>(kScalarTruncOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarFloor, std::make_shared<Primitive>(kScalarFloorOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarUadd, std::make_shared<Primitive>(kScalarUaddOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarUsub, std::make_shared<Primitive>(kScalarUsubOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarCast, std::make_shared<Primitive>(kScalarCastOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarExp, std::make_shared<Primitive>("scalar_exp"));
GVAR_DEF(PrimitivePtr, kPrimScalarLog, std::make_shared<Primitive>(kScalarLogOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarSin, std::make_shared<Primitive>("scalar_sin"));
GVAR_DEF(PrimitivePtr, kPrimScalarCos, std::make_shared<Primitive>("scalar_cos"));
GVAR_DEF(PrimitivePtr, kPrimScalarTan, std::make_shared<Primitive>("scalar_tan"));
GVAR_DEF(PrimitivePtr, kPrimLinearSumAssignment, std::make_shared<Primitive>(kLinearSumAssignmentOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarBitwiseAnd, std::make_shared<Primitive>(kScalarBitwiseAndOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarBitwiseOr, std::make_shared<Primitive>(kScalarBitwiseOrOpName));
GVAR_DEF(PrimitivePtr, kPrimTensorToScalar, std::make_shared<Primitive>(kTensorToScalarOpName));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_ARITHMETIC_OPS_H_
