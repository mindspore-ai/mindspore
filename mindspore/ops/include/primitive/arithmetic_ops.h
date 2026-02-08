/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_OPS_PRIMITIVE_ARITHMETIC_OPS_H_
#define MINDSPORE_OPS_PRIMITIVE_ARITHMETIC_OPS_H_

#include <memory>

#include "primitive/arithmetic_op_name.h"
#include "ir/anf.h"
#include "ir/primitive.h"

namespace mindspore {
namespace prim {
// Arithmetic
GVAR_DEF(PrimitivePtr, kPrimScalarTrunc, std::make_shared<Primitive>(kScalarTruncOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarFloor, std::make_shared<Primitive>(kScalarFloorOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarExp, std::make_shared<Primitive>("scalar_exp"));
GVAR_DEF(PrimitivePtr, kPrimScalarSin, std::make_shared<Primitive>("scalar_sin"));
GVAR_DEF(PrimitivePtr, kPrimScalarCos, std::make_shared<Primitive>("scalar_cos"));
GVAR_DEF(PrimitivePtr, kPrimScalarTan, std::make_shared<Primitive>("scalar_tan"));
GVAR_DEF(PrimitivePtr, kPrimLinearSumAssignment, std::make_shared<Primitive>(kLinearSumAssignmentOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarBitwiseAnd, std::make_shared<Primitive>(kScalarBitwiseAndOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarBitwiseOr, std::make_shared<Primitive>(kScalarBitwiseOrOpName));
GVAR_DEF(PrimitivePtr, kPrimTensorToScalar, std::make_shared<Primitive>(kTensorToScalarOpName));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_OPS_PRIMITIVE_ARITHMETIC_OPS_H_
