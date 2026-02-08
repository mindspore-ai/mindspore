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

#ifndef MINDSPORE_OPS_PRIMITIVE_COMPARISON_OPS_H_
#define MINDSPORE_OPS_PRIMITIVE_COMPARISON_OPS_H_

#include <memory>

#include "primitive/comparison_op_name.h"
#include "ir/anf.h"
#include "ir/primitive.h"

namespace mindspore {
namespace prim {
GVAR_DEF(PrimitivePtr, kPrimScalarNe, std::make_shared<Primitive>("scalar_ne"));
GVAR_DEF(PrimitivePtr, kPrimBoolAnd, std::make_shared<Primitive>("bool_and"));
GVAR_DEF(PrimitivePtr, kPrimBoolOr, std::make_shared<Primitive>("bool_or"));
GVAR_DEF(PrimitivePtr, kPrimBoolEq, std::make_shared<Primitive>("bool_eq"));
GVAR_DEF(PrimitivePtr, kPrimBitAnd, std::make_shared<Primitive>("bit_and"));
GVAR_DEF(PrimitivePtr, kPrimBitOr, std::make_shared<Primitive>("bit_or"));
GVAR_DEF(PrimitivePtr, kPrimBitXor, std::make_shared<Primitive>("bit_xor"));
GVAR_DEF(PrimitivePtr, kPrimBitLeftShift, std::make_shared<Primitive>("bit_left_shift"));
GVAR_DEF(PrimitivePtr, kPrimBitRightShift, std::make_shared<Primitive>("bit_right_shift"));
GVAR_DEF(PrimitivePtr, kPrimEqualCount, std::make_shared<Primitive>("EqualCount"));
GVAR_DEF(PrimitivePtr, kPrimApproximateEqual, std::make_shared<Primitive>("ApproximateEqual"));
GVAR_DEF(PrimitivePtr, kPrimDistribute, std::make_shared<Primitive>("distribute"));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_OPS_PRIMITIVE_COMPARISON_OPS_H_
