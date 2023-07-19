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

#ifndef MINDSPORE_CORE_BASE_COMPARISON_OPS_H_
#define MINDSPORE_CORE_BASE_COMPARISON_OPS_H_

#include <memory>
#include "ops/comparison_op_name.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace prim {
GVAR_DEF(PrimitivePtr, kPrimScalarEq, std::make_shared<Primitive>(kScalarEqOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarLt, std::make_shared<Primitive>(kScalarLtOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarGt, std::make_shared<Primitive>(kScalarGtOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarNe, std::make_shared<Primitive>("scalar_ne"));
GVAR_DEF(PrimitivePtr, kPrimScalarLe, std::make_shared<Primitive>(kScalarLeOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarGe, std::make_shared<Primitive>(kScalarGeOpName));
GVAR_DEF(PrimitivePtr, kPrimScalarBool, std::make_shared<Primitive>(kScalarBoolOpName));
GVAR_DEF(PrimitivePtr, kPrimBoolNot, std::make_shared<Primitive>(kBoolNotOpName));
GVAR_DEF(PrimitivePtr, kPrimBoolAnd, std::make_shared<Primitive>("bool_and"));
GVAR_DEF(PrimitivePtr, kPrimBoolOr, std::make_shared<Primitive>("bool_or"));
GVAR_DEF(PrimitivePtr, kPrimBoolEq, std::make_shared<Primitive>("bool_eq"));
GVAR_DEF(PrimitivePtr, kPrimBitAnd, std::make_shared<Primitive>("bit_and"));
GVAR_DEF(PrimitivePtr, kPrimBitOr, std::make_shared<Primitive>("bit_or"));
GVAR_DEF(PrimitivePtr, kPrimBitXor, std::make_shared<Primitive>("bit_xor"));
GVAR_DEF(PrimitivePtr, kPrimBitLeftShift, std::make_shared<Primitive>("bit_left_shift"));
GVAR_DEF(PrimitivePtr, kPrimBitRightShift, std::make_shared<Primitive>("bit_right_shift"));
GVAR_DEF(PrimitivePtr, kPrimGreater, std::make_shared<Primitive>("Greater"));
GVAR_DEF(PrimitivePtr, kPrimGreaterEqual, std::make_shared<Primitive>("GreaterEqual"));
GVAR_DEF(PrimitivePtr, kPrimLess, std::make_shared<Primitive>("Less"));
GVAR_DEF(PrimitivePtr, kPrimLessEqual, std::make_shared<Primitive>("LessEqual"));
GVAR_DEF(PrimitivePtr, kPrimEqual, std::make_shared<Primitive>("Equal"));
GVAR_DEF(PrimitivePtr, kPrimNotEqual, std::make_shared<Primitive>(kNotEqualOpName));
GVAR_DEF(PrimitivePtr, kPrimLogicalAnd, std::make_shared<Primitive>("LogicalAnd"));
GVAR_DEF(PrimitivePtr, kPrimLogicalOr, std::make_shared<Primitive>("LogicalOr"));
GVAR_DEF(PrimitivePtr, kPrimLogicalNot, std::make_shared<Primitive>("LogicalNot"));
GVAR_DEF(PrimitivePtr, kPrimLogicalXor, std::make_shared<Primitive>(kLogicalXorOpName));
GVAR_DEF(PrimitivePtr, kPrimEqualCount, std::make_shared<Primitive>("EqualCount"));
GVAR_DEF(PrimitivePtr, kPrimApproximateEqual, std::make_shared<Primitive>("ApproximateEqual"));
GVAR_DEF(PrimitivePtr, kPrimDistribute, std::make_shared<Primitive>("distribute"));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_COMPARISON_OPS_H_
