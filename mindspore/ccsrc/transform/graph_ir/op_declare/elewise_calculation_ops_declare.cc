/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/elewise_calculation_ops_declare.h"
#include <memory>

namespace mindspore::transform {
// Assign
INPUT_MAP(Assign) = {{1, INPUT_DESC(ref)}, {2, INPUT_DESC(value)}};
ATTR_MAP(Assign) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Assign) = {{0, OUTPUT_DESC(ref)}};
REG_ADPT_DESC(Assign, prim::kPrimAssign->name(), ADPT_DESC(Assign))
REG_ADPT_DESC(StateSetItem, prim::kPrimStateSetItem->name(), ADPT_DESC(Assign))

// add
INPUT_MAP(Add) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Add) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Add) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Add, prim::kPrimTensorAdd->name(),
              std::make_shared<OpAdapterDesc>(std::make_shared<OpAdapter<Add>>(ExtraAttr({{"mode", MakeValue(1)}})),
                                              std::make_shared<OpAdapter<Add>>(ExtraAttr({{"mode", MakeValue(1)}}))))

// GreaterEqual
INPUT_MAP(GreaterEqual) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(GreaterEqual) = EMPTY_ATTR_MAP;
OUTPUT_MAP(GreaterEqual) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(GreaterEqual, kNameGreaterEqual, ADPT_DESC(GreaterEqual))

// AssignAdd
INPUT_MAP(AssignAdd) = {{1, INPUT_DESC(ref)}, {2, INPUT_DESC(value)}};
ATTR_MAP(AssignAdd) = EMPTY_ATTR_MAP;
OUTPUT_MAP(AssignAdd) = {{0, OUTPUT_DESC(ref)}};
REG_ADPT_DESC(AssignAdd, kNameAssignAdd, ADPT_DESC(AssignAdd))

// AssignSub
INPUT_MAP(AssignSub) = {{1, INPUT_DESC(var)}, {2, INPUT_DESC(value)}};
ATTR_MAP(AssignSub) = EMPTY_ATTR_MAP;
OUTPUT_MAP(AssignSub) = {{0, OUTPUT_DESC(var)}};
REG_ADPT_DESC(AssignSub, kNameAssignSub, ADPT_DESC(AssignSub))

// Cos
INPUT_MAP(Cos) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Cos) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Cos) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Cos, kNameCos, ADPT_DESC(Cos))

// Acos
INPUT_MAP(Acos) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Acos) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Acos) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Acos, kNameACos, ADPT_DESC(Acos))

// AcosGrad
INPUT_MAP(AcosGrad) = {{1, INPUT_DESC(y)}, {2, INPUT_DESC(dy)}};
ATTR_MAP(AcosGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(AcosGrad) = {{0, OUTPUT_DESC(z)}};
REG_ADPT_DESC(AcosGrad, kNameACosGrad, ADPT_DESC(AcosGrad))

// Acosh
INPUT_MAP(Acosh) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Acosh) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Acosh) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Acosh, kNameAcosh, ADPT_DESC(Acosh))

// AcoshGrad
INPUT_MAP(AcoshGrad) = {{1, INPUT_DESC(y)}, {2, INPUT_DESC(dy)}};
ATTR_MAP(AcoshGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(AcoshGrad) = {{0, OUTPUT_DESC(z)}};
REG_ADPT_DESC(AcoshGrad, kNameAcoshGrad, ADPT_DESC(AcoshGrad))

// Floor
INPUT_MAP(Floor) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Floor) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Floor) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Floor, kNameFloor, ADPT_DESC(Floor))

// FloorDiv
INPUT_MAP(FloorDiv) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(FloorDiv) = EMPTY_ATTR_MAP;
OUTPUT_MAP(FloorDiv) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(FloorDiv, kNameFloorDiv, ADPT_DESC(FloorDiv))

// FloorMod
INPUT_MAP(FloorMod) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(FloorMod) = EMPTY_ATTR_MAP;
OUTPUT_MAP(FloorMod) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(FloorMod, kNameFloorMod, ADPT_DESC(FloorMod))

// Sin
INPUT_MAP(Sin) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Sin) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Sin) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Sin, kNameSin, ADPT_DESC(Sin))

// Exp
INPUT_MAP(Exp) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Exp) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Exp) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Exp, kNameExp, ADPT_DESC(Exp))

// BiasAdd
INPUT_MAP(BiasAdd) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(bias)}};
ATTR_MAP(BiasAdd) = {{"data_format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(BiasAdd) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(BiasAdd, kNameBiasAdd, ADPT_DESC(BiasAdd))

// ZerosLike
INPUT_MAP(ZerosLike) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ZerosLike) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ZerosLike) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ZerosLike, kNameZerosLike, ADPT_DESC(ZerosLike))

// OnesLike
INPUT_MAP(OnesLike) = {{1, INPUT_DESC(x)}};
ATTR_MAP(OnesLike) = EMPTY_ATTR_MAP;
OUTPUT_MAP(OnesLike) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(OnesLike, kNameOnesLike, ADPT_DESC(OnesLike))

// ArgMaxD
INPUT_MAP(ArgMaxD) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ArgMaxD) = {{"axis", ATTR_DESC(dimension, AnyTraits<int>())},
                     {"output_type", ATTR_DESC(dtype, AnyTraits<GEType>())}};
OUTPUT_MAP(ArgMaxD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ArgMaxD, kNameArgmax, ADPT_DESC(ArgMaxD))

// ArgMinD
INPUT_MAP(ArgMinD) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ArgMinD) = {{"axis", ATTR_DESC(dimension, AnyTraits<int>())},
                     {"output_type", ATTR_DESC(dtype, AnyTraits<GEType>())}};
OUTPUT_MAP(ArgMinD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ArgMinD, kNameArgmin, ADPT_DESC(ArgMinD))

// ArgMaxWithValue
INPUT_MAP(ArgMaxWithValue) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ArgMaxWithValue) = {{"axis", ATTR_DESC(dimension, AnyTraits<int>())},
                             {"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ArgMaxWithValue) = {{0, OUTPUT_DESC(indice)}, {1, OUTPUT_DESC(values)}};
REG_ADPT_DESC(ArgMaxWithValue, kNameArgMaxWithValue, ADPT_DESC(ArgMaxWithValue))

// ArgMinWithValue
INPUT_MAP(ArgMinWithValue) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ArgMinWithValue) = {{"axis", ATTR_DESC(dimension, AnyTraits<int>())},
                             {"keep_dims", ATTR_DESC(keep_dims, AnyTraits<bool>())}};
OUTPUT_MAP(ArgMinWithValue) = {{0, OUTPUT_DESC(indice)}, {1, OUTPUT_DESC(values)}};
REG_ADPT_DESC(ArgMinWithValue, kNameArgMinWithValue, ADPT_DESC(ArgMinWithValue))

// RsqrtGrad
INPUT_MAP(RsqrtGrad) = {{1, INPUT_DESC(y)}, {2, INPUT_DESC(dy)}};
ATTR_MAP(RsqrtGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(RsqrtGrad) = {{0, OUTPUT_DESC(z)}};
REG_ADPT_DESC(RsqrtGrad, kNameRsqrtGrad, ADPT_DESC(RsqrtGrad))

// SqrtGrad
INPUT_MAP(SqrtGrad) = {{1, INPUT_DESC(y)}, {2, INPUT_DESC(dy)}};
ATTR_MAP(SqrtGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SqrtGrad) = {{0, OUTPUT_DESC(z)}};
REG_ADPT_DESC(SqrtGrad, kNameSqrtGrad, ADPT_DESC(SqrtGrad))

// ReciprocalGrad
INPUT_MAP(ReciprocalGrad) = {{1, INPUT_DESC(y)}, {2, INPUT_DESC(dy)}};
ATTR_MAP(ReciprocalGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ReciprocalGrad) = {{0, OUTPUT_DESC(z)}};
REG_ADPT_DESC(ReciprocalGrad, kNameReciprocalGrad, ADPT_DESC(ReciprocalGrad))

// AddN
INPUT_MAP(AddN) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(AddN) = {{1, DYN_INPUT_DESC(x)}};
ATTR_MAP(AddN) = {{"n", ATTR_DESC(N, AnyTraits<int64_t>())}};
OUTPUT_MAP(AddN) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(AddN, kNameAddN, ADPT_DESC(AddN))

// Mul
INPUT_MAP(Mul) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Mul) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Mul) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Mul, prim::kPrimMul->name(), ADPT_DESC(Mul))

// RealDiv
INPUT_MAP(RealDiv) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(RealDiv) = EMPTY_ATTR_MAP;
OUTPUT_MAP(RealDiv) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(RealDiv, kNameRealDiv, ADPT_DESC(RealDiv))

// Cast
INPUT_MAP(Cast) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(Cast) = {{2, ATTR_DESC(dst_type, AnyTraits<GEType>())}};
ATTR_MAP(Cast) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Cast) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Cast, prim::kPrimCast->name(), ADPT_DESC(Cast))

// Reciprocal
INPUT_MAP(Reciprocal) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Reciprocal) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Reciprocal) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Reciprocal, kNameReciprocal, ADPT_DESC(Reciprocal))

// Sub
INPUT_MAP(Sub) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Sub) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Sub) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Sub, prim::kPrimSub->name(), ADPT_DESC(Sub))

// Neg
INPUT_MAP(Neg) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Neg) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Neg) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Neg, prim::kPrimNeg->name(), ADPT_DESC(Neg))

// Less
INPUT_MAP(Less) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Less) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Less) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Less, kNameLess, ADPT_DESC(Less))

// Rsqrt
INPUT_MAP(Rsqrt) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Rsqrt) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Rsqrt) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Rsqrt, kNameRsqrt, ADPT_DESC(Rsqrt))

// Sqrt
INPUT_MAP(Sqrt) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Sqrt) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Sqrt) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Sqrt, kNameSqrt, ADPT_DESC(Sqrt))

// Square
INPUT_MAP(Square) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Square) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Square) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Square, kNameSquare, ADPT_DESC(Square))

// SquareSumAll
INPUT_MAP(SquareSumAll) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(SquareSumAll) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SquareSumAll) = {{0, OUTPUT_DESC(y1)}, {1, OUTPUT_DESC(y2)}};
REG_ADPT_DESC(SquareSumAll, kNameSquareSumAll, ADPT_DESC(SquareSumAll))

// Maximum
INPUT_MAP(Maximum) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Maximum) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Maximum) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Maximum, prim::kPrimMaximum->name(), ADPT_DESC(Maximum))

// Minimum
INPUT_MAP(Minimum) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Minimum) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Minimum) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Minimum, prim::kPrimMinimum->name(), ADPT_DESC(Minimum))

// MaximumGrad
INPUT_MAP(MaximumGrad) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(grads)}};
ATTR_MAP(MaximumGrad) = {{"grad_x", ATTR_DESC(grad_x, AnyTraits<bool>())},
                         {"grad_y", ATTR_DESC(grad_y, AnyTraits<bool>())}};
OUTPUT_MAP(MaximumGrad) = {{0, OUTPUT_DESC(y1)}, {1, OUTPUT_DESC(y2)}};
REG_ADPT_DESC(MaximumGrad, prim::kPrimMaximumGrad->name(), ADPT_DESC(MaximumGrad))

// MinimumGrad
INPUT_MAP(MinimumGrad) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}, {3, INPUT_DESC(grads)}};
ATTR_MAP(MinimumGrad) = {{"grad_x", ATTR_DESC(grad_x, AnyTraits<bool>())},
                         {"grad_y", ATTR_DESC(grad_y, AnyTraits<bool>())}};
OUTPUT_MAP(MinimumGrad) = {{0, OUTPUT_DESC(y1)}, {1, OUTPUT_DESC(y2)}};
REG_ADPT_DESC(MinimumGrad, prim::kPrimMinimumGrad->name(), ADPT_DESC(MinimumGrad))

// Pow
INPUT_MAP(Pow) = {
  {1, INPUT_DESC(x1)},
  {2, INPUT_DESC(x2)},
};
ATTR_MAP(Pow) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Pow) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Pow, kNamePow, ADPT_DESC(Pow))

// Equal
INPUT_MAP(Equal) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Equal) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Equal) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Equal, kNameEqual, ADPT_DESC(Equal))

// NotEqual
INPUT_MAP(NotEqual) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(NotEqual) = EMPTY_ATTR_MAP;
OUTPUT_MAP(NotEqual) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(NotEqual, kNameNotEqual, ADPT_DESC(NotEqual))

// Log
INPUT_MAP(Log) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Log) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Log) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Log, kNameLog, ADPT_DESC(Log))

// LogicalAnd
INPUT_MAP(LogicalAnd) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(LogicalAnd) = EMPTY_ATTR_MAP;
OUTPUT_MAP(LogicalAnd) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(LogicalAnd, kNameLogicalAnd, ADPT_DESC(LogicalAnd))

// LogicalOr
INPUT_MAP(LogicalOr) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(LogicalOr) = EMPTY_ATTR_MAP;
OUTPUT_MAP(LogicalOr) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(LogicalOr, kNameLogicalOr, ADPT_DESC(LogicalOr))

// LogicalNot
INPUT_MAP(LogicalNot) = {{1, INPUT_DESC(x)}};
ATTR_MAP(LogicalNot) = EMPTY_ATTR_MAP;
OUTPUT_MAP(LogicalNot) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(LogicalNot, kNameLogicalNot, ADPT_DESC(LogicalNot))

// Greater
INPUT_MAP(Greater) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Greater) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Greater) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Greater, kNameGreater, ADPT_DESC(Greater))

// LessEqual
INPUT_MAP(LessEqual) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(LessEqual) = EMPTY_ATTR_MAP;
OUTPUT_MAP(LessEqual) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(LessEqual, kNameLessEqual, ADPT_DESC(LessEqual))

// Abs
INPUT_MAP(Abs) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Abs) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Abs) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Abs, kNameAbs, ADPT_DESC(Abs))

// AbsGrad
INPUT_MAP(AbsGrad) = {{1, INPUT_DESC(y)}, {2, INPUT_DESC(dy)}};
ATTR_MAP(AbsGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(AbsGrad) = {{0, OUTPUT_DESC(z)}};
REG_ADPT_DESC(AbsGrad, kNameAbsGrad, ADPT_DESC(AbsGrad))

// Sign
INPUT_MAP(Sign) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Sign) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Sign) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Sign, kNameSign, ADPT_DESC(Sign))

// Round
INPUT_MAP(Round) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Round) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Round) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Round, kNameRound, ADPT_DESC(Round))

// Atan2
INPUT_MAP(Atan2) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Atan2) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Atan2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Atan2, kNameAtan2, ADPT_DESC(Atan2))
}  // namespace mindspore::transform
