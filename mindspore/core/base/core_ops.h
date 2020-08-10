/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPERATOR_OPS_H_
#define MINDSPORE_CORE_OPERATOR_OPS_H_

#include <iostream>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"

namespace mindspore {
namespace prim {
// Maths
inline const PrimitivePtr kPrimTensorAdd = std::make_shared<Primitive>("TensorAdd");
inline const PrimitivePtr kPrimMatMul = std::make_shared<Primitive>("MatMul");
inline const PrimitivePtr kPrimBatchMatMul = std::make_shared<Primitive>("BatchMatMul");
inline const PrimitivePtr kPrimMaximumGrad = std::make_shared<Primitive>("MaximumGrad");
inline const PrimitivePtr kPrimMinimumGrad = std::make_shared<Primitive>("MinimumGrad");
inline const PrimitivePtr kPrimReduceMean = std::make_shared<Primitive>("ReduceMean");
inline const PrimitivePtr kPrimReduceSum = std::make_shared<Primitive>("ReduceSum");
inline const PrimitivePtr kPrimReduceAll = std::make_shared<Primitive>("ReduceAll");
inline const PrimitivePtr kPrimReduceAny = std::make_shared<Primitive>("ReduceAny");
inline const PrimitivePtr kPrimReduceMax = std::make_shared<Primitive>("ReduceMax");
inline const PrimitivePtr kPrimReduceMin = std::make_shared<Primitive>("ReduceMin");
inline const PrimitivePtr kPrimNeg = std::make_shared<Primitive>("Neg");
inline const PrimitivePtr kPrimSub = std::make_shared<Primitive>("Sub");
inline const PrimitivePtr kPrimMul = std::make_shared<Primitive>("Mul");
inline const PrimitivePtr kPrimMinimum = std::make_shared<Primitive>("Minimum");
inline const PrimitivePtr kPrimMaximum = std::make_shared<Primitive>("Maximum");
inline const PrimitivePtr kPrimSquare = std::make_shared<Primitive>("Square");
inline const PrimitivePtr kPrimCumSum = std::make_shared<Primitive>("CumSum");
inline const PrimitivePtr kPrimCumProd = std::make_shared<Primitive>("CumProd");
inline const PrimitivePtr kPrimSubscalar = std::make_shared<Primitive>("Subscalar");
inline const PrimitivePtr kPrimInplaceAdd = std::make_shared<Primitive>("InplaceAdd");
inline const PrimitivePtr kPrimInplaceSub = std::make_shared<Primitive>("InplaceSub");
inline const PrimitivePtr kPrimPow = std::make_shared<Primitive>("Pow");
inline const PrimitivePtr kPrimRealDiv = std::make_shared<Primitive>("RealDiv");
inline const PrimitivePtr kPrimSqrt = std::make_shared<Primitive>("Sqrt");
inline const PrimitivePtr kPrimReciprocal = std::make_shared<Primitive>("Reciprocal");
inline const PrimitivePtr kPrimExpandDims = std::make_shared<Primitive>("ExpandDims");

// Statements
inline const PrimitivePtr kPrimReturn = std::make_shared<Primitive>("return");
inline const PrimitivePtr kPrimSwitch = std::make_shared<Primitive>("switch");
inline const PrimitivePtr kPrimSwitchLayer = std::make_shared<Primitive>("switch_layer");
inline const PrimitivePtr kPrimAssign = std::make_shared<Primitive>("Assign");
inline const PrimitivePtr kPrimAssignAdd = std::make_shared<Primitive>("AssignAdd");
inline const PrimitivePtr kPrimAssignSub = std::make_shared<Primitive>("AssignSub");
inline const PrimitivePtr kPrimSelect = std::make_shared<Primitive>("Select");
inline const PrimitivePtr kPrimCall = std::make_shared<Primitive>("call");

// Structures
inline const PrimitivePtr kPrimStringEqual = std::make_shared<Primitive>("string_equal");
inline const PrimitivePtr kPrimStringConcat = std::make_shared<Primitive>("string_concat");
inline const PrimitivePtr kPrimMakeTuple = std::make_shared<Primitive>("make_tuple");
inline const PrimitivePtr kPrimMakeDict = std::make_shared<Primitive>("make_dict");
inline const PrimitivePtr kPrimMakeList = std::make_shared<Primitive>("make_list");
inline const PrimitivePtr kPrimMakeKeywordArg = std::make_shared<Primitive>("make_keyword_arg");
inline const PrimitivePtr kPrimMakeSlice = std::make_shared<Primitive>("make_slice");
inline const PrimitivePtr kPrimMakeRecord = std::make_shared<Primitive>("make_record");
inline const PrimitivePtr kPrimTupleGetItem = std::make_shared<Primitive>("tuple_getitem");
inline const PrimitivePtr kPrimListGetItem = std::make_shared<Primitive>("list_getitem");
inline const PrimitivePtr kPrimArrayGetItem = std::make_shared<Primitive>("array_getitem");
inline const PrimitivePtr kPrimTupleSetItem = std::make_shared<Primitive>("tuple_setitem");
inline const PrimitivePtr kPrimListSetItem = std::make_shared<Primitive>("list_setitem");
inline const PrimitivePtr kPrimArraySetItem = std::make_shared<Primitive>("array_setitem");
inline const PrimitivePtr kPrimDictGetItem = std::make_shared<Primitive>("dict_getitem");
inline const PrimitivePtr kPrimDictSetItem = std::make_shared<Primitive>("dict_setitem");
inline const PrimitivePtr kPrimListAppend = std::make_shared<Primitive>("list_append");
inline const PrimitivePtr kPrimGetAttr = std::make_shared<Primitive>("getattr");
inline const PrimitivePtr kPrimTupleLen = std::make_shared<Primitive>("tuple_len");
inline const PrimitivePtr kPrimDictLen = std::make_shared<Primitive>("dict_len");
inline const PrimitivePtr kPrimListLen = std::make_shared<Primitive>("list_len");
inline const PrimitivePtr kPrimArrayLen = std::make_shared<Primitive>("array_len");
inline const PrimitivePtr kPrimListMap = std::make_shared<Primitive>("list_map");
inline const PrimitivePtr kPrimListReduce = std::make_shared<Primitive>("list_reduce");
inline const PrimitivePtr kPrimTupleReversed = std::make_shared<Primitive>("tuple_reversed");
inline const PrimitivePtr kPrimTileShape = std::make_shared<Primitive>("tile_shape");
inline const PrimitivePtr kPrimReducedShape = std::make_shared<Primitive>("reduced_shape");
inline const PrimitivePtr kPrimTupleDiv = std::make_shared<Primitive>("tuple_div");
inline const PrimitivePtr kPrimTupleToArray = std::make_shared<Primitive>("tuple_to_array");
inline const PrimitivePtr kPrimShapeMul = std::make_shared<Primitive>("shape_mul");
inline const PrimitivePtr kPrimGenerateShapeIndex = std::make_shared<Primitive>("generate_shape_index");
inline const PrimitivePtr kPrimGenerateInverseIndex = std::make_shared<Primitive>("generate_inverse_index");
inline const PrimitivePtr kPrimTupleEqual = std::make_shared<Primitive>("tuple_equal");
inline const PrimitivePtr kPrimListEqual = std::make_shared<Primitive>("list_equal");
inline const PrimitivePtr kPrimMakeRange = std::make_shared<Primitive>("make_range");
inline const PrimitivePtr kPrimStopGradient = std::make_shared<Primitive>("stop_gradient");
inline const PrimitivePtr kPrimExtractKeywordArg = std::make_shared<Primitive>("extract_keyword_arg");

// Debug ops
inline const PrimitivePtr kPrimScalarSummary = std::make_shared<Primitive>("ScalarSummary");
inline const PrimitivePtr kPrimImageSummary = std::make_shared<Primitive>("ImageSummary");
inline const PrimitivePtr kPrimTensorSummary = std::make_shared<Primitive>("TensorSummary");
inline const PrimitivePtr kPrimHistogramSummary = std::make_shared<Primitive>("HistogramSummary");
inline const PrimitivePtr kPrimDebug = std::make_shared<Primitive>("Debug");

// Other miscellaneous
inline const PrimitivePtr kPrimJ = std::make_shared<Primitive>("J");
inline const PrimitivePtr kPrimDepend = std::make_shared<Primitive>("Depend");
inline const PrimitivePtr kPrimPartial = std::make_shared<Primitive>("Partial");
inline const PrimitivePtr kPrimIdentity = std::make_shared<Primitive>("identity");
inline const PrimitivePtr kPrimEnvSetItem = std::make_shared<Primitive>("env_setitem");
inline const PrimitivePtr kPrimEnvGetItem = std::make_shared<Primitive>("env_getitem");
inline const PrimitivePtr kPrimEnvAdd = std::make_shared<Primitive>("env_add");
inline const PrimitivePtr kPrimMakeRefKey = std::make_shared<Primitive>("MakeRefKey");
inline const PrimitivePtr kPrimGetRefKey = std::make_shared<Primitive>("get_ref_key");
inline const PrimitivePtr kPrimGetRefValue = std::make_shared<Primitive>("get_ref_value");
inline const PrimitivePtr kPrimInsertGradientOf = std::make_shared<Primitive>("InsertGradientOf");
inline const PrimitivePtr kPrimHookBackward = std::make_shared<Primitive>("HookBackward");
inline const PrimitivePtr kPrimPrintShapeType = std::make_shared<Primitive>("PrintShapeType");
inline const PrimitivePtr kPrimSameTypeShape = std::make_shared<Primitive>("SameTypeShape");
inline const PrimitivePtr kPrimCheckBprop = std::make_shared<Primitive>("CheckBprop");
inline const PrimitivePtr kPrimPrint = std::make_shared<Primitive>("Print");
inline const PrimitivePtr kPrimMakeRef = std::make_shared<Primitive>("make_ref");
inline const PrimitivePtr kPrimStateSetItem = std::make_shared<Primitive>("state_setitem");
inline const PrimitivePtr kPrimBroadcastGradientArgs = std::make_shared<Primitive>("BroadcastGradientArgs");
inline const PrimitivePtr kPrimControlDepend = std::make_shared<Primitive>("ControlDepend");
inline const PrimitivePtr kPrimIs_ = std::make_shared<Primitive>("is_");
inline const PrimitivePtr kPrimIsNot = std::make_shared<Primitive>("is_not");
inline const PrimitivePtr kPrimInDict = std::make_shared<Primitive>("in_dict");
inline const PrimitivePtr kPrimNotInDict = std::make_shared<Primitive>("not_in_dict");
inline const PrimitivePtr kPrimMixedPrecisionCast = std::make_shared<Primitive>("mixed_precision_cast");
inline const PrimitivePtr kPrimIsConsant = std::make_shared<Primitive>("is_constant");
inline const PrimitivePtr kPrimEquivFormat = std::make_shared<Primitive>("EquivFormat");

class DoSignaturePrimitive : public Primitive {
 public:
  explicit DoSignaturePrimitive(const std::string &name, const ValuePtr &function)
      : Primitive("S-Prim-" + name), function_(function) {}

  ~DoSignaturePrimitive() override = default;

  MS_DECLARE_PARENT(DoSignaturePrimitive, Primitive)

  const ValuePtr function() const { return function_; }

 private:
  ValuePtr function_;
};
using DoSignaturePrimitivePtr = std::shared_ptr<DoSignaturePrimitive>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPERATOR_OPS_H_
