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

#ifndef MINDSPORE_OPS_PRIMITIVE_STRUCTURE_OPS_H_
#define MINDSPORE_OPS_PRIMITIVE_STRUCTURE_OPS_H_

#include <iostream>
#include <memory>
#include <string>

#include "ir/anf.h"
#include "ir/primitive.h"
#include "primitive/structure_op_name.h"
#include "utils/hash_map.h"
#include "primitive/image_op_name.h"
#include "ir/core_ops_primitive.h"

namespace mindspore {
namespace prim {
// String
GVAR_DEF(PrimitivePtr, kPrimStringEq, std::make_shared<Primitive>(kStringEqOpName));
GVAR_DEF(PrimitivePtr, kPrimStringLt, std::make_shared<Primitive>(kStringLtOpName));
GVAR_DEF(PrimitivePtr, kPrimStringGt, std::make_shared<Primitive>(kStringGtOpName));
GVAR_DEF(PrimitivePtr, kPrimStringLe, std::make_shared<Primitive>(kStringLeOpName));
GVAR_DEF(PrimitivePtr, kPrimStringGe, std::make_shared<Primitive>(kStringGeOpName));
GVAR_DEF(PrimitivePtr, kPrimStringConcat, std::make_shared<Primitive>(kStringConcatOpName));
GVAR_DEF(PrimitivePtr, kPrimStringNot, std::make_shared<Primitive>(kStringNotOpName));
GVAR_DEF(PrimitivePtr, kPrimStringIn, std::make_shared<Primitive>(kStringInOpName));
GVAR_DEF(PrimitivePtr, kPrimStringMul, std::make_shared<Primitive>(kStringMulOpName));
GVAR_DEF(PrimitivePtr, kPrimStringGetItem, std::make_shared<Primitive>(kStringGetItemOpName));

// Stack ops
GVAR_DEF(PrimitivePtr, kPrimStackInit, std::make_shared<Primitive>("StackInit"));
GVAR_DEF(PrimitivePtr, kPrimStackDestroy, std::make_shared<Primitive>("StackDestroy"));
GVAR_DEF(PrimitivePtr, kPrimStackPush, std::make_shared<Primitive>("StackPush"));
GVAR_DEF(PrimitivePtr, kPrimStackPop, std::make_shared<Primitive>("StackPop"));

// TensorList
GVAR_DEF(PrimitivePtr, kPrimTensorListFromTensor, std::make_shared<Primitive>("TensorListFromTensor"));
GVAR_DEF(PrimitivePtr, kPrimTensorListReserve, std::make_shared<Primitive>("TensorListReserve"));
GVAR_DEF(PrimitivePtr, kPrimTensorListStack, std::make_shared<Primitive>("TensorListStack"));
GVAR_DEF(PrimitivePtr, kPrimTensorListSetItem, std::make_shared<Primitive>("TensorListSetItem"));

// Structures
GVAR_DEF(PrimitivePtr, kPrimMakeKeywordArg, std::make_shared<Primitive>("make_keyword_arg"));
GVAR_DEF(PrimitivePtr, kPrimReducedShape, std::make_shared<Primitive>("reduced_shape"));
GVAR_DEF(PrimitivePtr, kPrimStopGradient, std::make_shared<Primitive>("StopGradient"));
GVAR_DEF(PrimitivePtr, kPrimFakeBprop, std::make_shared<Primitive>("fake_bprop"));
GVAR_DEF(PrimitivePtr, kPrimBroadcastGradientArgs, std::make_shared<Primitive>("BroadcastGradientArgs"));
GVAR_DEF(PrimitivePtr, kPrimDynamicBroadcastGradientArgs,
         std::make_shared<Primitive>(kDynamicBroadcastGradientArgsOpName));
GVAR_DEF(PrimitivePtr, kPrimDtypeToEnum, std::make_shared<Primitive>("DtypeToEnum"));
GVAR_DEF(PrimitivePtr, kPrimEnumToDtype, std::make_shared<Primitive>("EnumToDtype"));

// Statements
GVAR_DEF(PrimitivePtr, kPrimUnroll, std::make_shared<Primitive>("Unroll"));
GVAR_DEF(PrimitivePtr, kPrimVmapStackAssign,
         std::make_shared<Primitive>(
           kVmapStackAssignOpName,
           mindspore::HashMap<std::string, ValuePtr>({{std::string(GRAPH_FLAG_SIDE_EFFECT_MEM), MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimVmapUnstackAssign,
         std::make_shared<Primitive>(
           kVmapUnstackAssignOpName,
           mindspore::HashMap<std::string, ValuePtr>({{std::string(GRAPH_FLAG_SIDE_EFFECT_MEM), MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimMakeSlice, std::make_shared<Primitive>(kMakeSliceOpName));
GVAR_DEF(PrimitivePtr, kPrimSliceGetItem, std::make_shared<Primitive>(kSliceGetItemOpName));
GVAR_DEF(PrimitivePtr, kPrimGetAttr, std::make_shared<Primitive>("getattr"));
GVAR_DEF(PrimitivePtr, kPrimTileShape, std::make_shared<Primitive>("tile_shape"));
GVAR_DEF(PrimitivePtr, kPrimGenerateShapeIndex, std::make_shared<Primitive>("generate_shape_index"));
GVAR_DEF(PrimitivePtr, kPrimGenerateInverseIndex, std::make_shared<Primitive>("generate_inverse_index"));
GVAR_DEF(PrimitivePtr, kPrimCond, std::make_shared<Primitive>(kCondOpName));
GVAR_DEF(PrimitivePtr, kPrimJoinedStr,
         std::make_shared<Primitive>(kJoinedStrOpName, mindspore::HashMap<std::string, ValuePtr>(
                                                         {{std::string("primitive_target"), MakeValue("CPU")},
                                                          {std::string("variable_length_inputs"), MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimTileSize, std::make_shared<Primitive>("TileSize"));
GVAR_DEF(PrimitivePtr, kPrimNormalizeSlice, std::make_shared<Primitive>("NormalizeSlice"));
GVAR_DEF(PrimitivePtr, kPrimNormalizeDimIndex, std::make_shared<Primitive>("NormalizeDimIndex"));
GVAR_DEF(PrimitivePtr, kPrimRemakeTupleIndex, std::make_shared<Primitive>("RemakeTupleIndex"));
GVAR_DEF(PrimitivePtr, kPrimNormalizeTupleIndex, std::make_shared<Primitive>("NormalizeTupleIndex"));
GVAR_DEF(PrimitivePtr, kPrimEllipsisToSlice, std::make_shared<Primitive>("EllipsisToSlice"));
GVAR_DEF(PrimitivePtr, kPrimGetSqueezeSliceShape, std::make_shared<Primitive>("GetSqueezeSliceShape"));
GVAR_DEF(PrimitivePtr, kPrimRemoveExpandedDims, std::make_shared<Primitive>("RemoveExpandedDims"));
GVAR_DEF(PrimitivePtr, kPrimGetTupleIndexInfo, std::make_shared<Primitive>("GetTupleIndexInfo"));

// Debug ops
GVAR_DEF(PrimitivePtr, kPrimAssert, std::make_shared<Primitive>("Assert"));
GVAR_DEF(PrimitivePtr, kPrimHistogramFixedWidth, std::make_shared<Primitive>("HistogramFixedWidth"));
GVAR_DEF(PrimitivePtr, kPrimTensorDump, std::make_shared<Primitive>(kTensorDump));
GVAR_DEF(PrimitivePtr, kPrimTensorReport, std::make_shared<Primitive>(kTensorReport));
GVAR_DEF(PrimitivePtr, kPrimDebug, std::make_shared<Primitive>("Debug"));

// Dynamic shape testing
GVAR_DEF(PrimitivePtr, kPrimConvertToDynamic, std::make_shared<Primitive>("ConvertToDynamic"));
GVAR_DEF(PrimitivePtr, kPrimGpuConvertToDynamicShape, std::make_shared<Primitive>("GpuConvertToDynamicShape"));
GVAR_DEF(PrimitivePtr, kPrimErrorOnDynamicShapeInput, std::make_shared<Primitive>("ErrorOnDynamicShapeInput"));

// Dynamic shape.
GVAR_DEF(PrimitivePtr, kPrimIsDimUnknown, std::make_shared<Primitive>("IsDimUnKnown"));
GVAR_DEF(PrimitivePtr, kPrimIsShapeUnknown, std::make_shared<Primitive>("IsShapeUnKnown"));
GVAR_DEF(PrimitivePtr, kPrimIsElementUnknown, std::make_shared<Primitive>("IsElementUnknown"));
GVAR_DEF(PrimitivePtr, kPrimIsTensorBoolCond, std::make_shared<Primitive>("IsTensorBoolCond"));

// GetNext
GVAR_DEF(PrimitivePtr, kPrimGetNext, std::make_shared<Primitive>(kGetNextOpName));
GVAR_DEF(PrimitivePtr, kPrimGetNextFromQueue, std::make_shared<Primitive>(kGetNextFromQueueOpName));
GVAR_DEF(PrimitivePtr, kPrimDynamicGetNextV2, std::make_shared<Primitive>(kDynamicGetNextV2OpName));
GVAR_DEF(PrimitivePtr, kPrimDynamicGetNextAscend, std::make_shared<Primitive>(kDynamicGetNextAscendOpName));

class DoTransPrimitiveFunction : public Primitive {
 public:
  explicit DoTransPrimitiveFunction(const PrimitivePtr &prim) : Primitive("T-PrimFunc-" + prim->name()), prim_(prim) {}

  ~DoTransPrimitiveFunction() override = default;

  MS_DECLARE_PARENT(DoTransPrimitiveFunction, Primitive)

  const PrimitivePtr function() const { return prim_; }

 private:
  PrimitivePtr prim_;
};
using DoTransPrimitiveFunctionPtr = std::shared_ptr<DoTransPrimitiveFunction>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_OPS_PRIMITIVE_STRUCTURE_OPS_H_
