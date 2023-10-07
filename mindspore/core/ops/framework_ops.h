/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_BASE_FRAMEWORK_OPS_H_
#define MINDSPORE_CORE_BASE_FRAMEWORK_OPS_H_

#include <iostream>
#include <memory>
#include <string>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/value.h"
#include "ops/framework_op_name.h"
#include "utils/flags.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace prim {
/*
 * The origin core_ops.h has been decomposed to following files:
 * arithmetic_ops.h, array_ops.h, comparison_ops.h,
 * image_ops.h, lite_ops.h, math_ops.h, nn_ops.h,
 * nn_optimizer_ops.h, other_ops.h, conv_pool_ops.h,
 * random_ops.h, sequence_ops.h, sparse_ops.h,
 * sparse_tensor_ops.h, structure_ops.h.
 *
 * The const strings, which were in core_ops.h and common/utils/utils.h
 * were moved to the following *_op_name files:
 * framework_op_name.h, arithmetic_op_name.h, array_op_name.h,
 * comparison_op_name.h, image_op_name.h, lite_op_name.h,
 * math_op_name.h, nn_op_name.h, nn_optimizer_op_name.h,
 * other_op_name.h, conv_pool_op_name.h, random_op_name.h,
 * sequence_op_name.h, sparse_op_name.h, structure_op_name.h.
 */
GVAR_DEF(ValuePtr, kValueOne, std::make_shared<Int64Imm>(1));
#define COMMA ,
GVAR_DEF(mindspore::HashMap<std::string COMMA ValuePtr>, kSideEffectPropagate,
         {{mindspore::GRAPH_FLAG_SIDE_EFFECT_PROPAGATE COMMA kValueOne}});
#undef COMMA
GVAR_DEF(PrimitivePtr, kPrimIdentityMath, std::make_shared<Primitive>("Identity", kSideEffectPropagate));

// Shape
GVAR_DEF(PrimitivePtr, kPrimShapeMul, std::make_shared<Primitive>("shape_mul"));
GVAR_DEF(PrimitivePtr, kPrimShapeMulGrad, std::make_shared<Primitive>("ShapeMulGrad"));
GVAR_DEF(PrimitivePtr, kPrimShape, std::make_shared<Primitive>("Shape"));
GVAR_DEF(PrimitivePtr, kPrimDType, std::make_shared<Primitive>("DType"));

// SideEffectPropagate
GVAR_DEF(PrimitivePtr, kPrimDepend, std::make_shared<Primitive>(kDependOpName, kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimPartial, std::make_shared<Primitive>("Partial", kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimIdentity, std::make_shared<Primitive>(kidentityOpName, kSideEffectPropagate));

// Other primitive not used by backend but used in core;
GVAR_DEF(PrimitivePtr, kPrimStateSetItem, std::make_shared<Primitive>("state_setitem"));
GVAR_DEF(PrimitivePtr, kPrimJ, std::make_shared<Primitive>(kJOpName, kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimVmap, std::make_shared<Primitive>(kVmapOpName, kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimShard, std::make_shared<Primitive>("Shard", kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimTaylor, std::make_shared<Primitive>(kTaylorOpName));

// Control ops
GVAR_DEF(PrimitivePtr, kPrimMerge, std::make_shared<Primitive>("Merge"));

// Other miscellaneous
GVAR_DEF(PrimitivePtr, kPrimEnvironCreate, std::make_shared<Primitive>(kEnvironCreateOpName));
GVAR_DEF(PrimitivePtr, kPrimEnvironSet, std::make_shared<Primitive>(kEnvironSetOpName));
GVAR_DEF(PrimitivePtr, kPrimEnvironGet, std::make_shared<Primitive>(kEnvironGetOpName));
GVAR_DEF(PrimitivePtr, kPrimEnvironAdd, std::make_shared<Primitive>(kEnvironAddOpName));
GVAR_DEF(PrimitivePtr, kPrimEnvironDestroyAll, std::make_shared<Primitive>(kEnvironDestroyAllOpName));
GVAR_DEF(PrimitivePtr, kPrimSetSize, std::make_shared<Primitive>(kSetSizeOpName));

// Other miscellaneous
GVAR_DEF(PrimitivePtr, kPrimPyFunc, std::make_shared<Primitive>("PyFunc"));
GVAR_DEF(PrimitivePtr, kPrimCheckValid, std::make_shared<Primitive>("CheckValid"));
GVAR_DEF(PrimitivePtr, kPrimReformat, std::make_shared<Primitive>("Reformat"));
GVAR_DEF(PrimitivePtr, kPrimLoad, std::make_shared<Primitive>(kLoadOpName));
GVAR_DEF(PrimitivePtr, kPrimMutable, std::make_shared<Primitive>(kMutableOpName));
GVAR_DEF(PrimitivePtr, kPrimGetGrad, std::make_shared<Primitive>(kGetGradOpName));
GVAR_DEF(PrimitivePtr, kPrimHookBackward, std::make_shared<Primitive>("HookBackward"));
GVAR_DEF(PrimitivePtr, kPrimCellBackwardHook, std::make_shared<Primitive>("CellBackwardHook"));
GVAR_DEF(PrimitivePtr, kPrimPrintShapeType, std::make_shared<Primitive>("PrintShapeType"));
GVAR_DEF(PrimitivePtr, kPrimSameTypeShape, std::make_shared<Primitive>("SameTypeShape"));
GVAR_DEF(PrimitivePtr, kPrimPrint, std::make_shared<Primitive>("Print"));
GVAR_DEF(PrimitivePtr, kPrimIs_, std::make_shared<Primitive>("is_"));
GVAR_DEF(PrimitivePtr, kPrimIsNot, std::make_shared<Primitive>("is_not"));
GVAR_DEF(PrimitivePtr, kPrimInDict, std::make_shared<Primitive>("in_dict"));
GVAR_DEF(PrimitivePtr, kPrimNotInDict, std::make_shared<Primitive>("not_in_dict"));
GVAR_DEF(PrimitivePtr, kPrimIsConstant, std::make_shared<Primitive>("IsConstant"));
GVAR_DEF(PrimitivePtr, kPrimEquivFormat, std::make_shared<Primitive>("EquivFormat"));
GVAR_DEF(PrimitivePtr, kPrimLshProjection, std::make_shared<Primitive>("LshProjection"));
GVAR_DEF(PrimitivePtr, kPrimHashtableLookup, std::make_shared<Primitive>("HashtableLookup"));
GVAR_DEF(PrimitivePtr, kPrimCustomPredict, std::make_shared<Primitive>("CustomPredict"));
GVAR_DEF(PrimitivePtr, kPrimPriorBox, std::make_shared<Primitive>("PriorBox"));
GVAR_DEF(PrimitivePtr, kPrimQuantDTypeCast, std::make_shared<Primitive>("QuantDTypeCast"));
GVAR_DEF(PrimitivePtr, kPrimWhile, std::make_shared<Primitive>("While"));
GVAR_DEF(PrimitivePtr, kPrimPull, std::make_shared<Primitive>("Pull"));
GVAR_DEF(PrimitivePtr, kPrimPush, std::make_shared<Primitive>("Push"));

// JIT Fallback ops
// We add IO side-effect for them in advance.
GVAR_DEF(PrimitivePtr, kPrimPyInterpret,
         std::make_shared<Primitive>("PyInterpret", mindspore::HashMap<std::string, ValuePtr>(
                                                      {{std::string(GRAPH_FLAG_SIDE_EFFECT_IO), MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimPyExecute,
         std::make_shared<Primitive>("PyExecute", mindspore::HashMap<std::string, ValuePtr>(
                                                    {{std::string(GRAPH_FLAG_SIDE_EFFECT_IO), MakeValue(true)},
                                                     {std::string("primitive_target"), MakeValue("CPU")}})));
GVAR_DEF(PrimitivePtr, kPrimSetAttr,
         std::make_shared<Primitive>(kSetAttrOpName, mindspore::HashMap<std::string, ValuePtr>(
                                                       {{std::string(GRAPH_FLAG_SIDE_EFFECT_IO), MakeValue(true)}})));

// Used to build graph which have keyword arguments
GVAR_DEF(PrimitivePtr, kPrimExtractKeywordArg, std::make_shared<Primitive>("extract_keyword_arg"));
GVAR_DEF(PrimitivePtr, kPrimMakeDict, std::make_shared<Primitive>("make_dict"));

// Custom
GVAR_DEF(PrimitivePtr, kPrimCustom, std::make_shared<Primitive>("Custom"));

// Type introspection
GVAR_DEF(PrimitivePtr, kPrimTypeOf, std::make_shared<Primitive>("typeof"));
GVAR_DEF(PrimitivePtr, kPrimTopTypeOf, std::make_shared<Primitive>("TopTypeof"));
GVAR_DEF(PrimitivePtr, kPrimHasType, std::make_shared<Primitive>("hastype"));
GVAR_DEF(PrimitivePtr, kPrimIsInstance, std::make_shared<Primitive>(kIsInstanceOpName));
GVAR_DEF(PrimitivePtr, kPrimResolve, std::make_shared<Primitive>("resolve"));
GVAR_DEF(PrimitivePtr, kPrimEmbed, std::make_shared<Primitive>("embed"));
GVAR_DEF(PrimitivePtr, kPrimRefToEmbed, std::make_shared<Primitive>("RefToEmbed"));
GVAR_DEF(PrimitivePtr, kPrimCreateInstance, std::make_shared<Primitive>("create_instance"));
GVAR_DEF(PrimitivePtr, kPrimWithEnter, std::make_shared<Primitive>("with_enter"));
GVAR_DEF(PrimitivePtr, kPrimWithExit, std::make_shared<Primitive>("with_exit"));

// Other miscellaneous
GVAR_DEF(PrimitivePtr, kPrimInsertGradientOf, std::make_shared<Primitive>("InsertGradientOf"));
GVAR_DEF(PrimitivePtr, kPrimCheckBprop, std::make_shared<Primitive>("CheckBprop"));
GVAR_DEF(PrimitivePtr, kPrimMixedPrecisionCast, std::make_shared<Primitive>("MixedPrecisionCast"));

// Sponge Ops
GVAR_DEF(PrimitivePtr, kPrimAngleAtomEnergy, std::make_shared<Primitive>("AngleAtomEnergy"));

// Framework ops
GVAR_DEF(PrimitivePtr, kPrimStreamSend, std::make_shared<Primitive>(kStreamSendOpName));
GVAR_DEF(PrimitivePtr, kPrimStreamRecv, std::make_shared<Primitive>(kStreamRecvOpName));
GVAR_DEF(PrimitivePtr, kPrimSliceToIndices, std::make_shared<Primitive>("SliceToIndices"));
GVAR_DEF(PrimitivePtr, kPrimTensorMove, std::make_shared<Primitive>("TensorMove"));
GVAR_DEF(PrimitivePtr, kPrimMemCpyAsync, std::make_shared<Primitive>("memcpy_async"));
GVAR_DEF(PrimitivePtr, kPrimSend, std::make_shared<Primitive>("Send"));
GVAR_DEF(PrimitivePtr, kPrimReceive, std::make_shared<Primitive>("Receive"));
GVAR_DEF(PrimitivePtr, kPrimRpcSend, std::make_shared<Primitive>("RpcSend"));
GVAR_DEF(PrimitivePtr, kPrimRpcRecv, std::make_shared<Primitive>("RpcRecv"));
GVAR_DEF(PrimitivePtr, kPrimUpdateState, std::make_shared<Primitive>(kUpdateStateOpName));
GVAR_DEF(PrimitivePtr, kPrimReturn, std::make_shared<Primitive>(kReturnOpName));
GVAR_DEF(PrimitivePtr, kPrimSwitch, std::make_shared<Primitive>(kSwitchOpName));
GVAR_DEF(PrimitivePtr, kPrimSelect, std::make_shared<Primitive>(kSelectOpName));
GVAR_DEF(PrimitivePtr, kPrimCall, std::make_shared<Primitive>(kCallOpName));
GVAR_DEF(PrimitivePtr, kPrimRaise,
         std::make_shared<Primitive>("raise", mindspore::HashMap<std::string, ValuePtr>(
                                                {{std::string(GRAPH_FLAG_SIDE_EFFECT_IO), MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimCallInline, std::make_shared<Primitive>("call_inline"));
GVAR_DEF(PrimitivePtr, kPrimSwitchLayer, std::make_shared<Primitive>("switch_layer"));
GVAR_DEF(PrimitivePtr, kPrimStringUpper, std::make_shared<Primitive>(kStringUpperOpName));
GVAR_DEF(PrimitivePtr, kPrimStringLower, std::make_shared<Primitive>(kStringLowerOpName));
GVAR_DEF(PrimitivePtr, kPrimFormat, std::make_shared<Primitive>("Format"));

// Pack
GVAR_DEF(PrimitivePtr, kPrimPackFunc, std::make_shared<Primitive>(kPackFuncOpName));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_FRAMEWORK_OPS_H_
