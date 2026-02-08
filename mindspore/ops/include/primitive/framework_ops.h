/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_OPS_PRIMITIVE_FRAMEWORK_OPS_H_
#define MINDSPORE_OPS_PRIMITIVE_FRAMEWORK_OPS_H_

#include <iostream>
#include <memory>
#include <string>

#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/value.h"
#include "primitive/framework_op_name.h"
#include "utils/flags.h"
#include "utils/hash_map.h"
#include "ir/core_ops_primitive.h"

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
GVAR_DEF(PrimitivePtr, kPrimIdentityMath, std::make_shared<Primitive>("Identity", kSideEffectPropagate));

// Shape
GVAR_DEF(PrimitivePtr, kPrimShapeMul, std::make_shared<Primitive>("shape_mul"));
GVAR_DEF(PrimitivePtr, kPrimShapeMulGrad, std::make_shared<Primitive>("ShapeMulGrad"));
GVAR_DEF(PrimitivePtr, kPrimDType, std::make_shared<Primitive>("DType"));
GVAR_DEF(PrimitivePtr, kPrimDTypeId, std::make_shared<Primitive>("DTypeId"));

// SideEffectPropagate
GVAR_DEF(PrimitivePtr, kPrimidentity, std::make_shared<Primitive>(kidentityOpName, kSideEffectPropagate));

// Other primitive not used by backend but used in core;
GVAR_DEF(PrimitivePtr, kPrimReshard, std::make_shared<Primitive>("Reshard"));
GVAR_DEF(PrimitivePtr, kPrimReusing, std::make_shared<Primitive>("Reusing"));
// Control ops
GVAR_DEF(PrimitivePtr, kPrimMerge, std::make_shared<Primitive>("Merge"));
GVAR_DEF(PrimitivePtr, kPrimWhileLoop, std::make_shared<Primitive>("WhileLoop"));
GVAR_DEF(PrimitivePtr, kPrimScan, std::make_shared<Primitive>("Scan"));
GVAR_DEF(PrimitivePtr, kPrimForiLoop, std::make_shared<Primitive>("ForiLoop"));

// Other miscellaneous
GVAR_DEF(PrimitivePtr, kPrimEnvironSet, std::make_shared<Primitive>(kEnvironSetOpName));
GVAR_DEF(PrimitivePtr, kPrimEnvironGet, std::make_shared<Primitive>(kEnvironGetOpName));
GVAR_DEF(PrimitivePtr, kPrimEnvironAdd, std::make_shared<Primitive>(kEnvironAddOpName));
GVAR_DEF(PrimitivePtr, kPrimEnvironDestroyAll, std::make_shared<Primitive>(kEnvironDestroyAllOpName));
GVAR_DEF(PrimitivePtr, kPrimSetSize, std::make_shared<Primitive>(kSetSizeOpName));

// Other miscellaneous
GVAR_DEF(PrimitivePtr, kPrimPyFunc, std::make_shared<Primitive>("PyFunc"));
GVAR_DEF(PrimitivePtr, kPrimCheckValid, std::make_shared<Primitive>("CheckValid"));
GVAR_DEF(PrimitivePtr, kPrimReformat, std::make_shared<Primitive>("Reformat"));
GVAR_DEF(PrimitivePtr, kPrimMutable, std::make_shared<Primitive>(kMutableOpName));
GVAR_DEF(PrimitivePtr, kPrimGetGrad, std::make_shared<Primitive>(kGetGradOpName));
GVAR_DEF(PrimitivePtr, kPrimHookBackward, std::make_shared<Primitive>("HookBackward"));
GVAR_DEF(PrimitivePtr, kPrimPrintShapeType, std::make_shared<Primitive>("PrintShapeType"));
GVAR_DEF(PrimitivePtr, kPrimSameTypeShape, std::make_shared<Primitive>("SameTypeShape"));
GVAR_DEF(PrimitivePtr, kPrimPrint, std::make_shared<Primitive>("Print"));
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

// GraphKernel ops
GVAR_DEF(PrimitivePtr, kPrimGraphKernel, std::make_shared<Primitive>("GraphKernel"));

// Custom
GVAR_DEF(PrimitivePtr, kPrimCustom, std::make_shared<Primitive>("Custom"));

// Type introspection
GVAR_DEF(PrimitivePtr, kPrimTypeOf, std::make_shared<Primitive>("typeof"));
GVAR_DEF(PrimitivePtr, kPrimTopTypeOf, std::make_shared<Primitive>("TopTypeof"));
GVAR_DEF(PrimitivePtr, kPrimHasType, std::make_shared<Primitive>("hastype"));
GVAR_DEF(PrimitivePtr, kPrimResolve, std::make_shared<Primitive>("resolve"));
GVAR_DEF(PrimitivePtr, kPrimEmbed, std::make_shared<Primitive>("embed"));
GVAR_DEF(PrimitivePtr, kPrimRefToEmbed, std::make_shared<Primitive>("RefToEmbed"));
GVAR_DEF(PrimitivePtr, kPrimCreateInstance, std::make_shared<Primitive>("create_instance"));
GVAR_DEF(PrimitivePtr, kPrimWithEnter, std::make_shared<Primitive>("with_enter"));
GVAR_DEF(PrimitivePtr, kPrimWithExit, std::make_shared<Primitive>("with_exit"));

// Other miscellaneous
GVAR_DEF(PrimitivePtr, kPrimInsertGradientOf, std::make_shared<Primitive>("InsertGradientOf"));
GVAR_DEF(PrimitivePtr, kPrimMorph, std::make_shared<Primitive>("Morph"));
GVAR_DEF(PrimitivePtr, kPrimCheckBprop, std::make_shared<Primitive>("CheckBprop"));
GVAR_DEF(PrimitivePtr, kPrimMixedPrecisionCast, std::make_shared<Primitive>("MixedPrecisionCast"));
GVAR_DEF(PrimitivePtr, kPrimDoUnpackCall, std::make_shared<Primitive>("DoUnpackCall"));

// Sponge Ops
GVAR_DEF(PrimitivePtr, kPrimAngleAtomEnergy, std::make_shared<Primitive>("AngleAtomEnergy"));

// Framework ops
GVAR_DEF(PrimitivePtr, kPrimStreamSend,
         std::make_shared<Primitive>(
           kStreamSendOpName,
           mindspore::HashMap<std::string, ValuePtr>({{std::string(ATTR_NO_ELIMINATE), MakeValue(true)},
                                                      {std::string(GRAPH_FLAG_SIDE_EFFECT_HIDDEN), MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimStreamRecv,
         std::make_shared<Primitive>(
           kStreamRecvOpName,
           mindspore::HashMap<std::string, ValuePtr>({{std::string(ATTR_NO_ELIMINATE), MakeValue(true)},
                                                      {std::string(GRAPH_FLAG_SIDE_EFFECT_HIDDEN), MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimGetStreamInfo,
         std::make_shared<Primitive>("GetStreamInfo", mindspore::HashMap<std::string, ValuePtr>(
                                                        {{std::string(ATTR_NO_ELIMINATE), MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimResLimit, std::make_shared<Primitive>("ResLimit"));
GVAR_DEF(PrimitivePtr, kPrimSliceToIndices, std::make_shared<Primitive>("SliceToIndices"));
GVAR_DEF(PrimitivePtr, kPrimTensorMove, std::make_shared<Primitive>("TensorMove"));
GVAR_DEF(PrimitivePtr, kPrimMemCpyAsync, std::make_shared<Primitive>("memcpy_async"));
GVAR_DEF(PrimitivePtr, kPrimSend, std::make_shared<Primitive>("Send"));
GVAR_DEF(PrimitivePtr, kPrimReceive, std::make_shared<Primitive>("Receive"));
GVAR_DEF(PrimitivePtr, kPrimCall, std::make_shared<Primitive>(kCallOpName));
GVAR_DEF(PrimitivePtr, kPrimRaise,
         std::make_shared<Primitive>(kRaiseOpName, mindspore::HashMap<std::string, ValuePtr>(
                                                     {{std::string(GRAPH_FLAG_SIDE_EFFECT_IO), MakeValue(true)},
                                                      {std::string("primitive_target"), MakeValue("CPU")},
                                                      {std::string("variable_length_inputs"), MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimSwitchLayer, std::make_shared<Primitive>("switch_layer"));
GVAR_DEF(PrimitivePtr, kPrimStringUpper, std::make_shared<Primitive>(kStringUpperOpName));
GVAR_DEF(PrimitivePtr, kPrimStringLower, std::make_shared<Primitive>(kStringLowerOpName));
GVAR_DEF(PrimitivePtr, kPrimFormat, std::make_shared<Primitive>(kFormatOpName));
GVAR_DEF(PrimitivePtr, kPrimMoveTo, std::make_shared<Primitive>(kMoveToOpName));
GVAR_DEF(PrimitivePtr, kPrimMoveAssign, std::make_shared<Primitive>(kMoveAssignOpName));
GVAR_DEF(PrimitivePtr, kPrimTraceGraph, std::make_shared<Primitive>(kTraceGraphOpName));

// Backend Inline
GVAR_DEF(PrimitivePtr, kPrimCallInline, std::make_shared<Primitive>("CallInline"));
GVAR_DEF(PrimitivePtr, kPrimPartialInline, std::make_shared<Primitive>("PartialInline"));
GVAR_DEF(PrimitivePtr, kPrimConditionSwitch, std::make_shared<Primitive>("ConditionSwitch"));
GVAR_DEF(PrimitivePtr, kPrimConditionGather, std::make_shared<Primitive>("ConditionGather"));

// Backend GE kernel
GVAR_DEF(PrimitivePtr, kPrimGEGraphOp, std::make_shared<Primitive>("GEGraphOp"));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_OPS_PRIMITIVE_FRAMEWORK_OPS_H_
