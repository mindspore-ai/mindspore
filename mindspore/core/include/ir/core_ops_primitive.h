/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_INCLUDE_CORE_IR_CORE_OPS_PRIMITIVE_H
#define MINDSPORE_CORE_INCLUDE_CORE_IR_CORE_OPS_PRIMITIVE_H

#include <iostream>
#include <memory>
#include <string>

#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/value.h"
#include "ir/core_ops_name.h"
#include "utils/flags.h"
#include "utils/hash_map.h"

namespace mindspore {
static constexpr char kDoSignaturePrimitivePrefix[] = "S_Prim_";
namespace prim {
GVAR_DEF(ValuePtr, kValueOne, std::make_shared<Int64Imm>(1));
#define COMMA ,
GVAR_DEF(mindspore::HashMap<std::string COMMA ValuePtr>, kSideEffectPropagate,
         {{mindspore::GRAPH_FLAG_SIDE_EFFECT_PROPAGATE COMMA kValueOne}});
#undef COMMA
GVAR_DEF(PrimitivePtr, kPrimDepend, std::make_shared<Primitive>(kDependOpName, kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimPartial, std::make_shared<Primitive>("Partial", kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimStateSetItem, std::make_shared<Primitive>("state_setitem"));
GVAR_DEF(PrimitivePtr, kPrimJ, std::make_shared<Primitive>(kJOpName, kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimVmap, std::make_shared<Primitive>(kVmapOpName, kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimShard, std::make_shared<Primitive>("Shard", kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimAddAttr, std::make_shared<Primitive>("AddAttr", kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimTaylor, std::make_shared<Primitive>(kTaylorOpName));
GVAR_DEF(PrimitivePtr, kPrimEnvironCreate, std::make_shared<Primitive>(kEnvironCreateOpName));
GVAR_DEF(PrimitivePtr, kPrimLoad, std::make_shared<Primitive>(kLoadOpName));
GVAR_DEF(PrimitivePtr, kPrimIs_, std::make_shared<Primitive>("is_"));
GVAR_DEF(PrimitivePtr, kPrimIsNot, std::make_shared<Primitive>("is_not"));
GVAR_DEF(PrimitivePtr, kPrimExtractKeywordArg, std::make_shared<Primitive>("extract_keyword_arg"));
GVAR_DEF(PrimitivePtr, kPrimMakeDict, std::make_shared<Primitive>("make_dict"));
GVAR_DEF(PrimitivePtr, kPrimIsInstance, std::make_shared<Primitive>(kIsInstanceOpName));
GVAR_DEF(PrimitivePtr, kPrimUpdateState, std::make_shared<Primitive>(kUpdateStateOpName));
GVAR_DEF(PrimitivePtr, kPrimReturn, std::make_shared<Primitive>(kReturnOpName));
GVAR_DEF(PrimitivePtr, kPrimSwitch, std::make_shared<Primitive>(kSwitchOpName));
GVAR_DEF(PrimitivePtr, kPrimVirtualViewGrad, std::make_shared<Primitive>(kVirtualViewGradOpName));

GVAR_DEF(PrimitivePtr, kPrimDynamicLossScale, std::make_shared<Primitive>("_DynamicLossScale"));

GVAR_DEF(PrimitivePtr, kPrimMakeTuple, std::make_shared<Primitive>(kMakeTupleOpName));
GVAR_DEF(PrimitivePtr, kPrimTupleGetItem, std::make_shared<Primitive>(kTupleGetItemOpName));
GVAR_DEF(PrimitivePtr, kPrimListGetItem, std::make_shared<Primitive>(kListGetItemOpName));
GVAR_DEF(PrimitivePtr, kPrimMakeList, std::make_shared<Primitive>(kMakeListNewOpName));

GVAR_DEF(PrimitivePtr, kPrimScalarSummary, std::make_shared<Primitive>("ScalarSummary"));
GVAR_DEF(PrimitivePtr, kPrimImageSummary, std::make_shared<Primitive>("ImageSummary"));
GVAR_DEF(PrimitivePtr, kPrimTensorSummary, std::make_shared<Primitive>("TensorSummary"));
GVAR_DEF(PrimitivePtr, kPrimHistogramSummary, std::make_shared<Primitive>("HistogramSummary"));

class MS_CORE_API DoSignaturePrimitive : public Primitive {
 public:
  explicit DoSignaturePrimitive(const std::string &name, const ValuePtr &function)
      : Primitive(kDoSignaturePrimitivePrefix + name), function_(function) {}

  ~DoSignaturePrimitive() override = default;

  MS_DECLARE_PARENT(DoSignaturePrimitive, Primitive)

  const ValuePtr function() const { return function_; }

 private:
  ValuePtr function_;
};
using DoSignaturePrimitivePtr = std::shared_ptr<DoSignaturePrimitive>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_INCLUDE_CORE_IR_CORE_OPS_PRIMITIVE_H
