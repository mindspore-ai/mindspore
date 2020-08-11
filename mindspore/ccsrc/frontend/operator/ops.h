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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_OPS_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_OPS_H_

#include <iostream>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "base/core_ops.h"

namespace mindspore {
// namespace to support primitive operators
namespace prim {
ValuePtr GetPythonOps(const std::string &op_name,
                      const std::string &module_name = "mindspore._extends.parse.standard_method",
                      bool use_signature = false);

// Primitives only used by frontend;
// Type introspection
inline const PrimitivePtr kPrimTypeOf = std::make_shared<Primitive>("typeof");
inline const PrimitivePtr kPrimHasType = std::make_shared<Primitive>("hastype");

inline const PrimitivePtr kPrimResolve = std::make_shared<Primitive>("resolve");
inline const PrimitivePtr kPrimEmbed = std::make_shared<Primitive>("embed");
inline const PrimitivePtr kPrimRefToEmbed = std::make_shared<Primitive>("RefToEmbed");
inline const PrimitivePtr kPrimCreateInstance = std::make_shared<Primitive>("create_instance");

// Other miscellaneous
inline const PrimitivePtr kPrimGetRefOrigin = std::make_shared<Primitive>("get_ref_origin");
inline const PrimitivePtr kPrimInsertGradientOf = std::make_shared<Primitive>("InsertGradientOf");
inline const PrimitivePtr kPrimCheckBprop = std::make_shared<Primitive>("CheckBprop");
inline const PrimitivePtr kPrimMixedPrecisionCast = std::make_shared<Primitive>("mixed_precision_cast");
inline const PrimitivePtr kPrimMakeRecord = std::make_shared<Primitive>("make_record");

// Structures

inline const PrimitivePtr kPrimListMap = std::make_shared<Primitive>("list_map");
inline const PrimitivePtr kPrimListReduce = std::make_shared<Primitive>("list_reduce");
inline const PrimitivePtr kPrimTupleReversed = std::make_shared<Primitive>("tuple_reversed");
inline const PrimitivePtr kPrimReducedShape = std::make_shared<Primitive>("reduced_shape");
inline const PrimitivePtr kPrimTupleDiv = std::make_shared<Primitive>("tuple_div");
inline const PrimitivePtr kPrimTupleToArray = std::make_shared<Primitive>("tuple_to_array");
inline const PrimitivePtr kPrimShapeMul = std::make_shared<Primitive>("shape_mul");
inline const PrimitivePtr kPrimTupleEqual = std::make_shared<Primitive>("tuple_equal");
inline const PrimitivePtr kPrimListEqual = std::make_shared<Primitive>("list_equal");
inline const PrimitivePtr kPrimMakeRange = std::make_shared<Primitive>("make_range");
inline const PrimitivePtr kPrimStopGradient = std::make_shared<Primitive>("stop_gradient");
inline const PrimitivePtr kPrimStringEqual = std::make_shared<Primitive>("string_equal");
inline const PrimitivePtr kPrimStringConcat = std::make_shared<Primitive>("string_concat");
inline const PrimitivePtr kPrimDictLen = std::make_shared<Primitive>("dict_len");

inline const PrimitivePtr kPrimFakeBprop = std::make_shared<Primitive>("fake_bprop");

inline const PrimitivePtr kPrimBroadcastGradientArgs = std::make_shared<Primitive>("BroadcastGradientArgs");

class UnpackGraphPrimitive : public Primitive {
 public:
  explicit UnpackGraphPrimitive(const std::string &name, const bool &with_sens, const bool &need_unpack_args)
      : Primitive("UnpackGraph"), with_sens_in_args_(with_sens), need_unpack_args_(need_unpack_args) {}
  ~UnpackGraphPrimitive() override = default;
  MS_DECLARE_PARENT(UnpackGraphPrimitive, Primitive)
  bool with_sens_in_args() const { return with_sens_in_args_; }
  bool need_unpack_args() const { return need_unpack_args_; }

 private:
  bool with_sens_in_args_;
  bool need_unpack_args_;
};
using UnpackGraphPrimitivePtr = std::shared_ptr<UnpackGraphPrimitive>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_OPS_H_
