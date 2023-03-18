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

#include "ops/in_sequence.h"

#include <vector>
#include <memory>
#include <set>
#include <string>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
AbstractBasePtr InSequenceInferInner(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr size_t input_len = 2;
  constexpr size_t seq_index = 1;
  constexpr size_t element_index = 0;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, prim_name);
  auto second_abs = input_args[seq_index];
  if (!second_abs->isa<abstract::AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', the second input should be tuple or list but got: " << second_abs->ToString();
  }
  auto seq_abs = second_abs->cast<abstract::AbstractSequencePtr>();
  auto ele_abs = input_args[element_index];
  if (!ele_abs->isa<abstract::AbstractScalar>() && !ele_abs->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "The prim '" << prim_name << "', element input must be scalar or tensor, but got "
                            << " target is " << ele_abs->ToString();
  }
  if (seq_abs->dynamic_len()) {
    auto data_element_abs = seq_abs->dynamic_len_element_abs();
    CheckAndConvertUtils::CheckAbstractTypeAndShapeSame({data_element_abs, ele_abs},
                                                        "For " + prim::kPrimInSequence->ToString(),
                                                        "mutable list existing item", "new added item");
    return std::make_shared<abstract::AbstractScalar>(kValueAny, kBool);
  }
  const auto &elements = seq_abs->elements();
  auto first_element = elements[0];
  CheckAndConvertUtils::CheckAbstractTypeAndShapeSame(
    {first_element, ele_abs}, "For " + prim::kPrimInSequence->ToString(), "list existing item", "new added item");
  return std::make_shared<abstract::AbstractScalar>(kValueAny, kBool);
}
}  // namespace

class InSequenceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return InSequenceInferInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return InSequenceInferInner(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return InSequenceInferInner(primitive, input_args);
  }
};
MIND_API_OPERATOR_IMPL(InSequence, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(InSequence, prim::kPrimInSequence, InSequenceInfer, false);
}  // namespace ops
}  // namespace mindspore
