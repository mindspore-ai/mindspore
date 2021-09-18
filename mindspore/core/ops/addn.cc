/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <set>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "ops/addn.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr AddNInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  if (!input_args[0]->isa<abstract::AbstractTuple>() && !input_args[0]->isa<abstract::AbstractList>()) {
    MS_EXCEPTION(TypeError) << "The input of AddN must be list or tuple of tensors.";
  }
  auto elements = input_args[0]->isa<abstract::AbstractTuple>()
                    ? input_args[0]->cast<abstract::AbstractTuplePtr>()->elements()
                    : input_args[0]->cast<abstract::AbstractListPtr>()->elements();
  (void)CheckAndConvertUtils::CheckInteger("concat element num", SizeToLong(elements.size()), kGreaterEqual, 1,
                                           primitive->name());
  (void)primitive->AddAttr("n", MakeValue(SizeToLong(elements.size())));
  auto shape_0 = elements[0]->BuildShape();
  auto element0_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(shape_0);
  for (size_t i = 0; i < elements.size(); ++i) {
    auto shape = elements[i]->BuildShape();
    if (shape->isa<abstract::Shape>() && shape_0->isa<abstract::Shape>()) {
      const auto &shape_vec = shape->cast<abstract::ShapePtr>()->shape();
      const auto &shape_0_vec = shape_0->cast<abstract::ShapePtr>()->shape();
      if ((shape_vec == ShapeVector({1}) && shape_0_vec == ShapeVector()) ||
          (shape_vec == ShapeVector() && shape_0_vec == ShapeVector({1}))) {
        MS_LOG(DEBUG) << "The primitive[" << primitive->name() << "]'s input[" << i << "] shape: " << shape->ToString()
                      << " are consistent with the shape of input[0]" << shape_0->ToString();
        continue;
      }
    }
    if (*shape != *shape_0) {
      MS_EXCEPTION(ValueError) << "The primitive[" << primitive->name() << "]'s input shape must be same, "
                               << "but got the shape of input[" << i << "]: " << shape->ToString()
                               << ", shape of input[0]:" << shape_0->ToString();
    }
  }
  auto in_shape = element0_shape_map[kShape];
  auto min_shape = element0_shape_map[kMinShape];
  auto max_shape = element0_shape_map[kMaxShape];
  return std::make_shared<abstract::Shape>(in_shape, min_shape, max_shape);
}

TypePtr AddNInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  if (!input_args[0]->isa<abstract::AbstractTuple>() && !input_args[0]->isa<abstract::AbstractList>()) {
    MS_EXCEPTION(TypeError) << "The input of AddN must be list or tuple of tensors.";
  }
  auto elements = input_args[0]->isa<abstract::AbstractTuple>()
                    ? input_args[0]->cast<abstract::AbstractTuplePtr>()->elements()
                    : input_args[0]->cast<abstract::AbstractListPtr>()->elements();
  (void)CheckAndConvertUtils::CheckInteger("concat element num", SizeToLong(elements.size()), kGreaterEqual, 1,
                                           prim->name());
  std::map<std::string, TypePtr> types;
  (void)types.emplace("element_0", elements[0]->BuildType());
  for (size_t i = 0; i < elements.size(); ++i) {
    if (elements[i]->BuildType()->type_id() == kObjectTypeUndeterminedType) {
      return elements[0]->BuildType();
    }
    std::string element_i = "element_" + std::to_string(i);
    (void)types.emplace(element_i, elements[i]->BuildType());
  }
  std::set<TypePtr> valid_types = common_valid_types;
  valid_types.insert(kBool);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
  return elements[0]->BuildType();
}
}  // namespace
AbstractBasePtr AddNInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, 1, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return abstract::MakeAbstract(AddNInferShape(primitive, input_args), AddNInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(AddN, prim::kPrimAddN, AddNInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
