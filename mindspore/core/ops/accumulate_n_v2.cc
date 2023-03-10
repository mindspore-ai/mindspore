/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "ops/accumulate_n_v2.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr AccumulateNV2InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  const auto &prim_name = primitive->name();
  AbstractBasePtrList elements = input_args;
  if (input_args.size() == 1) {
    if (!input_args[0]->isa<abstract::AbstractSequence>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the input data type must be list or tuple of tensors.";
    }
    elements = input_args[0]->cast<abstract::AbstractSequencePtr>()->elements();
  }
  (void)CheckAndConvertUtils::CheckInteger("concat element num", SizeToLong(elements.size()), kGreaterEqual, 1,
                                           primitive->name());
  (void)primitive->AddAttr("n", MakeValue(SizeToLong(elements.size())));
  auto shape_0 = elements[0]->BuildShape();
  auto element0_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(shape_0);
  for (size_t i = 0; i < elements.size(); ++i) {
    auto shape_i = CheckAndConvertUtils::ConvertShapePtrToShapeMap(elements[i]->BuildShape())[kShape];
    if (IsDynamicRank(shape_i)) {
      return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
    }
    if (IsDynamic(shape_i)) {
      element0_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(elements[i]->BuildShape());
    }
  }
  for (size_t i = 0; i < elements.size(); ++i) {
    auto shape = elements[i]->BuildShape();
    if (shape->isa<abstract::Shape>() && shape_0->isa<abstract::Shape>()) {
      const auto &shape_vec = shape->cast<abstract::ShapePtr>()->shape();
      const auto &shape_0_vec = shape_0->cast<abstract::ShapePtr>()->shape();
      if ((shape_vec == ShapeVector({1}) && shape_0_vec == ShapeVector()) ||
          (shape_vec == ShapeVector() && shape_0_vec == ShapeVector({1}))) {
        MS_LOG(DEBUG) << "For '" << primitive->name() << "', Shape of input[" << i
                      << "] must be consistent with the shape of input[0], but got shape of input[" << i
                      << "]: " << shape->ToString() << ", shape of input[0]: " << shape_0->ToString();
        continue;
      }
    }
    if (*shape != *shape_0) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', shape of input[" << i
                               << "] must be consistent with the shape of input[0], but got shape of input[" << i
                               << "]: " << shape->ToString() << ", shape of input[0]: " << shape_0->ToString() << ".";
    }
  }
  auto in_shape = element0_shape_map[kShape];
  return std::make_shared<abstract::Shape>(in_shape);
}

TypePtr AccumulateNV2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const auto &prim_name = prim->name();
  AbstractBasePtrList elements = input_args;
  if (input_args.size() == 1) {
    if (!input_args[0]->isa<abstract::AbstractSequence>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the input data type must be list or tuple of tensors.";
    }
    elements = input_args[0]->cast<abstract::AbstractSequencePtr>()->elements();
  }
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
  (void)valid_types.emplace(kBool);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
  return elements[0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(AccumulateNV2, BaseOperator);
AbstractBasePtr AccumulateNV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, 1, prim_name);
  auto infer_type = AccumulateNV2InferType(primitive, input_args);
  auto infer_shape = AccumulateNV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGAccumulateNV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AccumulateNV2InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AccumulateNV2InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AccumulateNV2Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AccumulateNV2, prim::kPrimAccumulateNV2, AGAccumulateNV2Infer, false);
}  // namespace ops
}  // namespace mindspore
