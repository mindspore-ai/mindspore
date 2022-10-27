/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/grad/multi_margin_loss_grad.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
TypePtr MultiMarginLossGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  (void)CheckAndConvertUtils::CheckTensorTypeValid("target", input_args[kInputIndex2]->BuildType(), {kInt64},
                                                   prim->name());
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("y_grad", input_args[kInputIndex0]->BuildType());
  (void)types.emplace("x", input_args[kInputIndex1]->BuildType());
  if (input_args.size() == kDim4 && input_args[kInputIndex3]->BuildType()->isa<TensorType>()) {
    auto tensor_type = input_args[kInputIndex3]->BuildType()->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    auto element = tensor_type->element();
    MS_EXCEPTION_IF_NULL(element);
    if (element->type_id() != kMetaTypeNone) {
      (void)types.emplace("weight", input_args[kInputIndex3]->BuildType());
    }
  }
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return input_args[kInputIndex1]->BuildType();
}

abstract::ShapePtr MultiMarginLossGradInferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto target_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape) || IsDynamicRank(target_shape)) {
    return std::make_shared<abstract::Shape>(x_shape);
  }

  if (x_shape.size() != kDim2 || target_shape.size() != kDim1) {
    MS_EXCEPTION(ValueError) << "For MultiMarginLossGrad, the rank of input x should be 2, and "
                                "the rank of target should be 1,"
                             << " while rank of x is " << x_shape.size() << ", rank of target is "
                             << target_shape.size();
  }
  if (x_shape[kInputIndex0] != target_shape[kInputIndex0]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << " x_shape[0] and target_shape[0] should be the same,"
                             << " while x_shape[0] is " << x_shape[kInputIndex0] << ", target_shape[0] is "
                             << target_shape[kInputIndex0];
  }
  if (input_args.size() == kDim4 && input_args[kInputIndex3]->BuildType()->isa<TensorType>()) {
    auto tensor_type = input_args[kInputIndex3]->BuildType()->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    auto element = tensor_type->element();
    MS_EXCEPTION_IF_NULL(element);
    if (element->type_id() != kMetaTypeNone) {
      auto weight_shape =
        CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
      if (weight_shape.size() != kDim1) {
        MS_EXCEPTION(ValueError) << "For " << prim_name << " the rank of weight should be 1,"
                                 << " but get " << weight_shape.size();
      }
      if (x_shape[kInputIndex1] != weight_shape[kInputIndex0]) {
        MS_EXCEPTION(ValueError) << "For " << prim_name << " x_shape[1] and weight_shape[0] should be the same,"
                                 << " while x_shape[1] is " << x_shape[kInputIndex1] << ", weight_shape[0] is "
                                 << weight_shape[kInputIndex0];
      }
    }
  }
  return std::make_shared<abstract::Shape>(x_shape);
}
}  // namespace

MIND_API_OPERATOR_IMPL(MultiMarginLossGrad, BaseOperator);

void MultiMarginLossGrad::set_p(int64_t p) { (void)AddAttr(kP, api::MakeValue(p)); }

void MultiMarginLossGrad::set_margin(float margin) { (void)AddAttr(kMargin, api::MakeValue(margin)); }

void MultiMarginLossGrad::set_reduction(const Reduction &reduction) {
  int64_t swi = reduction;
  (void)this->AddAttr(kReduction, api::MakeValue(swi));
}

int64_t MultiMarginLossGrad::get_p() const {
  auto value_ptr = GetAttr(kP);
  return GetValue<int64_t>(value_ptr);
}

float MultiMarginLossGrad::get_margin() const {
  auto value_ptr = GetAttr(kMargin);
  return GetValue<float>(value_ptr);
}

string MultiMarginLossGrad::get_reduction() const {
  auto value_ptr = GetAttr(kReduction);
  return GetValue<string>(value_ptr);
}

AbstractBasePtr MultiMarginLossGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  constexpr size_t kInputNumWithWeight = 4;
  constexpr size_t kInputNumWithoutWeight = 3;
  size_t input_num = input_args.size();
  if (input_num != kInputNumWithoutWeight && input_num != kInputNumWithWeight) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but MultiMarginLossGrad needs 3 or 4 inputs.";
  }
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex2]);
  if (input_args.size() == kInputNumWithWeight) {
    MS_EXCEPTION_IF_NULL(input_args[kInputIndex3]);
  }
  auto types = MultiMarginLossGradInferType(primitive, input_args);
  auto shapes = MultiMarginLossGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}
REGISTER_PRIMITIVE_EVAL_IMPL(MultiMarginLossGrad, prim::kPrimMultiMarginLossGrad, MultiMarginLossGradInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
