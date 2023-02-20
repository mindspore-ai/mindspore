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

#include "ops/multi_margin_loss.h"

#include <map>
#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/base/type_id.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
TypePtr MultiMarginLossInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  (void)CheckAndConvertUtils::CheckTensorTypeValid("target", input_args[kInputIndex1]->BuildType(), {kInt64},
                                                   prim->name());
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kInputIndex0]->BuildType());
  if (input_args.size() == kInputIndex3 && input_args[kInputIndex2]->BuildType()->isa<TensorType>()) {
    auto tensor_type = input_args[kInputIndex2]->BuildType()->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    auto element = tensor_type->element();
    MS_EXCEPTION_IF_NULL(element);
    if (element->type_id() != kMetaTypeNone) {
      (void)types.emplace("weight", input_args[kInputIndex2]->BuildType());
    }
  }
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return input_args[kInputIndex0]->BuildType();
}

abstract::ShapePtr MultiMarginLossInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto target_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];

  int64_t reduction = 0;
  CheckAndConvertUtils::GetReductionEnumValue(primitive->GetAttr(kReduction), &reduction);
  auto out_shape = target_shape;
  if (reduction == static_cast<int64_t>(REDUCTION_SUM) || reduction == static_cast<int64_t>(MEAN)) {
    out_shape.resize(kInputIndex0);
  }
  if (IsDynamic(x_shape) || IsDynamic(target_shape)) {
    return std::make_shared<abstract::Shape>(out_shape);
  }

  if (x_shape.size() != kDim2 || target_shape.size() != kDim1) {
    MS_EXCEPTION(ValueError) << "For MultiMarginLoss, the rank of input "
                                "x and target should be 2 and 1,"
                             << " while rank of x is " << x_shape.size() << ", rank of target is  "
                             << target_shape.size();
  }
  if (x_shape[kInputIndex0] != target_shape[kInputIndex0]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << " x_shape[0] and target_shape[0] should be the same,"
                             << " while x_shape[0] is " << x_shape[kInputIndex0] << ", target_shape[0] is "
                             << target_shape[kInputIndex0];
  }
  if (input_args.size() == kDim3 && input_args[kInputIndex2]->BuildType()->isa<TensorType>()) {
    auto tensor_type = input_args[kInputIndex2]->BuildType()->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    auto element = tensor_type->element();
    MS_EXCEPTION_IF_NULL(element);
    if (element->type_id() != kMetaTypeNone) {
      auto weight_shape =
        CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
      if (IsDynamic(weight_shape)) {
        return std::make_shared<abstract::Shape>(out_shape);
      }
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

  return std::make_shared<abstract::Shape>(out_shape);
}
}  // namespace

MIND_API_OPERATOR_IMPL(MultiMarginLoss, BaseOperator);

void MultiMarginLoss::Init(int64_t p, float margin, const Reduction &reduction) {
  set_p(p);
  set_margin(margin);
  set_reduction(reduction);
}

void MultiMarginLoss::set_p(int64_t p) { (void)AddAttr(kP, api::MakeValue(p)); }

void MultiMarginLoss::set_margin(float margin) { (void)AddAttr(kMargin, api::MakeValue(margin)); }

void MultiMarginLoss::set_reduction(const Reduction &reduction) {
  int64_t swi = reduction;
  (void)this->AddAttr(kReduction, api::MakeValue(swi));
}

int64_t MultiMarginLoss::get_p() const {
  auto value_ptr = GetAttr(kP);
  return GetValue<int64_t>(value_ptr);
}

float MultiMarginLoss::get_margin() const {
  auto value_ptr = GetAttr(kMargin);
  return GetValue<float>(value_ptr);
}

string MultiMarginLoss::get_reduction() const {
  auto value_ptr = GetAttr(kReduction);
  return GetValue<string>(value_ptr);
}

AbstractBasePtr MultiMarginLossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInRange("multi_margin_loss_input_nums", input_args.size(), kIncludeBoth, {kDim2, kDim3},
                                     primitive->name());
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  if (input_args.size() == kDim3) {
    MS_EXCEPTION_IF_NULL(input_args[kInputIndex2]);
  }
  auto types = MultiMarginLossInferType(primitive, input_args);
  auto shapes = MultiMarginLossInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

// AG means auto generated
class MIND_API AGMultiMarginLossInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MultiMarginLossInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MultiMarginLossInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MultiMarginLossInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MultiMarginLoss, prim::kPrimMultiMarginLoss, AGMultiMarginLossInfer, false);
}  // namespace ops
}  // namespace mindspore
