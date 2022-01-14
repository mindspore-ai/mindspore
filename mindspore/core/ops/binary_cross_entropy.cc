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

#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include <map>

#include "ops/binary_cross_entropy.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr BinaryCrossEntroyInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 3;
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto y_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto weight_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto x_shape_BaseShapePtr = input_args[kInputIndex0]->BuildShape();
  auto y_shape_BaseShapePtr = input_args[kInputIndex1]->BuildShape();
  auto weight_shape_BaseShapePtr = input_args[kInputIndex2]->BuildShape();
  auto x_shape_ptr = x_shape_BaseShapePtr->cast<abstract::ShapePtr>();
  auto y_shape_ptr = y_shape_BaseShapePtr->cast<abstract::ShapePtr>();
  auto weight_shape_ptr = weight_shape_BaseShapePtr->cast<abstract::ShapePtr>();
  if (!x_shape_ptr->IsDynamic() && !y_shape_ptr->IsDynamic())
    CheckAndConvertUtils::Check("x shape", x_shape, kEqual, "y shape", y_shape, prim_name, ValueError);
  if (weight_shape.size() > 0) {
    if (!y_shape_ptr->IsDynamic() && !weight_shape_ptr->IsDynamic())
      CheckAndConvertUtils::Check("y shape", y_shape, kEqual, "weight shape", weight_shape, prim_name, ValueError);
  }
  auto out_shape = x_shape;
  int64_t reduction;
  CheckAndConvertUtils::GetReductionEnumValue(primitive->GetAttr(kReduction), &reduction);
  if (reduction == REDUCTION_SUM || reduction == MEAN) {
    out_shape.resize(0);
    return std::make_shared<abstract::Shape>(out_shape);
  }
  return x_shape_ptr;
}

TypePtr BinaryCrossEntroyInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const int64_t kInputNum = 3;
  auto prim_name = prim->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> types1, types2;
  types1.emplace("x", input_args[kInputIndex0]->BuildType());
  types1.emplace("y", input_args[kInputIndex1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types1, valid_types, prim_name);
  auto weight_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  if (weight_shape.size() > 0) {
    types2.emplace("x", input_args[kInputIndex0]->BuildType());
    types2.emplace("weight", input_args[kInputIndex2]->BuildType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types2, valid_types, prim_name);
  }
  return input_args[kInputIndex0]->BuildType();
}
}  // namespace

void BinaryCrossEntropy::set_reduction(const Reduction &reduction) {
  int64_t swi = reduction;
  (void)this->AddAttr(kReduction, MakeValue(swi));
}

Reduction BinaryCrossEntropy::get_reduction() const {
  auto value_ptr = GetAttr(kReduction);
  return Reduction(GetValue<int64_t>(value_ptr));
}
void BinaryCrossEntropy::Init(const Reduction &reduction) { this->set_reduction(reduction); }

AbstractBasePtr BinaryCrossEntropyInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = BinaryCrossEntroyInferType(primitive, input_args);
  auto infer_shape = BinaryCrossEntroyInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(BinaryCrossEntropy, prim::kPrimBinaryCrossEntropy, BinaryCrossEntropyInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
