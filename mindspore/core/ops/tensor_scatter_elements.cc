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

#include "ops/tensor_scatter_elements.h"
#include <map>
#include <set>
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr TensorScatterElementsInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto input_x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(input_x_shape_ptr);
  auto indices_shape_ptr = input_args[kInputIndex1]->BuildShape();
  MS_EXCEPTION_IF_NULL(indices_shape_ptr);
  auto updates_shape_ptr = input_args[kInputIndex2]->BuildShape();
  MS_EXCEPTION_IF_NULL(updates_shape_ptr);
  if (input_x_shape_ptr->IsDynamic() || indices_shape_ptr->IsDynamic() || updates_shape_ptr->IsDynamic()) {
    return input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
  }
  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_x_shape_ptr)[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices_shape_ptr)[kShape];
  auto updates_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(updates_shape_ptr)[kShape];
  if (input_x_shape.size() < 1 || indices_shape.size() < 1 || updates_shape.size() < 1) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", 'input_x_shape', 'indices_shape' and "
                             << "'updates_shape' dims must be greater than 1. but got input_x_shape:" << input_x_shape
                             << ", indices_shape:" << indices_shape << ", updates_shape: " << updates_shape << ".";
  }

  if (updates_shape != indices_shape) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", "
                             << "'updates_shape' must be as same as 'indices_shape' but got indices_shape: "
                             << indices_shape << ", updates_shape: " << updates_shape << ".";
  }

  return input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
}

TypePtr TensorScatterElementsInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto indiecs_type_ptr = input_args[kInputIndex1]->BuildType();
  std::set<TypePtr> type_set = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices type", indiecs_type_ptr, type_set, prim_name);
  std::map<std::string, TypePtr> type_dict;
  (void)type_dict.emplace("input_x", input_args[kInputIndex0]->BuildType());
  (void)type_dict.emplace("updates", input_args[kInputIndex2]->BuildType());
  std::set<TypePtr> check_list(common_valid_types);
  (void)check_list.insert(kBool);
  return CheckAndConvertUtils::CheckTensorTypeSame(type_dict, check_list, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(TensorScatterElements, BaseOperator);
AbstractBasePtr TensorScatterElementsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto infer_type = TensorScatterElementsInferType(primitive, input_args);
  auto infer_shape = TensorScatterElementsInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
void TensorScatterElements::Init(const int64_t axis, const std::string reduction) {
  this->set_axis(axis);
  this->set_reduction(reduction);
}
void TensorScatterElements::set_axis(const int64_t axis) { (void)AddAttr(kAxis, api::MakeValue(axis)); }
int64_t TensorScatterElements::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }
void TensorScatterElements::set_reduction(const std::string reduction) {
  (void)this->AddAttr(kReduction, api::MakeValue(reduction));
}
std::string TensorScatterElements::get_reduction() const {
  auto value_ptr = GetAttr(kReduction);
  return GetValue<std::string>(value_ptr);
}

// AG means auto generated
class MIND_API AGTensorScatterElementsInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return TensorScatterElementsInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return TensorScatterElementsInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return TensorScatterElementsInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(TensorScatterElements, prim::kPrimTensorScatterElements, AGTensorScatterElementsInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
