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

#include "ops/grad/pdist_grad.h"

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr PdistGradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto pdist_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];
  auto x_size = x_shape.size();
  CheckAndConvertUtils::CheckValue("y_grad shape", grad_shape, kEqual, "y shape", pdist_shape, prim_name);
  const int64_t x_dim = 2;
  CheckAndConvertUtils::CheckInteger("x dim", x_size, kEqual, x_dim, "PdistGrad");
  auto out_shape = input_args[1]->BuildShape();
  MS_EXCEPTION_IF_NULL(out_shape);
  auto out_element = out_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(out_element);
  return out_element;
}

TypePtr PdistGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const size_t y_grad_index = 0;
  const size_t x_index = 1;
  const size_t y_index = 2;
  const std::set<TypePtr> valid_types = {kFloat64, kFloat32, kFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("y_grad", input_args[y_grad_index]->BuildType());
  (void)types.emplace("x", input_args[x_index]->BuildType());
  (void)types.emplace("y", input_args[y_index]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
}
}  // namespace

float PdistGrad::get_p() const {
  auto value_ptr = this->GetAttr(kP);
  return GetValue<float>(value_ptr);
}
void PdistGrad::set_p(const float p) { (void)this->AddAttr(kP, api::MakeValue(p)); }

MIND_API_OPERATOR_IMPL(PdistGrad, BaseOperator);
AbstractBasePtr PdistGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = PdistGradInferType(primitive, input_args);
  auto infer_shape = PdistGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(PdistGrad, prim::kPrimPdistGrad, PdistGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
