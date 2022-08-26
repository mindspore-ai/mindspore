/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/grad/maximum_grad.h"
#include <vector>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr MaximumGradInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x = input_args[0]->BuildShape();
  auto y = input_args[1]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(y);
  auto x_element = x->cast<abstract::ShapePtr>();
  auto y_element = y->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(x_element);
  MS_EXCEPTION_IF_NULL(y_element);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x_element, y_element});
}
TuplePtr MaximumGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t INPUT_GRADS_IDX = 2;
  auto x1 = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto x2 = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 1);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, INPUT_GRADS_IDX);
  (void)abstract::CheckDtypeSame(prim_name, x1, x2);
  auto x_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, common_valid_types, prim_name);
  std::vector<TypePtr> type_tuple;
  type_tuple.push_back(x_type);
  type_tuple.push_back(x_type);
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace

MIND_API_OPERATOR_IMPL(MaximumGrad, BaseOperator);
abstract::AbstractBasePtr MaximumGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = MaximumGradInferType(primitive, input_args);
  auto infer_shape = MaximumGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
void MaximumGrad::Init(const bool grad_x, const bool grad_y) {
  set_grad_x(grad_x);
  set_grad_y(grad_y);
}

void MaximumGrad::set_grad_x(const bool grad_x) { (void)this->AddAttr(kGradX, api::MakeValue(grad_x)); }

void MaximumGrad::set_grad_y(const bool grad_y) { (void)this->AddAttr(kGradY, api::MakeValue(grad_y)); }

bool MaximumGrad::get_grad_x() const {
  auto value_ptr = GetAttr(kGradX);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

bool MaximumGrad::get_grad_y() const {
  auto value_ptr = GetAttr(kGradY);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}
REGISTER_PRIMITIVE_EVAL_IMPL(MaximumGrad, prim::kPrimMaximumGrad, MaximumGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
