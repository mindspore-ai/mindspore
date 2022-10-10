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

#include "ops/grad/softmax_grad.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SoftmaxGradInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto axis = GetValue<int64_t>(primitive->GetAttr(kAxis));

  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto in_shape = shape_map[kShape];
  if (!IsDynamicRank(in_shape)) {
    auto rank = SizeToLong(in_shape.size());
    (void)CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis, kIncludeLeft, {-rank, rank}, primitive->name());
  }
  return std::make_shared<abstract::Shape>(in_shape);
}

TypePtr SoftmaxGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  MS_EXCEPTION_IF_NULL(input_args[1]);
  auto x_type = input_args[1]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  if (!x_type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input must be a Tensor, but got: " << x_type->ToString()
                            << ".";
  }
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(SoftmaxGrad, BaseOperator);
void SoftmaxGrad::Init(const int64_t axis) { this->set_axis(axis); }

void SoftmaxGrad::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

std::vector<int64_t> SoftmaxGrad::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

AbstractBasePtr SoftmaxGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto type = SoftmaxGradInferType(primitive, input_args);
  auto shape = SoftmaxGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SoftmaxGrad, prim::kPrimSoftmaxGrad, SoftmaxGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
