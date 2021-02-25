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

#include "ops/arg_max.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto prim = primitive->cast<PrimArgMaxPtr>();
  MS_EXCEPTION_IF_NULL(prim);
  auto axis = prim->get_axis();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  auto x_rank = SizeToLong(x_shape.size());
  CheckAndConvertUtils::CheckInRange<int64_t>("argmax axis", axis, kIncludeLeft, {-x_rank, x_rank}, prim_name);
  axis = axis < 0 ? axis + x_rank : axis;
  std::vector<int64_t> out_shape;
  for (size_t i = 0; i < x_shape.size(); ++i) {
    if (SizeToLong(i) != axis) {
      out_shape.emplace_back(x_shape[i]);
    }
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  CheckAndConvertUtils::CheckInteger("input number", input_args.size(), kEqual, 1, prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return kInt32;
}
}  // namespace

void ArgMax::Init(const int64_t axis, const TypeId output_type) {
  set_axis(axis);
  set_output_type(output_type);
}

void ArgMax::set_axis(const int64_t axis) { this->AddAttr(kAxis, MakeValue(axis)); }
void ArgMax::set_output_type(const TypeId output_type) { this->AddAttr(kOutputType, TypeIdToType(output_type)); }

int64_t ArgMax::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }
TypeId ArgMax::get_output_type() const {
  auto type_ptr = GetAttr(kOutputType)->cast<TensorTypePtr>()->element();
  return type_ptr->type_id();
}

AbstractBasePtr ArgMaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameArgMax, ArgMax);
}  // namespace ops
}  // namespace mindspore
