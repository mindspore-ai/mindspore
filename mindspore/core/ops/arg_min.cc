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
#include "ops/arg_min.h"

namespace mindspore {
namespace ops {
void ArgMin::Init(const int64_t axis, const TypeId output_type) {
  set_axis(axis);
  set_output_type(output_type);
}

void ArgMin::set_axis(const int64_t axis) { this->AddAttr(kAxis, MakeValue(axis)); }
void ArgMin::set_output_type(const TypeId output_type) { this->AddAttr(kOutputType, TypeIdToType(output_type)); }

int64_t ArgMin::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}

TypeId ArgMin::get_output_type() const {
  auto type_ptr = GetAttr(kOutputType)->cast<TensorTypePtr>()->element();
  return type_ptr->type_id();
}

AbstractBasePtr ArgMinInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto argmin_prim = primitive->cast<PrimArgMin>();
  MS_EXCEPTION_IF_NULL(argmin_prim);
  auto prim_name = argmin_prim->name();
  CheckAndConvertUtils::CheckInteger("arg_min_infer", input_args.size(), kEqual, 1, prim_name);

  // Infer shape
  auto axis = argmin_prim->get_axis();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  auto x_rank = SizeToLong(x_shape.size());
  CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis, kIncludeLeft, {-x_rank, x_rank}, prim_name);
  if (axis < 0) {
    axis += x_rank;
  }
  std::vector<int64_t> out_shape;
  for (int64_t i = 0; i < x_rank; i++) {
    if (i != axis) {
      out_shape.push_back(x_shape[i]);
    }
  }

  // Infer type
  auto x_dtype = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  std::set<TypePtr> template_types = {TypeIdToType(kObjectTypeTensorType)};
  CheckAndConvertUtils::CheckSubClass("x_dtype", x_dtype, template_types, prim_name);

  return std::make_shared<abstract::AbstractTensor>(x_dtype, std::make_shared<abstract::Shape>(out_shape));
}
REGISTER_PRIMITIVE_C(kNameArgMin, ArgMin);
}  // namespace ops
}  // namespace mindspore
