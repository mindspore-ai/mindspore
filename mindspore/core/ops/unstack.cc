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

#include "ops/unstack.h"

namespace mindspore {
namespace ops {

void Unstack::Init(const int64_t axis) { this->set_axis(axis); }
void Unstack::set_axis(const int64_t axis) { AddAttr(kAxis, MakeValue(axis)); }
int64_t Unstack::get_axis() const {
  auto value_ptr = this->GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}
AbstractBasePtr UnstackInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto unstack_prim = primitive->cast<PrimUnstackPtr>();
  MS_EXCEPTION_IF_NULL(unstack_prim);
  auto prim_name = unstack_prim->name();
  CheckAndConvertUtils::CheckSubClass("x", input_args[0]->BuildType(), {TypeIdToType(kObjectTypeTensorType)},
                                      prim_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  int64_t dim = x_shape.size();
  int64_t axis = unstack_prim->get_axis();
  //  CheckAndConvertUtils::CheckInRange("axis value", axis, kIncludeLeft, {-dim, dim}, prim_name);
  if (axis < 0) {
    axis = axis + dim;
  }
  auto output_num = x_shape[axis];
  CheckAndConvertUtils::CheckInteger("output_num", output_num, kGreaterThan, 0, prim_name);
  auto output_valid_check = x_shape[axis] - output_num;
  CheckAndConvertUtils::CheckInteger("The dimension which to unstack divides output_num", output_valid_check, kEqual, 0,
                                     prim_name);
  std::vector<int64_t> infer_shape(x_shape.begin(), x_shape.begin() + axis);
  infer_shape.insert(infer_shape.end(), x_shape.begin() + axis + 1, x_shape.end());
  AbstractBasePtrList output;
  auto tensor_type = input_args[0]->BuildType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto element = tensor_type->element();
  for (int64_t i = 0; i != output_num; i++) {
    output.push_back(std::make_shared<abstract::AbstractTensor>(element, infer_shape));
  }
  return std::make_shared<abstract::AbstractTuple>(output);
}
REGISTER_PRIMITIVE_C(kNameUnstack, Unstack);
}  // namespace ops
}  // namespace mindspore
