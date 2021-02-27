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

#include <map>
#include <string>
#include "ops/one_hot.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void OneHot::Init(const int64_t axis) { this->set_axis(axis); }
void OneHot::set_axis(const int64_t axis) { this->AddAttr(kAxis, MakeValue(axis)); }

int64_t OneHot::get_axis() const {
  auto value_ptr = this->GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}
namespace {
abstract::ShapePtr OneHotInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto OneHot_prim = primitive->cast<PrimOneHotPtr>();
  MS_EXCEPTION_IF_NULL(OneHot_prim);
  auto op_name = OneHot_prim->name();
  int64_t axis = OneHot_prim->get_axis();
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShape("input_shape", input_args[0]->BuildShape(), op_name);
  CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis, kIncludeBoth, {-1, SizeToLong(in_shape.size())}, op_name);
  auto depth_val = GetValue<int64_t>(input_args[1]->BuildValue());
  CheckAndConvertUtils::CheckInteger("depth", depth_val, kGreaterEqual, 0, op_name);
  if (axis >= 0) {
    in_shape.insert(in_shape.begin() + axis, depth_val);
  } else {
    in_shape.push_back(depth_val);
  }
  return std::make_shared<abstract::Shape>(in_shape);
}

TypePtr OneHotInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto OneHot_prim = prim->cast<PrimOneHotPtr>();
  MS_EXCEPTION_IF_NULL(OneHot_prim);
  auto op_name = OneHot_prim->name();
  CheckAndConvertUtils::CheckTensorTypeValid("indices", input_args[0]->BuildType(), {kNumberTypeInt32}, op_name);
  CheckAndConvertUtils::CheckTypeSame("depth", input_args[1]->BuildType(),
                                      {kNumberTypeInt8, kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64}, op_name);
  auto value_type = input_args[2]->BuildType();
  auto tensor_type = value_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto element = tensor_type->element();
  MS_EXCEPTION_IF_NULL(element);
  std::map<std::string, TypePtr> args = {{"on_value", value_type}, {"off_dtype", input_args[3]->BuildType()}};
  CheckAndConvertUtils::CheckTensorTypeSame(args, {kNumberTypeFloat16, kNumberTypeFloat32}, op_name);
  return element;
}
}  // namespace
AbstractBasePtr OneHotInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(OneHotInferType(primitive, input_args),
                                                    OneHotInferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameOneHot, OneHot);
}  // namespace ops
}  // namespace mindspore
