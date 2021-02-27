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

#include "ops/op_utils.h"
#include "ops/reverse_v2.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto reverseV2_prim = primitive->cast<PrimReverseV2Ptr>();
  MS_EXCEPTION_IF_NULL(reverseV2_prim);
  auto prim_name = reverseV2_prim->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  //  auto axis = reverseV2_prim->get_axis();
  //  int dim = x_shape.size();
  //  for (auto &axis_value : axis) {
  //    CheckAndConvertUtils::CheckInRange("axis value", axis_value, kIncludeLeft, {-dim, dim}, prim_name);
  //  }
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypeId> valid_types = {kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,   kNumberTypeInt64,
                                        kNumberTypeUInt8,   kNumberTypeUInt16,  kNumberTypeUInt32,  kNumberTypeUInt64,
                                        kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeBool};
  auto infer_type = input_args[0]->BuildType();
  CheckAndConvertUtils::CheckTensorTypeValid("x type", infer_type, valid_types, prim->name());
  MS_EXCEPTION_IF_NULL(infer_type);
  auto tensor_type = infer_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto data_type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(data_type);
  return data_type;
}
}  // namespace

void ReverseV2::Init(const std::vector<int64_t> &axis) { this->set_axis(axis); }
void ReverseV2::set_axis(const std::vector<int64_t> &axis) { this->AddAttr(kAxis, MakeValue(axis)); }
std::vector<int64_t> ReverseV2::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

AbstractBasePtr ReverseV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameReverseV2, ReverseV2);
}  // namespace ops
}  // namespace mindspore
