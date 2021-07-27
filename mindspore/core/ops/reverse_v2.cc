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
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,   kUInt16,
                                         kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kBool};
  auto infer_type = input_args[0]->BuildType();
  return CheckAndConvertUtils::CheckTensorTypeValid("x type", infer_type, valid_types, prim->name());
}
}  // namespace

void ReverseV2::Init(const std::vector<int64_t> &axis) { this->set_axis(axis); }
void ReverseV2::set_axis(const std::vector<int64_t> &axis) { (void)this->AddAttr(kAxis, MakeValue(axis)); }
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
