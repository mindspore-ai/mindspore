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

#include "ops/constant_of_shape.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  (void)CheckAndConvertUtils::CheckInteger("input args size", SizeToLong(input_args.size()), kEqual, 1,
                                           "ConstantOfShape");
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  return std::make_shared<abstract::Shape>(input_shape);
}

TypePtr InferType(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto data_type = TypeId(GetValue<int64_t>(primitive->GetAttr(kDataType)));
  return TypeIdToType(data_type);
}
}  // namespace

void ConstantOfShape::Init(int64_t data_type, const std::vector<float> &value) {
  this->set_data_type(data_type);
  this->set_value(value);
}

void ConstantOfShape::set_data_type(int64_t data_type) { (void)this->AddAttr(kDataType, MakeValue(data_type)); }

int64_t ConstantOfShape::get_data_type() const {
  auto value_ptr = this->GetAttr(kDataType);
  return GetValue<int64_t>(value_ptr);
}

void ConstantOfShape::set_value(const std::vector<float> &value) { (void)this->AddAttr(kValue, MakeValue(value)); }

std::vector<float> ConstantOfShape::get_value() const {
  auto value_ptr = this->GetAttr(kValue);
  return GetValue<std::vector<float>>(value_ptr);
}
AbstractBasePtr ConstantOfShapeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive), InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameConstantOfShape, ConstantOfShape);
}  // namespace ops
}  // namespace mindspore
