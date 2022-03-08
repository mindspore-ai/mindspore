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

#include "ops/dynamic_quant.h"

namespace mindspore {
namespace ops {
void DynamicQuant::set_symmetric(const bool symmetric) { (void)AddAttr(kSymmetric, MakeValue(symmetric)); }
bool DynamicQuant::get_symmetric() const {
  auto value_ptr = this->GetAttr(kSymmetric);
  return GetValue<bool>(value_ptr);
}
void DynamicQuant::set_dst_type(const int64_t dst_type) { (void)AddAttr(kDstType, MakeValue(dst_type)); }
int64_t DynamicQuant::get_dst_type() const { return GetValue<int64_t>(GetAttr(kDstType)); }
void DynamicQuant::Init(const bool symmetric, const int64_t dst_type) {
  this->set_symmetric(symmetric);
  this->set_dst_type(dst_type);
}
AbstractBasePtr DynamicQuantInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  const size_t x_index = 0;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto input_type = CheckAndConvertUtils::GetTensorInputType(primitive->name(), input_args, x_index);
  auto dst_type = TypeIdToType(TypeId(GetValue<int64_t>(primitive->GetAttr(kDstType))));
  MS_EXCEPTION_IF_NULL(dst_type);
  if (input_type->type_id() != kNumberTypeFloat16 && input_type->type_id() != kNumberTypeFloat32) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                            << "', Input type should be kNumberTypeFloat16 or kNumberTypeFloat32"
                            << ", but " << input_type->ToString();
  }
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  return std::make_shared<abstract::AbstractTensor>(dst_type, input_shape);
}
REGISTER_PRIMITIVE_C(kNameDynamicQuant, DynamicQuant);
}  // namespace ops
}  // namespace mindspore
