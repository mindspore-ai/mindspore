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

#include "ops/quant_dtype_cast.h"

namespace mindspore {
namespace ops {
void QuantDTypeCast::set_src_t(const int64_t src_t) { (void)AddAttr(kSrcT, MakeValue(src_t)); }
int64_t QuantDTypeCast::get_src_t() const {
  auto value_ptr = this->GetAttr(kSrcT);
  return GetValue<int64_t>(value_ptr);
}
void QuantDTypeCast::set_dst_t(const int64_t dst_t) { (void)AddAttr(kDstT, MakeValue(dst_t)); }
int64_t QuantDTypeCast::get_dst_t() const { return GetValue<int64_t>(GetAttr(kDstT)); }
void QuantDTypeCast::Init(const int64_t src_t, const int64_t dst_t) {
  this->set_src_t(src_t);
  this->set_dst_t(dst_t);
}
AbstractBasePtr QuantDTypeCastInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  const int64_t x_index = 0;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto input_type = CheckAndConvertUtils::GetInputTensorType(input_args, x_index, primitive->name());
  auto dst_type = TypeIdToType(TypeId(GetValue<int64_t>(primitive->GetAttr(kDstT))));
  MS_EXCEPTION_IF_NULL(dst_type);
  if (input_type != dst_type) {
    MS_EXCEPTION(TypeError) << "Input type should be " << dst_type->ToString() << ", but " << input_type->ToString();
  }
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  return std::make_shared<abstract::AbstractTensor>(dst_type, input_shape);
}
REGISTER_PRIMITIVE_C(kNameQuantDTypeCast, QuantDTypeCast);
}  // namespace ops
}  // namespace mindspore
