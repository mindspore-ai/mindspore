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

#include "ops/quant_dtype_cast.h"

namespace mindspore {
namespace ops {
void QuantDTypeCast::set_src_t(const int64_t src_t) { AddAttr(kSrcT, MakeValue(src_t)); }
int64_t QuantDTypeCast::get_src_t() const {
  auto value_ptr = this->GetAttr(kSrcT);
  return GetValue<int64_t>(value_ptr);
}
void QuantDTypeCast::set_dst_t(const int64_t dst_t) { AddAttr(kDstT, MakeValue(dst_t)); }
int64_t QuantDTypeCast::get_dst_t() const {
  auto value_ptr = this->GetAttr(kDstT);
  return GetValue<int64_t>(value_ptr);
}
void QuantDTypeCast::Init(const int64_t src_t, const int64_t dst_t) {
  this->set_src_t(src_t);
  this->set_dst_t(dst_t);
}
AbstractBasePtr QuantDTypeCastInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto QuantDTypeCast_prim = primitive->cast<PrimQuantDTypeCastPtr>();
  MS_EXCEPTION_IF_NULL(QuantDTypeCast_prim);
  auto op_name = QuantDTypeCast_prim->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto input_type = input_args[0]->BuildType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_type);
  MS_ASSERT(input_type->element() == TypeIdToType(TypeId(QuantDTypeCast_prim->get_dst_t())));
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShape("input_shape", input_args[0]->BuildShape(), op_name);
  return std::make_shared<abstract::AbstractTensor>(TypeIdToType(TypeId(QuantDTypeCast_prim->get_dst_t())),
                                                    input_shape);
}
REGISTER_PRIMITIVE_C(kNameQuantDTypeCast, QuantDTypeCast);
}  // namespace ops
}  // namespace mindspore
