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
#include "ops/topk.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void TopK::Init(const bool sorted) { this->set_sorted(sorted); }
void TopK::set_sorted(const bool sorted) { (void)this->AddAttr(kSorted, MakeValue(sorted)); }

bool TopK::get_sorted() const {
  auto value_ptr = this->GetAttr(kSorted);
  return GetValue<bool>(value_ptr);
}
AbstractBasePtr TopKInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("top_k_infer", SizeToLong(input_args.size()), kEqual, input_num, prim_name);

  // Infer dtype
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto output1_type = kInt32;
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  auto output0_type =
    CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_args[0]->BuildType(), valid_types, prim_name);

  // Infer shape
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto k_v = GetValue<int>(input_args[1]->BuildValue());
  auto ndims = x_shape.size() - 1;
  x_shape[ndims] = k_v;

  auto output0 = std::make_shared<abstract::AbstractTensor>(output0_type, x_shape);
  auto output1 = std::make_shared<abstract::AbstractTensor>(output1_type, x_shape);
  AbstractBasePtrList output = {output0, output1};
  return std::make_shared<abstract::AbstractTuple>(output);
}
REGISTER_PRIMITIVE_C(kNameTopK, TopK);
}  // namespace ops
}  // namespace mindspore
